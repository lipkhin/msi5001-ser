"""Minimal AESDD baseline: six sklearn defaults + a small, untuned CNN."""
# %% 1. Imports and configuration
from pathlib import Path
import re, zipfile, hashlib
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import gdown
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

SOURCE = 'local'                         # 'local' or 'drive'
LOCAL_PATH = r'./AESDD'                   # WAV directory, ZIP, or folder of ZIPs
DRIVE_URL = 'PASTE_PUBLIC_FILE_OR_FOLDER_URL_HERE'
EPOCHS, BATCH_SIZE = 10, 32               # Required CNN baseline choices, not tuned
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CNN device:', DEVICE)

# %% 2. Load local data or download a public Google Drive copy (no mounting)
root = Path(LOCAL_PATH).expanduser().resolve()
if SOURCE == 'drive':
    root = Path('./downloaded_AESDD').resolve()
    root.mkdir(exist_ok=True)
    if '/folders/' in DRIVE_URL:
        downloaded = gdown.download_folder(DRIVE_URL, output=str(root), use_cookies=False)
    else:
        downloaded = gdown.download(DRIVE_URL, output=str(root/'dataset.zip'), use_cookies=False)
    if not downloaded:
        raise RuntimeError('Download failed. Use a public anyone-with-the-link Drive URL.')
elif SOURCE != 'local':
    raise ValueError("SOURCE must be 'local' or 'drive'.")
if not root.exists():
    raise FileNotFoundError(f'Edit LOCAL_PATH: {root}')

def unzip(path):
    destination = path.with_suffix('')
    if not (destination/'.extracted').exists():
        with zipfile.ZipFile(path) as archive:
            if any(not (destination/name).resolve().is_relative_to(destination.resolve())
                   for name in archive.namelist()):
                raise ValueError('Unsafe archive path')
            archive.extractall(destination)
        (destination/'.extracted').touch()
    return destination

if root.is_file():
    root = unzip(root)
for _ in range(3):                       # Also handles ZIPs inside a downloaded folder
    for archive in list(root.rglob('*.zip')):
        unzip(archive)

# %% 3. Read WAVs: 40 MFCC statistics for ML; log-mel images for CNN
codes = {'a':'Anger', 'd':'Disgust', 'f':'Fear', 'h':'Happiness', 's':'Sadness'}
pattern = re.compile(r'([adfhs])\s*[_ -]?(\d{1,2})\s*[_ -]?\(\s*(\d+)\s*\)\s*[a-z]*', re.I)
rows, features, images, seen = [], [], [], set()
for path in sorted(root.rglob('*.wav')):
    match = pattern.fullmatch(path.stem)
    if not match or '__MACOSX' in path.parts:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen:
        continue
    try:
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=16000)
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-7:
            raise ValueError('Empty, invalid or silent audio')
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
        mel = torch.tensor(librosa.power_to_db(mel), dtype=torch.float32)[None, None]
        image = nn.functional.interpolate(mel, size=(64,128), mode='bilinear', align_corners=False)[0].numpy()
    except (RuntimeError, ValueError, OSError) as error:
        print('Skipped:', path.name, error)
        continue
    emotion, utterance, speaker = match.groups()
    emotion, speaker = emotion.lower(), str(int(speaker))
    rows.append({'emotion':codes[emotion], 'speaker':speaker,
                 'group':f'{speaker}_{emotion}_{int(utterance)}'})
    features.append(np.r_[mfcc.mean(axis=1), mfcc.std(axis=1)])
    images.append(image)
    seen.add(digest)
if not rows:
    raise ValueError('No valid AESDD WAVs found. Check the dataset path and filenames.')
metadata = pd.DataFrame(rows)
X, Z = np.asarray(features), np.asarray(images)
encoder = LabelEncoder()
y = encoder.fit_transform(metadata.emotion)
print('Valid recordings:', len(y), '| features:', X.shape[1], '| speakers:', metadata.speaker.nunique())

# %% 4. Six default classifiers and a small untuned CNN
models = {'DummyClassifier':DummyClassifier(), 'Logistic Regression':LogisticRegression(),
          'KNN':KNeighborsClassifier(), 'SVM':SVC(),
          'Random Forest':RandomForestClassifier(), 'MLP':MLPClassifier()}

def cnn_predict(train, test):
    torch.manual_seed(42)
    mean, std = Z[train].mean(), max(float(Z[train].std()), 1e-8)
    inputs = torch.tensor((Z[train]-mean)/std)
    targets = torch.tensor(y[train], dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=BATCH_SIZE, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32,len(encoder.classes_))
    ).to(DEVICE)
    optimizer, loss_fn = torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor((Z[test[i:i+64]]-mean)/std).to(DEVICE))
                               .argmax(1).cpu().numpy() for i in range(0,len(test),64)])

# %% 5. One stratified group random split and six LOSO folds; no tuning
train, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                   .split(X, y, groups=metadata.group))
splits = [('Random', train, test)]
splits += [('LOSO', tr, te) for tr, te in LeaveOneGroupOut().split(X, y, metadata.speaker)]
predictions = []
for setting, train, test in splits:
    print(setting, '| test speakers:', ','.join(metadata.speaker.iloc[test].unique()))
    for name, estimator in models.items():
        np.random.seed(42)
        model = make_pipeline(StandardScaler(), clone(estimator))
        model.fit(X[train], y[train])
        predictions.append((setting, name, y[test], model.predict(X[test])))
    predictions.append((setting, 'CNN', y[test], cnn_predict(train, test)))

# %% 6. Accuracy and macro F1 (LOSO pooled over all held-out speakers)
results = []
for name in [*models, 'CNN']:
    row = {'Model':name}
    for setting in ['Random', 'LOSO']:
        selected = [p for p in predictions if p[0]==setting and p[1]==name]
        actual = np.concatenate([p[2] for p in selected])
        predicted = np.concatenate([p[3] for p in selected])
        row[f'{setting} accuracy'] = accuracy_score(actual, predicted)
        row[f'{setting} macro F1'] = f1_score(actual, predicted, average='macro',
                                            labels=np.arange(len(encoder.classes_)), zero_division=0)
    results.append(row)
summary = pd.DataFrame(results).set_index('Model')
print(summary.round(3).to_string())
summary.to_csv('SER_Minimal_Results.csv')
