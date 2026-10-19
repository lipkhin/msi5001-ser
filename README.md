# MSI5001 2026/2027 Group Project - Speech Emotion Recognition (SER)

## Team Members

- Chow Lip Khin
- Manish Agarwal
- Tan Sumengyuan
- Xu Jin

## Project Brief

Speech conveys both the words being spoken and information about the speaker's emotional expression. This project investigates how machine learning can classify emotions from speech recordings using the AESDD dataset. The classification task covers five emotions: **Anger, Disgust, Fear, Happiness, and Sadness**.

We compare seven models, ranging from a simple reference baseline and classical machine learning methods to an artificial neural network and a convolutional neural network. The project follows a complete experimental workflow: loading and cleaning audio, exploring the dataset, extracting features, splitting and scaling data, training models, and evaluating predictions using **accuracy** and **macro F1**.

A key objective is to examine how well the models generalise to different speakers. We compare five-fold stratified group random cross-validation, where speakers can appear in both training and testing, with **Leave-One-Speaker-Out (LOSO)** evaluation, where the test speaker is excluded from training. The project documents the progression from a predefined richer-feature baseline with default model settings to validation-based model optimisation.

## Running Locally or in Google Colab

### Local: VS Code

1. Clone this repository and open its folder in VS Code.
2. Install the Microsoft **Python** and **Jupyter** extensions.
3. Create or select a Python environment and choose it as the notebook kernel.
4. Install the dependencies using the optional cell in section 1, then restart the kernel after upgrades if required. If the environment is already configured, skip that cell. For NVIDIA GPU execution, install a compatible CUDA-enabled PyTorch build using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
5. Set `SOURCE = 'local'` and point `LOCAL_PATH` to your AESDD directory or ZIP. The default path is `./AESDD` relative to the notebook's working directory.
6. Set `RUN_MODE='train'` to train/save, or `RUN_MODE='load'` to reproduce an existing completed archive, then run sections 1–11 in order.

### Google Colab

1. Open the notebook from this repository in Colab.
2. Select a GPU runtime for CNN and both baseline/tuned PyTorch MLP training if required.
3. Set `SOURCE = 'drive'` and paste your public dataset folder or ZIP URL into `DRIVE_URL`. The sharing setting must allow **Anyone with the link → Viewer**.
4. Run the required installation cell, then run the remaining sections in order.

Drive downloads are copied to the runtime's local storage. Local CPU execution is also supported.

## Models

| Model | Role |
| --- | --- |
| DummyClassifier | Reference baseline that ignores the input features |
| K-Nearest Neighbours (KNN) | Distance-based classifier |
| Logistic Regression | Linear classifier |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Nonlinear classifier using the default RBF kernel |
| Multilayer Perceptron (MLP) | Artificial neural network |
| Convolutional Neural Network (CNN) | Neural network using 40 MFCC sequences only in basic mode, or log-mel only in rich mode |

Sections 1–8 establish a **predefined baseline using audio features**. The five sklearn models use default settings; MLP uses a fixed PyTorch configuration, and CNN uses a fixed architecture with training duration calibrated by inner validation. Section 9 implements validation-based hyperparameter tuning; section 10 compares baseline and optimised predictions; section 11 verifies model saving/loading. The five scikit-learn classifiers retain their default constructor settings. Baseline MLP uses the same PyTorch estimator as tuning: one 100-unit hidden layer, ReLU, Adam, alpha=0.0001, LR=0.001 and a shared 1500-iteration upper limit with training-loss stopping. The predefined CNN uses two convolution layers (16/32 channels), BatchNorm, Dense 64, dropout 0.4, Adam LR=0.001 and weight_decay=0.0001, batch size 32, and the same 50-epoch maximum/checkpoint/stopping rules as tuned CNNs. CUDA is selected automatically for CNN and both baseline/tuned PyTorch MLP when available. The five scikit-learn baseline models run on CPU. Finished PyTorch MLP models and predictions use CPU.

## Notebook Workflow

Open [MSI5001_SER.ipynb](MSI5001_SER.ipynb) and run sections 1–11 in order:

1. **Install dependency & import library** — prepare the Python environment.
2. **Load dataset** — read a local WAV directory or ZIP, or download a local copy from a public Google Drive link without mounting Drive.
3. **Explore and clean dataset** — remove byte-identical duplicates and skip damaged, empty, invalid, or silent WAVs; inspect tables and graphs; listen to one randomly selected recording per emotion.
4. **Data preprocessing** — convert audio to mono, resample to 16 kHz, trim quiet leading/trailing regions for every recording, and choose basic (40 MFCC summary features + a single-channel CNN input of 40 MFCC sequences) or rich (120 acoustic features + log-mel-only CNN input).
5. **Train test split (random & LOSO)** — prepare all five random CV folds and all six speaker-held-out LOSO folds.
6. **Feature scaling** — fit scaling statistics separately on each split's training data and apply them to its training and test inputs.
7. **Model training** — train fresh model instances for every split; define shared CNN/inner-split settings and calibrate the fixed two-layer CNN duration using only outer training data.
8. **Model prediction and evaluation** — display pooled out-of-fold baseline accuracy/macro F1, fold statistics, per-speaker LOSO baseline accuracy, training/test diagnostics and charts.
9. **Model Optimisation & Hyperparameter Tuning** — preserve baseline results, set up inner validation, and tune each model:
   - **9.1 Baseline results**
   - **9.2 Optimisation method**
   - **9.3 KNN optimisation**
   - **9.4 Logistic Regression optimisation**
   - **9.5 Random Forest optimisation**
   - **9.6 SVM optimisation**
   - **9.7 MLP optimisation**
   - **9.8 CNN optimisation**
10. **Final model prediction and evaluation** — compare both stages on the same outer test recordings, including pooled score changes, fold mean ± standard deviation, per-speaker LOSO accuracy/macro F1 and charts.
11. **Save models and reproduce results** — reload selected fold models, regenerate held-out predictions and verify that they match the saved run.

### Quiet-edge trimming

Section 4 applies the same fixed preprocessing to every speaker in both basic and rich modes: mono conversion, resampling to 16 kHz, then `librosa.effects.trim(audio, top_db=30, frame_length=400, hop_length=160)`. The threshold is 30 dB below each recording's maximum frame RMS, using 25 ms frames with a 10 ms hop. Only leading/trailing quiet regions are removed; internal pauses remain. Both ML and CNN features are extracted from the retained audio. Empty trimmed audio raises an error so feature rows and labels cannot silently become misaligned.

A table shows mean original/trimmed durations and leading/trailing amounts removed for each speaker. `metadata.duration_s` retains original file duration; `trimmed_duration_s`, `leading_trim_s` and `trailing_trim_s` describe the model input. Pause statistics use the trimmed audio duration. This per-recording rule does not learn statistics from other recordings. Re-run the section 1 configuration cell and **sections 4–11** after changing trimming or feature settings; keep the outer folds and seeds fixed for before/after comparisons. Improved accuracy is not guaranteed.

### Input Features

One selector controls both the acoustic feature vectors (including MLP) and CNN inputs:

| `FEATURE_SET` | Feature-vector models (including MLP) | CNN input per recording |
| --- | --- | --- |
| `'basic'` | **40 MFCC mean/std features** | **40 MFCC sequences only: 1 × 40 × 128** |
| `'rich'` (default) | **120 acoustic features** | **Log-mel only: 1 × 64 × 128** |

Select the preset in the dedicated **Run configuration** code cell after imports in section 1:

```python
FEATURE_SET = 'rich'  # 'basic' or 'rich'
```

`basic` uses only the MFCC row below and skips the other acoustic feature extraction. `rich` uses every family. Both modes use the same recordings, evaluation splits and training budgets. Both stages use all features from the selected preset; tuning changes model settings only. The former `'mfcc'` selector is now called `'basic'`.

| Category | Feature | Summary statistics | Count | What it captures | Possible emotional relevance |
| --- | --- | --- | ---: | --- | --- |
| Cepstral | MFCC | Mean/std of 20 coefficients | 40 | Voice timbre and variation in the spectral envelope | Can describe timbre differences across all five emotions; individual coefficients have no simple emotion label. |
| | Delta MFCC | Std of 20 first-order delta coefficients | 20 | Changes in the spectral envelope over time | May distinguish changing or abrupt delivery from steadier delivery; also captures phonetic changes. |
| Spectral | Spectral centroid | Mean/std | 2 | Centre of spectral energy; spectral brightness | May describe brighter versus darker delivery; emotion patterns overlap. |
| | Spectral bandwidth | Mean/std | 2 | Spread of energy around the centroid | May describe differences in spectral spread associated with articulation and delivery. |
| | Spectral roll-off | Mean/std | 2 | Frequency below which 85% of spectral energy lies | May describe differences in upper-frequency energy; not specific to any one emotion. |
| | Spectral flatness | Mean/std | 2 | Noise-like versus tonal spectral structure | May describe voice texture; noise and silence also affect it, so it is not a direct breathiness measure. |
| | Spectral contrast | Mean/std across 7 frequency bands | 14 | Differences between spectral peaks and valleys | Adds spectral structure that may help separate overlapping emotion classes; no universal higher/lower direction. |
| Prosodic | Pitch (F0) | Mean, std, P5, P95 and P5–P95 range in semitones | 5 | Pitch height and variation | Helps describe arousal and intonation; see the emotion examples below. |
| | Energy | Active-frame RMS dBFS: mean, std, P5 and P95 | 4 | Digital signal level and its variation | May distinguish forceful and subdued delivery; recording gain and normalisation also affect it. |
| | Timing and voicing | Pause ratio, pauses per second and voiced-frame ratio | 3 | Pausing and proportion of detected voiced speech | May distinguish interrupted and continuous delivery; patterns depend on the speaker and spoken content. |
| Harmonic | Chroma | Mean/std across 12 pitch classes | 24 | Distribution and variation of pitch-class energy | Candidate complementary pitch-class cues for all five emotions; no fixed pitch class identifies an emotion. |
| Time-domain | Zero-crossing rate | Mean/std | 2 | Frequency of waveform sign changes | May describe noisy/unvoiced articulation; a high value is not uniquely associated with any emotion. |
| **Total** | | | **120** | | |

| Category | Count |
| --- | ---: |
| Cepstral | 60 |
| Spectral | 22 |
| Prosodic | 12 |
| Harmonic | 24 |
| Time-domain | 2 |
| **Total** | **120** |

The following table is an **interpretation guide, not measured AESDD results or classification rules**. The broad prosodic tendencies are informed by vocal-emotion research; links to the added descriptors are hypotheses to assess through training-data validation. Expression intensity, speaker, language, phonetic content and recording conditions can change these patterns.

| Emotion | Possible pitch, energy and timing patterns | How the additional descriptors could help |
| --- | --- | --- |
| Anger | Intense anger can involve elevated pitch, stronger delivery and abrupt changes; restrained anger can differ. | Delta MFCCs may capture transitions; ZCR, flatness and contrast may add information about the spectral texture of forceful delivery. |
| Fear | High pitch is a common candidate cue; energy and timing vary between anxious and panicked delivery. | Delta MFCCs may describe changing articulation; ZCR, flatness and contrast may distinguish spectral texture from equally high-pitched happiness. |
| Happiness | Excited happiness can have elevated, varied pitch and animated delivery; quiet happiness can differ. | Delta MFCCs may describe expressive changes; ZCR, flatness and contrast may help distinguish it from anger or fear with similar arousal. |
| Sadness | Subdued sadness can involve lower, less varied pitch, weaker delivery and slower timing; distressed crying can differ. | Delta MFCCs may describe steadier delivery; ZCR, flatness and contrast provide complementary texture cues without a fixed expected direction. |
| Disgust | Voice quality and articulation can change, but pitch, level and timing do not have a reliable universal direction. | MFCCs, ZCR, flatness and contrast may capture those timbre differences; delta MFCCs may capture changes in their delivery. |

No individual feature identifies an emotion. The classifiers learn combinations of features from labelled recordings, and speaker-separated validation is needed to check whether those combinations generalise.

Chroma extraction uses `tuning=0.0`: a fixed A440 reference that bypasses the automatic tuning-estimation path responsible for the reported Numba cast warning. It does not change the audio waveform or the 120-feature count.

### CNN input channels

| Channel | Included in | What it captures |
| --- | --- | --- |
| 40 MFCC coefficients per frame | Basic | Cepstral representation of spectral shape over time |
| Log-mel | Rich | Energy across 64 mel-frequency bands over time |

The **CNN basic input is a sequence**, not the 40-value summary vector used by classical ML. Compute a 64-band log-mel spectrum at 16 kHz, then retain 40 MFCC coefficients using DCT type 2 and orthonormal scaling. Keep these MFCC sequences as a single channel, retaining all 40 coefficient rows and resizing only time to 128 bins: **1 × 40 × 128**. No delta or delta-delta channels are added to the CNN input.

The **CNN rich input is log-mel only**, with 64 mel bands and time resized to 128 bins: **1 × 64 × 128**. Both presets use `n_fft=2048`, `hop_length=512`, a Hann window, power=2 and dB conversion with ref=1, amin=1e-10 and top_db=80. The basic/rich names select presets; they do not imply a guaranteed CNN performance ranking. These CNN changes do not alter the classical 40/120-feature definitions.

Section 4 displays the selected input shapes and plots every CNN channel. Both baseline and tuned CNNs infer the input channel count automatically. Fit one mean and standard deviation **per channel**, across training recordings, coefficient/frequency rows and time bins. Apply those statistics to the corresponding validation/test inputs; clip zero standard deviations to `1e-8`. Inner validation fits its own training-channel statistics, and the selected CNN's final refit stores statistics from all outer training images.

## Evaluation

| Protocol | How it works | What it assesses |
| --- | --- | --- |
| Five-fold stratified group random CV | Use all five folds from a shuffled `StratifiedGroupKFold` (seed 42). Each fold trains on approximately 80% and tests on 20%. Group by speaker, emotion, and utterance to keep alternate takes together. | Performance when speakers may be shared between training and testing |
| LOSO | Hold out one complete speaker, train on the remaining speakers, and repeat for every speaker: six folds for AESDD. | Generalisation to unseen speakers |

Both protocols test **every recording exactly once**. Random CV has five outer folds; LOSO has six outer folds because AESDD has six speakers. This creates **11 separately scaled and trained outer splits**. Baseline and optimised models use identical outer folds. Class proportions in random CV are preserved as closely as grouping constraints allow. LOSO follows speaker boundaries and does not redistribute recordings to enforce stratification.

The primary comparison reports **pooled out-of-fold scores for both protocols**: combine the held-out predictions from all five random folds, or all six LOSO folds, then calculate accuracy and macro F1 once. Each recording has one prediction per model, protocol and stage. The notebook checks this coverage explicitly and retains recording indices to align baseline and optimised predictions.

A second table reports the **unweighted mean ± sample standard deviation (`ddof=1`) of fold accuracy and macro F1**, using five random-fold values and six LOSO-fold values. Pooled macro F1 differs from mean fold macro F1; pooled accuracy also differs from the unweighted fold mean when fold sizes differ. Fold standard deviations describe variation across folds and are not confidence intervals. Macro F1 gives each emotion equal weight. Charts and improvement tables use pooled scores.

Section 8 shows baseline LOSO accuracy separately for every held-out speaker, using the Model/Stage row layout from section 10. Section 10 shows both stages' per-speaker accuracy and macro F1. Class IDs for per-speaker F1 are generated explicitly from the encoder, and chart annotations use `bar_labels`, so rerunning the evaluation cell cannot replace class IDs with formatted bar text.

## Model Optimisation & Hyperparameter Tuning

Section 9 tunes classical model settings using all features from the selected preset, and compares CNN architectures with validation-based training. It snapshots the section 8 baseline and uses the same outer test recordings for both stages. The CNN baseline is the predefined two-layer model with validation-calibrated training; sklearn baselines retain default settings. DummyClassifier remains unchanged. Improvement is not guaranteed; negative changes are retained.

### Validation and model selection

Every outer training split gets its own tuning run: **five random runs and six LOSO runs**.

| Outer protocol | Default inner validation | What stays separate |
| --- | --- | --- |
| Random | **4 shuffled stratified group folds** inside each outer training fold | Alternate takes stay together; the outer test clips are excluded |
| LOSO | **All remaining speakers**, one inner validation speaker at a time: **5 holdouts for AESDD** | The outer test speaker never enters tuning; inner training uses the other 4 speakers |

Defaults, defined in section 7 and reused in 9.2, are `INNER_RANDOM_FOLDS = 4` and `INNER_LOSO_FOLDS = None`. The same inner folds are shared across models. Section 9.2 displays the validation plan and speaker coverage; with full inner LOSO, each AESDD speaker serves as validation in five eligible outer runs. Reducing the inner holdout count is possible but reduces speaker coverage.

Select model settings by **mean inner-validation macro F1**, with accuracy retained alongside it. Each outer fold can select a different configuration. Feature-vector searches use `Pipeline(StandardScaler, model)` on raw X, retaining every feature from the selected preset. KNN, Logistic Regression, Random Forest and SVM use exhaustive grids. Baseline and tuned MLP use the same cloneable PyTorch estimator; tuning samples two Adam/ReLU configurations per architecture. Each inner fold fits its own scaler; the selected pipeline then refits on all outer training clips. CNN channel normalisation is also fitted only on the appropriate training images.

### Inputs used during tuning

Keep `FEATURE_SET = 'basic'` or `'rich'` in the section 1 configuration cell. Every classical model uses all **40 MFCC features** in basic mode or all **120 acoustic features** in rich mode, throughout baseline training and optimisation. The feature preset stays fixed across models and outer folds; there is no automatic feature-group selection. The same toggle sets CNN input to 40 MFCC sequences only in basic mode, or log-mel only in rich mode.

### Implemented searches

Candidate counts are the same in basic and rich modes. Sklearn baseline constructors remain unchanged. The CNN baseline is a predefined, validation-calibrated 16/32-channel model, evaluated separately from the four configurations of two larger tuning architectures.

| Subsection | Model | Search and settings | Candidates per outer split |
| --- | --- | --- | ---: |
| 9.3 | KNN | Grid: k=3/5/7/9/11 × uniform/distance × Manhattan/Euclidean | 20 |
| 9.4 | Logistic Regression | Grid: C=0.001/0.005/0.01/0.05/0.1/0.5/1/5/10/50/100/200, all at max_iter=1500 | 12 |
| 9.5 | Random Forest | Grid: n_estimators=100/300 × max_depth=None × min_samples_leaf=1/2 × max_features=sqrt/0.5 | 8 |
| 9.6 | SVM | Grid: 10 C values × 7 gamma values for RBF (70); 10 C values for linear | 80 |
| 9.7 | PyTorch MLP | Adam/ReLU only: 2 sampled alpha/LR settings per architecture across five architectures | 10 |
| 9.8 | CNN | Two-layer (32/64) or three-layer (32/64/128), both Dense 128 × augmentation off/on; fixed initial LR=0.001 and weight_decay=0.0001 | 4 |

### PyTorch MLP architecture and randomised search

Baseline and optimisation use the **same PyTorch MLP implementation**. Section 7 defines the shared cloneable estimator and fits a fixed configuration: `(100,)`, ReLU, Adam, alpha=0.0001 and initial LR=0.001. Both stages share `MLP_MAX_ITER=1500`, `MLP_DEVICE='auto'`, batches of min(200, N_train), training-loss tolerance 0.0001 and patience 10 for Adam. Section 9.7 reuses the class and shared settings rather than redefining them. No skorch dependency is required.

The baseline uses scaled outer-training features; tuning fits a separate scaler inside each inner-training fold and refits its selected pipeline on all outer training clips. Training-loss stopping creates no internal validation holdout.

| Setting | Search space |
| --- | --- |
| Hidden layers | `(64,)`, `(128,)`, `(256,)`, `(64, 32)`, `(128, 64)` |
| Activation | ReLU only |
| Optimiser | Adam only |
| L2 alpha | 0.0001, 0.001, 0.01 |
| Adam learning rate | 0.0003, 0.001 |
| Training limit | 1500 Adam epochs |

There are **30 possible configurations**: five architectures × three alpha values × two learning rates. `ParameterSampler` draws **2 distinct configurations per architecture**, without replacement, for **10 candidates**. A single random stream seeded at 42 advances between architectures; the resulting candidate list is reused across all outer folds. Each architecture receives an equal budget covering two of its six combinations; individual alpha/LR values are not guaranteed coverage. Singleton parameter grids let the existing `GridSearchCV` runner evaluate only these candidates with shared explicit CV, scoring, diagnostics and one selected refit. `MLP_SAMPLES_PER_ARCHITECTURE=2` controls the budget (valid range 1–6); `MLP_ALPHAS` and `MLP_LEARNING_RATES` expose the search values. The fixed `(100,)` baseline is evaluated separately and is not a tuning candidate. No extra reference candidate is added. All sampled configurations retain the shared 1500-iteration limit; tanh and L-BFGS are not searched.

The explicit objective is **mean cross-entropy + alpha/(2 × N_train) × sum of squared weight matrices**. Biases are excluded and Adam weight_decay is zero. The alpha values therefore have a defined interpretation; copying them directly into weight_decay is avoided. This is a sklearn-inspired objective, not identical sklearn minibatch behaviour. Initialisation, optimisers and convergence behaviour also differ between frameworks.

Adam uses shuffled batches of min(200, N_train), and training-loss convergence with tolerance 0.0001 and patience 10; max_iter is an upper epoch limit. Budget warnings remain visible. No internal validation holdout is created and outer test data are never monitored. Every CV fit and selected outer refit starts with fresh weights.

`MLP_DEVICE='auto'` selects CUDA if available, otherwise CPU; `'cpu'` or `'cuda'` can be requested explicitly. MLP baseline fits and tuning use **one fitting job**, regardless of TUNING_JOBS. Finished models are moved to CPU to release GPU allocations, and small tabular predictions run on CPU. Local and Colab use the same code; install a suitable CUDA-enabled PyTorch build locally or choose a Colab GPU runtime.

### CNN architecture and training

The section 7 **predefined CNN baseline** uses Conv 16/32 with BatchNorm and ReLU after each convolution, 2×2 max pooling after both convolutions, 4×4 adaptive average pooling, Flatten, Dense 64, ReLU, Dropout 0.4 and five output logits. Convolutions use 3×3 kernels, stride 1, padding 1 and bias=False before BatchNorm. Its fixed Adam settings are initial LR=0.001, weight_decay=0.0001 and batch size 32.

The baseline and tuned CNNs share normalisation rules and a **50-epoch upper limit**, plus checkpointing, stopping and LR scheduling. The baseline remains unaugmented. Tuning compares two larger architectures, each with augmentation off/on:

| Stage | Channels | Hidden dense | Initial LR | Weight decay | Augmentation |
| --- | --- | --- | --- | --- | --- |
| Baseline | 16 → 32 | 64 neurons + ReLU | 0.001 | 0.0001 | Off |
| Candidate 1 | 32 → 64 | 128 neurons + ReLU | 0.001 | 0.0001 | Off |
| Candidate 2 | 32 → 64 | 128 neurons + ReLU | 0.001 | 0.0001 | On |
| Candidate 3 | 32 → 64 → 128 | 128 neurons + ReLU | 0.001 | 0.0001 | Off |
| Candidate 4 | 32 → 64 → 128 | 128 neurons + ReLU | 0.001 | 0.0001 | On |

This gives **4 candidates**: 2 architectures × augmentation off/on. Initial LR and weight decay are fixed. Every convolution is followed by BatchNorm and ReLU; every network uses dropout 0.4, **two max-pooling layers** after convolutions 1 and 2, and 4×4 adaptive pooling. The third convolution adds no further max pooling. All use Adam, batch size 32 and CUDA when available (CPU otherwise). The validation-based scheduler can reduce LR during training. The baseline is evaluated separately and is not a tuning candidate. Both tuning architectures use 128 hidden dense units, but their convolution channels and dense input sizes differ, so this does not isolate depth alone.

**Training-only spectrogram masking:** select each example independently with probability 0.5. Draw one frequency-mask width uniformly from 0–4 bins and one time-mask width from 0–8 frames, at random valid positions. Both presets have one input channel: MFCC sequences in basic mode and log-mel in rich mode. The row mask operates on coefficient rows for MFCC and frequency rows for log-mel. Mask with zero after channel normalisation, representing the training-channel mean. Clone each batch to preserve cached images. A separate seeded CPU generator works with CPU/CUDA batches and avoids consuming model/dropout/shuffle random draws. Refresh masks during each inner training fit and selected outer refit. Validation, test and diagnostic metric inputs remain unaugmented. The unaugmented baseline and augmentation-off candidate paths do not draw augmentation randomness.

The shared section 7 training helper now includes optional masking. **Rerun sections 4–11 after changing CNN inputs** to load the helper and refresh the matched baseline/tuning state.

Shared settings are defined in **section 7**, then reused by section 9.2 and 9.8:

| Setting | Default | Role |
| --- | ---: | --- |
| `CNN_MAX_EPOCHS` | 50 | Shared upper limit per inner training run |
| `CNN_MIN_EPOCHS` | 20 | Delay stopping; does not prevent an earlier best checkpoint |
| `CNN_PATIENCE` | 15 | Non-improving validation-loss epochs before stopping, after the minimum duration |
| `CNN_LR_PATIENCE` | 5 | ReduceLROnPlateau halves LR after its patience is exceeded |
| `CNN_MIN_LR` | 0.00001 | Minimum scheduled LR |
| `INNER_RANDOM_FOLDS` | 4 | Shared group-aware random validation folds |
| `INNER_LOSO_FOLDS` | None | Validate on all remaining speakers |

For the baseline and every tuning candidate, each inner fold fits channel normalisation **only on inner training images**, trains up to 50 epochs and restores a copied lowest-validation-loss checkpoint. Checkpoint improvement and LR scheduling use the same absolute threshold 0.0001. For the baseline, architecture, initial LR, weight decay and augmentation are fixed: validation calibrates duration and the LR schedule only. In tuning, mean checkpoint macro F1 selects the configuration, with candidate order resolving ties.

Both stages then fit **fresh weights** on all outer training clips for the rounded-up median inner best epoch and the median inner learning-rate schedule. Shorter LR histories extend their last rate. No outer test data select settings, stopping, checkpoints or refit duration. The 50-epoch limit is shared; **actual stopping/checkpoint/refit epochs may differ**. A checkpoint/refit may occur before epoch 20. Thus the CNN baseline is validation-calibrated, rather than an untuned fixed-epoch model.

Section 7 reports baseline inner checkpoint/refit epochs and stores its histories/bundles in `baseline_cnn_bundles`. Section 9 verifies shared settings/inner folds have not changed. Finished CNNs are stored on CPU and transferred for prediction as needed. **Rerun sections 7–11** when changing these shared settings.

### Diagnostics and result display

- **Section 8/9.1:** baseline training/test accuracy and macro F1, averaged across outer folds, alongside pooled held-out scores.
- **Each classical tuning section:** model-parameter candidates, complete settings, Top 10 & Bottom 10 candidates across all Random folds and separately across all LOSO folds, selected settings for every outer fold, and an inner-training versus validation macro F1 graph. Full candidate tables remain in `validation_results`. After defining the helper, these displays can be regenerated with `show_candidate_rankings(model_name, validation_results[model_name])`, without any new fitting.
- **Candidate rankings for every tuned model:** independently display Top 10 and Bottom 10 by the unweighted mean inner-validation macro F1 across all Random outer folds, then across all LOSO outer folds. Include sample standard deviation across the outer-fold means, fold coverage and selected-fold count. This replaces the Random-fold-1 Top 15 table. These are inner-validation diagnostics, not pooled F1 or outer-test scores; selection rules are unchanged. Show fewer than ten when necessary (eight RF and four CNN candidates); top/bottom lists can overlap when fewer than twenty candidates exist.
- **CNN:** training/validation loss and macro F1, LR histories, selected checkpoint/refit epochs and F1 gaps. Curves use the first outer fold and first inner fold of each protocol, fixed independently of outer test performance. Inner training/validation losses and F1 are evaluated in evaluation mode; optimisation loss is retained separately.
- **Combined selected settings:** a standalone cell immediately after CNN tuning displays `selected_settings` for every model/outer split and can be rerun without retraining.
- **Section 10:** pooled baseline/tuned scores and changes, fold mean ± sample standard deviation, and separate per-speaker LOSO accuracy and macro F1 tables.

Pandas displays full settings text and all columns. Training scores are resubstitution diagnostics, not generalisation estimates; distance-weighted KNN can score perfectly on its own training data. Compare training/validation gaps and per-speaker variation before increasing model size or training duration.

### Running optimisation

1. Run sections 1–8 for the selected basic/rich preset. Retain the original baseline for learning-progress comparisons.
2. Review shared CNN/inner-fold settings in **7**, and search compute settings in **9.2**. `TUNING_JOBS = 2` runs up to two classical candidate/fold fits concurrently. RF keeps its own n_jobs=1; reduce the shared job count if CPU/RAM capacity is limited. Tuned PyTorch MLP always uses one fitting job; MLP and CNN select CPU/CUDA separately.
3. Run **9.3–9.8**, then **10**. Re-running 9.2 clears previous tuning state, so re-run the remaining tuning/evaluation cells afterwards.
4. Re-run the section 1 configuration cell and **4–11** when changing basic/rich inputs. Within a run, every classical model retains all features from the selected preset and the outer test clips stay fixed.

Model-parameter searches and full inner LOSO require substantial computation. There are 120 classical grid candidates and 10 PyTorch MLP candidates per outer training split in either preset (130 total). With five random outer folds × four inner folds and six LOSO outer folds × five inner holdouts, these require 6500 inner fits, plus 55 selected refits. The MLP contributes 500 inner fits plus 11 selected refits; a full 30-candidate MLP grid would require 1500 inner fits. CNN tuning has 200 inner candidate runs across the 11 outer folds, each with up to 50 epochs, plus 11 selected refits. The predefined baseline adds 50 inner calibration runs and 11 fresh refits using the same protocol. CPU execution can be lengthy; GPU acceleration applies to CNN and both baseline/tuned PyTorch MLP; the five sklearn baselines stay on CPU. The notebook is saved without outputs; generate updated nested scores by executing it. Exploratory fixed-epoch CNN results are not hard-coded as final tuned results.

## Save models and reproduce results

Set both the feature preset and these run controls in the dedicated configuration code cell after imports in section 1:

```python
FEATURE_SET = 'rich'        # 'basic' or 'rich'
RUN_MODE = 'train'          # 'train' or 'load'
SAVE_MODELS = True
MODEL_ROOT = Path('./saved_models')
MODEL_RUN = 'latest'        # Or the folder name of a specific completed run
```

**Train/save:** choose `FEATURE_SET` in the section 1 configuration cell and run sections 1–11. Section 7 saves baseline outer-fold models after baseline training completes; section 9 saves each selected refit. Section 11 writes the evaluation state, reloads every model from disk and recomputes its held-out predictions. It checks exact predicted-label agreement and identical pooled accuracy/macro F1 before marking the run complete and updating `latest.json`. This is a prediction round-trip check, not a comparison of cached score tables.

**Load/reproduce:** retain the original dataset and the complete `saved_models` folder beside the notebook. Set `RUN_MODE='load'`, select the matching `FEATURE_SET`, and run all sections. Dataset loading, cleaning and feature extraction still run; model fitting and hyperparameter searches are skipped. Sections 8 and 10 regenerate baseline/final tables and charts from loaded models. Section 9 displays the saved validation results, settings and CNN curves. Section 11 checks all regenerated predictions against the saved reference predictions. `SAVE_MODELS` is ignored in load mode.

In Colab, download and retain the complete saved_models folder before resetting or losing the runtime.

Each train/save run gets a new folder under `saved_models/basic/<run>/` or `saved_models/rich/<run>/`. Earlier completed runs remain available. Save one complete run for each preset you report. Incomplete folders are not eligible for load mode; `latest.json` points only to a successfully verified run. Saving must be enabled before training; switching it on only at section 11 cannot recover checkpoints that were not exported.

| Saved file | Contents |
| --- | --- |
| sklearn `.joblib` files | Fitted outer-fold classifiers/pipelines, including training-fitted scaling |
| MLP `.pt` and scaler `.joblib` files | CPU model weights, architecture/settings, class mapping and fitted scaler; no pickled notebook-defined MLP class |
| CNN `.pt` and bundle `.joblib` files | CPU model weights, architecture, normalisation, selected settings and available training history |
| `evaluation_state.joblib` | Recording identities, class order, exact outer indices, reference predictions, selected settings and validation histories |
| `manifest.json` | Input/preprocessing configuration, package versions, model count and SHA-256 file checksums |

For AESDD, a completed run contains **154 selected model entries**: 7 models × 11 outer folds × 2 stages. DummyClassifier is included at both stages without tuning. Inner-search candidate weights are not saved. Baseline scalers are retained from section 6. Model weights load onto CPU and the existing CNN inference helper uses the available device; CUDA training is not required for loading.

Only load your own trusted archives because joblib deserialises Python objects. Load mode checks dataset bytes, labels/order, feature definitions, class mapping and outer splits. The dataset may move to a different Windows/Colab path. Package differences are reported, and prediction verification detects result changes. Retain the recorded versions to reproduce the submitted run. Basic CNN checkpoints saved with MFCC/delta/delta-delta inputs have three channels and are incompatible with the new single-channel basic input; train a new basic run. This change leaves the rich input definition unchanged.

Section 11 lists saved file sizes and identifies individual files above **10 MB** for Google Drive plus `links.txt`. It does not upload files or generate the remaining submission documents. The code saves weights when you execute training; this repository update does not include newly trained production weights or new accuracy claims.

## References

- [Librosa audio feature extraction](https://librosa.org/doc/latest/feature.html)
- [Librosa spectral feature definitions](https://librosa.org/doc/main/_modules/librosa/feature/spectral.html)
- [Librosa chroma STFT](https://librosa.org/doc/main/api/generated/librosa.feature.chroma_stft.html)
- [Librosa MFCC extraction](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)
- [Librosa delta feature definition](https://librosa.org/doc/main/_modules/librosa/feature/utils.html)
- [Banse & Scherer (1996), Acoustic profiles in vocal emotion expression](https://doi.org/10.1037/0022-3514.70.3.614) — background for broad vocal-emotion tendencies, not evidence of the specific AESDD feature relationships proposed here.
- [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
- [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [PyTorch](https://pytorch.org/)
- [PyTorch model checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [gdown public Google Drive downloads](https://github.com/wkentaro/gdown)
- [Scikit-learn RBF SVM parameter selection](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
- [Scikit-learn MLP practical guidance](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Scikit-learn ParameterSampler](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html)
- [PyTorch L-BFGS](https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)
- [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
