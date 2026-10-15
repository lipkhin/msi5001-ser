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
6. Run all sections in order.

### Google Colab

1. Open the notebook from this repository in Colab.
2. Select a GPU runtime if required for CNN training.
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
| Convolutional Neural Network (CNN) | Neural network trained on log-mel spectrograms |

Sections 1–8 establish an **untuned model baseline using predefined audio features**. Section 9 implements validation-based hyperparameter tuning; section 10 compares baseline and optimised predictions. The six scikit-learn classifiers retain their default constructor settings, including MLP's 200-iteration limit. The custom CNN uses two convolution layers, Adam, a batch size of 32, and 10 epochs. CUDA is selected automatically when available; the scikit-learn models run on CPU.

## Notebook Workflow

Open [MSI5001_SER.ipynb](MSI5001_SER.ipynb) and run sections 1–10 in order:

1. **Install dependency & import library** — prepare the Python environment.
2. **Load dataset** — read a local WAV directory or ZIP, or download a local copy from a public Google Drive link without mounting Drive.
3. **Explore and clean dataset** — remove byte-identical duplicates and skip damaged, empty, invalid, or silent WAVs; inspect tables and graphs; listen to one randomly selected recording per emotion.
4. **Data preprocessing** — convert audio to mono, resample to 16 kHz, and choose 120 rich acoustic features or 40 MFCC-only features, plus CNN inputs.
5. **Train test split (random & LOSO)** — prepare all five random CV folds and all six speaker-held-out LOSO folds.
6. **Feature scaling** — fit scaling statistics separately on each split's training data and apply them to its training and test inputs.
7. **Model training** — train fresh model instances for every split.
8. **Model prediction and evaluation** — display pooled out-of-fold baseline accuracy/macro F1, fold mean ± standard deviation and charts.
9. **Model Optimisation & Hyperparameter Tuning** — preserve baseline results, set up inner validation, and tune each model:
   - **9.1 Baseline results**
   - **9.2 Optimisation method**
   - **9.3 KNN optimisation**
   - **9.4 Logistic Regression optimisation**
   - **9.5 Random Forest optimisation**
   - **9.6 SVM optimisation**
   - **9.7 MLP optimisation**
   - **9.8 CNN optimisation**
10. **Final model prediction and evaluation** — compare both stages on the same outer test recordings, including pooled score changes, fold mean ± standard deviation, per-speaker LOSO macro F1 and charts.

### Input Features

| Models | Representation |
| --- | --- |
| Six scikit-learn models | **120 acoustic features** (`rich`, default) or **40 MFCC-only features** (`mfcc`) per recording |
| CNN | **64 × 128 log-mel spectrogram**, with one input channel |

Select the feature set in section 4:

```python
FEATURE_SET = 'rich'  # 'rich': all 120 features; 'mfcc': 40 MFCC mean/std features
```

`rich` uses every family in the table below. `mfcc` uses only the MFCC row and skips the other feature extraction. Both modes use the same recordings, evaluation splits and model settings. CNN always uses the same log-mel images.

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

Re-run sections 4–10 after updating the notebook or changing `FEATURE_SET`, so features, scaling, models and results all match the selection. The model settings, split definitions and CNN log-mel representation are unchanged.

## Evaluation

| Protocol | How it works | What it assesses |
| --- | --- | --- |
| Five-fold stratified group random CV | Use all five folds from a shuffled `StratifiedGroupKFold` (seed 42). Each fold trains on approximately 80% and tests on 20%. Group by speaker, emotion, and utterance to keep alternate takes together. | Performance when speakers may be shared between training and testing |
| LOSO | Hold out one complete speaker, train on the remaining speakers, and repeat for every speaker: six folds for AESDD. | Generalisation to unseen speakers |

Both protocols test **every recording exactly once**. Random CV has five outer folds; LOSO has six outer folds because AESDD has six speakers. This creates **11 separately scaled and trained outer splits**. Baseline and optimised models use identical outer folds. Class proportions in random CV are preserved as closely as grouping constraints allow. LOSO follows speaker boundaries and does not redistribute recordings to enforce stratification.

The primary comparison reports **pooled out-of-fold scores for both protocols**: combine the held-out predictions from all five random folds, or all six LOSO folds, then calculate accuracy and macro F1 once. Each recording has one prediction per model, protocol and stage. The notebook checks this coverage explicitly and retains recording indices to align baseline and optimised predictions.

A second table reports the **unweighted mean ± sample standard deviation (`ddof=1`) of fold accuracy and macro F1**, using five random-fold values and six LOSO-fold values. Pooled macro F1 differs from mean fold macro F1; pooled accuracy also differs from the unweighted fold mean when fold sizes differ. Fold standard deviations describe variation across folds and are not confidence intervals. Macro F1 gives each emotion equal weight. Charts and improvement tables use pooled scores.

## Model Optimisation & Hyperparameter Tuning

Section 9 implements a compact, reproducible search for the selected feature set. It preserves the default baseline and keeps the outer random and LOSO test recordings fixed. Candidate lists are starting search spaces, not established best settings or a guarantee of improvement. DummyClassifier remains unchanged.

### Validation and model selection

Every outer training split gets its own tuning run: **five random runs and six LOSO runs**. Inner validation is separate from outer evaluation:

| Outer protocol | Inner validation | Default run |
| --- | --- | --- |
| Random | Shuffled `StratifiedGroupKFold` within each of the five outer training folds, keeping alternate takes together | 3 inner folds per outer fold |
| LOSO | Hold out training speakers one at a time; the outer test speaker never enters tuning | 2 predetermined inner speaker holdouts per outer fold; set `INNER_LOSO_FOLDS = None` to use all 5 remaining speakers in AESDD |

The same inner folds are used by the classical models and CNN. The compact default limits runtime; using all inner speaker holdouts provides broader validation coverage at higher cost. Settings are selected by **mean validation macro F1**, with validation accuracy retained alongside it. Each outer fold can select different settings.

Classical searches use a `Pipeline(StandardScaler, model)` with raw acoustic features. Scaling is fitted within each inner training fold, then fitted again on all outer training recordings during the selected model's refit. CNN image mean/std are also computed separately from the appropriate inner training images. The outer-scaled arrays in section 6 are not used for tuning. Outer test scores never select hyperparameters, checkpoints or epoch counts.

### Implemented searches

| Subsection | Model | Compact search |
| --- | --- | --- |
| 9.3 | KNN | 3/5/11 neighbours × uniform/distance weights × Manhattan/Euclidean distance (12 candidates) |
| 9.4 | Logistic Regression | Default C=1/max_iter=100 reference; C=0.1/1/10 with max_iter=1500 (4 candidates) |
| 9.5 | Random Forest | Default tree settings, depth 10 with leaf size 2, unrestricted depth with leaf size 4, and 300 trees with leaf size 2 (4 candidates) |
| 9.6 | SVM | RBF C=1/10 with gamma=scale; RBF C=10/100 with gamma=0.01; linear C=1/10 (6 candidates) |
| 9.7 | MLP | Default 100-neuron/200-iteration reference; 100/64/128 neurons with max_iter=1500; (64,32) with alpha=0.001 and learning_rate_init=0.0003 (5 candidates) |
| 9.8 | CNN | Original architecture with longer training; lower learning rate; 2×2 spatial pooling with dropout and weight decay (3 candidates) |

Random Forest and MLP tuning use seed 42. MLP's internal random early stopping is disabled; its validation stays under the explicit group-aware procedure. Default-parameter references are included in the classical searches. Convergence warnings remain visible even with a higher iteration budget.

CNN retains the original baseline in section 7. Tuning uses Adam, batch size 32, a maximum of 20 epochs and patience 4 by default. Validation macro F1 selects a copied best-epoch checkpoint inside each inner fold. The candidate with the highest mean best-epoch macro F1 is selected; ties use candidate order. Its median inner best epoch, rounded up, determines the number of epochs for a fresh refit on all outer training images. No outer test data are used for early stopping. Training and validation curves illustrate the selected candidate in random outer fold 1, inner fold 1; that example is fixed before viewing test scores. The spatial CNN and two-layer MLP candidates change multiple settings together; further ablations are needed to separate those settings' effects.

### Running and reviewing optimisation

1. Run sections 1–8 to establish the baseline.
2. Choose the validation and compute settings in **9.2** before starting searches. `TUNING_JOBS = 1` is the default; CNN uses CUDA automatically when available.
3. Run **9.3–9.8** in order. Each model shows its candidates, validation scores for random outer fold 1 as an illustration, and selected settings for all 11 outer splits. Full candidate results for every fold remain in `validation_results`. CNN adds validation curves.
4. Run **10** after all models finish. It reports pooled out-of-fold baseline/optimised accuracy and macro F1 for both protocols, fold mean ± sample standard deviation, pooled accuracy changes in percentage points, pooled macro F1 changes, per-speaker LOSO macro F1 and comparison charts.
5. To compare `mfcc` and `rich`, run **4–10** separately for each selection and record both runs. The feature set is kept fixed within a tuning run.

Re-running 9.2 clears previous tuning state, so re-run 9.3–10 afterwards. Using five random outer folds increases random-protocol baseline training and tuning from one outer run to five. MLP, CNN and full inner LOSO validation can take substantially longer, especially on CPU. The notebook is saved without outputs; results are generated when run locally or in Colab.

### Further experiments

After recording this compact search, refine promising validation ranges, test feature selection inside inner training folds, perform CNN augmentation/regularisation ablations, and repeat the predefined procedure with multiple training seeds. Keep the outer splits fixed and use inner validation to choose changes; retain negative outer-test changes in the report rather than discarding them.

## References

- [Librosa audio feature extraction](https://librosa.org/doc/latest/feature.html)
- [Librosa spectral feature definitions](https://librosa.org/doc/main/_modules/librosa/feature/spectral.html)
- [Librosa chroma STFT](https://librosa.org/doc/main/api/generated/librosa.feature.chroma_stft.html)
- [Librosa delta feature definition](https://librosa.org/doc/main/_modules/librosa/feature/utils.html)
- [Banse & Scherer (1996), Acoustic profiles in vocal emotion expression](https://doi.org/10.1037/0022-3514.70.3.614) — background for broad vocal-emotion tendencies, not evidence of the specific AESDD feature relationships proposed here.
- [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
- [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [PyTorch](https://pytorch.org/)
- [PyTorch model checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [gdown public Google Drive downloads](https://github.com/wkentaro/gdown)





