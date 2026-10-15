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
| Convolutional Neural Network (CNN) | Neural network trained on log-mel inputs, with temporal derivatives in rich mode |

Sections 1–8 establish an **untuned model baseline using predefined audio features**. Section 9 implements validation-based hyperparameter tuning; section 10 compares baseline and optimised predictions. The six scikit-learn classifiers retain their default constructor settings, including MLP's 200-iteration limit. The custom CNN uses two convolution layers, Adam, a batch size of 32, and 10 epochs. CUDA is selected automatically when available; the scikit-learn models run on CPU.

## Notebook Workflow

Open [MSI5001_SER.ipynb](MSI5001_SER.ipynb) and run sections 1–10 in order:

1. **Install dependency & import library** — prepare the Python environment.
2. **Load dataset** — read a local WAV directory or ZIP, or download a local copy from a public Google Drive link without mounting Drive.
3. **Explore and clean dataset** — remove byte-identical duplicates and skip damaged, empty, invalid, or silent WAVs; inspect tables and graphs; listen to one randomly selected recording per emotion.
4. **Data preprocessing** — convert audio to mono, resample to 16 kHz, and choose basic (40 MFCC features + log-mel CNN input) or rich (120 acoustic features + log-mel/delta/delta-delta CNN input).
5. **Train test split (random & LOSO)** — prepare all five random CV folds and all six speaker-held-out LOSO folds.
6. **Feature scaling** — fit scaling statistics separately on each split's training data and apply them to its training and test inputs.
7. **Model training** — train fresh model instances for every split.
8. **Model prediction and evaluation** — display pooled out-of-fold baseline accuracy/macro F1, fold statistics, training/test diagnostics and charts.
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

### Input Features

One selector controls both the scikit-learn features and CNN inputs:

| `FEATURE_SET` | Six scikit-learn models | CNN input per recording |
| --- | --- | --- |
| `'basic'` | **40 MFCC mean/std features** | **Log-mel only: 1 × 64 × 128** |
| `'rich'` (default) | **120 acoustic features** | **Log-mel + delta + delta-delta: 3 × 64 × 128** |

Select the mode in section 4:

```python
FEATURE_SET = 'rich'  # 'basic' or 'rich'
```

`basic` uses only the MFCC row below and skips the other acoustic feature extraction. `rich` uses every family. Both modes use the same recordings, evaluation splits and training budgets. Both stages start from the selected inputs; classical tuning may select feature subsets within that preset. The former `'mfcc'` selector is now called `'basic'`.

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
| Log-mel | Basic and rich | Energy across 64 mel-frequency bands over time |
| Delta of log-mel | Rich | First-order changes across neighbouring time frames |
| Delta-delta of log-mel | Rich | Second-order changes across neighbouring time frames |

Extract log-mel at 16 kHz using the existing mel-spectrogram settings. In rich mode, compute first- and second-order temporal derivatives with `librosa.feature.delta`, a 9-frame window, `axis=-1` and nearest-edge padding. Compute derivatives on the original log-mel frames **before** resizing the stacked channels together to 64 × 128. Channel order is log-mel, delta, delta-delta. These derivatives are distinct from the delta MFCC summaries used by the classical models; CNN does not receive an MFCC sequence.

Section 4 displays the selected input shapes and plots every CNN channel. Both baseline and tuned CNNs infer the number of input channels automatically. Fit one mean and standard deviation **per channel**, across training recordings, mel bands and time bins. Apply those statistics to the corresponding validation/test inputs; clip zero standard deviations to `1e-8`. Inner validation fits its own training-channel statistics, and the selected CNN's final refit stores statistics from all outer training images.

Re-run sections **4–10** after changing `FEATURE_SET`, so features, scaling, models and results match the selection. Rich adds an explicit view of temporal changes; it is not guaranteed to improve accuracy. Comparing basic and rich changes the inputs for both model families, so it measures the complete feature preset rather than isolating hyperparameter tuning. Baseline-versus-optimised comparisons within one mode keep the available input preset and outer folds fixed; classical tuning may select a feature subset.

## Evaluation

| Protocol | How it works | What it assesses |
| --- | --- | --- |
| Five-fold stratified group random CV | Use all five folds from a shuffled `StratifiedGroupKFold` (seed 42). Each fold trains on approximately 80% and tests on 20%. Group by speaker, emotion, and utterance to keep alternate takes together. | Performance when speakers may be shared between training and testing |
| LOSO | Hold out one complete speaker, train on the remaining speakers, and repeat for every speaker: six folds for AESDD. | Generalisation to unseen speakers |

Both protocols test **every recording exactly once**. Random CV has five outer folds; LOSO has six outer folds because AESDD has six speakers. This creates **11 separately scaled and trained outer splits**. Baseline and optimised models use identical outer folds. Class proportions in random CV are preserved as closely as grouping constraints allow. LOSO follows speaker boundaries and does not redistribute recordings to enforce stratification.

The primary comparison reports **pooled out-of-fold scores for both protocols**: combine the held-out predictions from all five random folds, or all six LOSO folds, then calculate accuracy and macro F1 once. Each recording has one prediction per model, protocol and stage. The notebook checks this coverage explicitly and retains recording indices to align baseline and optimised predictions.

A second table reports the **unweighted mean ± sample standard deviation (`ddof=1`) of fold accuracy and macro F1**, using five random-fold values and six LOSO-fold values. Pooled macro F1 differs from mean fold macro F1; pooled accuracy also differs from the unweighted fold mean when fold sizes differ. Fold standard deviations describe variation across folds and are not confidence intervals. Macro F1 gives each emotion equal weight. Charts and improvement tables use pooled scores.

## Model Optimisation & Hyperparameter Tuning

Section 9 jointly selects classical feature groups and model settings, and compares CNN architectures with validation-based training. It preserves the original baseline and uses the same outer test recordings for both stages. DummyClassifier remains unchanged. Improvement is not guaranteed; negative changes are retained.

### Validation and model selection

Every outer training split gets its own tuning run: **five random runs and six LOSO runs**.

| Outer protocol | Default inner validation | What stays separate |
| --- | --- | --- |
| Random | **4 shuffled stratified group folds** inside each outer training fold | Alternate takes stay together; the outer test clips are excluded |
| LOSO | **All remaining speakers**, one inner validation speaker at a time: **5 holdouts for AESDD** | The outer test speaker never enters tuning; inner training uses the other 4 speakers |

Defaults are `INNER_RANDOM_FOLDS = 4` and `INNER_LOSO_FOLDS = None`. The same inner folds are shared across models. Section 9.2 displays the validation plan and speaker coverage; with full inner LOSO, each AESDD speaker serves as validation in five eligible outer runs. Reducing the inner holdout count is possible but reduces speaker coverage.

Select model settings by **mean inner-validation macro F1**, with accuracy retained alongside it. Each outer fold can select a different configuration. Classical searches use `Pipeline(feature selector, StandardScaler, model)` on raw X. Each inner fold fits its own scaler; the selected pipeline then refits on all outer training clips. CNN channel normalisation is also fitted only on the appropriate training images. Section 6's outer-scaled arrays are not used for inner tuning.

Outer test results never select parameters, features, checkpoints, learning-rate schedules or refit epochs. Do not choose baseline versus tuned separately for each outer fold using its test score. Further designs made after inspecting outer results remain exploratory until independently confirmed.

### Feature-group selection

Keep `FEATURE_SET = 'basic'` or `'rich'` in section 4. It still controls the predefined inputs for both model families. Classical tuning additionally selects from the feature groups available within that preset:

| Feature-group candidate | Rich-mode count | Purpose |
| --- | ---: | --- |
| All available features | 120 | Full predefined representation and default-parameter reference |
| MFCC | 40 | Test the simpler cepstral representation |
| MFCC + spectral | 62 | Add spectral brightness, spread, roll-off, flatness and contrast |
| MFCC + prosodic | 52 | Add pitch, energy, pauses and voicing |
| All except chroma | 96 | Test whether pitch-class descriptors contribute useful information |

Basic mode has only 40 MFCC features, so all duplicate feature-group candidates collapse to a single option. No unavailable features are extracted or invented. Rich-mode family selection does not change the CNN's three-channel input. Feature groups and model parameters are selected jointly inside inner validation, independently for each model and outer fold. Section 9 records the selected group, count and complete settings.

### Implemented searches

The counts below include default-parameter references for classical models. Rich mode has five feature-group options; basic has one.

| Subsection | Model | Model settings searched | Model combinations | Total candidates: basic / rich |
| --- | --- | --- | ---: | ---: |
| 9.3 | KNN | k = **3/5/7/9/11**; uniform/distance weighting; Manhattan/Euclidean distance | 20 | 20 / 100 |
| 9.4 | Logistic Regression | C=1/max_iter=100 reference; C=0.1/1/10 with max_iter=1500 | 4 | 4 / 20 |
| 9.5 | Random Forest | Six structural configurations: default; depth 10/leaf 2; unrestricted depth/leaf 4; 300 trees/leaf 2; depth 6/leaf 8/max_features=0.5; depth 10/leaf 4/max_features=0.5 | 6 | 6 / 30 |
| 9.6 | SVM | RBF C=0.01/0.1/1/10/100/1000 × gamma=scale/0.0001/0.001/0.01/0.1; linear C=1/10 | 32 | 32 / 160 |
| 9.7 | MLP | 100-neuron Adam reference at 200 and 1500 iterations; 32/64-neuron networks × Adam/L-BFGS × alpha=0.0001/0.01/0.1 at up to 1500 iterations | 14 | 14 / 70 |
| 9.8 | CNN | Compact spatial reference; stronger CNN at LR=0.001; stronger CNN at LR=0.0003 | 3 | 3 / 3 |

The first four Random Forest configurations use max_features='sqrt'; all use 100 trees except the explicit 300-tree candidate. RF and MLP tuning use seed 42. MLP's internal random early stopping is disabled so validation stays under the explicit group-aware procedure. L-BFGS does not use Adam's learning-rate setting. Convergence warnings remain visible; allowing more iterations does not guarantee better held-out performance.

### CNN architecture and training

The original section 7 CNN stays unchanged: two convolutions, global average pooling and 10 epochs. Section 9 compares:

| Candidate | Architecture | Initial LR | Dropout | Weight decay |
| --- | --- | ---: | ---: | ---: |
| Compact spatial reference | Conv 16/32, two max-pooling layers, 2×2 adaptive pooling and linear classifier | 0.001 | 0.3 | 0.0001 |
| Stronger CNN | Conv 16/32/64 with batch normalisation, two max-pooling layers, 4×4 adaptive pooling and 64-neuron dense layer | 0.001 | 0.2 | 0.0001 |
| Stronger CNN, lower learning rate | Same stronger architecture | 0.0003 | 0.2 | 0.0001 |

All infer one input channel in basic mode or three in rich mode, use Adam and batch size 32, and automatically use CUDA when available. The stronger architecture changes several components together; further ablations are needed to attribute any gain to a single component.

Training defaults in section 9.2:

| Setting | Default | Role |
| --- | ---: | --- |
| `CNN_MAX_EPOCHS` | 100 | Upper limit for each inner training run |
| `CNN_MIN_EPOCHS` | 20 | Delay stopping until at least this many epochs have run, capped by the run's epoch limit |
| `CNN_PATIENCE` | 15 | Stop after this many epochs without sufficient validation-loss improvement, once minimum training is complete |
| `CNN_LR_PATIENCE` | 5 | `ReduceLROnPlateau` halves LR after this patience is exceeded without sufficient validation-loss improvement |
| `CNN_MIN_LR` | 0.00001 | Minimum scheduled LR |

Checkpointing and stopping use a validation-loss improvement threshold of 0.0001. The scheduler uses the same absolute threshold. The minimum training duration delays stopping; it **does not** force the best checkpoint or final refit to occur at epoch 20 or later.

For every inner fold, restore a copied **lowest-validation-loss checkpoint** and record its macro F1/accuracy. Select the candidate with the highest mean checkpoint macro F1, using candidate order to resolve ties. Its median inner best epoch, rounded up, determines a fresh refit's duration on all outer training clips.

The refit's LR schedule is the median recorded inner LR at each refit epoch; shorter inner histories are extended with their last recorded LR. This transfers a validation-derived schedule without monitoring the outer test set. No inner model weights are reused. The fitted bundle stores the model, outer-training normalisation, configuration, epoch count and LR schedule.

### Diagnostics and result display

- **Section 8/9.1:** baseline training/test accuracy and macro F1, averaged across outer folds, alongside pooled held-out scores.
- **Each classical tuning section:** feature-group candidates, complete settings, the top 15 candidates for random outer fold 1, selected settings for every outer fold, and an inner-training versus validation macro F1 graph. Full candidate tables remain in `validation_results`.
- **CNN:** training/validation loss and macro F1, LR histories, selected checkpoint/refit epochs and F1 gaps. Curves use the first outer fold and first inner fold of each protocol, fixed independently of outer test performance. Inner training/validation losses and F1 are evaluated in evaluation mode; optimisation loss is retained separately.
- **Section 10:** pooled baseline/tuned scores and changes, fold mean ± sample standard deviation, and separate per-speaker LOSO accuracy and macro F1 tables.

Pandas displays full settings text and all columns. Training scores are resubstitution diagnostics, not generalisation estimates; distance-weighted KNN can score perfectly on its own training data. Compare training/validation gaps and per-speaker variation before increasing model size or training duration.

### Running optimisation

1. Run sections 1–8 for the selected basic/rich preset. Retain the original baseline for learning-progress comparisons.
2. Review validation and compute settings in **9.2**. `TUNING_JOBS = 1` is the conservative default; larger values parallelise classical searches, while CNN selects CPU/CUDA separately.
3. Run **9.3–9.8**, then **10**. Re-running 9.2 clears previous tuning state, so re-run the remaining tuning/evaluation cells afterwards.
4. Re-run **4–10** when changing basic/rich inputs. Within a run, the baseline input preset and outer test clips stay fixed, while classical tuning may select subsets of its available features.

Expanded feature/parameter searches and full inner LOSO require substantially more computation than the compact search. In rich mode there are 380 classical candidates per outer training split, with four or five inner fits per candidate plus refitting. CNN has 150 inner candidate runs across the 11 outer folds, each with up to 100 epochs. CPU execution can be lengthy; GPU acceleration applies to CNN, not the scikit-learn MLP. The notebook is saved without outputs; generate updated nested scores by executing it. Exploratory fixed-epoch CNN results are not hard-coded as final tuned results.

### Further experiments

After recording the predefined search, test feature selection/PCA refinements, mild training-only CNN augmentation, architecture/regularisation ablations and multiple predefined training seeds. Keep related or augmented copies within their source recording's training partition. Select changes through inner validation and retain negative outer-test changes. More diverse speakers may be needed if unseen-speaker performance remains limited.


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








- [Scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Scikit-learn RBF SVM parameter selection](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
- [Scikit-learn MLP practical guidance](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
