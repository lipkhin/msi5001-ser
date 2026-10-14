# MSI5001 2026/2027 Group Project - Speech Emotion Recognition (SER)

## Team Members

- Chow Lip Khin
- Manish Agarwal
- Tan Sumengyuan
- Xu Jin

## Project Brief

Speech conveys both the words being spoken and information about the speaker's emotional expression. This project investigates how machine learning can classify emotions from speech recordings using the AESDD dataset. The classification task covers five emotions: **Anger, Disgust, Fear, Happiness, and Sadness**.

We compare seven models, ranging from a simple reference baseline and classical machine learning methods to an artificial neural network and a convolutional neural network. The project follows a complete experimental workflow: loading and cleaning audio, exploring the dataset, extracting features, splitting and scaling data, training models, and evaluating predictions using **accuracy** and **macro F1**.

A key objective is to examine how well the models generalise to different speakers. We compare a stratified group random split, where speakers can appear in both training and testing, with **Leave-One-Speaker-Out (LOSO)** evaluation, where the test speaker is excluded from training. The project documents the progression from a predefined richer-feature baseline with default model settings to planned model optimisation, with earlier feature baselines available in the Git history for comparison.

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

The current notebook is an **untuned model baseline using predefined richer audio features**. The six scikit-learn classifiers retain their default constructor settings, including MLP's 200-iteration limit. The custom CNN uses two convolution layers, Adam, a batch size of 32, and 10 epochs. CUDA is selected automatically when available; the scikit-learn models run on CPU.

## Notebook Workflow

Open [MSI5001_SER.ipynb](MSI5001_SER.ipynb) and run its eight sections in order:

1. **Install dependency & import library** — prepare the Python environment.
2. **Load dataset** — read a local WAV directory or ZIP, or download a local copy from a public Google Drive link without mounting Drive.
3. **Explore and clean dataset** — remove byte-identical duplicates and skip damaged, empty, invalid, or silent WAVs; inspect tables and graphs; listen to one randomly selected recording per emotion.
4. **Data preprocessing** — convert audio to mono, resample to 16 kHz, and choose 97 rich acoustic features or 40 MFCC-only features, plus CNN inputs.
5. **Train test split (random & LOSO)** — prepare the evaluation splits.
6. **Feature scaling** — fit scaling statistics separately on each split's training data and apply them to its training and test inputs.
7. **Model training** — train fresh model instances for every split.
8. **Model prediction and evaluation** — display accuracy and macro F1 in a comparison table and charts.

### Input Features

| Models | Representation |
| --- | --- |
| Six scikit-learn models | **97 acoustic features** (`rich`, default) or **40 MFCC-only features** (`mfcc`) per recording |
| CNN | **64 × 128 log-mel spectrogram**, with one input channel |

Select the feature set in section 4:

```python
FEATURE_SET = 'rich'  # 'rich': all 97 features; 'mfcc': 40 MFCC mean/std features
```

`rich` uses every family in the table below. `mfcc` uses only the MFCC row and skips the other feature extraction. Both modes use the same recordings, evaluation splits and model settings. CNN always uses the same log-mel images.

| Feature family | Statistics | Count | What it captures | Possible emotional relevance |
| --- | --- | ---: | --- | --- |
| MFCC | Mean/std of 20 coefficients | 40 | Overall spectral envelope (timbre) and its variation | Can capture timbre differences across all five emotions; individual coefficients have no simple emotion label. |
| Pitch (F0) | Voiced F0 mean, std, P5, P95; P5–P95 semitone range | 5 | Voice pitch and the extent of pitch variation | Helps describe arousal and intonation; see the emotion examples below. |
| Energy | Active-frame RMS dBFS mean, std, P5, P95 | 4 | Digital signal level and its variation | Can describe forceful versus subdued delivery; recording gain and normalisation also affect it. |
| Spectral shape | Centroid, bandwidth and 85% roll-off mean/std | 6 | Where spectral energy lies, how widely it spreads, and its upper-frequency extent | May capture differences between brighter, sharper delivery and darker delivery; patterns overlap across emotions. |
| Timing | Duration, internal pause ratio, pauses per second, voiced-frame ratio | 4 | Clip length, low-energy interruptions and detected voicing | Can distinguish slower or interrupted delivery from more continuous delivery; duration also depends on the words spoken. |
| Zero-crossing rate (ZCR) | Mean/std | 2 | Frequency of waveform sign changes | May help describe noisy/unvoiced articulation; a high value is not uniquely anger, fear or any other emotion. |
| Spectral flatness | Mean/std | 2 | Noise-like versus tonal spectral structure | May help describe differences in voice texture; background noise and silence affect it, so it is not a direct breathiness measure. |
| Delta MFCC | Std of 20 first-order delta coefficients | 20 | Variation in local changes of the MFCC spectral envelope over time | May help distinguish changing or abrupt delivery from steadier delivery; it also captures phonetic changes. |
| Spectral contrast | Mean/std in each of 7 frequency bands | 14 | Differences between spectral peaks and valleys within each band | Adds spectral structure that may help separate overlapping emotion classes; there is no universal higher/lower direction for an emotion. |
| **Total** | | **97** | | |

The following table is an **interpretation guide, not measured AESDD results or classification rules**. The broad prosodic tendencies are informed by vocal-emotion research; links to the added descriptors are hypotheses to assess through training-data validation. Expression intensity, speaker, language, phonetic content and recording conditions can change these patterns.

| Emotion | Possible pitch, energy and timing patterns | How the additional descriptors could help |
| --- | --- | --- |
| Anger | Intense anger can involve elevated pitch, stronger delivery and abrupt changes; restrained anger can differ. | Delta MFCCs may capture transitions; ZCR, flatness and contrast may add information about the spectral texture of forceful delivery. |
| Fear | High pitch is a common candidate cue; energy and timing vary between anxious and panicked delivery. | Delta MFCCs may describe changing articulation; ZCR, flatness and contrast may distinguish spectral texture from equally high-pitched happiness. |
| Happiness | Excited happiness can have elevated, varied pitch and animated delivery; quiet happiness can differ. | Delta MFCCs may describe expressive changes; ZCR, flatness and contrast may help distinguish it from anger or fear with similar arousal. |
| Sadness | Subdued sadness can involve lower, less varied pitch, weaker delivery and slower timing; distressed crying can differ. | Delta MFCCs may describe steadier delivery; ZCR, flatness and contrast provide complementary texture cues without a fixed expected direction. |
| Disgust | Voice quality and articulation can change, but pitch, level and timing do not have a reliable universal direction. | MFCCs, ZCR, flatness and contrast may capture those timbre differences; delta MFCCs may capture changes in their delivery. |

No individual feature identifies an emotion. The classifiers learn combinations of features from labelled recordings, and speaker-separated validation is needed to check whether those combinations generalise.

Pitch uses pYIN over a fixed 50–600 Hz range. Unvoiced or non-finite estimates are excluded; when none remain, all five pitch statistics are zero. The robust pitch range is `12 * log2(P95 / P5)` semitones.

Energy and pause features use 25 ms RMS windows and a 10 ms hop. Active frames are within 30 dB of the maximum frame RMS level in that recording. Energy statistics are calculated over active frames, using dBFS relative to digital full scale. Similar peak levels do not imply equal RMS levels; these measurements do not recover original physical loudness.

Internal low-energy runs lasting at least 150 ms count as pause proxies. Leading and trailing silence are excluded. Pause ratio is the summed duration of qualifying pauses divided by full clip duration; pause rate is their count divided by full clip duration. Voiced ratio is the fraction of pYIN frames flagged as voiced. These thresholds are predefined per-recording rules, not parameters selected from test performance.

Jitter and shimmer are omitted because they require reliable vocal-cycle measurements. Onset rate is omitted because it is not a direct speaking-rate measure. A separate energy-range feature is omitted because the percentile summaries already describe the spread.

The four added feature groups contribute **38 features** (2 ZCR + 2 flatness + 20 delta MFCC + 14 contrast), bringing the previous 59-feature baseline to **97**. ZCR, flatness, delta MFCC and contrast are summarised over the whole recording, including quiet frames. Delta MFCCs use a fixed 9-frame first-order filter with nearest-edge padding, including for short clips. Contrast uses `n_bands=6`, which returns 7 bands. All these statistics are calculated independently for each recording.

Re-run sections 4–8 after updating the notebook or changing `FEATURE_SET`, so features, scaling, models and results all match the selection. The model settings, split definitions and CNN log-mel representation are unchanged. The 40-feature MFCC-only reference is available through the selector; earlier 59-feature implementations remain in Git history. Adding features does not guarantee better accuracy or macro F1; learned scaling is fitted on training data only.

## Evaluation

| Protocol | How it works | What it assesses |
| --- | --- | --- |
| Stratified group random split | Use the first split from a shuffled five-fold `StratifiedGroupKFold`, giving approximately 80% training and 20% testing. Group by speaker, emotion, and utterance to keep alternate takes together. | Performance when speakers may be shared between training and testing |
| LOSO | Hold out one complete speaker, train on the remaining speakers, and repeat for every speaker. | Generalisation to unseen speakers |

The random protocol uses **one split**, rather than averaging five-fold cross-validation results. Class proportions are preserved as closely as the grouping constraints allow. LOSO follows speaker boundaries and does not redistribute recordings to enforce stratification.

The notebook reports **pooled LOSO** scores: combine the held-out predictions from all speakers, then calculate accuracy and macro F1 once. This differs from averaging speaker-level scores. Macro F1 gives each emotion equal weight.

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

## Optimisation Roadmap — Planned Work

These experiments are proposed next steps; they are **not implemented in the baseline notebook**. The suggested parameter values are starting search ranges, not established best settings.

1. **Preserve the richer baseline.** Run `FEATURE_SET = 'rich'` (97 features) and record its package versions, split definitions, accuracy, and macro F1. Use `FEATURE_SET = 'mfcc'` for the 40-feature reference and earlier commits for the historical 59-feature baseline.
2. **Add validation within the training data.** For the random protocol, use group-aware cross-validation inside its training portion. For each outer LOSO fold, tune using only the remaining speakers: with six speakers, the inner loop can train on four and validate on the fifth. Refit the selected configuration on all five outer training speakers before testing the sixth. Fit scaling and feature selection inside each inner training fold. Use validation macro F1 as the main selection metric and retain accuracy alongside it.
3. **Keep features fixed during initial model tuning.** Use the predefined 97-feature baseline first so that parameter changes can be assessed separately. Investigate further feature selection or alternative feature sets later through the same inner validation procedure.
4. **Tune the models.** Start with a small search, then refine around promising validation results.

| Model | Suggested optimisation sequence |
| --- | --- |
| DummyClassifier | Keep unchanged as the reference baseline. |
| KNN | Test neighbour counts such as 3, 5, 7, 11, and 15; compare uniform versus distance weighting and Manhattan versus Euclidean distance. |
| Logistic Regression | Allow sufficient iterations for convergence, then test `C` values such as 0.01, 0.1, 1, 10, and 100. |
| Random Forest | Test tree depth and minimum leaf size before increasing tree count. Start with `max_depth` = None, 5, 10, 20; `min_samples_leaf` = 1, 2, 4; and 100–500 trees. |
| SVM | Compare linear and RBF kernels. For RBF, search `C` and `gamma` jointly, starting with `C` = 0.1, 1, 10, 100 and `gamma` = `scale`, 0.001, 0.01, 0.1. |
| MLP | Increase `max_iter` to 1500 to address convergence, then compare small hidden-layer layouts such as (64,), (128,), and (64, 32), followed by regularisation and learning rate. Retain explicit group-aware validation when selecting settings. |
| CNN | First inspect training/validation loss and macro F1. Test learning rates and a longer epoch budget with validation-based early stopping. Then assess modest capacity changes, dropout/weight decay, and training-only time/frequency masking in separate experiments. Use speaker-separated validation for LOSO development. |

5. **Track each change.** Keep an experiment table with the model, feature set, changed settings, validation metrics, and final outer-test metrics. Change one main component at a time when demonstrating the learning progression.
6. **Check stability and speaker-level behaviour.** For stochastic models, repeat the fixed evaluation procedure with several predetermined training seeds. Report per-speaker LOSO metrics alongside pooled results to show whether improvements are consistent.
7. **Compare baseline and optimised results.** Keep the same outer test splits and document any feature or training changes. Select parameters from training-data validation rather than from outer-test scores; nested evaluation measures the tuning procedure itself.

## References

- [Librosa audio feature extraction](https://librosa.org/doc/latest/feature.html)
- [Librosa spectral feature definitions](https://librosa.org/doc/main/_modules/librosa/feature/spectral.html)
- [Librosa delta feature definition](https://librosa.org/doc/main/_modules/librosa/feature/utils.html)
- [Banse & Scherer (1996), Acoustic profiles in vocal emotion expression](https://doi.org/10.1037/0022-3514.70.3.614) — background for broad vocal-emotion tendencies, not evidence of the specific AESDD feature relationships proposed here.
- [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
- [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [PyTorch](https://pytorch.org/)
- [gdown public Google Drive downloads](https://github.com/wkentaro/gdown)


