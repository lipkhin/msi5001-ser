# MSI5001 2026/2027 Group Project - Speech Emotion Recognition (SER)

## Team Members

- Chow Lip Khin
- Manish Agarwal
- Tan Sumengyuan
- Xu Jin

## Project Brief

Speech conveys both the words being spoken and information about the speaker's emotional expression. This project investigates how machine learning can classify emotions from speech recordings using the AESDD dataset. The classification task covers five emotions: **Anger, Disgust, Fear, Happiness, and Sadness**.

We compare seven models, ranging from a simple reference baseline and classical machine learning methods to an artificial neural network and a convolutional neural network. The project follows a complete experimental workflow: loading and cleaning audio, exploring the dataset, extracting features, splitting and scaling data, training models, and evaluating predictions using **accuracy** and **macro F1**.

A key objective is to examine how well the models generalise to different speakers. We compare a stratified group random split, where speakers can appear in both training and testing, with **Leave-One-Speaker-Out (LOSO)** evaluation, where the test speaker is excluded from training. The project also documents the learning progression from default model settings to planned feature and model optimisation, so that changes can be assessed against a consistent baseline.

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

The current notebook is an **untuned baseline**. The six scikit-learn classifiers retain their default constructor settings, including MLP's 200-iteration limit. The custom CNN uses two convolution layers, Adam, a batch size of 32, and 10 epochs. CUDA is selected automatically when available; the scikit-learn models run on CPU.

## Notebook Workflow

Open [MSI5001_SER.ipynb](MSI5001_SER.ipynb) and run its eight sections in order:

1. **Install dependency & import library** — prepare the Python environment.
2. **Load dataset** — read a local WAV directory or ZIP, or download a local copy from a public Google Drive link without mounting Drive.
3. **Explore and clean dataset** — remove byte-identical duplicates and skip damaged, empty, invalid, or silent WAVs; inspect tables and graphs; listen to one randomly selected recording per emotion.
4. **Data preprocessing** — convert audio to mono, resample to 16 kHz, and extract model inputs.
5. **Train test split (random & LOSO)** — prepare the evaluation splits.
6. **Feature scaling** — fit scaling statistics separately on each split's training data and apply them to its training and test inputs.
7. **Model training** — train fresh model instances for every split.
8. **Model prediction and evaluation** — display accuracy and macro F1 in a comparison table and charts.

### Input Features

| Models | Representation |
| --- | --- |
| Six scikit-learn models | 20 MFCC means + 20 MFCC standard deviations: **40 features per recording** |
| CNN | **64 × 128 log-mel spectrogram**, with one input channel |

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

1. **Preserve the baseline.** Record its configuration, package versions, split definitions, accuracy, and macro F1 before changing features or parameters.
2. **Add validation within the training data.** For the random protocol, use group-aware cross-validation inside its training portion. For each outer LOSO fold, tune using only the remaining speakers: with six speakers, the inner loop can train on four and validate on the fifth. Refit the selected configuration on all five outer training speakers before testing the sixth. Fit scaling and feature selection inside each inner training fold. Use validation macro F1 as the main selection metric and retain accuracy alongside it.
3. **Test richer features.** Compare the 40-feature MFCC baseline with MFCC plus pitch, RMS energy, and selected spectral descriptors. Assess feature choices through the same inner validation procedure.
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
- [Scikit-learn model selection](https://scikit-learn.org/stable/model_selection.html)
- [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [PyTorch](https://pytorch.org/)
- [gdown public Google Drive downloads](https://github.com/wkentaro/gdown)
