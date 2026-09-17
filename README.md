# elec5305-project-540800355
# ELEC5305 Project

# ELEC5305 Project

## Power Quality Disturbance Detection and Classification

This project was developed for **ELEC5305**.

The aim is to analyse and classify common power quality disturbances using signal processing and machine learning techniques.

## Signal Classes

Six types of power quality signals are considered:

- Normal
- Voltage Sag
- Voltage Swell
- Interruption
- Harmonics
- Transient

## Methods

The project uses:

- Fast Fourier Transform (FFT)
- Short-Time Fourier Transform (STFT)
- Discrete Wavelet Transform (DWT)
- Feature extraction
- Machine learning classification
- SNR robustness analysis

## Repository Structure

```text
data/
results/

ELEC5305_Project_Proposal_Power_Quality.pdf
README.md

signal_generation.py
fft_analysis.py
stft_analysis.py
dwt_analysis.py
feature_extraction.py
classification.py
snr_experiment.py
```

## File Description

### `signal_generation.py`

Generates the six power quality signal classes and saves the signal data.

### `fft_analysis.py`

Performs FFT analysis and generates frequency spectra.

### `stft_analysis.py`

Performs STFT analysis and generates time-frequency spectrograms.

### `dwt_analysis.py`

Performs discrete wavelet transform analysis.

### `feature_extraction.py`

Extracts numerical signal features for classification.

### `classification.py`

Performs machine learning classification and evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### `snr_experiment.py`

Evaluates classification performance under different noise levels.

## Data

The `data/` folder contains the generated signal data and extracted features used for analysis and classification.

## Results

The `results/` folder contains:

- Signal waveform figures
- FFT results
- STFT results
- DWT results
- Classification results
- Confusion matrix
- SNR experiment results

## Requirements

Install the required Python packages using:

```bash
pip install numpy pandas matplotlib scipy pywavelets scikit-learn
```

## How to Run

Run the Python files in the following order:

```bash
python signal_generation.py
python fft_analysis.py
python stft_analysis.py
python dwt_analysis.py
python feature_extraction.py
python classification.py
python snr_experiment.py
```

## Project Workflow

```text
Signal Generation
      ↓
FFT / STFT / DWT
      ↓
Feature Extraction
      ↓
Classification
      ↓
SNR Evaluation

## References

1. IEEE, *IEEE Recommended Practice for Monitoring Electric Power Quality*, IEEE Std 1159-2019, 2019.

2. S. Santoso, E. J. Powers, W. M. Grady, and P. Hofmann, “Power quality assessment via wavelet transform analysis,” *IEEE Transactions on Power Delivery*, vol. 11, no. 2, pp. 924–930, 1996.

3. W.-M. Lin, C.-H. Wu, C.-H. Lin, and F.-S. Cheng, “Detection and Classification of Multiple Power-Quality Disturbances With Wavelet Multiclass SVM,” *IEEE Transactions on Power Delivery*, vol. 23, no. 4, pp. 2575–2582, 2008.

## Course
ELEC5305
## Author
minghang shi
