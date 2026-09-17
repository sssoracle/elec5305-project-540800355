# elec5305-project-540800355
# ELEC5305 Project

## Detection and Classification of Power Quality Disturbances Using Time-Frequency Signal Analysis and Machine Learning

This project develops a signal-processing-based system for detecting and classifying common power quality disturbances from voltage waveforms.

The project is completed as part of **ELEC5305**.

## Project Overview

Power quality disturbances can affect the reliable operation of electrical equipment, industrial systems, and sensitive electronic loads.

This project investigates six types of power quality signals:

* Normal voltage
* Voltage sag
* Voltage swell
* Voltage interruption
* Harmonic distortion
* Transient disturbance

The signals are generated using mathematical models based on a 50 Hz fundamental power frequency.

## Methodology

The project applies several digital signal processing techniques to analyse power quality disturbances.

### 1. Signal Generation

Six types of voltage signals are generated using Python:

* Normal
* Voltage Sag
* Voltage Swell
* Interruption
* Harmonics
* Transient

The sampling frequency is 6400 Hz and the fundamental power frequency is 50 Hz.

### 2. FFT Analysis

Fast Fourier Transform (FFT) is used to analyse the frequency components of each power quality signal.

For the harmonic disturbance, the signal contains:

* 50 Hz fundamental frequency
* 150 Hz third harmonic
* 250 Hz fifth harmonic

FFT analysis allows these harmonic components to be clearly identified in the frequency domain.

### 3. STFT Analysis

Short-Time Fourier Transform (STFT) will be used to analyse how the frequency components change over time.

This method is particularly useful for analysing short-duration and non-stationary disturbances.

### 4. Wavelet Analysis

Discrete Wavelet Transform (DWT) will be used to extract multi-resolution time-frequency features from the signals.

### 5. Machine Learning Classification

Features extracted from FFT, STFT and DWT will be used to train machine learning classifiers including:

* Support Vector Machine (SVM)
* Random Forest

The classification performance will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Noise robustness will also be investigated under different signal-to-noise ratio conditions.

## Project Structure

```text
ELEC5305 Project
│
├── README.md
├── ELEC5305_Project_Proposal.pdf
├── requirements.txt
│
├── src
│   ├── signal_generation.py
│   └── fft_analysis.py
│
├── data
│   └── power_quality_signals.csv
│
└── results
    ├── waveforms
    └── fft
```

## Current Progress

Completed:

* Power quality signal generation
* Six disturbance waveform visualisations
* FFT analysis
* FFT spectrum visualisations

In progress:

* STFT analysis
* DWT analysis
* Feature extraction
* SVM classification
* Random Forest classification
* Noise robustness analysis

## Software

The project is implemented using Python.

Main libraries:

```text
NumPy
Pandas
Matplotlib
SciPy
PyWavelets
Scikit-learn
```

## Installation

Install the required Python packages using:

```bash
pip install numpy pandas matplotlib scipy pywavelets scikit-learn
```

## Running the Project

Generate the power quality signals:

```bash
python src/signal_generation.py
```

Run the FFT analysis:

```bash
python src/fft_analysis.py
```

## Example Results

The generated signals include normal voltage, voltage sag, voltage swell, interruption, harmonic distortion, and transient disturbances.

FFT analysis is used to identify the frequency characteristics of the signals, particularly the harmonic components at 50 Hz, 150 Hz, and 250 Hz.


## References

1. IEEE, *IEEE Recommended Practice for Monitoring Electric Power Quality*, IEEE Std 1159-2019, 2019.

2. S. Santoso, E. J. Powers, W. M. Grady, and P. Hofmann, “Power quality assessment via wavelet transform analysis,” *IEEE Transactions on Power Delivery*, vol. 11, no. 2, pp. 924–930, 1996.

3. W.-M. Lin, C.-H. Wu, C.-H. Lin, and F.-S. Cheng, “Detection and Classification of Multiple Power-Quality Disturbances With Wavelet Multiclass SVM,” *IEEE Transactions on Power Delivery*, vol. 23, no. 4, pp. 2575–2582, 2008.

## Course
ELEC5305
## Author
minghang shi
