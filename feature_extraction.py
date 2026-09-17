"""
ELEC5305 Project
Feature Extraction for Power Quality Disturbance Classification

This script:
1. Generates multiple randomized samples for six signal classes
2. Extracts time-domain, frequency-domain, STFT, and DWT features
3. Saves the feature dataset as data/features.csv

Outputs:
    data/features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.signal import stft


# ============================================================
# 1. Basic Settings
# ============================================================

FS = 6400
F0 = 50
DURATION = 0.2
SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURE_FILE = DATA_DIR / "features.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

t = np.arange(0, DURATION, 1 / FS)
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. Signal Generation
# ============================================================

def base_signal(amplitude=1.0, phase=0.0):
    return amplitude * np.sin(2 * np.pi * F0 * t + phase)


def event_window():
    start = rng.uniform(0.035, 0.085)
    duration = rng.uniform(0.04, 0.09)
    end = min(start + duration, DURATION)
    return start, end


def generate_signal(label):
    amplitude = rng.uniform(0.95, 1.05)
    phase = rng.uniform(-0.1, 0.1)
    signal = base_signal(amplitude, phase)

    if label == "Normal":
        return signal

    start, end = event_window()
    mask = (t >= start) & (t <= end)

    if label == "Voltage Sag":
        level = rng.uniform(0.35, 0.75)
        signal[mask] *= level

    elif label == "Voltage Swell":
        level = rng.uniform(1.2, 1.8)
        signal[mask] *= level

    elif label == "Interruption":
        level = rng.uniform(0.0, 0.12)
        signal[mask] *= level

    elif label == "Harmonics":
        third = rng.uniform(0.08, 0.22) * np.sin(2 * np.pi * 150 * t + phase)
        fifth = rng.uniform(0.05, 0.16) * np.sin(2 * np.pi * 250 * t + phase)
        seventh = rng.uniform(0.0, 0.08) * np.sin(2 * np.pi * 350 * t + phase)
        signal = signal + third + fifth + seventh

    elif label == "Transient":
        start = rng.uniform(0.045, 0.12)
        mask = t >= start
        transient_amplitude = rng.uniform(0.3, 0.9)
        transient_frequency = rng.uniform(500, 1200)
        decay = rng.uniform(50, 140)
        transient = np.zeros_like(t)
        transient[mask] = (
            transient_amplitude
            * np.exp(-decay * (t[mask] - start))
            * np.sin(2 * np.pi * transient_frequency * (t[mask] - start))
        )
        signal = signal + transient

    else:
        raise ValueError(f"Unknown signal label: {label}")

    return signal


def add_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


# ============================================================
# 3. Feature Extraction
# ============================================================

def extract_time_features(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": np.ptp(signal),
        "crest_factor": peak / (rms + 1e-12),
        "skewness": stats.skew(signal),
        "kurtosis": stats.kurtosis(signal),
        "energy": np.sum(signal ** 2),
    }


def extract_fft_features(signal):
    frequency = np.fft.rfftfreq(len(signal), d=1 / FS)
    amplitude = np.abs(np.fft.rfft(signal)) / len(signal)
    if len(amplitude) > 2:
        amplitude[1:-1] *= 2

    total = np.sum(amplitude) + 1e-12
    spectral_centroid = np.sum(frequency * amplitude) / total
    dominant_index = np.argmax(amplitude[1:]) + 1

    def band_energy(low, high):
        mask = (frequency >= low) & (frequency < high)
        return np.sum(amplitude[mask] ** 2)

    return {
        "dominant_freq": frequency[dominant_index],
        "dominant_amp": amplitude[dominant_index],
        "spectral_centroid": spectral_centroid,
        "band_0_100": band_energy(0, 100),
        "band_100_300": band_energy(100, 300),
        "band_300_1000": band_energy(300, 1000),
        "harmonic_150_amp": amplitude[np.argmin(np.abs(frequency - 150))],
        "harmonic_250_amp": amplitude[np.argmin(np.abs(frequency - 250))],
        "transient_800_amp": amplitude[np.argmin(np.abs(frequency - 800))],
    }


def extract_stft_features(signal):
    frequency, _, values = stft(
        signal,
        fs=FS,
        window="hann",
        nperseg=256,
        noverlap=192,
        boundary=None,
    )
    magnitude = np.abs(values)

    low_mask = frequency <= 100
    mid_mask = (frequency > 100) & (frequency <= 300)
    high_mask = (frequency > 300) & (frequency <= 1000)

    return {
        "stft_mean": np.mean(magnitude),
        "stft_max": np.max(magnitude),
        "stft_low_energy": np.sum(magnitude[low_mask] ** 2),
        "stft_mid_energy": np.sum(magnitude[mid_mask] ** 2),
        "stft_high_energy": np.sum(magnitude[high_mask] ** 2),
    }


def extract_dwt_features(signal):
    coeffs = pywt.wavedec(signal, "db4", level=4)
    features = {}

    for index, coeff in enumerate(coeffs):
        name = "cA4" if index == 0 else f"cD{5 - index}"
        features[f"dwt_{name}_energy"] = np.sum(coeff ** 2)
        features[f"dwt_{name}_std"] = np.std(coeff)

    return features


def extract_features(signal):
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_fft_features(signal))
    features.update(extract_stft_features(signal))
    features.update(extract_dwt_features(signal))
    return features


# ============================================================
# 4. Build Feature Dataset
# ============================================================

def build_dataset():
    labels = [
        "Normal",
        "Voltage Sag",
        "Voltage Swell",
        "Interruption",
        "Harmonics",
        "Transient",
    ]

    rows = []

    for label in labels:
        for sample_index in range(SAMPLES_PER_CLASS):
            signal = generate_signal(label)
            features = extract_features(signal)
            features["label"] = label
            features["sample_id"] = f"{label.lower().replace(' ', '_')}_{sample_index:03d}"
            rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# 5. Main Program
# ============================================================

def main():
    print("\n==============================================")
    print("ELEC5305 Feature Extraction")
    print("==============================================")
    print(f"Samples per class : {SAMPLES_PER_CLASS}")
    print(f"Output file       : {FEATURE_FILE}")
    print("==============================================\n")

    df = build_dataset()
    df.to_csv(FEATURE_FILE, index=False)

    print(f"Saved feature dataset: {FEATURE_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nClass counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
