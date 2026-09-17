"""
ELEC5305 Project
Power Quality Disturbance Signal Generation

Generates six power-quality signal classes:
1. Normal
2. Voltage Sag
3. Voltage Swell
4. Interruption
5. Harmonics
6. Transient

Outputs:
- Combined waveform figure
- Individual waveform figures
- CSV file containing all signals
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 6400
F0 = 50
DURATION = 0.2

t = np.arange(0, DURATION, 1 / FS)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
WAVEFORM_DIR = RESULTS_DIR / "waveforms"
DATA_DIR = BASE_DIR / "data"

SAVE_OUTPUTS = True
OPEN_IMAGES = True

if SAVE_OUTPUTS:
    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def open_image(image_file):
    if OPEN_IMAGES:
        os.startfile(str(image_file))


def normal_signal():
    return np.sin(2 * np.pi * F0 * t)


def sag_signal(start=0.05, end=0.12, level=0.5):
    signal = np.sin(2 * np.pi * F0 * t)
    mask = (t >= start) & (t <= end)
    signal[mask] = level * signal[mask]
    return signal


def swell_signal(start=0.05, end=0.12, level=1.5):
    signal = np.sin(2 * np.pi * F0 * t)
    mask = (t >= start) & (t <= end)
    signal[mask] = level * signal[mask]
    return signal


def interruption_signal(start=0.05, end=0.12, level=0.05):
    signal = np.sin(2 * np.pi * F0 * t)
    mask = (t >= start) & (t <= end)
    signal[mask] = level * signal[mask]
    return signal


def harmonic_signal():
    fundamental = np.sin(2 * np.pi * 50 * t)
    third = 0.15 * np.sin(2 * np.pi * 150 * t)
    fifth = 0.10 * np.sin(2 * np.pi * 250 * t)
    return fundamental + third + fifth


def transient_signal(start=0.08, amplitude=0.5, transient_frequency=800, decay=80):
    signal = np.sin(2 * np.pi * F0 * t)
    transient = np.zeros_like(t)
    mask = t >= start
    transient[mask] = (
        amplitude
        * np.exp(-decay * (t[mask] - start))
        * np.sin(2 * np.pi * transient_frequency * (t[mask] - start))
    )
    return signal + transient


signals = {
    "Normal": normal_signal(),
    "Voltage Sag": sag_signal(),
    "Voltage Swell": swell_signal(),
    "Interruption": interruption_signal(),
    "Harmonics": harmonic_signal(),
    "Transient": transient_signal(),
}


def save_signals_to_csv():
    if not SAVE_OUTPUTS:
        print("CSV save skipped. Set SAVE_OUTPUTS = True to save files.")
        return

    data = {"Time_s": t}
    for name, signal in signals.items():
        data[name.replace(" ", "_")] = signal

    df = pd.DataFrame(data)
    output_file = DATA_DIR / "power_quality_signals.csv"
    df.to_csv(output_file, index=False)
    print(f"CSV saved to: {output_file}")


def plot_all_signals():
    return

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, signal) in zip(axes, signals.items()):
        ax.plot(t, signal)
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (p.u.)")
        ax.grid(True)

    fig.suptitle("Six Types of Power Quality Signals", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if SAVE_OUTPUTS:
        output_file = WAVEFORM_DIR / "six_power_quality_signals.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Combined figure saved to: {output_file}")

    plt.close(fig)


def plot_individual_signals():
    if not SAVE_OUTPUTS:
        print("Individual figure save skipped. Set SAVE_OUTPUTS = True to save files.")
        return

    for name, signal in signals.items():
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal)
        plt.title(name)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (p.u.)")
        plt.grid(True)

        filename = name.lower().replace(" ", "_") + ".png"
        output_file = WAVEFORM_DIR / filename

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_file}")
        open_image(output_file)


def print_signal_information():
    print("\n==========================================")
    print("ELEC5305 Power Quality Signal Generation")
    print("==========================================")
    print(f"Sampling frequency : {FS} Hz")
    print(f"Power frequency    : {F0} Hz")
    print(f"Signal duration    : {DURATION} s")
    print(f"Number of samples  : {len(t)}")
    print("\nGenerated signal classes:")
    for i, name in enumerate(signals.keys(), start=1):
        print(f"{i}. {name}")
    print("==========================================\n")


def main():
    print_signal_information()
    save_signals_to_csv()
    plot_individual_signals()
    plot_all_signals()
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
