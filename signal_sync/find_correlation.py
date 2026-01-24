import os
import numpy as np
import heartpy as hp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

FS = 250

WINDOW_15S = 15
NUM_WINDOWS = 7
TOTAL_ANALYSIS_TIME = WINDOW_15S * NUM_WINDOWS  # 105 s

STEP_SECONDS = 1
STEP_SAMPLES = STEP_SECONDS * FS

ECG_FILE = "/home/soph/rppg/rPPG/get_ground_truth/ecg_signal_L9_16-02_16-08.csv"
HR_PRED_FOLDER = "/home/soph/rppg/rPPG/signal_sync/hr_pred"

# =========================
# Utility functions
# =========================

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def estimate_hr_heartpy(ecg_segment, fs):
    try:
        wd, m = hp.process(
            ecg_segment,
            sample_rate=fs
        )
        hr_value = m['bpm']
        return hr_value
    except Exception:
        return np.nan

# =========================
# Load ECG once
# =========================

ecg = hp.get_data(ECG_FILE)

window_15s_samples = WINDOW_15S * FS
ecg_window_samples = TOTAL_ANALYSIS_TIME * FS

# =========================
# Load HR files
# =========================

hr_files = sorted([
    f for f in os.listdir(HR_PRED_FOLDER)
    if f.endswith(".txt")
])

num_files = len(hr_files)

# Storage
correlation_results = {}
best_hr_matches = {}

# =========================
# Main processing loop
# =========================

summary_results = {}

for hr_file in hr_files:

    print(hr_file)

    ppg_hr = np.loadtxt(os.path.join(HR_PRED_FOLDER, hr_file))
    if len(ppg_hr) != NUM_WINDOWS:
        raise ValueError(f"{hr_file} must contain exactly {NUM_WINDOWS} values")

    best_corr = -np.inf
    best_start_sec = None
    best_ecg_hr = None
    correlation_trace = []

    for start in range(0, len(ecg) - ecg_window_samples, STEP_SAMPLES):

        ecg_block = ecg[start:start + ecg_window_samples]
        ecg_hr = []

        for i in range(NUM_WINDOWS):
            seg = ecg_block[
                i * window_15s_samples:(i + 1) * window_15s_samples
            ]
            hr = estimate_hr_heartpy(seg, FS)
            ecg_hr.append(hr)

        ecg_hr = np.array(ecg_hr)

        if np.any(np.isnan(ecg_hr)):
            continue

        corr, _ = pearsonr(zscore(ecg_hr), zscore(ppg_hr))
        correlation_trace.append((start / FS, corr))

        if corr > best_corr:
            best_corr = corr
            best_start_sec = start / FS
            best_ecg_hr = ecg_hr.copy()
        
        summary_results[hr_file] = (best_corr, best_start_sec)


    correlation_results[hr_file] = (correlation_trace, best_corr, best_start_sec)
    best_hr_matches[hr_file] = (ppg_hr, best_ecg_hr)

# ======================================================
# FIGURE 1 — Correlation traces
# ======================================================

fig1, axes1 = plt.subplots(
    num_files, 1,
    figsize=(10, 3 * num_files),
    sharex=True
)

if num_files == 1:
    axes1 = [axes1]

for ax, hr_file in zip(axes1, hr_files):
    trace, best_corr, best_start = correlation_results[hr_file]

    if trace:
        times, corrs = zip(*trace)
        ax.plot(times, corrs)

    ax.set_title(
        f"{hr_file} | max corr = {best_corr:.3f} @ {best_start:.1f}s"
    )
    ax.set_ylabel("Correlation")
    ax.grid(True)

axes1[-1].set_xlabel("ECG start time (s)")
# plt.tight_layout()
plt.show()

# ======================================================
# FIGURE 2 — Best HR sequence matches
# ======================================================

fig2, axes2 = plt.subplots(
    num_files, 1,
    figsize=(8, 3 * num_files),
    sharex=True
)

if num_files == 1:
    axes2 = [axes2]

for ax, hr_file in zip(axes2, hr_files):
    ppg_hr, ecg_hr = best_hr_matches[hr_file]

    if ecg_hr is not None:
        ax.plot(ppg_hr, marker='o', label="rPPG HR")
        ax.plot(ecg_hr, marker='s', label="ECG HR (HeartPy)")

    ax.set_title(hr_file)
    ax.set_ylabel("Heart Rate (bpm)")
    ax.legend()
    ax.grid(True)

axes2[-1].set_xlabel("15 s Window Index")
# plt.tight_layout()
plt.show()

print("\n===== SUMMARY =====")
for fname, (corr, start) in summary_results.items():
    print(f"{fname:25s}  corr={corr:.4f}  start={start:.2f}s")