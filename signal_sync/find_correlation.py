import os
import numpy as np
import heartpy as hp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================

FS = 250                    # ECG sampling rate (Hz)

WINDOW_15S = 15             # seconds
NUM_WINDOWS = 7
TOTAL_ANALYSIS_TIME = WINDOW_15S * NUM_WINDOWS  # 105 s

STEP_SECONDS = 1            # ECG sliding step
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
            sample_rate=fs,
            windowsize=1
        )

        hr_values = wd['hr']
        if len(hr_values) == 0:
            return np.nan

        return np.median(hr_values)

    except Exception:
        return np.nan

# =========================
# Load ECG once
# =========================

ecg = np.loadtxt(ECG_FILE)

window_15s_samples = WINDOW_15S * FS
ecg_window_samples = TOTAL_ANALYSIS_TIME * FS

# =========================
# Iterate over HR prediction files
# =========================

hr_files = sorted([
    f for f in os.listdir(HR_PRED_FOLDER)
    if f.endswith(".txt")
])

num_files = len(hr_files)

fig, axes = plt.subplots(
    num_files, 1,
    figsize=(10, 3 * num_files),
    sharex=True
)

if num_files == 1:
    axes = [axes]

summary_results = {}

for ax, hr_file in zip(axes, hr_files):
    print(hr_file)

    ppg_hr = np.loadtxt(os.path.join(HR_PRED_FOLDER, hr_file))

    if len(ppg_hr) != NUM_WINDOWS:
        raise ValueError(f"{hr_file} must contain exactly {NUM_WINDOWS} values")

    best_corr = -np.inf
    best_start_sec = None
    correlation_trace = []

    for start in range(0, len(ecg) - ecg_window_samples, STEP_SAMPLES):

        ecg_block = ecg[start:start + ecg_window_samples]
        ecg_hr = []

        for i in range(NUM_WINDOWS):
            seg_start = i * window_15s_samples
            seg_end = seg_start + window_15s_samples
            segment = ecg_block[seg_start:seg_end]

            hr = estimate_hr_heartpy(segment, FS)
            ecg_hr.append(hr)

        ecg_hr = np.array(ecg_hr)

        if np.any(np.isnan(ecg_hr)):
            continue

        corr, _ = pearsonr(zscore(ecg_hr), zscore(ppg_hr))
        correlation_trace.append((start / FS, corr))

        if corr > best_corr:
            best_corr = corr
            best_start_sec = start / FS

    summary_results[hr_file] = (best_corr, best_start_sec)

    # -------------------------
    # Plot correlation trace
    # -------------------------

    if correlation_trace:
        times, corrs = zip(*correlation_trace)
        ax.plot(times, corrs)

    ax.set_title(
        f"{hr_file} | max corr = {best_corr:.3f} @ {best_start_sec:.1f}s"
    )
    ax.set_ylabel("Correlation")
    ax.grid(True)

# =========================
# Final plot formatting
# =========================

axes[-1].set_xlabel("ECG start time (s)")
plt.tight_layout()
plt.show()

# =========================
# Print summary
# =========================

print("\n===== SUMMARY =====")
for fname, (corr, start) in summary_results.items():
    print(f"{fname:25s}  corr={corr:.4f}  start={start:.2f}s")
