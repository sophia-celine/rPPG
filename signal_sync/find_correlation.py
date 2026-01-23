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
PPG_HR_FILE = "/home/soph/rppg/rPPG/signal_sync/hr_pred/HR_CHROM.txt"

# =========================
# Utility functions
# =========================

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def estimate_hr_heartpy(ecg_segment, fs):
    """
    Estimate heart rate (bpm) from a 15 s ECG segment using HeartPy.
    Uses median HR for robustness.
    """
    # print('ecg segment', ecg_segment)
    try:
        wd, m = hp.process(
            ecg_segment,
            sample_rate=fs,
            windowsize=1
        )

        # HeartPy returns bpm values per beat
        hr_values = wd['hr']

        if len(hr_values) == 0:
            return np.nan

        return np.median(hr_values)

    except hp.exceptions.BadSignalWarning:
        return np.nan
    except Exception:
        return np.nan

# =========================
# Load data
# =========================

ecg = np.loadtxt(ECG_FILE)
ppg_hr = np.loadtxt(PPG_HR_FILE)

if len(ppg_hr) != NUM_WINDOWS:
    raise ValueError("PPG HR file must contain exactly 7 values")

# =========================
# Sliding-window correlation
# =========================

window_15s_samples = WINDOW_15S * FS
ecg_window_samples = TOTAL_ANALYSIS_TIME * FS

best_corr = -np.inf
best_start_sec = None
best_ecg_hr = None

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

    # Reject windows with failed HR estimation
    if np.any(np.isnan(ecg_hr)):
        # print('nan')
        continue

    corr, _ = pearsonr(zscore(ecg_hr), zscore(ppg_hr))
    print('correlation', corr)
    correlation_trace.append((start / FS, corr))

    if corr > best_corr:
        best_corr = corr
        best_start_sec = start / FS
        best_ecg_hr = ecg_hr.copy()

# =========================
# Results
# =========================

print("====================================")
print(f"Best correlation: {best_corr:.4f}")
print(f"Best ECG window start time: {best_start_sec:.2f} seconds")
print("====================================")

# =========================
# Visualization
# =========================

# Correlation vs ECG time
if correlation_trace:
    times, corrs = zip(*correlation_trace)

    plt.figure(figsize=(10, 4))
    plt.plot(times, corrs)
    plt.xlabel("ECG start time (s)")
    plt.ylabel("Correlation")
    plt.title("rPPG–ECG HR Correlation vs ECG Window Position")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# HR comparison at best alignment
if best_ecg_hr is not None:
    plt.figure(figsize=(8, 4))
    plt.plot(ppg_hr, marker='o', label="rPPG HR")
    plt.plot(best_ecg_hr, marker='s', label="ECG HR (HeartPy)")
    plt.xlabel("15 s Window Index")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Best-Matching HR Sequences")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
