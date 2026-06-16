import numpy as np
import matplotlib.pyplot as plt

# =========================
# Global parameters
# =========================
fs = 200  # sampling frequency (Hz)
t = np.arange(0, 20, 1/fs)

f_resp = 0.25  # respiration frequency (Hz)
f_card = 1.2   # cardiac frequency (Hz)

resp = np.sin(2 * np.pi * f_resp * t)

# =========================
# PPG (base)
# =========================
def generate_ppg(t, f_card):
    base = np.sin(2 * np.pi * f_card * t)
    dicrotic = 0.4 * np.sin(2 * np.pi * f_card * t + np.pi/4)
    return (base + dicrotic) ** 2


# =========================
# ECG (base)
# =========================
def generate_ecg(t, f_card):
    ecg = np.zeros_like(t)
    peaks = np.arange(0, t[-1], 1/f_card)

    for p in peaks:
        ecg += 1.0 * np.exp(-((t - p) / 0.01)**2)               # R (pico central e agudo)
        ecg -= 0.15 * np.exp(-((t - (p - 0.02)) / 0.01)**2)     # Q (pequena deflexão negativa antes)
        ecg -= 0.2 * np.exp(-((t - (p + 0.02)) / 0.01)**2)      # S (deflexão negativa após o R)
        ecg += 0.15 * np.exp(-((t - (p - 0.16)) / 0.03)**2)     # P (onda arredondada lenta)
        ecg += 0.3 * np.exp(-((t - (p + 0.3)) / 0.05)**2)       # T (onda mais larga que a P)

    return ecg


# =========================
# FM generators (FIXED)
# =========================
def generate_ppg_fm(t, f_inst):
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    base = np.sin(phase)
    dicrotic = 0.4 * np.sin(phase + np.pi/4)
    return (base + dicrotic) ** 2


def generate_ecg_fm(t, f_inst):
    ecg = np.zeros_like(t)

    # Phase accumulation
    phase = 2 * np.pi * np.cumsum(f_inst) / fs

    # Detect R peaks from phase wrap
    cycles = np.floor(phase / (2*np.pi))
    peak_indices = np.where(np.diff(cycles) > 0)[0]

    for idx in peak_indices:
        p = t[idx]

        ecg += 1.0 * np.exp(-((t - p) / 0.01)**2)               # R
        ecg -= 0.15 * np.exp(-((t - (p - 0.02)) / 0.01)**2)     # Q
        ecg -= 0.2 * np.exp(-((t - (p + 0.02)) / 0.01)**2)      # S
        ecg += 0.15 * np.exp(-((t - (p - 0.16)) / 0.03)**2)     # P
        ecg += 0.3 * np.exp(-((t - (p + 0.3)) / 0.05)**2)       # T

    return ecg


# =========================
# Base signals
# =========================
ppg_base = generate_ppg(t, f_card)
ecg_base = generate_ecg(t, f_card)

# =========================
# Respiratory modulations
# =========================

# --- Baseline Wander (BW)
ppg_bw = ppg_base + 0.5 * resp
ecg_bw = ecg_base + 0.2 * resp

# --- Amplitude Modulation (AM)
ppg_am = (1 + 0.5 * resp) * ppg_base
ecg_am = (1 + 0.3 * resp) * ecg_base

# --- Frequency Modulation (FM)
f_inst_ppg = f_card + 0.2 * resp
f_inst_ecg = f_card + 0.1 * resp

ppg_fm = generate_ppg_fm(t, f_inst_ppg)
ecg_fm = generate_ecg_fm(t, f_inst_ecg)

# =========================
# Plotting (clean style)
# =========================
fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True)

titles = ["Sem\nmodulação", "Modulação da\nlinha de base", "Modulação de\namplitude", "Modulação de\nfrequência"]
ppg_signals = [ppg_base, ppg_bw, ppg_am, ppg_fm]
ecg_signals = [ecg_base, ecg_bw, ecg_am, ecg_fm]

for i in range(4):
    # PPG
    axes[i, 0].plot(t, ppg_signals[i], linewidth=2)
    if i == 0:
        axes[i, 0].set_title("PPG")

    # ECG
    axes[i, 1].plot(t, ecg_signals[i], linewidth=2)
    if i == 0:
        axes[i, 1].set_title("ECG")

    # Row labels (left side)
    axes[i, 0].text(
        -0.05, 0.5, titles[i],
        transform=axes[i, 0].transAxes,
        va='center', ha='right', fontsize=10
    )

# Remove EVERYTHING except the lines
for ax in axes.flatten():
    ax.set_xlim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.show()
