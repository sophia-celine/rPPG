import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------
# 1. Parâmetros Globais
# -----------------------------
fs = 100  # Aumentado para melhor resolução nos gráficos
t = np.linspace(0, 20, 20 * fs)
f_resp = 0.25
f_card = 1.2
resp = np.sin(2 * np.pi * f_resp * t)

# Componentes base do PPG (sem modulação)
ppg_base = 0.6 * np.sin(2 * np.pi * f_card * t)
ppg_base += 0.2 * np.sin(2 * np.pi * 2 * f_card * t)

def get_peaks_and_pairs(sig, fs):
    """Detecta picos, vales e extrai pares (vale, pico)."""
    peaks, _ = find_peaks(sig, distance=fs//2)
    troughs, _ = find_peaks(-sig, distance=fs//2)
    pairs = []
    for tr in troughs:
        next_peaks = peaks[peaks > tr]
        if len(next_peaks) > 0:
            pk = next_peaks[0]
            pairs.append((tr, pk))
    return peaks, troughs, pairs

# =========================================================
# FIGURA 1 — BW (Baseline Wander)
# =========================================================
baseline = 0.8 * resp
signal_bw = ppg_base + baseline
peaks, troughs, pairs = get_peaks_and_pairs(signal_bw, fs)

plt.figure(figsize=(10, 5))
plt.plot(t, signal_bw, label="Sinal rPPG", alpha=0.7)
plt.plot(t[peaks], signal_bw[peaks], "ro", markersize=4)
plt.plot(t[troughs], signal_bw[troughs], "go", markersize=4)

bw_t, bw_v = [], []
for tr, pk in pairs:
    bw = (signal_bw[tr] + signal_bw[pk]) / 2
    tm = (t[tr] + t[pk]) / 2
    bw_t.append(tm)
    bw_v.append(bw)

plt.plot(bw_t, bw_v, "k--", label="Modulação de Linha de Base")

# Anotação explicativa (Exemplo único)
ex_tr, ex_pk = pairs[2]
ex_bw = (signal_bw[ex_tr] + signal_bw[ex_pk]) / 2
ex_tm = (t[ex_tr] + t[ex_pk]) / 2
plt.annotate('Pico (P)', xy=(t[ex_pk], signal_bw[ex_pk]), xytext=(t[ex_pk]+0.5, signal_bw[ex_pk]+0.3), arrowprops=dict(arrowstyle='->'))
plt.annotate('Vale (V)', xy=(t[ex_tr], signal_bw[ex_tr]), xytext=(t[ex_tr]-1.5, signal_bw[ex_tr]-0.3), arrowprops=dict(arrowstyle='->'))
plt.scatter(ex_tm, ex_bw, color='black', s=50, zorder=5)
plt.text(ex_tm + 0.1, ex_bw - 0.2, r'$BW = \frac{P+V}{2}$', fontsize=10, fontweight='bold', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()


# =========================================================
# FIGURA 2 — AM (Amplitude Modulation)
# =========================================================
# Sinal puramente AM (sem BW)
signal_am = (1 + 0.4 * resp) * ppg_base
peaks, troughs, pairs = get_peaks_and_pairs(signal_am, fs)

plt.figure(figsize=(10, 5))
plt.plot(t, signal_am, label="Sinal rPPG", alpha=0.7)
plt.plot(t[peaks], signal_am[peaks], "ro", markersize=4)
plt.plot(t[troughs], signal_am[troughs], "go", markersize=4)

am_t, am_v = [], []
for tr, pk in pairs:
    amp = signal_am[pk] - signal_am[tr]
    tm = (t[tr] + t[pk]) / 2
    am_t.append(tm)
    am_v.append(amp)

# Plot da modulação de amplitude por cima (offset para visibilidade se necessário)
plt.plot(am_t, am_v, "b-", linewidth=2, label="Modulação de Amplitude (Pico-Vale)")

# Anotação explicativa (Exemplo único)
ex_tr, ex_pk = pairs[2]
plt.vlines(t[ex_pk], signal_am[ex_tr], signal_am[ex_pk], colors='blue', linestyles='solid', linewidth=3)
plt.text(t[ex_pk] + 0.1, (signal_am[ex_pk] + signal_am[ex_tr])/2, 'AM = P - V', 
         color='blue', fontweight='bold', verticalalignment='center',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()


# =========================================================
# FIGURA 3 — FM (Frequency Modulation)
# =========================================================
# Sinal puramente FM (sem BW ou AM) usando acúmulo de fase
f_inst = f_card + 0.3 * resp
phase = 2 * np.pi * np.cumsum(f_inst) / fs
signal_fm = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)

peaks, _ = find_peaks(signal_fm, distance=fs//2)

plt.figure(figsize=(10, 5))
plt.plot(t, signal_fm, label="Sinal rPPG", alpha=0.7)
plt.plot(t[peaks], signal_fm[peaks], "ro", markersize=4)

fm_t, fm_v = [], []
for i in range(len(peaks) - 1):
    dt = t[peaks[i+1]] - t[peaks[i]]
    tm = (t[peaks[i+1]] + t[peaks[i]]) / 2
    fm_t.append(tm)
    fm_v.append(dt)

# Plot da modulação de frequência (período entre picos)
plt.plot(fm_t, fm_v, "r-", linewidth=2, label="Modulação de Frequência")

# Anotação explicativa (Exemplo único)
idx = 2
p1, p2 = peaks[idx], peaks[idx+1]
h = 0.7
plt.hlines(h, t[p1], t[p2], colors='red', linestyles='solid', linewidth=3)
plt.text((t[p1] + t[p2])/2, h-0.1, f'FM = $\Delta t$', 
         color='red', fontweight='bold', horizontalalignment='center',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
