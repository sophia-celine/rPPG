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

ECG_FILE = "../get_ground_truth/ECG/ecg_signal_L9_16-02_16-08.csv"
HR_PRED_FOLDER = "./hr_pred_L9"

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
all_correlations_list = []
common_times = []

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
    file_corrs = []
    file_times = []

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
        file_corrs.append(corr)
        file_times.append(start / FS)

        if corr > best_corr:
            best_corr = corr
            best_start_sec = start / FS
            best_ecg_hr = ecg_hr.copy()
        
        summary_results[hr_file] = (best_corr, best_start_sec)


    correlation_results[hr_file] = (correlation_trace, best_corr, best_start_sec)
    best_hr_matches[hr_file] = (ppg_hr, best_ecg_hr)
    
    all_correlations_list.append(file_corrs)
    if not common_times and file_times:
        common_times = file_times

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
        f"{hr_file} | corr máx = {best_corr:.3f} @ {best_start:.1f}s"
    )
    ax.set_ylabel("Correlação")
    ax.grid(True)

axes1[-1].set_xlabel("Tempo de início do ECG (s)")
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
        ax.plot(ppg_hr, marker='o', label="FC rPPG")
        ax.plot(ecg_hr, marker='s', label="FC ECG (HeartPy)")

    ax.set_title(hr_file)
    ax.set_ylabel("Frequência Cardíaca (bpm)")
    ax.legend()
    ax.grid(True)

axes2[-1].set_xlabel("Índice da Janela de 15 s")
# plt.tight_layout()
plt.show()

print("\n===== SUMMARY =====")
for fname, (corr, start) in summary_results.items():
    print(f"{fname:25s}  corr={corr:.4f}  start={start:.2f}s")

# ======================================================
# FIGURE 3 — Mean Correlation
# ======================================================

if all_correlations_list and len(all_correlations_list[0]) > 0:
    # Ensure alignment (truncate to minimum length if any discrepancies occurred)
    min_len = min(len(c) for c in all_correlations_list)
    all_correlations_arr = np.array([c[:min_len] for c in all_correlations_list])
    common_times = common_times[:min_len]

    mean_corrs = np.mean(all_correlations_arr, axis=0)
    best_global_idx = np.argmax(mean_corrs)
    best_global_val = mean_corrs[best_global_idx]
    best_global_time = common_times[best_global_idx]
    idx_second = np.argsort(mean_corrs)[-2]
    best_second_val = mean_corrs[idx_second]
    best_second_time = common_times[idx_second]
    idx_third = np.argsort(mean_corrs)[-3]
    best_third_val = mean_corrs[idx_third]
    best_third_time = common_times[idx_third]

    print(f"\n===== GLOBAL BEST FIT =====")
    print(f"Best interval start: {best_global_time:.2f}s")
    print(f"Mean correlation:    {best_global_val:.4f}")
    print(f"Second best interval start: {best_second_time:.2f}s")
    print(f"Mean correlation:    {best_second_val:.4f}")
    print(f"Third best interval start: {best_third_time:.2f}s")
    print(f"Mean correlation:    {best_third_val:.4f}")


 # plt.figure(figsize=(10, 5))
    plt.plot(common_times, mean_corrs, label="Correlação Média", color='black', linewidth=2)
    plt.axvline(best_global_time, color='r', linestyle='--', label=f"Melhor Início: {best_global_time:.2f}s")
    plt.axvline(best_second_time, color='g', linestyle='--', label=f"Segundo Melhor Início: {best_second_time:.2f}s")
    plt.axvline(best_third_time, color='b', linestyle='--', label=f"Terceiro Melhor Início: {best_third_time:.2f}s")
    plt.title(f"Correlação Média de Todos os Sinais (Máx: {best_global_val:.3f})")
    plt.xlabel("Tempo de início do ECG (s)")
    plt.ylabel("Coeficiente de Correlação")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ======================================================
    # FIGURA 4 — Visualização da Translação e Encaixe de Sinais
    # ======================================================
    
    print("Gerando visualização de translação...")
    # 1. Perfil de FC do ECG completo (calculado a cada segundo)
    full_ecg_hr_profile = []
    profile_times = []
    for s in range(0, len(ecg) - window_15s_samples, STEP_SAMPLES):
        seg = ecg[s : s + window_15s_samples]
        full_ecg_hr_profile.append(estimate_hr_heartpy(seg, FS))
        profile_times.append(s / FS)
    
    full_ecg_hr_profile = np.array(full_ecg_hr_profile)
    profile_times = np.array(profile_times)

    # Dados do rPPG (Template)
    example_hr_file = hr_files[0]
    ppg_hr_points = best_hr_matches[example_hr_file][0]
    best_t = best_global_time

    # Tempos relativos dos pontos do rPPG (7.5s, 22.5s, etc.)
    ppg_rel_times = np.arange(NUM_WINDOWS) * WINDOW_15S + (WINDOW_15S / 2)

    fig4, (v_ax1, v_ax2, v_ax3) = plt.subplots(3, 1, figsize=(12, 12))
    # fig4.suptitle(f"Explicação da Sincronização: {example_hr_file}", fontsize=16)

    # SUBFIGURA 1: Perfil de FC do ECG ao longo do tempo total
    v_ax1.plot(profile_times, full_ecg_hr_profile, color='blue', alpha=0.6, label='FC extraída do ECG (Ground Truth)')
    v_ax1.set_title("1. Perfil de Frequência Cardíaca do ECG (Tempo Total)")
    v_ax1.set_ylabel("FC (bpm)")
    v_ax1.grid(True, linestyle=':')
    v_ax1.legend()

    # SUBFIGURA 2: Previsões do rPPG (Template)
    # Mostra os 105 segundos de dados vindos do TXT
    v_ax2.plot(ppg_rel_times, ppg_hr_points, 'ro-', label='FC rPPG (janelas de 15s)')
    for i, txt in enumerate(ppg_hr_points):
        v_ax2.annotate(f"{txt:.1f}", (ppg_rel_times[i], ppg_hr_points[i]+1), ha='center')
        # Desenha as bordas das janelas de 15s para clareza
        v_ax2.axvspan(i*WINDOW_15S, (i+1)*WINDOW_15S, color='gray', alpha=0.05)
    
    v_ax2.set_xlim(0, TOTAL_ANALYSIS_TIME)
    v_ax2.set_title(f"2. Sequência de FC de um dos métodos rPPG")
    v_ax2.set_xlabel("Tempo Relativo (s)")
    v_ax2.set_ylabel("FC (bpm)")
    v_ax2.grid(True, linestyle=':')
    v_ax2.legend()

    # SUBFIGURA 3: Translação e Alinhamento
    # Aqui plotamos o ECG novamente, mas sobrepomos o rPPG deslocado pelo melhor tempo
    v_ax3.plot(profile_times, full_ecg_hr_profile, color='blue', alpha=0.3, label='Perfil ECG')
    
    # rPPG transladado (Tempo Original + Deslocamento Encontrado)
    shifted_ppg_times = ppg_rel_times + best_t
    v_ax3.plot(shifted_ppg_times, ppg_hr_points, 'ro-', linewidth=2, label='rPPG Transladado (Melhor Encaixe)')
    
    # Destacar a área do "match"
    v_ax3.axvspan(best_t, best_t + TOTAL_ANALYSIS_TIME, color='green', alpha=0.1, label='Janela Sincronizada')
    
    v_ax3.set_title(f"3. Translação: rPPG alinhado ao ECG (Offset: {best_t:.1f}s)")
    v_ax3.set_xlabel("Tempo no ECG (s)")
    v_ax3.set_ylabel("FC (bpm)")
    v_ax3.legend()
    v_ax3.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
