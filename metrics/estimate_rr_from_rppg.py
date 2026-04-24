import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d

# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================
FS_RPPG = 25.0        # Frequência de amostragem original (Câmera)
FS_RESAMP = 4.0       # Frequência de re-amostragem das modulações (Hz)
WINDOW_SEC = 30       # Tamanho da janela de análise (Segundos)
RR_MIN_BPM = 4.0      # Frequência respiratória mínima (BPM)
RR_MAX_BPM = 65.0     # Frequência respiratória máxima (BPM)
SD_THRESHOLD = 4.0    # Limite de desvio padrão para fusão robusta (BPM)

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    """Aplica filtro Butterworth para isolar a banda do pulso."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_bvp_features(segment, fs):
    """Detecta picos e vales no sinal rPPG para extração de modulações."""
    # Filtro para realçar o sinal de pulso (0.7 a 3.5 Hz)
    clean_sig = butter_bandpass(segment, 0.7, 3.5, fs)
    
    # Distância mínima entre picos baseada em fisiologia (~130 BPM)
    min_dist = int(fs * 0.45)
    peaks, _ = find_peaks(clean_sig, distance=min_dist)
    troughs, _ = find_peaks(-clean_sig, distance=min_dist)
    
    # Alinhamento de pares (Vale anterior ao Pico correspondente)
    pairs = []
    for tr in troughs:
        next_pks = peaks[peaks > tr]
        if len(next_pks) > 0:
            pairs.append((tr, next_pks[0]))
            
    return peaks, troughs, pairs

def berger_algorithm(peak_times, fs_target, duration):
    """Implementa o Algoritmo de Berger para re-amostragem da modulação de frequência."""
    t_new = np.arange(0, duration, 1/fs_target)
    y_new = np.zeros_like(t_new)
    dt = 1/fs_target
    
    for i, t_start in enumerate(t_new):
        t_end = t_start + dt
        acc = 0
        for j in range(len(peak_times) - 1):
            p1, p2 = peak_times[j], peak_times[j+1]
            # Calcula a contribuição de cada intervalo IBI na janela do novo passo de tempo
            overlap = max(0, min(t_end, p2) - max(t_start, p1))
            if overlap > 0:
                acc += overlap * (1.0 / (p2 - p1))
        y_new[i] = acc / dt
    return t_new, y_new

def estimate_rr_fft(sig, fs):
    """Estima a RR encontrando o pico de energia no espectro FFT."""
    n = len(sig)
    n_fft = max(512, 1 << (n-1).bit_length()) # Zero padding para resolução
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    mag = np.abs(np.fft.rfft(sig - np.mean(sig), n=n_fft))
    
    # Máscara para a banda de 4 a 65 RPM
    mask = (freqs >= RR_MIN_BPM/60) & (freqs <= RR_MAX_BPM/60)
    if not np.any(mask):
        return np.nan
    
    peak_idx = np.argmax(mag[mask])
    return freqs[mask][peak_idx] * 60

def process_rppg_file(file_path):
    """Processa um arquivo individual e extrai as sequências de RR."""
    try:
        raw_data = np.loadtxt(file_path)
        signal = raw_data[0, :] if raw_data.ndim > 1 and raw_data.shape[0] < raw_data.shape[1] else raw_data
        if signal.ndim > 1: signal = signal[:, 0]
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        return None

    win_samples = int(WINDOW_SEC * FS_RPPG)
    n_windows = len(signal) // win_samples
    
    results = {'bw': [], 'am': [], 'fm': [], 'fusion': [], 'is_reliable': []}
    
    for i in range(n_windows):
        seg = signal[i*win_samples : (i+1)*win_samples]
        p_idx, t_idx, pairs = get_bvp_features(seg, FS_RPPG)
        
        if len(pairs) < 8: # Mínimo de pulsos para análise espectral válida
            for k in results: results[k].append(np.nan)
            continue
            
        t_orig = np.arange(len(seg)) / FS_RPPG
        t_target = np.arange(0, WINDOW_SEC, 1/FS_RESAMP)
        
        # --- Extração e Re-amostragem das Modulações ---
        
        # 1. Baseline Wander (BW) - Interpolação Linear
        bw_t = [(t_orig[tr] + t_orig[pk])/2 for tr, pk in pairs]
        bw_v = [(seg[tr] + seg[pk])/2 for tr, pk in pairs]
        sig_bw = interp1d(bw_t, bw_v, kind='linear', fill_value='extrapolate')(t_target)
        
        # 2. Amplitude (AM) - Interpolação Linear
        am_t = [(t_orig[tr] + t_orig[pk])/2 for tr, pk in pairs]
        am_v = [seg[pk] - seg[tr] for tr, pk in pairs]
        sig_am = interp1d(am_t, am_v, kind='linear', fill_value='extrapolate')(t_target)
        
        # 3. Frequência (FM) - Algoritmo de Berger
        fm_t = t_orig[p_idx]
        _, sig_fm = berger_algorithm(fm_t, FS_RESAMP, WINDOW_SEC)
        
        # --- Estimativa Espectral (FFT) ---
        rr_bw = estimate_rr_fft(sig_bw, FS_RESAMP)
        rr_am = estimate_rr_fft(sig_am, FS_RESAMP)
        rr_fm = estimate_rr_fft(sig_fm, FS_RESAMP)
        
        # --- Fusão Robusta ---
        ests = np.array([rr_bw, rr_am, rr_fm])
        if not np.any(np.isnan(ests)):
            mean_rr = np.mean(ests)
            std_rr = np.std(ests)
            results['bw'].append(rr_bw)
            results['am'].append(rr_am)
            results['fm'].append(rr_fm)
            results['fusion'].append(mean_rr)
            results['is_reliable'].append(std_rr < SD_THRESHOLD)
        else:
            for k in results: results[k].append(np.nan)
        print(results)

    return results

def run_batch_analysis(folder_path):
    """Varre a pasta e gera os gráficos comparativos."""
    if not os.path.exists(folder_path):
        print(f"Erro: Pasta {folder_path} não encontrada.")
        return
        
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    for f in files:
        res = process_file_data = process_rppg_file(os.path.join(folder_path, f))
        if not res or len(res['fusion']) == 0: continue
        
        wins = np.arange(len(res['fusion']))
        plt.figure(figsize=(12, 6))
        
        # Plot das estimativas individuais
        plt.plot(wins, res['bw'], 'g--', alpha=0.4, label='RR (Mod. Linha de Base)')
        plt.plot(wins, res['am'], 'b--', alpha=0.4, label='RR (Mod. Amplitude)')
        plt.plot(wins, res['fm'], 'r--', alpha=0.4, label='RR (Mod. Frequência)')
        
        # Plot da fusão com destaque para confiabilidade
        fusion = np.array(res['fusion'])
        reliable = np.array(res['is_reliable'], dtype=bool)
        
        plt.scatter(wins[reliable], fusion[reliable], color='black', s=50, label='Fusão (Confiável)', zorder=5)
        plt.scatter(wins[~reliable], fusion[~reliable], color='red', marker='x', label='Fusão (Desvio > 4)', zorder=5)
        plt.plot(wins, fusion, 'k-', alpha=0.2)
        
        plt.title(f"Análise de Frequência Respiratória - Método: {f.split('_')[1] if '_' in f else f}")
        plt.xlabel("Janela (30 segundos)")
        plt.ylabel("Frequência Respiratória (rpm)")
        plt.ylim(RR_MIN_BPM - 2, RR_MAX_BPM + 2)
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Altere para o caminho da sua pasta de sinais BVP
    TARGET_FOLDER = r"C:\Users\Sophia\Documents\rPPG\preliminary_results\L9\bvp"
    run_batch_analysis(TARGET_FOLDER)