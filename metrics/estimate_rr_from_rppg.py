import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.interpolate import interp1d

# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================
FS_RPPG = 25.0        # Frequência de amostragem original (Câmera)
FS_REF = 125.0        # Frequência de amostragem do sinal de referência (Impedância)
FS_RESAMP = 4.0       # Frequência de re-amostragem das modulações (Hz)
WINDOW_SEC = 30       # Tamanho da janela de análise (Segundos)
RR_MIN_BPM = 4.0      # Frequência respiratória mínima (BPM)
RR_MAX_BPM = 65.0     # Frequência respiratória máxima (BPM)
SD_THRESHOLD = 4.0    # Limite de desvio padrão para fusão robusta (BPM)

# Configurações de visualização
PLOT_CONFIG = {
    'rr_estimates': True,  # Estimativas de RR por janela
    'sig_bw': False,        # Sinal temporal da modulação BW
    'sig_am': False,        # Sinal temporal da modulação AM
    'sig_fm': False,        # Sinal temporal da modulação FM
    'psd_bw': False,        # Espectro de potência (PSD) da modulação BW
    'psd_am': False,        # Espectro de potência (PSD) da modulação AM
    'psd_fm': False,        # Espectro de potência (PSD) da modulação FM
    'error_per_window': False, # Erro absoluto por janela comparando modulações
    'psd_ref': True        # Espectro de potência (PSD) do sinal de referência
}

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    """Aplica filtro Butterworth para isolar a banda do pulso."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_bvp_features(segment, fs):
    """Detecta picos e vales no sinal rPPG para extração de modulações."""
    # Filtro para realçar o sinal de pulso
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

def calculate_metrics(y_true, y_pred):
    """Calcula MAE, RMSE e MAPE ignorando NaNs."""
    print(y_true)
    print(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    if len(y_t) == 0:
        return np.nan, np.nan, np.nan
    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p)**2))
    mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100
    return mae, rmse, mape

def process_reference_signal(file_path, fs_ref):
    """Lê o arquivo de referência e calcula RR por janelas."""
    try:
        sig = np.loadtxt(file_path)
        t = np.linspace(0, len(sig)/fs_ref, len(sig))
        if sig.ndim > 1:
            sig = sig[0, :] if sig.shape[0] < sig.shape[1] else sig[:, 0]
            t = np.linspace(0, len(sig)/fs_ref, len(sig))
    except Exception as e:
        print(f"Erro ao ler referência {file_path}: {e}")
        return None, None
    plt.plot(t, sig)
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo (s)')
    plt.show()
    win_samples = int(WINDOW_SEC * fs_ref)
    n_windows = len(sig) // win_samples
    ref_rr = [estimate_rr_fft(sig[i*win_samples : (i+1)*win_samples], fs_ref) 
              for i in range(n_windows)]
    return np.array(ref_rr), sig

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
    
    results = {
        'bw': [], 'am': [], 'fm': [], 'fusion': [], 'is_reliable': [],
        'sig_bw': [], 'sig_am': [], 'sig_fm': []
    }
    
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

        results['sig_bw'].append(sig_bw)
        results['sig_am'].append(sig_am)
        results['sig_fm'].append(sig_fm)
        
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

    # Concatenar sinais para plotagem temporal
    for k in ['sig_bw', 'sig_am', 'sig_fm']:
        if results[k]:
            results[k] = np.concatenate(results[k])
        else:
            results[k] = np.array([])

    return results

def run_batch_analysis(folder_path, ref_path=None):
    """Varre a pasta e gera os gráficos em subplots."""
    if not os.path.exists(folder_path):
        print(f"Erro: Pasta {folder_path} não encontrada.")
        return
        
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    all_results = []

    print(f"Processando {len(files)} arquivos...")
    for f in files:
        res = process_rppg_file(os.path.join(folder_path, f))
        if res and len(res['fusion']) > 0:
            all_results.append((f, res))

    if not all_results:
        print("Nenhum resultado válido processado.")
        return

    ref_rr = None
    ref_sig = None
    if ref_path:
        print(f"Processando sinal de referência: {ref_path}")
        ref_rr, ref_sig = process_reference_signal(ref_path, FS_REF)

    final_stats = []

    num_methods = len(all_results)
    cols = 2
    rows = int(np.ceil(num_methods / cols))

    plots = {}
    def setup_fig(key, title):
        if PLOT_CONFIG.get(key):
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
            fig.suptitle(title, fontsize=16)
            plots[key] = (fig, axes.flatten())

    setup_fig('rr_estimates', 'Estimativas de Frequência Respiratória (RR) por Janela')
    setup_fig('sig_bw', 'Sequências de Modulação de Linha de Base (BW)')
    setup_fig('sig_am', 'Sequências de Modulação de Amplitude (AM)')
    setup_fig('sig_fm', 'Sequências de Modulação de Frequência (FM)')
    setup_fig('psd_bw', 'Espectro de Potência (PSD) - Modulação BW')
    setup_fig('psd_am', 'Espectro de Potência (PSD) - Modulação AM')
    setup_fig('psd_fm', 'Espectro de Potência (PSD) - Modulação FM')
    setup_fig('error_per_window', 'Erro Absoluto por Janela (BPM) - Comparação de Modulações')

    for i, (filename, res) in enumerate(all_results):
        method_name = filename.split('_')[1] if '_' in filename else filename

        # Plot Figura 1 (RR por janela)
        if 'rr_estimates' in plots:
            ax1 = plots['rr_estimates'][1][i]
            wins = np.arange(len(res['fusion']))
            ax1.plot(wins, res['bw'], 'g--', alpha=0.4, label='BW')
            ax1.plot(wins, res['am'], 'b--', alpha=0.4, label='AM')
            ax1.plot(wins, res['fm'], 'r--', alpha=0.4, label='FM')

            fusion = np.array(res['fusion'])
            reliable = np.array(res['is_reliable'], dtype=bool)
            ax1.scatter(wins[reliable], fusion[reliable], color='black', s=30, label='Fusão (OK)')
            ax1.scatter(wins[~reliable], fusion[~reliable], color='red', marker='x', label='Fusão (Instável)')
            ax1.plot(wins, fusion, 'k-', alpha=0.2)

            # Plot da Referência e Cálculo de Métricas
            title_metrics = ""
            if ref_rr is not None:
                min_len = min(len(res['fusion']), len(ref_rr))
                ax1.plot(np.arange(min_len), ref_rr[:min_len], 'k-', linewidth=1.5, label='GT (Ref)', alpha=0.7)
                
                mae, rmse, mape = calculate_metrics(ref_rr[:min_len], fusion[:min_len])
                title_metrics = f"\nMAE:{mae:.1f} RMSE:{rmse:.1f} MAPE:{mape:.1f}%"

                # Adicionar às estatísticas finais para exportação
                final_stats.append({
                    'method': method_name,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'n': min_len
                })

            ax1.set_title(f"RR: {method_name}{title_metrics}")
            ax1.set_ylabel("RPM")
            ax1.set_ylim(RR_MIN_BPM - 5, RR_MAX_BPM + 5)
            ax1.legend(loc='upper right', fontsize='x-small', ncol=2)
            ax1.grid(True, alpha=0.3)

        # Plot Modulação BW (Figura 2)
        norm = lambda x: (x - np.mean(x)) / np.std(x) if len(x) > 0 else x
        
        if 'sig_bw' in plots:
            ax_bw = plots['sig_bw'][1][i]
            t_mod = np.arange(len(res['sig_bw'])) / FS_RESAMP
            ax_bw.plot(t_mod, norm(res['sig_bw']), 'g', alpha=0.7)
            ax_bw.set_title(f"Método: {method_name}")
            ax_bw.set_xlabel("Tempo (s)")
            ax_bw.set_ylabel("Amp. Normalizada")
            ax_bw.grid(True, alpha=0.3)

        # Plot Modulação AM (Figura 3)
        if 'sig_am' in plots:
            ax_am = plots['sig_am'][1][i]
            t_mod = np.arange(len(res['sig_am'])) / FS_RESAMP
            ax_am.plot(t_mod, norm(res['sig_am']), 'b', alpha=0.7)
            ax_am.set_title(f"Método: {method_name}")
            ax_am.set_xlabel("Tempo (s)")
            ax_am.set_ylabel("Amp. Normalizada")
            ax_am.grid(True, alpha=0.3)

        # Plot Modulação FM (Figura 4)
        if 'sig_fm' in plots:
            ax_fm = plots['sig_fm'][1][i]
            t_mod = np.arange(len(res['sig_fm'])) / FS_RESAMP
            ax_fm.plot(t_mod, norm(res['sig_fm']), 'r', alpha=0.7)
            ax_fm.set_title(f"Método: {method_name}")
            ax_fm.set_xlabel("Tempo (s)")
            ax_fm.set_ylabel("Amp. Normalizada")
            ax_fm.grid(True, alpha=0.3)

        def plot_modulation_psd(ax, sig, color, title):
            if len(sig) == 0: return
            fs = FS_RESAMP
            freqs, psd = welch(sig - np.mean(sig), fs=fs, nperseg=len(sig)//2, nfft=2048)
            rpm = freqs * 60
            ax.plot(rpm, psd, color=color, lw=2)
            ax.set_xlim(0, RR_MAX_BPM + 20)
            ax.set_title(f"PSD {title}: {method_name}")
            ax.set_xlabel("RPM")
            ax.set_ylabel("Densidade de Potência")
            ax.axvspan(RR_MIN_BPM, RR_MAX_BPM, color='gray', alpha=0.1, label='Banda Resp.')
            ax.grid(True, alpha=0.3)

        # Plot PSDs (Figuras 5, 6 e 7)
        if 'psd_bw' in plots:
            plot_modulation_psd(plots['psd_bw'][1][i], res['sig_bw'], 'g', 'BW')
        if 'psd_am' in plots:
            plot_modulation_psd(plots['psd_am'][1][i], res['sig_am'], 'b', 'AM')
        if 'psd_fm' in plots:
            plot_modulation_psd(plots['psd_fm'][1][i], res['sig_fm'], 'r', 'FM')

        # Plot Figura de Erro por Janela
        if 'error_per_window' in plots and ref_rr is not None:
            ax_err = plots['error_per_window'][1][i]
            min_len = min(len(res['fusion']), len(ref_rr))
            wins = np.arange(min_len)
            
            # Cálculo dos erros absolutos em relação à referência
            err_bw = np.abs(np.array(res['bw'][:min_len]) - ref_rr[:min_len])
            err_am = np.abs(np.array(res['am'][:min_len]) - ref_rr[:min_len])
            err_fm = np.abs(np.array(res['fm'][:min_len]) - ref_rr[:min_len])
            err_fusion = np.abs(np.array(res['fusion'][:min_len]) - ref_rr[:min_len])
            
            ax_err.plot(wins, err_bw, 'g--', alpha=0.5, label='Erro BW')
            ax_err.plot(wins, err_am, 'b--', alpha=0.5, label='Erro AM')
            ax_err.plot(wins, err_fm, 'r--', alpha=0.5, label='Erro FM')
            ax_err.plot(wins, err_fusion, 'k-', linewidth=1.5, label='Erro Fusão')
            
            ax_err.set_title(f"Erro: {method_name}")
            ax_err.set_ylabel("Erro (BPM)")
            ax_err.set_xlabel("Janela")
            ax_err.legend(loc='upper right', fontsize='x-small', ncol=2)
            ax_err.grid(True, alpha=0.3)

    # Plot do Espectro de Referência (Figura Independente)
    if PLOT_CONFIG.get('psd_ref') and ref_sig is not None:
        fig_psd_ref, ax_ref = plt.subplots(figsize=(10, 6))
        fig_psd_ref.suptitle('Espectro de Potência (PSD) - Sinal de Referência (GT)', fontsize=16)
        
        # Usando Welch no sinal completo para ver a frequência respiratória dominante
        fs = FS_REF
        nfft = 4096 if WINDOW_SEC * fs < 4096 else WINDOW_SEC * fs
        freqs, psd = welch(ref_sig - np.mean(ref_sig), fs=fs, nperseg=int(WINDOW_SEC * fs), nfft=nfft)
        rpm = freqs * 60
        
        ax_ref.plot(rpm, psd, color='black', lw=2, label='Impedância Torácica')
        ax_ref.set_xlim(0, RR_MAX_BPM + 20)
        ax_ref.set_xlabel("RPM")
        ax_ref.set_ylabel("Densidade de Potência")
        ax_ref.axvspan(RR_MIN_BPM, RR_MAX_BPM, color='gray', alpha=0.1, label='Banda Respiratória')
        ax_ref.grid(True, alpha=0.3)
        ax_ref.legend()
        fig_psd_ref.tight_layout()

    # Remover eixos extras
    for key in plots:
        fig, axes = plots[key]
        for j in range(num_methods, len(axes)):
            fig.delaxes(axes[j])
        fig.tight_layout()

    # Salvar resultados em arquivo txt
    if final_stats:
        output_path = os.path.join(os.path.dirname(folder_path), "metrics_rr.txt")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                header = f"{'MÉTODO':<20} | {'MAE (RPM)':<10} | {'RMSE (RPM)':<10} | {'MAPE (%)':<10} | {'N'}"
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for s in final_stats:
                    line = f"{s['method']:<20} | {s['mae']:<10.2f} | {s['rmse']:<10.2f} | {s['mape']:<10.2f} | {s['n']}"
                    f.write(line + "\n")
            print(f"\n[INFO] Métricas de erro exportadas para: {output_path}")
        except Exception as e:
            print(f"Erro ao salvar arquivo de métricas: {e}")

    print("Exibindo gráficos...")
    plt.show()

if __name__ == "__main__":
    # Altere para o caminho da sua pasta de sinais BVP
    TARGET_FOLDER = r"../preliminary_results/L9/bvp_dl"
    REF_FILE = r"../get_ground_truth/thoracic_impedance/L9_16-05-26_16-07-25.txt"
    run_batch_analysis(TARGET_FOLDER, ref_path=REF_FILE)