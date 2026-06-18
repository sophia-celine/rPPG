import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

# =========================================================
# CONFIGURAÇÕES GLOBAIS
# =========================================================
FS_RPPG = 25.0        # Frequência de amostragem original (Câmera)
FS_REF = 125.0        # Frequência de amostragem do sinal de referência (Impedância)
FS_RESAMP = 4.0       # Frequência de re-amostragem das modulações (Hz)
WINDOW_SEC = 30       # Tamanho da janela de análise (Segundos)
RR_MIN_BPM = 4.0      # Frequência respiratória mínima (RPM)
RR_MAX_BPM = 65.0     # Frequência respiratória máxima (RPM)
SD_THRESHOLD = 4.0    # Limite de desvio padrão para fusão robusta (RPM)
SAVE_TXT = True        
PLOT_LANG = 'pt'      # Idioma dos gráficos: 'en' ou 'pt'

# Dicionário de tradução para os gráficos e logs
TEXT = {
    'en': {
        'method': 'METHOD',
        'mae': 'MAE (RPM)',
        'rmse': 'RMSE (RPM)',
        'mape': 'MAPE (%)',
        'n_samples': 'N',
        'title_bw': 'Baseline Wander Modulation Sequences (BW)',
        'title_am': 'Amplitude Modulation Sequences (AM)',
        'title_fm': 'Frequency Modulation Sequences (FM)',
        'psd_bw': 'Power Spectral Density (PSD) - BW Modulation',
        'psd_am': 'Power Spectral Density (PSD) - AM Modulation',
        'psd_fm': 'Power Spectral Density (PSD) - FM Modulation',
        'rpm': 'RPM',
        'window': 'Window',
        'time': 'Time (s)',
        'norm_amp': 'Normalized Amp.',
        'psd_label': 'Power Density',
        'resp_band': 'Resp. Band',
        'error': 'Error (RPM)',
        'err_bw': 'BW',
        'err_am': 'AM',
        'err_fm': 'FM',
        'err_fusion': 'Fusion',
        'ref_label': 'Thoracic Impedance',
        'resp_band_full': 'Respiratory Band',
        'fusion': 'Fusion',
        'gt_ref': 'GT (Ref)',
    },
    'pt': {
        'method': 'MÉTODO',
        'mae': 'MAE (RPM)',
        'rmse': 'RMSE (RPM)',
        'mape': 'MAPE (%)',
        'n_samples': 'N',
        'title_bw': 'Sequências de Modulação de Linha de Base (BW)',
        'title_am': 'Sequências de Modulação de Amplitude (AM)',
        'title_fm': 'Sequências de Modulação de Frequência (FM)',
        'psd_bw': 'Espectro de Potência (PSD) - Modulação BW',
        'psd_am': 'Espectro de Potência (PSD) - Modulação AM',
        'psd_fm': 'Espectro de Potência (PSD) - Modulação FM',
        'rpm': 'RPM',
        'window': 'Janela',
        'time': 'Tempo (s)',
        'norm_amp': 'Amp. Normalizada',
        'psd_label': 'Densidade de Potência',
        'resp_band': 'Banda Resp.',
        'error': 'Erro (RPM)',
        'err_bw': 'Erro BW',
        'err_am': 'Erro AM',
        'err_fm': 'Erro FM',
        'err_fusion': 'Erro Fusão',
        'ref_label': 'Impedância Torácica',
        'resp_band_full': 'Banda Respiratória',
        'fusion': 'Fusão',
        'gt_ref': 'GT (Ref)',
    }
}

def t(key):
    """Função auxiliar para tradução rápida."""
    return TEXT.get(PLOT_LANG, TEXT['en']).get(key, key)

# Configurações de visualização
PLOT_CONFIG = {
    'rr_estimates': True,  # Estimativas de RR por janela
    'sig_bw': False,        # Sinal temporal da modulação BW
    'sig_am': False,        # Sinal temporal da modulação AM
    'sig_fm': False,        # Sinal temporal da modulação FM
    'psd_bw': True,        # Espectro de potência (PSD) da modulação BW
    'psd_am': True,        # Espectro de potência (PSD) da modulação AM
    'psd_fm': True,        # Espectro de potência (PSD) da modulação FM
    'error_per_window': True, # Erro absoluto por janela comparando modulações
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
    
    # Distância mínima entre picos baseada em fisiologia (~130 RPM)
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

def _next_power_of_2(x):
    """Calcula a potência de 2 mais próxima."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def estimate_rr_fft(sig, fs):
    """Estima a RR encontrando o pico de energia no espectro usando periodograma."""
    low_pass = RR_MIN_BPM / 60
    high_pass = RR_MAX_BPM / 60

    sig_exp = np.expand_dims(sig, 0)
    N = _next_power_of_2(sig_exp.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(sig_exp, fs=fs, nfft=N, detrend='constant')

    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    if len(fmask_ppg) == 0:
        return np.nan

    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    return np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60


# def estimate_rr_fft(sig, fs):
#     """Estima a RR encontrando o pico de energia no espectro usando Welch."""

#     low_pass = RR_MIN_BPM / 60
#     high_pass = RR_MAX_BPM / 60

#     sig = np.asarray(sig)

#     N = _next_power_of_2(len(sig))

#     f_ppg, pxx_ppg = scipy.signal.welch(
#         sig,
#         fs=fs,
#         window='hann',
#         nperseg=len(sig)//2,
#         noverlap=len(sig)//4,
#         nfft=N,
#         detrend='constant'
#     )

#     mask = (f_ppg >= low_pass) & (f_ppg <= high_pass)

#     if not np.any(mask):
#         return np.nan

#     f_band = f_ppg[mask]
#     pxx_band = pxx_ppg[mask]

#     return f_band[np.argmax(pxx_band)] * 60


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
    cols = 1
    rows = int(np.ceil(num_methods / cols))

    plots = {}
    def setup_fig(key, title=""):
        if PLOT_CONFIG.get(key):
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
            if title: 
                fig.suptitle(title, fontsize=16)
            plots[key] = (fig, axes.flatten())

    setup_fig('rr_estimates')
    setup_fig('sig_bw', t('title_bw'))
    setup_fig('sig_am', t('title_am'))
    setup_fig('sig_fm', t('title_fm'))
    setup_fig('psd_bw', t('psd_bw'))
    setup_fig('psd_am', t('psd_am'))
    setup_fig('psd_fm', t('psd_fm'))
    setup_fig('error_per_window')

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
            # reliable = np.array(res['is_reliable'], dtype=bool)
            # ax1.scatter(wins[reliable], fusion[reliable], color='black', s=30, label='Fusão (OK)')
            # ax1.scatter(wins[~reliable], fusion[~reliable], color='red', marker='x', label='Fusão (Instável)')
            ax1.plot(wins, fusion, color='orange', label=t('fusion'))

            # Plot da Referência e Cálculo de Métricas
            title_metrics = ""
            if ref_rr is not None:
                min_len = min(len(res['fusion']), len(ref_rr))
                ax1.plot(np.arange(min_len), ref_rr[:min_len], 'k-', linewidth=1.5, label=t('gt_ref'), alpha=0.7)
                
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

            ax1.set_title(f"{method_name}{title_metrics}", fontsize='xx-large')
            ax1.set_ylabel(t('rpm'), fontsize='xx-large')
            ax1.set_xlabel(t('window'), fontsize='xx-large')
            ax1.tick_params(axis='both', labelsize=16)
            ax1.set_ylim(RR_MIN_BPM - 5, RR_MAX_BPM + 5)
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax1.legend(loc='upper left', fontsize='xx-large', ncol=2)
            ax1.grid(True, alpha=0.3)

        # Plot Modulação BW (Figura 2)
        norm = lambda x: (x - np.mean(x)) / np.std(x) if len(x) > 0 else x
        
        if 'sig_bw' in plots:
            ax_bw = plots['sig_bw'][1][i]
            t_mod = np.arange(len(res['sig_bw'])) / FS_RESAMP
            ax_bw.plot(t_mod, norm(res['sig_bw']), 'g', alpha=0.7)
            ax_bw.set_title(f"{t('method')}: {method_name}")
            ax_bw.set_xlabel(t('time'))
            ax_bw.set_ylabel(t('norm_amp'))
            ax_bw.grid(True, alpha=0.3)

        # Plot Modulação AM (Figura 3)
        if 'sig_am' in plots:
            ax_am = plots['sig_am'][1][i]
            t_mod = np.arange(len(res['sig_am'])) / FS_RESAMP
            ax_am.plot(t_mod, norm(res['sig_am']), 'b', alpha=0.7)
            ax_am.set_title(f"{t('method')}: {method_name}")
            ax_am.set_xlabel(t('time'))
            ax_am.set_ylabel(t('norm_amp'))
            ax_am.grid(True, alpha=0.3)

        # Plot Modulação FM (Figura 4)
        if 'sig_fm' in plots:
            ax_fm = plots['sig_fm'][1][i]
            t_mod = np.arange(len(res['sig_fm'])) / FS_RESAMP
            ax_fm.plot(t_mod, norm(res['sig_fm']), 'r', alpha=0.7)
            ax_fm.set_title(f"{t('method')}: {method_name}")
            ax_fm.set_xlabel(t('time'))
            ax_fm.set_ylabel(t('norm_amp'))
            ax_fm.grid(True, alpha=0.3)

        def plot_modulation_psd(ax, sig, color, title):
            if len(sig) == 0: return
            fs = FS_RESAMP
            nfft = 2048
            # Periodograma em vez de FFT simples para consistência com estimativa
            freqs, psd = scipy.signal.periodogram(sig, fs=fs, nfft=nfft, detrend='constant')
            rpm = freqs * 60
            ax.plot(rpm, psd, color=color, lw=2)
            ax.set_xlim(0, RR_MAX_BPM + 20)
            ax.set_title(f"Spectrum {title}: {method_name}")
            ax.set_xlabel(t('rpm'))
            ax.set_ylabel(t('psd_label'))
            ax.axvspan(RR_MIN_BPM, RR_MAX_BPM, color='gray', alpha=0.1, label=t('resp_band'))
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
            
            ax_err.plot(wins, err_bw, 'g--', alpha=0.5, label=t('err_bw'))
            ax_err.plot(wins, err_am, 'b--', alpha=0.5, label=t('err_am'))
            ax_err.plot(wins, err_fm, 'r--', alpha=0.5, label=t('err_fm'))
            ax_err.plot(wins, err_fusion, 'k-', linewidth=1.5, label=t('err_fusion'))
            
            ax_err.set_title(f"{method_name}", fontsize='xx-large')
            ax_err.set_ylabel(t('error'), fontsize='xx-large')
            ax_err.set_xlabel(t('window'), fontsize='xx-large')
            ax_err.legend(loc='upper left', fontsize='xx-large', ncol=2)
            ax_err.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax_err.tick_params(axis='both', labelsize=14)
            ax_err.grid(True, alpha=0.3)

    # Plot do Espectro de Referência (Figura Independente)
    if PLOT_CONFIG.get('psd_ref') and ref_sig is not None:
        fig_psd_ref, ax_ref = plt.subplots(figsize=(10, 6))
        # fig_psd_ref.suptitle('Espectro de Potência (PSD) - Sinal de Referência (GT)', fontsize=16)
        
        # Periodograma para ver a frequência respiratória dominante
        fs = FS_REF
        nfft = _next_power_of_2(len(ref_sig))
        freqs, psd = scipy.signal.periodogram(ref_sig, fs=fs, nfft=nfft, detrend='constant')
        rpm = freqs * 60
        
        ax_ref.plot(rpm, psd, color='black', lw=2, label=t('ref_label'))
        ax_ref.set_xlim(0, RR_MAX_BPM + 20)
        ax_ref.set_xlabel(t('rpm'))
        ax_ref.set_ylabel(t('psd_label'))
        ax_ref.axvspan(RR_MIN_BPM, RR_MAX_BPM, color='gray', alpha=0.1, label=t('resp_band_full'))
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
    if SAVE_TXT and final_stats:
        output_path = os.path.join(os.path.dirname(folder_path), "metrics_rr.txt")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                header = f"{t('method'):<20} | {t('mae'):<10} | {t('rmse'):<10} | {t('mape'):<10} | {t('n_samples')}"
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for s in final_stats:
                    line = f"{s['method']:<20} | {s['mae']:<10.2f} | {s['rmse']:<10.2f} | {s['mape']:<10.2f} | {s['n']}"
                    f.write(line + "\n")
            msg = "[INFO] Metrics exported to:" if PLOT_LANG == 'en' else "[INFO] Métricas de erro exportadas para:"
            print(f"\n{msg} {output_path}")
        except Exception as e:
            msg_err = "Error saving metrics file:" if PLOT_LANG == 'en' else "Erro ao salvar arquivo de métricas:"
            print(f"{msg_err} {e}")

    msg_show = "Displaying plots..." if PLOT_LANG == 'en' else "Exibindo gráficos..."
    print(msg_show)
    plt.show()

if __name__ == "__main__":
    # Altere para o caminho da sua pasta de sinais BVP
    TARGET_FOLDER = r"../preliminary_results/L8/bvp"
    REF_FILE = r"../get_ground_truth/thoracic_impedance/L8_16-45-38_16-47-38.txt"
    run_batch_analysis(TARGET_FOLDER, ref_path=REF_FILE)