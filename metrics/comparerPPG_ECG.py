
import os
import numpy as np
import heartpy as hp
import scipy.signal
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def power2db(mag):
    """Convert power to dB."""
    return 10 * np.log10(mag)

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum 
        around the first and second harmonics of the ground truth HR frequency.
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0:
        return power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    return 0

def filter_ecg(data, sample_rate):
    """Filters ECG data using HeartPy's remove_baseline_wander."""
    filtered = hp.remove_baseline_wander(data, sample_rate)
    return filtered

def estimate_hr_heartpy(segment, fs):
    """
    Estimates Heart Rate (HR) from a signal segment using HeartPy.
    Returns NaN if the process fails.
    """
    try:
        wd, m = hp.process(segment, sample_rate=fs)
        return m['bpm']
    except Exception:
        return np.nan

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_evaluation():
    # =========================
    # Configuration
    # =========================
    # Path to the ground truth ECG CSV 
    ecg_csv = "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L8_16-45-38_16-47-38.csv"
    # Define se os gráficos do ECG (sinal e HR baseline) serão exibidos
    SHOW_ECG_PLOT = True
    # Define o número de colunas nos gráficos de subplots
    PLOT_COLS = 2
    # Folder containing the 7 prediction txt files
    predictions_folder = "/home/soph/rppg/rPPG/preliminary_results/L8/dl_hr_preds"
    # Folder containing the raw BVP signals (waveforms) for SNR calculation
    bvp_signals_folder = "/home/soph/rppg/rPPG/preliminary_results/L8/bvp_dl"

    DL_analysis = True
    
    fs_ecg = 250      # Sample rate of the input ECG
    fs_camera = 25     # FPS do vídeo/rPPG
    window_sec = 15    # Window size in seconds

    if not os.path.exists(ecg_csv):
        print(f"Error: ECG file not found at {ecg_csv}")
        return

    # List all txt files in the folder first to determine plot grid size
    if not os.path.exists(predictions_folder):
        print(f"Error: Predictions folder not found at {predictions_folder}")
        return
    txt_files = [f for f in os.listdir(predictions_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in {predictions_folder}")
        return

    # =========================
    # 1. Calculate Ground Truth HR from ECG
    # =========================
    print(f"Loading ECG data from: {ecg_csv}")
    sig = hp.get_data(ecg_csv)
    
    # Sempre filtra o sinal para garantir qualidade na estimativa da FC
    filtered_sig = filter_ecg(sig, sample_rate=fs_ecg)

    if SHOW_ECG_PLOT:
        # Plot do sinal ECG original e filtrado
        plt.figure(figsize=(12,3))
        plt.title('Sinal ECG Original e Filtrado (Baseline Wander Removido)')
        plt.plot(sig, label='Original', alpha=0.7)
        plt.plot(filtered_sig, label='Filtrado', color='red')
        plt.legend()
        plt.show()
    
    sig = filtered_sig
    
    window_len = int(window_sec * fs_ecg)
    n_windows = len(sig) // window_len
    
    print(f"Calculating Ground Truth HR for {n_windows} windows...")
    ecg_hr_values = []
    for i in range(n_windows):
        segment = sig[i * window_len : (i + 1) * window_len]
        hr = estimate_hr_heartpy(segment, fs_ecg)
        ecg_hr_values.append(hr)
    
    ecg_hr_values = np.array(ecg_hr_values)

    # Figura 1: Comparação de FC (Ground Truth + Métodos)
    num_hr = (1 if SHOW_ECG_PLOT else 0) + len(txt_files)
    rows_hr = int(np.ceil(num_hr / PLOT_COLS))
    fig_hr, axes_hr = plt.subplots(rows_hr, PLOT_COLS, figsize=(7 * PLOT_COLS, 4 * rows_hr), squeeze=False)
    axes_hr = axes_hr.flatten()

    # Figura 2: Erro Absoluto por Janela
    num_err = len(txt_files)
    rows_err = int(np.ceil(num_err / PLOT_COLS))
    fig_err, axes_err = plt.subplots(rows_err, PLOT_COLS, figsize=(7 * PLOT_COLS, 4 * rows_err), squeeze=False)
    axes_err = axes_err.flatten()
    
    hr_plot_offset = 0
    if SHOW_ECG_PLOT:
        # Plot do Ground Truth na Figura de FC
        axes_hr[0].plot(ecg_hr_values, marker='o', linestyle='-', markersize=4, color='black', label='ECG Ground Truth')
        axes_hr[0].set_title('ECG')
        axes_hr[0].set_ylabel('Frequência cardíaca (bpm)')
        axes_hr[0].set_xlabel('Janela de Amostragem')
        axes_hr[0].grid(True)
        hr_plot_offset = 1
    
    # =========================
    # 2. Compare with Prediction Files
    # =========================
    print("\nEvaluating files in folder...")
    results = []

    for i, file_name in enumerate(txt_files):
        file_path = os.path.join(predictions_folder, file_name)
        
        # Identificar o método para buscar o sinal BVP correspondente
        method_name = os.path.splitext(file_name)[0].split('_')[1] if '_' in file_name else os.path.splitext(file_name)[0]
        
        # Load predictions
        try:
            predictions = np.loadtxt(file_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
            
        # Ensure predictions is 1D
        if predictions.ndim == 0:
            predictions = np.array([predictions])
            
        # Match lengths (in case predictions or ECG windows are longer)
        min_len = min(len(ecg_hr_values), len(predictions))
        y_true = ecg_hr_values[:min_len]
        y_pred = predictions[:min_len]
        
        # Handle NaNs: only calculate metrics where both values are valid
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            print(f"File {file_name}: No valid HR pairs for comparison.")
            continue

        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape = calculate_mape(y_true_clean, y_pred_clean)
        
        # --- Cálculo de SNR ---
        avg_snr = np.nan
        if DL_analysis:
            bvp_file = os.path.join(bvp_signals_folder, f"BVP_{method_name}_subject1.txt")
        else:
            bvp_file = os.path.join(bvp_signals_folder, f"BVP_{method_name}_0_0.txt")
        # print(bvp_file)
        if os.path.exists(bvp_file):
            # print('found bvp file')
            try:
                bvp_sig = np.loadtxt(bvp_file)
                if bvp_sig.ndim > 1: bvp_sig = bvp_sig[0, :] if bvp_sig.shape[0] < bvp_sig.shape[1] else bvp_sig[:, 0]
                
                snr_values = []
                win_len_cam = int(window_sec * fs_camera)
                for win_idx in range(min_len):
                    if np.isnan(ecg_hr_values[win_idx]): continue
                    
                    start, end = win_idx * win_len_cam, (win_idx + 1) * win_len_cam
                    if end > len(bvp_sig): break
                    
                    snr = _calculate_SNR(bvp_sig[start:end], ecg_hr_values[win_idx], fs=fs_camera)
                    snr_values.append(snr)
                
                if snr_values: avg_snr = np.mean(snr_values)
            except Exception as e:
                print(f"  Warning: Could not calculate SNR for {method_name}: {e}")

        results.append({
            'file': file_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'snr': avg_snr,
            'samples': len(y_true_clean)
        })
        
        print(f"--- {file_name} ---")
        print(f"  MAE:  {mae:.2f} BPM")
        print(f"  RMSE: {rmse:.2f} BPM")
        print(f"  MAPE: {mape:.2f}%")
        if not np.isnan(avg_snr): print(f"  SNR:  {avg_snr:.2f} dB")
        
        # Plot de comparação de FC na Figura 1
        ax_hr = axes_hr[i + hr_plot_offset]
        ax_hr.plot(y_true, label='ECG', marker='o', linestyle='-', markersize=3, alpha=0.7)
        ax_hr.plot(y_pred, label='rPPG', marker='x', linestyle='--', markersize=3, alpha=0.9)
        ax_hr.set_title(f"{os.path.splitext(file_name)[0].split('_')[1]}")
        ax_hr.set_ylabel('Frequência cardíaca (bpm)')
        ax_hr.set_xlabel('Janela de Amostragem')
        ax_hr.legend(fontsize='small')
        ax_hr.grid(True)

        # Calcular o erro absoluto por janela para os dados válidos
        abs_error_per_window = np.abs(y_true_clean - y_pred_clean)
        
        # Plot do Erro Absoluto na Figura 2 (Gráfico de Linha)
        ax_mae = axes_err[i]
        window_indices = np.arange(min_len)[mask] # Obter os índices originais das janelas para os dados válidos
        ax_mae.plot(window_indices, abs_error_per_window, color='tab:red', marker='s', linestyle='-', markersize=3, alpha=0.8)
        ax_mae.set_title(f"{os.path.splitext(file_name)[0].split('_')[1]} - Erro Absoluto por Janela")
        ax_mae.set_ylabel('Erro Absoluto (BPM)')
        ax_mae.set_xlabel('Janela de Amostragem')
        ax_mae.grid(True)

    # Ocultar subplots não utilizados em ambas as figuras
    for j in range(num_hr, len(axes_hr)):
        axes_hr[j].axis('off')
    for j in range(num_err, len(axes_err)):
        axes_err[j].axis('off')

    fig_hr.tight_layout()
    fig_err.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_evaluation()
