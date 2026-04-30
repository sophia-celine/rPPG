import os
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Aplica um filtro passa-banda Butterworth no sinal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def estimate_hr_fft(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
    # we recommend low_pass=0.75 and high_pass=2.5 instead of the defaults above.
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


# def estimate_hr_fft(segment, fs):
#     """
#     Estima a Frequência Cardíaca (HR) usando FFT.
#     Filtra o sinal e busca o pico de magnitude na faixa de 45-210 BPM.
#     """
#     try:
#         # 1. Filtro passa-banda (0.75 Hz a 3.5 Hz ~ 45 a 210 BPM)
#         filtered = bandpass_filter(segment, 0.6, 3.3, fs)
        
#         # 2. Janelamento (Hamming) para reduzir leakage espectral
#         windowed = filtered * np.hamming(len(filtered))
        
#         # 3. FFT
#         n = len(windowed)
#         freqs = np.fft.rfftfreq(n, d=1/fs)
#         fft_mag = np.abs(np.fft.rfft(windowed))
        
#         # 4. Encontrar o pico de frequência na faixa de interesse
#         mask = (freqs >= 0.6) & (freqs <= 3.3)
#         if not np.any(mask):
#             return np.nan
            
#         peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
#         return peak_freq * 60  # Converte Hz para BPM
#     except Exception:
#         return np.nan

def calculate_mape(y_true, y_pred):
    """Calcula o Erro Médio Absoluto Percentual."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_evaluation_ppg_fft():
    # =========================
    # Configuração
    # =========================
    # Caminho para o TXT do PPG (Ground Truth)
    # ppg_txt = r"C:\Users\sophi\OneDrive\Documentos\rPPG\get_ground_truth\spo2\original_spo2_L7_16-22-48_16-24-47.txt"
    ppg_txt = "/home/soph/rppg/rPPG/get_ground_truth/spo2/sp02wave_L7_16-22-48_16-24-47.txt"
    # Pasta contendo os arquivos txt de predição
    predictions_folder = "/home/soph/rppg/rPPG/preliminary_results/L7/dl_hr_preds"
    
    fs = 25        # Frequência do sensor SpO2 (PPG)
    window_sec = 15   # Tamanho da janela em segundos

    if not os.path.exists(ppg_txt):
        print(f"Erro: Arquivo PPG não encontrado em {ppg_txt}")
        return

    if not os.path.exists(predictions_folder):
        print(f"Erro: Pasta de predições não encontrada em {predictions_folder}")
        return
        
    txt_files = [f for f in os.listdir(predictions_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"Nenhum arquivo .txt encontrado em {predictions_folder}")
        return

    # =========================
    # 1. Calcular Estimativas FFT do PPG
    # =========================
    print(f"Carregando dados de PPG: {ppg_txt}")
    sig = np.loadtxt(ppg_txt) 
    
    # Tratamento para garantir que o sinal seja 1D
    if sig.ndim > 1:
        sig = sig[0, :] if sig.shape[0] < sig.shape[1] else sig[:, 0]
    
    window_len = int(window_sec * fs)
    n_windows = len(sig) // window_len
    
    print(f"Calculando HR via FFT para {n_windows} janelas...")
    ppg_fft_hr_values = []
    for i in range(n_windows):
        segment = sig[i * window_len : (i + 1) * window_len]
        hr = estimate_hr_fft(segment, fs)
        ppg_fft_hr_values.append(hr)
    
    ppg_fft_hr_values = np.array(ppg_fft_hr_values)
    print(ppg_fft_hr_values)

    # Configuração dos Gráficos
    num_hr = 1 + len(txt_files)
    rows_hr = int(np.ceil(num_hr / 2))
    fig_hr, axes_hr = plt.subplots(rows_hr, 2, figsize=(16, 4 * rows_hr))
    axes_hr = axes_hr.flatten()

    num_err = len(txt_files)
    rows_err = int(np.ceil(num_err / 2))
    fig_err, axes_err = plt.subplots(rows_err, 2, figsize=(16, 4 * rows_err))
    axes_err = axes_err.flatten()
    
    # Plot do Baseline (PPG FFT)
    axes_hr[0].plot(ppg_fft_hr_values, marker='o', color='green', label='PPG (FFT Baseline)')
    axes_hr[0].set_title('PPG - Estimativa via FFT')
    axes_hr[0].set_ylabel('BPM')
    axes_hr[0].grid(True)

    # =========================
    # 2. Comparação com Predições TXT
    # =========================
    for i, file_name in enumerate(txt_files):
        file_path = os.path.join(predictions_folder, file_name)
        try:
            predictions = np.loadtxt(file_path)
        except Exception as e:
            print(f"Erro ao ler {file_name}: {e}")
            continue
            
        if predictions.ndim == 0: predictions = np.array([predictions])
            
        min_len = min(len(ppg_fft_hr_values), len(predictions))
        y_true = ppg_fft_hr_values[:min_len]
        y_pred = predictions[:min_len]
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0: continue

        num_test_samples = len(y_true_clean)

        # Calculate MAE and Standard Error
        mae = np.mean(np.abs(y_pred_clean - y_true_clean))
        mae_se = np.std(np.abs(y_pred_clean - y_true_clean)) / np.sqrt(num_test_samples)

        # Calculate RMSE and Standard Error
        squared_errors = np.square(y_pred_clean - y_true_clean)
        rmse = np.sqrt(np.mean(squared_errors))
        rmse_se = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))

        # Calculate MAPE and Standard Error
        mape = np.mean(np.abs((y_pred_clean - y_true_clean) / y_true_clean)) * 100
        mape_se = (np.std(np.abs((y_pred_clean - y_true_clean) / y_true_clean)) / np.sqrt(num_test_samples)) * 100

        print(f"--- {file_name} vs FFT ---")
        print("  FFT MAE (FFT Label): {0:.2f} +/- {1:.2f}".format(mae, mae_se))
        print("  FFT RMSE (FFT Label): {0:.2f} +/- {1:.2f}".format(rmse, rmse_se))
        print("  FFT MAPE (FFT Label): {0:.2f}% +/- {1:.2f}%".format(mape, mape_se))

        # Plot Comparação
        ax_hr = axes_hr[i + 1]
        ax_hr.plot(y_true, label='PPG FFT', marker='o', alpha=0.7, color='green')
        ax_hr.plot(y_pred, label='rPPG (Pred)', marker='x', linestyle='--', alpha=0.9)
        ax_hr.set_title(f"{file_name} vs FFT")
        ax_hr.legend()
        ax_hr.grid(True)

        # Plot Erro Absoluto
        ax_mae = axes_err[i]
        ax_mae.plot(np.abs(y_true_clean - y_pred_clean), color='red', marker='s')
        ax_mae.set_title(f"Erro Absoluto: {file_name}")
        ax_mae.grid(True)

    for j in range(num_hr, len(axes_hr)): axes_hr[j].axis('off')
    for j in range(num_err, len(axes_err)): axes_err[j].axis('off')

    fig_hr.tight_layout()
    fig_err.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_evaluation_ppg_fft()
