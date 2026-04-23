
import os
import numpy as np
import heartpy as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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
    ecg_csv = r"C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video023_ecg.csv"
    noisy_ecg = False
    # Folder containing the 7 prediction txt files
    predictions_folder = r"C:\Users\Sophia\Documents\rPPG\preliminary_results\vin023\hr_preds"
    
    fs = 1000         # Sample rate of the input ECG
    window_sec = 15   # Window size in seconds

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
    if noisy_ecg:
        # Optionally plot the filtered ECG if noisy_ecg is True
        plt.figure(figsize=(12,3))
        plt.title('Sinal ECG Original e Filtrado (Baseline Wander Removido)')
        plt.plot(sig, label='Original', alpha=0.7)
        sig = filter_ecg(sig, sample_rate=fs)
        plt.plot(sig, label='Filtrado', color='red')
        plt.legend()
        plt.show()
    
    window_len = int(window_sec * fs)
    n_windows = len(sig) // window_len
    
    print(f"Calculating Ground Truth HR for {n_windows} windows of {window_sec}s...")
    ecg_hr_values = []
    for i in range(n_windows):
        segment = sig[i * window_len : (i + 1) * window_len]
        hr = estimate_hr_heartpy(segment, fs)
        ecg_hr_values.append(hr)
    
    ecg_hr_values = np.array(ecg_hr_values)

    # Figura 1: Comparação de FC (Ground Truth + Métodos)
    num_hr = 1 + len(txt_files)
    rows_hr = int(np.ceil(num_hr / 2))
    fig_hr, axes_hr = plt.subplots(rows_hr, 2, figsize=(16, 4 * rows_hr))
    axes_hr = axes_hr.flatten()

    # Figura 2: Erro Absoluto por Janela
    num_err = len(txt_files)
    rows_err = int(np.ceil(num_err / 2))
    fig_err, axes_err = plt.subplots(rows_err, 2, figsize=(16, 4 * rows_err))
    axes_err = axes_err.flatten()
    
    # Plot do Ground Truth na Figura de FC
    axes_hr[0].plot(ecg_hr_values, marker='o', linestyle='-', markersize=4, color='black', label='ECG Ground Truth')
    axes_hr[0].set_title('ECG')
    axes_hr[0].set_ylabel('Frequência cardíaca (bpm)')
    axes_hr[0].set_xlabel('Janela de Amostragem')
    axes_hr[0].grid(True)
    
    # =========================
    # 2. Compare with Prediction Files
    # =========================
    print("\nEvaluating files in folder...")
    results = []

    for i, file_name in enumerate(txt_files):
        file_path = os.path.join(predictions_folder, file_name)
        
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
        
        results.append({
            'file': file_name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'samples': len(y_true_clean)
        })
        
        print(f"--- {file_name} ---")
        print(f"  MAE:  {mae:.2f} BPM")
        print(f"  RMSE: {rmse:.2f} BPM")
        print(f"  MAPE: {mape:.2f}%")
        
        # Plot de comparação de FC na Figura 1
        ax_hr = axes_hr[i + 1]
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
