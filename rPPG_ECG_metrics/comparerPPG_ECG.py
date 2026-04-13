
import os
import numpy as np
import heartpy as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def filter_ecg(data, sample_rate):
    '''
    function that filters using remove_baseline_wander 
    and visualises result
    '''
    filtered = hp.remove_baseline_wander(data, sample_rate)
    plt.figure(figsize=(12,3))
    plt.title('signal with baseline wander removed')
    plt.plot(filtered)
    plt.show()
    
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
    ecg_csv = r"C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video004_ecg.csv"
    noisy_ecg = False
    # Folder containing the 7 prediction txt files
    predictions_folder = r"C:\Users\Sophia\Documents\rPPG\preliminary_results\vin004\hr_preds"
    
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
        sig = filter_ecg(sig, sample_rate=fs)
    
    window_len = int(window_sec * fs)
    n_windows = len(sig) // window_len
    
    print(f"Calculating Ground Truth HR for {n_windows} windows of {window_sec}s...")
    ecg_hr_values = []
    for i in range(n_windows):
        segment = sig[i * window_len : (i + 1) * window_len]
        hr = estimate_hr_heartpy(segment, fs)
        ecg_hr_values.append(hr)
    
    ecg_hr_values = np.array(ecg_hr_values)

    # Initialize Subplots: 1 (Ground Truth) + number of prediction files
    num_plots = len(txt_files) + 1
    cols = 2
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    # 1st Subplot: Overall Ground Truth HR
    axes[0].plot(ecg_hr_values, marker='o', linestyle='-', markersize=4, color='black', label='ECG Ground Truth')
    axes[0].set_title('ECG')
    axes[0].set_ylabel('Frequência cardíaca (bpm)')
    axes[0].set_xlabel('Janela de Amostragem')
    axes[0].grid(True)
    
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
        
        # Plot in the next available subplot
        ax = axes[i + 1]
        ax.plot(y_true, label='ECG', marker='o', linestyle='-', markersize=3, alpha=0.7)
        ax.plot(y_pred, label='rPPG', marker='x', linestyle='--', markersize=3, alpha=0.9)
        ax.set_title(os.path.splitext(file_name)[0].split("_")[1])
        ax.set_ylabel('Frequência cardíaca (bpm)')
        ax.set_xlabel('Janela de Amostragem')
        ax.legend(fontsize='small')
        ax.grid(True)

    # Hide any unused subplots (if any)
    for j in range(i + 2, len(axes)):
        axes[j].axis('off')

    # plt.suptitle(f"Heart Rate Evaluation - Subject {os.path.basename(os.path.dirname(predictions_folder))}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_evaluation()
