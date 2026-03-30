import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp

# =========================
# Configuration
# =========================

FS = 250  # Sample rate from find_correlation.py
WINDOW_15S = 15  # 15-second window for HR calculation
STEP_SECONDS = 1  # 1-second step for sliding window
ECG_FILE = "../get_ground_truth/ECG/ecg_signal_L9_16-02_16-08.csv" # Path relative to signal_sync folder

# Duration for plotting (e.g., 5 minutes)
PLOT_DURATION_SECONDS = 360

# =========================
# Utility functions
# =========================

def estimate_hr_heartpy(ecg_segment, fs):
    """
    Estimates Heart Rate (HR) from an ECG segment using HeartPy.
    """
    try:
        # HeartPy's process function expects a 1D numpy array
        wd, m = hp.process(
            ecg_segment,
            sample_rate=fs
        )
        hr_value = m['bpm']
        return hr_value
    except Exception:
        return np.nan

# =========================
# Main script
# =========================

def plot_hr_calculation():
    print(f"Loading ECG data from: {ECG_FILE}")
    try:
        ecg = hp.get_data(ECG_FILE)
    except FileNotFoundError:
        print(f"Error: ECG file not found at {ECG_FILE}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return

    print(f"ECG data loaded. Total samples: {len(ecg)}, Duration: {len(ecg)/FS:.2f} seconds")

    # Limit ECG data to PLOT_DURATION_SECONDS for visualization
    ecg_to_plot = ecg[:PLOT_DURATION_SECONDS * FS]
    ecg_time_vector = np.arange(len(ecg_to_plot)) / FS

    window_samples = WINDOW_15S * FS
    step_samples = STEP_SECONDS * FS

    calculated_hrs = []
    hr_times = []

    print("Calculating HR using sliding windows...")
    # Iterate through the ECG data with a sliding window
    # Ensure there's enough data for at least one full window
    if len(ecg) < window_samples:
        print("ECG data is too short to calculate HR with the specified window size.")
        return

    for i in range(0, len(ecg) - window_samples + 1, step_samples):
        segment = ecg[i : i + window_samples]
        hr = estimate_hr_heartpy(segment, FS)
        calculated_hrs.append(hr)
        # Time point for HR is the center of the window
        hr_times.append((i + window_samples / 2) / FS)

    calculated_hrs = np.array(calculated_hrs)
    hr_times = np.array(hr_times)

    # Filter out NaN values for plotting HR
    valid_hr_mask = ~np.isnan(calculated_hrs)
    valid_hr_values = calculated_hrs[valid_hr_mask]
    valid_hr_times = hr_times[valid_hr_mask]

    print(f"HR calculation complete. Found {len(valid_hr_values)} valid HR values.")

    # =========================
    # Plotting
    # =========================

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    # fig.suptitle('Sinal de ECG e Frequência Cardíaca Calculada (Janela Deslizante de 15s)', fontsize=16)

    # Plot raw ECG signal
    ax1.plot(ecg_time_vector, ecg_to_plot, color='blue', linewidth=0.8)
    ax1.set_ylabel('Amplitude do ECG')
    ax1.set_title('Sinal de ECG Bruto')
    ax1.grid(True)

    # Plot calculated HR
    ax2.plot(valid_hr_times, valid_hr_values, color='red', marker='o', linestyle='-', markersize=4, label='FC Calculada (bpm)')
    ax2.set_xlabel('Tempo (segundos)')
    ax2.set_ylabel('Frequência Cardíaca (bpm)')
    ax2.set_title('Frequência Cardíaca ao Longo do Tempo')
    ax2.grid(True)
    ax2.legend()

    # Optional: Highlight an example window on the ECG plot
    # Let's pick a window around the middle of the plotted duration
    example_window_start_sec = PLOT_DURATION_SECONDS / 2 - WINDOW_15S / 2
    example_window_end_sec = PLOT_DURATION_SECONDS / 2 + WINDOW_15S / 2
    
    # Find the closest actual window start time
    # This ensures we highlight an *actual* window that was processed
    closest_hr_time_idx = np.argmin(np.abs(hr_times - (example_window_start_sec + WINDOW_15S/2)))
    if closest_hr_time_idx < len(hr_times):
        actual_window_start_sec = hr_times[closest_hr_time_idx] - WINDOW_15S/2
        actual_window_end_sec = hr_times[closest_hr_time_idx] + WINDOW_15S/2
        
        ax1.axvspan(actual_window_start_sec, actual_window_end_sec, color='green', alpha=0.2, label='Exemplo de Janela de 15s')
        ax1.legend()
        
        # Add a point on the HR plot for this specific window
        if not np.isnan(calculated_hrs[closest_hr_time_idx]):
            ax2.plot(hr_times[closest_hr_time_idx], calculated_hrs[closest_hr_time_idx], 'x', color='green', markersize=10, markeredgewidth=2, label='FC para Janela de Exemplo')
            ax2.legend()


    plt.xlim(0, PLOT_DURATION_SECONDS) # Ensure x-axis limit matches the plotted ECG duration
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    plot_hr_calculation()