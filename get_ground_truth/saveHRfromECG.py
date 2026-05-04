import numpy as np
import heartpy as hp
from scipy.signal import resample
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import time
import cv2


def filter_and_visualise(data, sample_rate):
    '''
    function that filters using remove_baseline_wander 
    and visualises result
    '''
    
    filtered = hp.remove_baseline_wander(data, sample_rate)

    plt.figure(figsize=(12,3))
    plt.title('signal with baseline wander removed')
    plt.plot(filtered)
    plt.show()

    #And let's plot both original and filtered signal, and zoom in to show peaks are not moved
    #We'll also scale both signals with hp.scale_data
    #This is so that they have the same amplitude so that the overlap is better visible
    plt.figure(figsize=(12,3))
    plt.title('zoomed in signal with baseline wander removed, original signal overlaid')
    plt.plot(data)
    # plt.plot(hp.scale_data(filtered[200:1200]))
    plt.show()
    
    return filtered

def estimate_hr_heartpy(segment, fs):
    """
    Estimates Heart Rate (HR) from a signal segment using HeartPy.
    Returns NaN if the process fails.
    """
    try:
        wd, m = hp.process(segment, sample_rate=fs)
        hr_value = m['bpm']
        return hr_value
    except Exception:
        return np.nan
    
def count_video_frames(vid_path):
    cap = cv2.VideoCapture(vid_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def save_hr_from_ecg():
    # =========================
    # Configuration
    # =========================
    # Path to the CSV file
    input_csv = "/home/soph/rppg/rPPG/get_ground_truth/ECG/vinicius_video023_ecg.csv" #r"C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video023_ecg.csv"
    output_txt = "/home/soph/rppg/rPPG/get_ground_truth/ECG/vinicius_video023_ecg.txt" #r"C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video023_ecg.txt"
    vid_path = "/media/soph/UTIrPPG/20260423_Coleta Vinicius/video023-cropped.avi" #r"C:\Users\Sophia\Videos\Baumer Video Records\VCXU.2-57C\video023-cropped.avi"
    
    fs = 1000       # Sample rate of the input ECG (e.g., 1000 Hz)
    n_points = count_video_frames(vid_path)  # Target number of points for the output (to match video duration/frames)
    window_sec = 15  # Sliding window size in seconds
    step_sec = 1     # Step size in seconds
    noisy = False

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        return

    # =========================
    # Data Loading
    # =========================
    print(f"Loading ECG data from: {input_csv}")
    # hp.get_data reads the CSV into a 1D numpy array
    sig = hp.get_data(input_csv)
    wd, m = hp.process(sig, sample_rate=fs)
    hp.plotter(wd, m)

    if noisy:
        sig = filter_and_visualise(sig, sample_rate=fs)
        wd, m = hp.process(hp.scale_data(sig), sample_rate=fs)
        plt.figure(figsize=(12,4))
        hp.plotter(wd, m)
    
    total_samples = len(sig)
    duration = total_samples / fs
    t_rel = np.arange(total_samples) / fs

    # =========================
    # Processing
    # =========================
    # 1. Resample ECG Signal (Line 1 of output)
    print(f"Resampling signal to {n_points} points...")
    ecg_resampled = resample(sig, n_points)

    # 2. Calculate Timesteps for resampled length (Line 3 of output)
    t_new = np.linspace(t_rel[0], t_rel[-1], n_points)

    # 3. Calculate HR Estimates (Line 2 of output)
    print("Calculating HR using HeartPy sliding window...")
    window_len = int(window_sec * fs)
    step_len = int(step_sec * fs)

    hr_times = []
    hr_values = []

    for i in range(0, total_samples - window_len, step_len):
        segment = sig[i : i + window_len]
        hr = estimate_hr_heartpy(segment, fs)
        hr_values.append(hr)
        # Time point for HR is the center of the window
        hr_times.append(t_rel[i + window_len // 2])

    # Interpolate calculated HR values to match the resampled signal length
    hr_times = np.array(hr_times)
    hr_values = np.array(hr_values)
    valid_mask = ~np.isnan(hr_values)

    if np.sum(valid_mask) > 1:
        f_hr = interp1d(hr_times[valid_mask], hr_values[valid_mask], kind='linear', fill_value="extrapolate")
        hr_resampled = f_hr(t_new)
    else:
        print("Warning: Not enough valid HR values found. Filling with NaN.")
        hr_resampled = np.full(n_points, np.nan)
    plt.plot(hr_times, hr_values)
    plt.xlabel("Tempo (s)")
    plt.ylabel('Frequência cardíaca (bpm)')
    plt.show()
    # =========================
    # Saving
    # =========================
    # Save to file: Line 1=ECG, Line 2=HR, Line 3=Time
    data_save = np.vstack((ecg_resampled, hr_resampled, t_new))
    
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    np.savetxt(output_txt, data_save, fmt='%.7e')
    print(f"Success! Data saved to: {output_txt}")

if __name__ == "__main__":
    save_hr_from_ecg()