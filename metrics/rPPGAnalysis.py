import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import heartpy as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as ticker

class rPPGAnalysis:
    def __init__(self, video_path: Path, 
                 ecg_data_path: Path, 
                 ppg_data_path: Path, 
                 respiration_data_path: Path, 
                 rPPG_folder_path: Path, 
                 hr_window_size: int, 
                 respiration_window_size: int):
        self.video_path = video_path
        self.gt_ecg_path = ecg_data_path
        self.gt_ppg_path = ppg_data_path
        self.gt_respiration_path = respiration_data_path
        self.rPPG_folder_path = rPPG_folder_path
        self.ecg_sample_rate = 250
        self.gt_ppg_sample_rate = 62.5
        self.gt_respiration_sample_rate = 125
        self.frame_count = self._get_frame_count()
        self.video_fps = self._get_video_fps()
        self.hr_window_size = hr_window_size
        self.respiration_window_size = respiration_window_size
        self.rppg_signals = self._load_rppg_signals()

    def _load_rppg_signals(self):
        rppg_signals = {}

        txt_files = [
            f for f in os.listdir(self.rPPG_folder_path)
            if f.endswith('.txt')
        ]

        for file_name in txt_files:
            file_path = os.path.join(self.rPPG_folder_path, file_name)

            method_name = (
                os.path.splitext(file_name)[0].split('_')[1]
                if '_' in file_name
                else os.path.splitext(file_name)[0]
            )

            signal = self._load_signal(file_path)

            rppg_signals[method_name] = signal

        return rppg_signals
    

    def _get_frame_count(self):
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return frame_count
    
    def _get_video_fps(self):
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        return fps

    def _calculate_fft_hr(self, low_pass=0.6, high_pass=3.3):
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
        N = 1500 #_next_power_of_2(ppg_signal.shape[1])     # zero-padding para resolução de 1 bpm
        f_ppg, pxx_ppg = scipy.signal.periodogram(self.rppg_signal, fs=self.video_fps, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr

    @staticmethod
    def _next_power_of_2(x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    @staticmethod
    def normalize_signal(s):
        return (s - np.mean(s)) / np.std(s)

    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def estimate_hr_heartpy(segment, fs):
        try:
            wd, m = hp.process(segment, sample_rate=fs)
            return m['bpm']
        except Exception:
            return np.nan

    @staticmethod
    def filter_ecg(data, sample_rate):
        return hp.remove_baseline_wander(data, sample_rate)

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    @staticmethod
    def load_signal(file_path):
        data = np.loadtxt(file_path)
        if data.ndim > 1:
            return data[0, :] if data.shape[0] < data.shape[1] else data[:, 0]
        return data

    def compare_hr_rppg_ecg(self):

        results = {}

        window_len_rppg = int(self.hr_window_size * self.video_fps)
        n_windows_rppg = len(rppg_signal) // window_len_rppg
        window_len_ecg = int(self.hr_window_size * self.ecg_sample_rate)
        n_windows_ecg = len(ecg) // window_len_ecg

        ecg = hp.get_data(self.gt_ecg_path)
        ecg = self.filter_ecg(ecg, sample_rate=self.ecg_sample_rate)

        print(f"Calculating Ground Truth HR for {n_windows_ecg} windows...")
        ecg_hr_values = []
        for i in range(n_windows_ecg):
            segment = ecg[i * window_len_ecg : (i + 1) * window_len_ecg]
            hr = self.estimate_hr_heartpy(segment, self.ecg_sample_rate)
            ecg_hr_values.append(hr)
        
        ecg_hr_values = np.array(ecg_hr_values)

        for i in self.rppg_signals:
            rppg_signal = self.rppg_signals[i]
            rppg_signal = self.bandpass_filter(rppg_signal, lowcut=0.6, highcut=3.3, fs=self.video_fps)
            
            print(f"Calculating rPPG HR for {n_windows_rppg} windows using method '{i}'...")
            rppg_hr_values = []
            for j in range(n_windows_rppg):
                segment = rppg_signal[j * window_len_rppg : (j + 1) * window_len_rppg]
                hr = self.estimate_hr_heartpy(segment, self.video_fps)
                rppg_hr_values.append(hr)

            rppg_hr_values = np.array(rppg_hr_values)

            # Align lengths
            min_length = min(len(ecg_hr_values), len(rppg_hr_values))
            ecg_hr_aligned = ecg_hr_values[:min_length]
            rppg_hr_aligned = rppg_hr_values[:min_length]

            mae = mean_absolute_error(ecg_hr_aligned, rppg_hr_aligned)
            rmse = np.sqrt(mean_squared_error(ecg_hr_aligned, rppg_hr_aligned))
            mape = self.calculate_mape(ecg_hr_aligned, rppg_hr_aligned)

            snr_values = []
            for win_idx in range(min_length):
                if np.isnan(ecg_hr_values[win_idx]): continue
                
                start, end = win_idx * window_len_rppg, (win_idx + 1) * window_len_rppg
                if end > len(rppg_signal): break
                
                snr = self._calculate_SNR(rppg_signal[start:end], ecg_hr_values[win_idx], fs=self.video_fps)
                snr_values.append(snr)
            if snr_values: avg_snr = np.mean(snr_values)
            
            results[i] = ({
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'snr': avg_snr,
            })
        
        return results

    def compare_rr_rppg_ref(self):
        pass

    def compare_rppg_ppg(self):
        pass