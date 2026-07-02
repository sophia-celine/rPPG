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
from scipy.sparse import spdiags

class rPPGAnalysis:
    def __init__(self, video_path: Path,
                 ecg_data_path: Path,
                 ppg_data_path: Path,
                 respiration_data_path: Path,
                 rPPG_folder_path: Path,
                 hr_window_size: int,
                 respiration_window_size: int,
                 auto_run: bool = False):
        """Create analysis object.

        If `auto_run` is False (default), heavy operations are deferred
        and must be triggered by calling `run()`.
        """
        self.video_path = video_path
        self.gt_ecg_path = ecg_data_path
        self.gt_ppg_path = ppg_data_path
        self.gt_respiration_path = respiration_data_path
        self.rPPG_folder_path = rPPG_folder_path
        self.ecg_sample_rate = 250
        self.gt_ppg_sample_rate = 62.5
        self.gt_respiration_sample_rate = 125

        # lightweight config
        self.hr_window_size = hr_window_size
        self.respiration_window_size = respiration_window_size

        # deferred/computed values (initialized to None)
        self.frame_count = None
        self.video_fps = None
        self.rppg_signals = None
        self.ecg_hr_values = None
        self.rppg_hr_values = None
        self.hr_results = None
        self.NCC_results = None

        if auto_run:
            self.run()

    def run(self):
        """Execute the heavy loading and processing steps.

        Call this explicitly when you want to perform I/O and compute results.
        """
        # populate video metadata
        self.frame_count = self._get_frame_count()
        self.video_fps = self._get_video_fps()

        # load signals and compute HR values
        self.rppg_signals = self._load_rppg_signals()
        self.ecg_hr_values = self._estimate_hr_ecg()
        self.rppg_hr_values = self._estimate_hr_rppg()
        self.hr_results = self.compare_hr_rppg_ecg()
        self.NCC_results = self.compare_rppg_ppg()

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

    def _calculate_fft_hr(self, segment, low_pass=0.6, high_pass=3.3):
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
        segment = np.expand_dims(segment, 0)
        N = self._next_power_of_2(segment.shape[1])     # @TODO colocar zero-padding para resolução de 1 bpm
        f_ppg, pxx_ppg = scipy.signal.periodogram(segment, fs=self.video_fps, nfft=N, detrend=False)
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
        [b, a] = butter(1, [lowcut / fs * 2, highcut / fs * 2], btype='bandpass')
        return filtfilt(b, a, np.double(data))
    
    @staticmethod   
    def power2db(mag):
        """Convert power to dB."""
        return 10 * np.log10(mag)

    def _load_signal(self, file_path):
        data = np.loadtxt(file_path)
        if data.ndim > 1:
            return data[0, :] if data.shape[0] < data.shape[1] else data[:, 0]
        return data
    
    def _detrend(self, input_signal, lambda_value):
        """Detrend PPG signal."""
        signal_length = input_signal.shape[0]
        # observation matrix
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index,
                    (signal_length - 2), signal_length).toarray()
        detrended_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
        return detrended_signal
    
    def _calculate_SNR(self, pred_ppg_signal, hr_label, low_pass=0.6, high_pass=3.3):
        """Calculate SNR as the ratio of the area under the curve of the frequency spectrum 
            around the first and second harmonics of the ground truth HR frequency.
        """
        # Get the first and second harmonics of the ground truth HR in Hz
        first_harmonic_freq = hr_label / 60
        second_harmonic_freq = 2 * first_harmonic_freq
        deviation = 6 / 60  # 6 beats/min converted to Hz

        # Calculate FFT
        pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
        N = self._next_power_of_2(pred_ppg_signal.shape[1])
        f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=self.video_fps, nfft=N, detrend=False)

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
            return self.power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
        return 0

    def _estimate_hr_ecg(self):

        ecg = hp.get_data(self.gt_ecg_path)
        ecg = self.filter_ecg(ecg, sample_rate=self.ecg_sample_rate)
        window_len_ecg = int(self.hr_window_size * self.ecg_sample_rate)
        n_windows_ecg = len(ecg) // window_len_ecg

        print(f"Calculating Ground Truth HR for {n_windows_ecg} windows...")
        ecg_hr_values = []
        for i in range(n_windows_ecg):
            segment = ecg[i * window_len_ecg : (i + 1) * window_len_ecg]
            hr = self.estimate_hr_heartpy(segment, self.ecg_sample_rate)
            ecg_hr_values.append(hr)
        
        return np.array(ecg_hr_values)
    
    def _estimate_hr_rppg(self):

        hr_rppg = {}

        for i in self.rppg_signals:
            rppg_signal = self.rppg_signals[i]
            # rppg_signal = self._detrend(rppg_signal, 100)
            # rppg_signal = self.bandpass_filter(rppg_signal, lowcut=0.6, highcut=3.3, fs=self.video_fps, order=1)
            window_len_rppg = int(self.hr_window_size * self.video_fps)
            n_windows_rppg = len(rppg_signal) // window_len_rppg

            rppg_hr_values = []
            for j in range(n_windows_rppg):
                segment = rppg_signal[j * window_len_rppg : (j + 1) * window_len_rppg]
                segment = self._detrend(segment, 100)
                segment = self.bandpass_filter(segment, lowcut=0.6, highcut=3.3, fs=self.video_fps, order=1)
                hr = self._calculate_fft_hr(segment)
                rppg_hr_values.append(hr)

            rppg_hr_values = np.array(rppg_hr_values)
            hr_rppg[i] = rppg_hr_values
        print(f"Calculating rPPG HR for {n_windows_rppg} windows")
            
        return hr_rppg
    
    def _sync_and_correlate(self, raw_rppg, raw_ppg):
        try:

            # Tratamento para arquivos com múltiplas linhas (ex: Sinal, HR, Time)
            if raw_ppg.ndim > 1:
                raw_ppg = raw_ppg[0, :] if raw_ppg.shape[0] < raw_ppg.shape[1] else raw_ppg[:, 0]
            if raw_rppg.ndim > 1:
                raw_rppg = raw_rppg[0, :] if raw_rppg.shape[0] < raw_rppg.shape[1] else raw_rppg[:, 0]
                
        except Exception as e:
            print(f"Erro ao carregar arquivos: {e}")
            return None, None, None, None

        # 2. Reamostragem baseada no tempo real (considerando frequências diferentes)
        fs_common = max(self.video_fps, self.gt_ppg_sample_rate)
        
        # Duração em segundos baseada na frequência de cada um
        dur_rppg = (len(raw_rppg) - 1) / self.video_fps
        dur_ppg = (len(raw_ppg) - 1) / self.gt_ppg_sample_rate
        
        t_rppg_orig = np.linspace(0, dur_rppg, len(raw_rppg))
        t_ppg_orig = np.linspace(0, dur_ppg, len(raw_ppg))
        
        # Novos eixos de tempo na frequência comum (usar round e garantir >=1 ponto)
        n_new_rppg = max(1, int(np.round(dur_rppg * fs_common)))
        n_new_ppg = max(1, int(np.round(dur_ppg * fs_common)))

        t_new_rppg = np.linspace(0, dur_rppg, n_new_rppg)
        t_new_ppg = np.linspace(0, dur_ppg, n_new_ppg)

        rppg_resampled = interp1d(t_rppg_orig, raw_rppg, kind='cubic')(t_new_rppg)
        ppg_resampled = interp1d(t_ppg_orig, raw_ppg, kind='cubic')(t_new_ppg)

        # Garantir mesmo comprimento após reamostragem — cortar pontos extras se necessário
        if len(rppg_resampled) != len(ppg_resampled):
            min_len_res = min(len(rppg_resampled), len(ppg_resampled))
            rppg_resampled = rppg_resampled[:min_len_res]
            ppg_resampled = ppg_resampled[:min_len_res]

        # 3. Normalização (Essencial para Cross-Correlation e Pearson)
        rppg_norm = self.normalize_signal(rppg_resampled)
        ppg_norm = self.normalize_signal(ppg_resampled)

        # 4. Cross-Correlation para Sincronização
        # A função correlate retorna a correlação para todos os possíveis deslocamentos (lags)
        correlation = signal.correlate(ppg_norm, rppg_norm, mode='full')
        lags = signal.correlation_lags(len(ppg_norm), len(rppg_norm), mode='full')

        # Calcular o coeficiente de correlação cruzada normalizado para cada lag
        coeffs = []
        for lag in lags:
            if lag > 0:
                a = ppg_norm[lag:]
                b = rppg_norm[:len(a)]
            elif lag < 0:
                b = rppg_norm[abs(lag):]
                a = ppg_norm[:len(b)]
            else:
                a = ppg_norm
                b = rppg_norm

            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                coeffs.append(0.0)
            else:
                coeffs.append(np.dot(a, b) / denom)

        coeffs = np.array(coeffs)

        # O lag ideal é onde o coeficiente normalizado tem magnitude máxima
        idx_best = np.argmax(np.abs(coeffs))
        best_lag = lags[idx_best]
        best_coeff = coeffs[idx_best]
        
        # 5. Alinhamento dos sinais com base no lag
        if best_lag > 0:
            # PPG está atrasado em relação ao rPPG
            synced_ppg = ppg_norm[best_lag:]
            synced_rppg = rppg_norm[:len(synced_ppg)]
        elif best_lag < 0:
            # rPPG está atrasado em relação ao PPG
            synced_rppg = rppg_norm[abs(best_lag):]
            synced_ppg = ppg_norm[:len(synced_rppg)]
        else:
            synced_rppg = rppg_norm
            synced_ppg = ppg_norm

        # Garantir que terminem com o mesmo tamanho após o corte
        min_len = min(len(synced_ppg), len(synced_rppg))
        synced_ppg = synced_ppg[:min_len]
        synced_rppg = synced_rppg[:min_len]

        # 6. Usar o coeficiente de correlação cruzada máximo como métrica
        coef_cross = best_coeff

        return coef_cross, best_lag, synced_rppg, synced_ppg

    def compare_hr_rppg_ecg(self):

        hr_results = {}
        window_len_rppg = int(self.hr_window_size * self.video_fps)
        
        for method in self.rppg_hr_values:
            min_length = min(len(self.ecg_hr_values), len(self.rppg_hr_values[method]))
            ecg_hr_aligned = self.ecg_hr_values[:min_length]
            rppg_hr_aligned = self.rppg_hr_values[method][:min_length]

            mae = mean_absolute_error(ecg_hr_aligned, rppg_hr_aligned)
            rmse = np.sqrt(mean_squared_error(ecg_hr_aligned, rppg_hr_aligned))
            mape = self.calculate_mape(ecg_hr_aligned, rppg_hr_aligned)

            snr_values = []
            for win_idx in range(min_length):
                if np.isnan(self.ecg_hr_values[win_idx]): continue
                
                start, end = win_idx * window_len_rppg, (win_idx + 1) * window_len_rppg
                if end > len(self.rppg_signals[method]): break
                
                snr = self._calculate_SNR(self.rppg_signals[method][start:end], self.ecg_hr_values[win_idx])
                snr_values.append(snr)
            if snr_values: avg_snr = np.mean(snr_values)
            
            hr_results[method] = ({
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'snr': avg_snr,
            })
        
        return hr_results

    def compare_rr_rppg_ref(self):

        pass    

    def compare_rppg_ppg(self):
        
        print('Running NCR analysis...')

        gt_ppg_signal = self._load_signal(self.gt_ppg_path)

        NCC_results = {}    
        
        for method in self.rppg_hr_values:
            coef_cross, best_lag, synced_rppg, synced_ppg = self._sync_and_correlate(self.rppg_signals[method], gt_ppg_signal)
            NCC_results[method] = ({
                'coef_cross': coef_cross,
                'best_lag': best_lag,
                'synced_rppg': synced_rppg,
                'synced_ppg': synced_ppg
            })
        
        return NCC_results