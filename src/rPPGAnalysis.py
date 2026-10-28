import json
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
import neurokit2 as nk

class rPPGAnalysis:
    def __init__(self, video_path: Path,
                 ecg_data_path: Path,
                 ppg_data_path: str | Path | None,
                 respiration_data_path: str | Path | None,
                 rPPG_folder_path: Path,
                 hr_window_size: int,
                 respiration_window_size: int,
                 auto_run: bool = False,
                 language: str = "pt"):
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
        self.resp_modulation_resample = 4
        self.language = language
        self.translations = self._load_translations(language)
        self.hr_lowcut = 0.5
        self.hr_highcut = 3.3

        # lightweight config
        self.hr_window_size = hr_window_size
        self.respiration_window_size = respiration_window_size

        self.filtered_methods = {
            'CHROM'
        }

        # deferred/computed values (initialized to None)
        self.frame_count = None
        self.video_fps = None
        self.rppg_signals = None
        self.ecg_hr_values = None
        self.rppg_hr_values = None
        self.hr_results = None
        self.correlation_results = None
        self.gt_rr_values = None
        self.ref_rr_signal = None
        self.rppg_rr_values = None
        self.rr_results = None

        if auto_run:
            self.run()

    @staticmethod
    def _1bpm_resolution(x, fs):
        return max(int(x), int(np.ceil(60 * fs)))

    @staticmethod
    def normalize_signal(s):
        return (s - np.mean(s)) / np.std(s)

    @staticmethod
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def _load_translations(self, language):
        translations_dir = Path(__file__).resolve().parent / "translations"
        file_path = translations_dir / f"{language}.json"

        if not file_path.exists():
            fallback_path = translations_dir / "en.json"
            if fallback_path.exists():
                file_path = fallback_path
            else:
                return {}

        with file_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _t(self, key):
        return self.translations.get(key, key)

    @staticmethod
    def estimate_hr_heartpy(segment, fs):
        try:
            wd, m = hp.process(segment, sample_rate=fs, bpmmin=30, bpmmax=200)
            hp.plotter(wd, m)
            plt.show()
            return m['bpm']
        except Exception:
            return np.nan

    @staticmethod
    def estimate_hr_neurokit2(segment, fs):
        try:
            ecg_cleaned = nk.ecg_clean(segment, sampling_rate=fs)
            signals, info = nk.ecg_peaks(
                ecg_cleaned,
                sampling_rate=fs,
                correct_artifacts=True,
                method='promac',
                # method='engzeemod2012', #-- fc mt bom, picos mt bons mas errou alguns picos
                # method='manikandan2012', #-- fc mt bom, pico com um pouco de thresh
                # method='khamis2016', #-- fc mt bom, pico com mt thresh
                # method='promac', #-- fc mt bom, pico bom
            )
            signals["ECG_Clean"] = ecg_cleaned
            signals["ECG_Raw"] = segment
            signals["ECG_Quality"] = nk.ecg_quality(
                ecg_cleaned,
                sampling_rate=fs,
            )
            signals["ECG_Rate"] = nk.signal_rate(
                info["ECG_R_Peaks"],
                sampling_rate=fs,
                desired_length=len(segment),
            )
            # nk.ecg_plot(signals, info)
            # plt.show()
            return float(np.nanmean(signals["ECG_Rate"]))
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

    @staticmethod
    def _load_signal_from_txt(file_path):
        data = np.loadtxt(file_path)
        if data.ndim > 1:
            return data[0, :] if data.shape[0] < data.shape[1] else data[:, 0]
        return data
    

    @staticmethod
    def _berger_algorithm(peak_times, fs_target, duration):
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

            signal = self._load_signal_from_txt(file_path)

            rppg_signals[method_name] = signal

        return rppg_signals

    def _filter_rppg_signals(self, lowcut=0.3, highcut=3.3, order=3):
        filtered_signals = {}
        for i in self.rppg_signals:
            rppg_signal = self.rppg_signals[i]
            if i not in self.filtered_methods:
                rppg_signal = self.bandpass_filter(rppg_signal, lowcut=lowcut, highcut=highcut, fs=self.video_fps, order=order)
            filtered_signals[i] = rppg_signal
        return filtered_signals
    
    def _load_respiratory_signal(self):
        """Lê o arquivo de referência e calcula RR por janelas."""
        if not self.gt_respiration_path:
            return None, None

        try:
            sig = np.loadtxt(self.gt_respiration_path)
            t = np.linspace(0, len(sig)/self.gt_respiration_sample_rate, len(sig))
            if sig.ndim > 1:
                sig = sig[0, :] if sig.shape[0] < sig.shape[1] else sig[:, 0]
                t = np.linspace(0, len(sig)/self.gt_respiration_sample_rate, len(sig))
        except Exception as e:
            print(f"Erro ao ler referência {self.gt_respiration_path}: {e}")
            return None, None
        win_samples = int(self.respiration_window_size * self.gt_respiration_sample_rate)
        n_windows = len(sig) // win_samples
        ref_rr = [self._estimate_rr_fft(sig[i*win_samples : (i+1)*win_samples], self.gt_respiration_sample_rate) 
                for i in range(n_windows)]
        return np.array(ref_rr), sig
    
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

    def _get_bvp_features(self, segment, fs):
        """Detecta picos e vales no sinal rPPG para extração de modulações."""
        # Filtro para realçar o sinal de pulso
        clean_sig = self.bandpass_filter(segment, 0.1, 4, fs)
        
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
    
    def _calculate_SNR(self, pred_ppg_signal, hr_label):
        """Calculate SNR as the ratio of the area under the curve of the frequency spectrum 
            around the first and second harmonics of the ground truth HR frequency.
        """
        # Get the first and second harmonics of the ground truth HR in Hz
        first_harmonic_freq = hr_label / 60
        second_harmonic_freq = 2 * first_harmonic_freq
        deviation = 6 / 60  # 6 beats/min converted to Hz
        low_pass = self.hr_lowcut
        high_pass = self.hr_highcut

        # Calculate FFT
        pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
        N = self._1bpm_resolution(pred_ppg_signal.shape[1], self.video_fps)
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

        # ecg = hp.get_data(self.gt_ecg_path)
        # ecg = self.filter_ecg(ecg, sample_rate=self.ecg_sample_rate)
        ecg = np.loadtxt(self.gt_ecg_path, dtype=np.float64, ndmin=1)
        window_len_ecg = int(self.hr_window_size * self.ecg_sample_rate)
        n_windows_ecg = len(ecg) // window_len_ecg
        print(f"Calculating Ground Truth HR for {n_windows_ecg} windows...")
        
        ecg_hr_values = []
        for i in range(n_windows_ecg):
            segment = ecg[i * window_len_ecg : (i + 1) * window_len_ecg]
            # hr = self.estimate_hr_heartpy(segment, self.ecg_sample_rate)
            hr = self.estimate_hr_neurokit2(segment, self.ecg_sample_rate)
            ecg_hr_values.append(hr)
        
        return np.array(ecg_hr_values)
    
    def _estimate_hr_rppg(self):

        hr_rppg = {}
        filtered_signals = self._filter_rppg_signals(lowcut=self.hr_lowcut, highcut=self.hr_highcut)

        for i in filtered_signals:
            rppg_signal = filtered_signals[i]
            # rppg_signal = self._detrend(rppg_signal, 100)
            print('----------------- ESTIMATING HR FOR METHOD:', i, '-----------------')
            window_len_rppg = int(self.hr_window_size * self.video_fps)
            n_windows_rppg = len(rppg_signal) // window_len_rppg

            rppg_hr_values = []
            for j in range(n_windows_rppg):
                segment = rppg_signal[j * window_len_rppg : (j + 1) * window_len_rppg]
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

        if np.all(np.asarray(raw_ppg) == 0):
            print("Null PPG signal.")
            return None, None, None, None

        # 2. Reamostragem baseada no tempo real (considerando frequências diferentes)
        fs_common = max(self.video_fps, self.gt_ppg_sample_rate)
                
        # Duração em segundos baseada na frequência de cada um
        dur_rppg = (len(raw_rppg) - 1) / self.video_fps
        dur_ppg = (len(raw_ppg) - 1) / self.gt_ppg_sample_rate
        
        t_rppg_orig = np.linspace(0, dur_rppg, len(raw_rppg))
        t_ppg_orig = np.linspace(0, dur_ppg, len(raw_ppg))
        
        # Novos eixos de tempo na frequência comum
        t_new_rppg = np.linspace(0, dur_rppg, int(dur_rppg * fs_common))
        t_new_ppg = np.linspace(0, dur_ppg, int(dur_ppg * fs_common))
        
        rppg_resampled = interp1d(t_rppg_orig, raw_rppg, kind='cubic')(t_new_rppg)
        ppg_resampled = interp1d(t_ppg_orig, raw_ppg, kind='cubic')(t_new_ppg)

        # 3. Normalização (Essencial para Cross-Correlation e Pearson)
        rppg_norm = self.normalize_signal(rppg_resampled)
        ppg_norm = self.normalize_signal(ppg_resampled)

        # 4. Cross-Correlation para Sincronização
        # A função correlate retorna a correlação para todos os possíveis deslocamentos (lags)
        correlation = signal.correlate(ppg_norm, rppg_norm, mode='full')
        lags = signal.correlation_lags(len(ppg_norm), len(rppg_norm), mode='full')
        
        # O lag ideal é onde a correlação é máxima
        best_lag = lags[np.argmax(correlation)]
        
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

        # 6. Cálculo do Coeficiente de Pearson
        coef_pearson, p_valor = pearsonr(synced_ppg, synced_rppg)

        return coef_pearson, best_lag, synced_rppg, synced_ppg

    def _calculate_fft_hr(self, segment):
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
        low_pass = self.hr_lowcut
        high_pass = self.hr_highcut
        seg_td = np.asarray(segment).flatten()
        seg_exp = np.expand_dims(seg_td, 0)
        N = self._1bpm_resolution(seg_td.shape[0], self.video_fps)
        f_ppg, pxx_ppg = scipy.signal.periodogram(seg_exp, fs=self.video_fps, nfft=N, detrend='constant')
        pxx_ppg = np.squeeze(pxx_ppg)
        plot_window = False
        if plot_window:
            # Create a figure with two stacked plots: time-domain signal and spectrum
            fig, (ax_sig, ax_spec) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

            # Time-domain signal (top)
            t = np.arange(seg_td.shape[0]) / float(self.video_fps)
            ax_sig.plot(t, seg_td)
            ax_sig.set_ylabel(self._t("amplitude") if hasattr(self, "_t") else "Amplitude")
            ax_sig.set_title(self._t("ppg_signal_and_spectrum") if hasattr(self, "_t") else "PPG Signal and Spectrum")

            # Frequency-domain (bottom) limited to 0-3 Hz
            ax_spec.plot(f_ppg, pxx_ppg)
            ax_spec.set_xlim(0, 3.0)
            ax_spec.set_xlabel(self._t("frequency_hz") if hasattr(self, "_t") else "Frequency (Hz)")
            ax_spec.set_ylabel(self._t("power") if hasattr(self, "_t") else "Power")

            plt.tight_layout()
            plt.show(block=True)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr
    
    def _estimate_rr_fft(self, sig, fs, rr_min=4, rr_max=65):
        """Estima a RR encontrando o pico de energia no espectro usando periodograma."""
        low_pass = rr_min / 60
        high_pass = rr_max / 60

        sig_exp = np.expand_dims(sig, 0)
        N = self._1bpm_resolution(sig_exp.shape[1], fs)
        f_ppg, pxx_ppg = scipy.signal.periodogram(sig_exp, fs=fs, nfft=N, detrend='constant')

        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        if len(fmask_ppg) == 0:
            return np.nan

        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        return np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    def _estimate_rr_rppg(self):

        rr_rppg = {}
        
        for method in self.rppg_signals:
            results = {
            'bw': [], 'am': [], 'fm': [], 'fusion': [],
            'sig_bw': [], 'sig_am': [], 'sig_fm': []
            }
            win_samples = int(self.respiration_window_size * self.video_fps)
            n_windows = len(self.rppg_signals[method]) // win_samples
            
            for i in range(n_windows):
                seg = self.rppg_signals[method][i*win_samples : (i+1)*win_samples]
                p_idx, t_idx, pairs = self._get_bvp_features(seg, self.video_fps)
                
                if len(pairs) < 8: # Mínimo de pulsos para análise espectral válida
                    for k in results: results[k].append(np.nan)
                    continue
                    
                t_orig = np.arange(len(seg)) / self.video_fps
                t_target = np.arange(0, self.respiration_window_size, 1/self.resp_modulation_resample)
                
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
                _, sig_fm = self._berger_algorithm(fm_t, self.resp_modulation_resample, self.respiration_window_size)

                results['sig_bw'].append(sig_bw)
                results['sig_am'].append(sig_am)
                results['sig_fm'].append(sig_fm)
                
                # --- Estimativa Espectral (FFT) ---
                rr_bw = self._estimate_rr_fft(sig_bw, self.resp_modulation_resample)
                rr_am = self._estimate_rr_fft(sig_am, self.resp_modulation_resample)
                rr_fm = self._estimate_rr_fft(sig_fm, self.resp_modulation_resample)
                
                # --- Fusão Robusta ---
                ests = np.array([rr_bw, rr_am, rr_fm])
                if not np.any(np.isnan(ests)):
                    mean_rr = np.mean(ests)
                    # std_rr = np.std(ests)
                    results['bw'].append(rr_bw)
                    results['am'].append(rr_am)
                    results['fm'].append(rr_fm)
                    results['fusion'].append(mean_rr)
                else:
                    for k in results: results[k].append(np.nan)

            # Concatenar sinais para plotagem temporal
            for k in ['sig_bw', 'sig_am', 'sig_fm']:
                if results[k]:
                    results[k] = np.concatenate(results[k])
                else:
                    results[k] = np.array([])

            rr_rppg[method] = results

        return rr_rppg

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
        
    def compare_rr_rppg_respiration(self):
        rr_results = {}
        
        for method in self.rppg_rr_values:
            if self.gt_rr_values is None:
                rr_results[method] = ({
                    'mae': None,
                    'rmse': None,
                    'mape': None
                })
            else:
                rr_rppg_values = self.rppg_rr_values[method]['fusion']
                min_length = min(len(self.gt_rr_values), len(rr_rppg_values))
                gt_rr_aligned = self.gt_rr_values[:min_length]
                rppg_rr_aligned = rr_rppg_values[:min_length]

                mae = mean_absolute_error(gt_rr_aligned, rppg_rr_aligned)
                rmse = np.sqrt(mean_squared_error(gt_rr_aligned, rppg_rr_aligned))
                mape = self.calculate_mape(gt_rr_aligned, rppg_rr_aligned)

                rr_results[method] = ({
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                })
        
        return rr_results

    def compare_rppg_ppg(self):
        if not self.gt_ppg_path:
            return {}

        print('Running linear correlation analysis...')

        gt_ppg_signal = self._load_signal_from_txt(self.gt_ppg_path)

        correlation_results = {}
        filtered_signals = self._filter_rppg_signals(lowcut=self.hr_lowcut, highcut=self.hr_highcut)    
        
        for method in self.rppg_hr_values:
            pearson_coef, best_lag, synced_rppg, synced_ppg = self._sync_and_correlate(filtered_signals[method], gt_ppg_signal)
            if pearson_coef is None:
                continue
            correlation_results[method] = ({
                'pearson_coef': pearson_coef,
                'best_lag': best_lag,
                'synced_rppg': synced_rppg,
                'synced_ppg': synced_ppg
            })
        
        return correlation_results

    def plot_gt(self, title=True, fontsize='large', labelsize='large'):

        if self.gt_ecg_path:
            ecg = hp.get_data(self.gt_ecg_path)

            fig1, ax1 = plt.subplots()
            ax1.plot(ecg)
            if title: ax1.set_title(self._t("raw_ecg"), fontsize=fontsize)
            ax1.set_xlabel(self._t("samples"), fontsize=fontsize)
            ax1.set_ylabel(self._t("amplitude"), fontsize=fontsize)
            ax1.tick_params(axis='both', labelsize=labelsize)
            plt.tight_layout()
            plt.show(block=False)

            ecg = self.filter_ecg(ecg, sample_rate=self.ecg_sample_rate)
            wd, m = hp.process(ecg, self.ecg_sample_rate)

            hp.plotter(wd, m)
            plt.tight_layout()
            plt.show(block=False)

        if self.gt_ppg_path:
            ppg = self._load_signal_from_txt(self.gt_ppg_path)
            ppg = self.normalize_signal(ppg)

            fig2, ax2 = plt.subplots()
            ax2.plot(ppg)
            if title: ax2.set_title(self._t("raw_ppg"), fontsize=fontsize)
            ax2.set_xlabel(self._t("samples"), fontsize=fontsize)
            ax2.set_ylabel(self._t("normalized_amplitude"), fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=labelsize)
            plt.tight_layout()
            plt.show(block=False)

        if self.gt_respiration_path:
            resp = self._load_signal_from_txt(self.gt_respiration_path)

            fig3, ax3 = plt.subplots()
            ax3.plot(resp)
            if title: ax3.set_title(self._t("respiration_signal"), fontsize=fontsize)
            ax3.set_xlabel(self._t("samples"), fontsize=fontsize)
            ax3.set_ylabel(self._t("amplitude"), fontsize=fontsize)
            ax3.tick_params(axis='both', labelsize=labelsize)
            plt.tight_layout()
    
        plt.show()


    def plot_spectral_analysis(self, title=True, fontsize='large', labelsize='large', methods='all'):

        if methods == 'all' or len(methods) > 3:
            if methods == 'all' and self.rppg_signals is not None: methods = list(self.rppg_signals.keys())
            cols = 2
        else:
            cols = 1
        rows = int(np.ceil(len(methods) / cols))
    
        # --- FIGURA 1: Espectrogramas ---
        fig_spec, axes_spec = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
        axes_spec = axes_spec.flatten()
    
        # --- FIGURA 2: Espectro de Potência (PSD) ---
        fig_psd, axes_psd = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
        axes_psd = axes_psd.flatten()
        filtered_signals = self._filter_rppg_signals(lowcut=self.hr_lowcut, highcut=self.hr_highcut)

        for i in range(len(methods)):
            method_name = methods[i]
            rppg_signal = filtered_signals[method_name]
    
            # Definir NFFT para alta resolução (Zero-padding)
            nfft_val = self._1bpm_resolution(len(rppg_signal), self.video_fps)

            # 1. Cálculo do Espectrograma
            # Usamos nfft para suavizar a imagem do espectrograma
            f, t, Sxx = signal.spectrogram(rppg_signal, fs=self.video_fps, nperseg=min(len(rppg_signal), 256), 
                                            noverlap=128, nfft=nfft_val)
            
            ax_s = axes_spec[i]
            im = ax_s.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='jet')
            if title: ax_s.set_title(f"{method_name}", fontsize=fontsize)
            ax_s.set_xlabel(self._t("time") + " (s)", fontsize=fontsize)
            ax_s.set_ylabel(self._t("frequency") + " (Hz)", fontsize=fontsize)
            ax_s.tick_params(axis='both', labelsize=labelsize)
            ax_s.set_ylim(self.hr_lowcut, self.hr_highcut)
            cbar = fig_spec.colorbar(im, ax=ax_s)
            cbar.set_label('dB', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=labelsize)

            # 2. Cálculo do Espectro de Potência (Welch ou Periodograma)
            freqs, psd = signal.periodogram(rppg_signal, fs=self.video_fps, nfft=nfft_val, detrend='constant')
            
            ax_p = axes_psd[i]
            ax_p.plot(freqs, psd, color='tab:green', linewidth=2)
            
            # Encontrar e marcar o pico principal (Frequência Cardíaca estimada)
            mask = (freqs >= self.hr_lowcut) & (freqs <= self.hr_highcut)
            if np.any(mask):
                peak_idx = np.argmax(psd[mask])
                peak_f = freqs[mask][peak_idx]
                ax_p.axvline(peak_f, color='red', linestyle='--', alpha=0.6)
                ax_p.text(peak_f + 0.1, np.max(psd)*0.8, f"{peak_f*60:.1f} BPM", color='red')

            ax_p.set_title(f"{method_name}", fontsize=fontsize)
            ax_p.set_xlabel(self._t("frequency") + " (Hz)", fontsize=fontsize)
            ax_p.set_ylabel(self._t("psd"), fontsize=fontsize)
            ax_p.tick_params(axis='both', labelsize=labelsize)
            ax_p.set_xlim(0, self.hr_highcut + 1)
            ax_p.grid(True, alpha=0.3)
    
        # Remover eixos extras
        for j in range(len(self.rppg_signals), len(axes_spec)):
            fig_spec.delaxes(axes_spec[j])
            fig_psd.delaxes(axes_psd[j])
    
        fig_spec.tight_layout()
        fig_psd.tight_layout()
        
        plt.show()
        
    def plot_rppg_ppg_correlation(self, title=True, fontsize='large', labelsize='large', methods='all'):
        
        if self.correlation_results:
            if methods == 'all' or len(methods) > 3:
                cols = 2
                if methods == 'all': methods = list(self.rppg_signals.keys())
            else:
                cols = 1
            rows = int(np.ceil(len(methods) / cols))
            fig, axes = plt.subplots(
                rows, cols, figsize=(15, 4 * rows), squeeze=False,
                sharex=True, sharey=True,
            )
            axes = axes.flatten()
            fs_common = max(self.video_fps, self.gt_ppg_sample_rate)

            for i in range(len(methods)):
                method = methods[i]
                result = self.correlation_results[method]
                synced_rppg = result['synced_rppg']
                synced_ppg = result['synced_ppg']
                pearson_coef = result['pearson_coef']
                best_lag = result['best_lag']/fs_common
                ax = axes[i]
                time_seconds = np.arange(len(synced_rppg)) / fs_common

                ax.plot(time_seconds, synced_rppg, label='rPPG', alpha=0.7)
                ax.plot(time_seconds, synced_ppg, label='PPG', alpha=0.7)
                if title: ax.set_title(f"{method} - Pearson: {pearson_coef:.3f}, Lag: {best_lag} s", fontsize=fontsize)
                ax.set_xlabel(self._t("time") + " (s)", fontsize=fontsize)
                ax.set_ylabel(self._t("normalized_amplitude"), fontsize=fontsize)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=labelsize)

            for unused_axis in axes[len(self.correlation_results):]:
                fig.delaxes(unused_axis)

            fig.tight_layout()
            plt.show()
        else:
            print("No correlation results available. Please run the analysis first.")

    def plot_bvp(self, title=True, fontsize='large', labelsize='large', methods='all', same_scale=True):
        if methods == 'all' or len(methods) > 3:
            cols = 2
            if methods == 'all': methods = list(self.rppg_signals.keys())
        else:
            cols = 1
        rows = int(np.ceil(len(methods) / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(15, 4 * rows), squeeze=False,
            sharex=same_scale, sharey=same_scale,
        )
        axes = axes.flatten()
        filtered_signals = self._filter_rppg_signals(lowcut=self.hr_lowcut, highcut=self.hr_highcut)

        for i in range(len(methods)):
            method_name = methods[i]
            rppg_signal = filtered_signals[method_name]
            ax = axes[i]
            time_seconds = np.arange(len(rppg_signal)) / self.video_fps
            ax.plot(time_seconds, rppg_signal, label=f'{method_name} rPPG', alpha=0.7)
            if title: ax.set_title(f"{method_name}", fontsize=fontsize)
            ax.set_xlabel(self._t("time") + " (s)", fontsize=fontsize)
            ax.set_ylabel(self._t("amplitude"), fontsize=fontsize)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=labelsize)

        for unused_axis in axes[len(methods):]:
            fig.delaxes(unused_axis)

        fig.tight_layout()
        plt.show()

    def plot_all(self, method: str, title=True, fontsize='large', labelsize='large', methods='all', same_scale=True):
        plots = []
        if self.gt_ecg_path:
            ecg = hp.get_data(self.gt_ecg_path)
            plots.append((ecg, self._t("raw_ecg"), self._t("amplitude"), self.ecg_sample_rate, None))

        if self.gt_ppg_path:
            ppg = self._load_signal_from_txt(self.gt_ppg_path)
            ppg = self.normalize_signal(ppg)
            plots.append((ppg, self._t("raw_ppg"), self._t("normalized_amplitude"), self.gt_ppg_sample_rate, None))

        if method in self.correlation_results:
            rppg_signal = self.correlation_results[method]['synced_rppg']
            # window_size = 5
            # if len(rppg_signal) > 0:
            #     weights = np.ones(window_size) / window_size
            #     padded_signal = np.pad(rppg_signal, (window_size // 2,), mode='edge')
            #     rppg_signal = np.convolve(padded_signal, weights, mode='valid')
            ppg_signal = self.correlation_results[method]['synced_ppg']
            plots.append((ppg_signal, self._t("raw_ppg"), self._t("normalized_amplitude"), self.gt_ppg_sample_rate, None))
            plots.append((rppg_signal, method, self._t("normalized_amplitude"), self.gt_ppg_sample_rate, f'{method} rPPG'))

        if plots:
            fig, axes = plt.subplots(len(plots), 1, figsize=(15, 4 * len(plots)), squeeze=False, sharex=True)
            for ax, (signal, plot_title, ylabel, sample_rate, label) in zip(axes.flatten(), plots):
                time_seconds = np.arange(len(signal)) / sample_rate
                ax.plot(time_seconds, signal, label=label, alpha=0.7)
                if title:
                    ax.set_title(plot_title, fontsize=fontsize)
                ax.set_xlabel(self._t("time") + " (s)", fontsize=fontsize)
                ax.set_ylabel(ylabel, fontsize=fontsize)
                if label:
                    ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=labelsize)

            fig.tight_layout()
            plt.show()

    def run(self):
        """Execute the heavy loading and processing steps.

        Call this explicitly when you want to perform I/O and compute results.
        """
        # populate video metadata
        self.frame_count = self._get_frame_count()
        self.video_fps = self._get_video_fps()

        if self.frame_count / self.video_fps < self.hr_window_size:
            print(f"Warning: Video duration ({self.frame_count / self.video_fps:.2f}s) is shorter than HR window size ({self.hr_window_size}s).")
            return

        # load signals and compute HR values
        self.rppg_signals = self._load_rppg_signals()
        self.ecg_hr_values = self._estimate_hr_ecg()
        self.rppg_hr_values = self._estimate_hr_rppg()
        self.hr_results = self.compare_hr_rppg_ecg()
        self.correlation_results = self.compare_rppg_ppg()
        self.gt_rr_values, self.ref_rr_signal = self._load_respiratory_signal()
        self.rppg_rr_values = self._estimate_rr_rppg()
        self.rr_results = self.compare_rr_rppg_respiration()