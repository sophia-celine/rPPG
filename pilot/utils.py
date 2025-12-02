import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram


def extract_rgb_signals_rect(video_path, rect):
    """
    Extracts mean R, G, B signals from all pixels inside a rectangle across video frames.

    Parameters:
    - video_path (str): Path to input video.
    - rect (tuple): (x1, y1, x2, y2) defining the rectangle (top-left to bottom-right).
    - output_csv (str): Output CSV filename.
    """
    x1, y1, x2, y2 = rect
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Error opening video file.")

    signals = {"Mean_R": [], "Mean_G": [], "Mean_B": []}
    frames = 0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Unable to read the first frame from the video.")

    cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("First Frame with ROI", first_frame)
    print("Displaying first frame with ROI... Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        roi = frame[y1:y2, x1:x2]

        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])

        signals["Mean_R"].append(mean_r)
        signals["Mean_G"].append(mean_g)
        signals["Mean_B"].append(mean_b)

    cap.release()
    print(frames)

    df = pd.DataFrame(signals)

    return df

def plot_rgb_signals(df, fps=30):

    """
    Plot R, G, B signals in time and frequency domains using subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame with extracted mean signals.
    - fps (int): Frames per second of the video.
    """
    r = df["Mean_R"].values
    g = df["Mean_G"].values
    b = df["Mean_B"].values

    n = len(r)
    t = np.arange(n) / fps

    freqs = np.fft.rfftfreq(n, d=1/fps)
    r_fft = np.abs(np.fft.rfft(r - np.mean(r)))
    g_fft = np.abs(np.fft.rfft(g - np.mean(g)))
    b_fft = np.abs(np.fft.rfft(b - np.mean(b)))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("RGB Signals (raw)", fontsize=14)

    axes[0, 0].plot(t, r, color='red')
    axes[0, 0].set_title('R - Time')
    axes[0, 0].set_ylabel('Intensity')

    axes[1, 0].plot(t, g, color='green')
    axes[1, 0].set_title('G - Time')
    axes[1, 0].set_ylabel('Intensity')

    axes[2, 0].plot(t, b, color='blue')
    axes[2, 0].set_title('B - Time')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Intensity')

    axes[0, 1].plot(freqs, r_fft, color='red')
    axes[0, 1].set_title('R - Frequency')

    axes[1, 1].plot(freqs, g_fft, color='green')
    axes[1, 1].set_title('G - Frequency')

    axes[2, 1].plot(freqs, b_fft, color='blue')
    axes[2, 1].set_title('B - Frequency')
    axes[2, 1].set_xlabel('Frequency (Hz)')

    for ax in axes[:, 1]:
        ax.set_xlim(0, 5)  
        ax.set_ylabel('Magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_spectrum(df, lowcut, highcut, filter_order=4, fps=30):
    """
    Plot filtered R, G, B signals in time and frequency domains (3x2 subplots)
    and also plot their spectrograms.
    """
    print('will get spectrum')
    r_raw = df["Mean_R"].values
    g_raw = df["Mean_G"].values
    b_raw = df["Mean_B"].values

    r = butter_bandpass_filter(r_raw, lowcut, highcut, fs=fps, order=filter_order)
    g = butter_bandpass_filter(g_raw, lowcut, highcut, fs=fps, order=filter_order)
    b = butter_bandpass_filter(b_raw, lowcut, highcut, fs=fps, order=filter_order)

    n = len(r)
    t = np.arange(n) / fps

    freqs = np.fft.rfftfreq(n, d=1/fps)
    r_fft = np.abs(np.fft.rfft(r - np.mean(r)))
    g_fft = np.abs(np.fft.rfft(g - np.mean(g)))
    b_fft = np.abs(np.fft.rfft(b - np.mean(b)))

    snr_r, fund_win_r, harm_win_r = getSNR(r_fft, freqs, lowcut, highcut)
    snr_g, fund_win_g, harm_win_g = getSNR(g_fft, freqs, lowcut, highcut)
    snr_b, fund_win_b, harm_win_b = getSNR(b_fft, freqs, lowcut, highcut)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"RGB Signals (filtered {lowcut}-{highcut} Hz)", fontsize=14)

    axes[0,0].plot(t, r, color='red')
    axes[0,0].set_title('R - Time')
    axes[0,0].set_ylabel('Intensity')
    
    axes[1,0].plot(t, g, color='green')
    axes[1,0].set_title('G - Time')
    axes[1,0].set_ylabel('Intensity')
    
    axes[2,0].plot(t, b, color='blue')
    axes[2,0].set_title('B - Time')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_ylabel('Intensity')

    axes[0,1].plot(freqs, r_fft, color='red')
    axes[0,1].set_title('R - Frequency')
    axes[0,1].text(0.05, 0.95, f"SNR: {snr_r:.2f} dB", transform=axes[0,1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    axes[1,1].plot(freqs, g_fft, color='green')
    axes[1,1].set_title('G - Frequency')
    axes[1,1].axvspan(lowcut, highcut, color='gray', alpha=0.2, label='Noise Band')
    axes[1,1].axvspan(fund_win_g[0], fund_win_g[1], color='green', alpha=0.4, label='Signal (Fundamental)')
    axes[1,1].axvspan(harm_win_g[0], harm_win_g[1], color='yellow', alpha=0.4, label='Signal (Harmonic)')
    axes[1,1].legend(loc='upper right')
    axes[1,1].text(0.05, 0.95, f"SNR: {snr_g:.2f} dB", transform=axes[1,1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    axes[2,1].plot(freqs, b_fft, color='blue')
    axes[2,1].set_title('B - Frequency')
    axes[2,1].set_xlabel('Frequency (Hz)')
    axes[2,1].text(0.05, 0.95, f"SNR: {snr_b:.2f} dB", transform=axes[2,1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    for ax in axes[:,1]:
        ax.set_xlim(0, 5)
        ax.set_ylabel('Magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    band_indices_g = np.where((freqs >= lowcut) & (freqs <= highcut))
    peak_index_g = np.argmax(g_fft[band_indices_g])
    peak_freq_g = freqs[band_indices_g][peak_index_g]

    print(f'Estimated HR from Green channel: {peak_freq_g * 60:.2f} bpm')
    print(f"SNR for R channel: {snr_r:.2f} dB")
    print(f"SNR for G channel: {snr_g:.2f} dB")
    print(f"SNR for B channel: {snr_b:.2f} dB")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Spectrograms of Filtered RGB Signals ({lowcut}-{highcut} Hz)", fontsize=14)

    signals = [(r, 'R', 'Reds'), (g, 'G', 'Greens'), (b, 'B', 'Blues')]
    for ax, (sig, label, cmap) in zip(axes, signals):
        f, tt, Sxx = spectrogram(sig, fs=fps, nperseg=256, noverlap=128)
        im = ax.pcolormesh(tt, f, 10*np.log10(Sxx+1e-12), shading='gouraud', cmap=cmap)
        ax.set_ylabel('Freq (Hz)')
        ax.set_ylim(0, 5)
        ax.set_title(f'{label} channel')
        fig.colorbar(im, ax=ax, orientation='vertical', label='Power (dB)')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return r, g, b, r_fft, g_fft, b_fft, freqs

def getSNR(fft_mag, freqs, lowcut, highcut, signal_window_hz=0.2):
    """
    Calculates the Signal-to-Noise Ratio (SNR) of a signal from its FFT spectrum.

    The signal is the power in a narrow band around the peak frequency (fundamental)
    and a wider band around its first harmonic. The noise is the power in the
    rest of the band.

    Parameters:
    - fft_mag (np.ndarray): Magnitude of the FFT of the signal.
    - freqs (np.ndarray): Frequencies corresponding to the fft_mag.
    - lowcut (float): The lower bound of the frequency band of interest (in Hz).
    - highcut (float): The upper bound of the frequency band of interest (in Hz).
    - signal_window_hz (float): The width of the window around the fundamental
                                frequency. The window around the first harmonic
                                will be twice this width.

    Returns:
    - float: The SNR in decibels (dB). Returns -inf if noise power is zero.
        - snr (float): The SNR in decibels (dB).
        - fundamental_window (tuple): (low_freq, high_freq) for the fundamental.
        - harmonic_window (tuple): (low_freq, high_freq) for the harmonic.
    """
    band_indices = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    if len(band_indices) == 0:
        return -np.inf, (0, 0), (0, 0)

    peak_index_in_band = np.argmax(fft_mag[band_indices])
    peak_index = band_indices[peak_index_in_band]
    fundamental_freq = freqs[peak_index]

    fund_window_low = fundamental_freq - signal_window_hz / 2
    fund_window_high = fundamental_freq + signal_window_hz / 2
    fund_indices = np.where((freqs >= fund_window_low) & (freqs <= fund_window_high))[0]

    harmonic_freq = 2 * fundamental_freq
    harmonic_window_hz = 2 * signal_window_hz
    harm_window_low = harmonic_freq - harmonic_window_hz / 2
    harm_window_high = harmonic_freq + harmonic_window_hz / 2
    harm_indices = np.where((freqs >= harm_window_low) & (freqs <= harm_window_high))[0]

    signal_indices = np.union1d(fund_indices, harm_indices)
    noise_indices = np.setdiff1d(band_indices, signal_indices)

    signal_power = np.sum(fft_mag[signal_indices] ** 2)
    noise_power = np.sum(fft_mag[noise_indices] ** 2)

    if noise_power == 0:
        return np.inf, (fund_window_low, fund_window_high), (harm_window_low, harm_window_high)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr, (fund_window_low, fund_window_high), (harm_window_low, harm_window_high)