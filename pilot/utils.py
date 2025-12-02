import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram


def extract_rgb_signals_rect(video_path, rect, output_csv="rgb_signals.csv"):
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

    # Draw the rectangle (in red, thickness=2)
    cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the frame
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
    df.to_csv(output_csv, index=False)

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
    plt.show()

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

    r_raw = df["Mean_R"].values
    g_raw = df["Mean_G"].values
    b_raw = df["Mean_B"].values

    r = butter_bandpass_filter(r_raw, lowcut, highcut, fs=fps, order=filter_order)
    g = butter_bandpass_filter(g_raw, lowcut, highcut, fs=fps, order=filter_order)
    b = butter_bandpass_filter(b_raw, lowcut, highcut, fs=fps, order=filter_order)

    n = len(r)
    t = np.arange(n) / fps

    print(f'HR from frequency spectrum: {freqs[np.argmax(g_fft)]*60} bpm')

    freqs = np.fft.rfftfreq(n, d=1/fps)
    r_fft = np.abs(np.fft.rfft(r - np.mean(r)))
    g_fft = np.abs(np.fft.rfft(g - np.mean(g)))
    b_fft = np.abs(np.fft.rfft(b - np.mean(b)))

    # === Time & Frequency plots ===
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
    
    axes[1,1].plot(freqs, g_fft, color='green')
    axes[1,1].set_title('G - Frequency')
    
    axes[2,1].plot(freqs, b_fft, color='blue')
    axes[2,1].set_title('B - Frequency')
    axes[2,1].set_xlabel('Frequency (Hz)')

    for ax in axes[:,1]:
        ax.set_xlim(0, 5)  
        ax.set_ylabel('Magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # === Spectrograms ===
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
    plt.show()

    return r, g, b, r_fft, g_fft, b_fft, freqs

def getSNR():
    return