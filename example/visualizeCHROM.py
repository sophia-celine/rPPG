import cv2
import numpy as np
import math
from scipy import signal
from scipy.interpolate import interp1d
from typing import Tuple, List

# =========================
# CHROM
# =========================
def extract_rgb_from_video(video_path: str, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extracts mean RGB signals from video without loading the whole file into RAM."""
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    RGB = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_roi = frame[y:y+h, x:x+w]
        RGB.append(np.mean(face_roi, axis=(0, 1)))
    cap.release()
    return np.asarray(RGB)

def CHROME_DEHAAN(RGB: np.ndarray, FS: float):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    
    FN = RGB.shape[0]
    NyquistF = FS / 2
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if (WinL % 2):
        WinL += 1

    NWin = math.floor((FN - WinL//2) / (WinL//2))
    WinS = 0
    WinM = WinL//2
    WinE = WinL

    S = np.zeros(FN)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE], axis=0)
        RGBNorm = RGB[WinS:WinE] / RGBBase

        # OpenCV is BGR: 0=Blue, 1=Green, 2=Red
        # Haan 2013: Xs = 3Rn - 2Gn; Ys = 1.5Rn + Gn - 1.5Bn
        Xs = 3 * RGBNorm[:, 2] - 2 * RGBNorm[:, 1]
        Ys = 1.5 * RGBNorm[:, 2] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 0]

        Xf = signal.filtfilt(B, A, Xs)
        Yf = signal.filtfilt(B, A, Ys)

        alpha = np.std(Xf) / np.std(Yf)
        SWin = (Xf - alpha * Yf) * signal.windows.hann(WinL)

        S[WinS:WinM] += SWin[:WinL//2]
        S[WinM:WinE] = SWin[WinL//2:]

        WinS = WinM
        WinM = WinS + WinL//2
        WinE = WinS + WinL

    return S


# =========================
# Pipeline completo
# =========================
def visualize_bvp_on_video(video_path: str, output_path: str = "output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        return

    fs = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret:
        return

    print('Selecione a ROI')
    roi = cv2.selectROI("Selecione a ROI", first_frame)
    cv2.destroyWindow("Selecione a ROI")

    if roi == (0, 0, 0, 0):
        roi = (0, 0, width, height)

    print("Passo 1: Extraindo sinais RGB...")
    # Reset capture to start for signal extraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rgb_signals = extract_rgb_from_video(video_path, roi)
    n_frames = len(rgb_signals)

    print("Passo 2: Processando CHROM...")
    bvp = CHROME_DEHAAN(rgb_signals, fs)

    # Normalização
    t_bvp = np.linspace(0, n_frames/fs, len(bvp))
    t_frames = np.arange(n_frames) / fs
    bvp_interp = interp1d(t_bvp, bvp, fill_value="extrapolate")(t_frames)
    bvp_norm = (bvp_interp - np.min(bvp_interp)) / (np.max(bvp_interp) - np.min(bvp_interp))

    print("Passo 3: Gerando vídeo de saída...")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fs, (width, height + 200))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # O(1) plotting: Use a persistent canvas for the signal
    plot_canvas = np.ones((200, width, 3), dtype=np.uint8) * 255
    x_roi, y_roi, w_roi, h_roi = roi

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Incremental Plotting
        if i > 0:
            pt1 = (int((i-1)/n_frames * width), int(200 - bvp_norm[i-1]*180))
            pt2 = (int(i/n_frames * width), int(200 - bvp_norm[i]*180))
            cv2.line(plot_canvas, pt1, pt2, (255, 0, 0), 2)

        # Visual Feedback
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)

        # =========================
        # Overlay de cor (pulsação)
        # =========================
        intensity = bvp_norm[i]
        roi_slice = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        color_overlay = np.zeros_like(roi_slice)
        color_overlay[:, :, 2] = (intensity * 255).astype(np.uint8) 
        frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = cv2.addWeighted(roi_slice, 1.0, color_overlay, 0.3, 0)

        # =========================
        # Plot do sinal
        # =========================
        current_plot = plot_canvas.copy()
        x_curr = int(i/n_frames * width)
        cv2.line(current_plot, (x_curr, 0), (x_curr, 200), (0, 0, 255), 2)

        combined = np.vstack((frame, current_plot))
        out.write(combined)

    cap.release()
    out.release()
    print("Vídeo salvo em:", output_path)

if __name__ == "__main__":
     visualize_bvp_on_video(r"C:\Users\Sophia\Documents\rPPG\initial_tests\videos\60s.avi", output_path="output.avi")
