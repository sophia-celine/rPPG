"""Extracao de rPPG pelo metodo POS a partir de um video AVI.

O video e selecionado por uma janela grafica. Apos a extracao, o programa
mostra o sinal POS filtrado entre 0.6 e 3.3 Hz no tempo, seu espectro e seu
espectrograma.
"""

import math
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


LOWCUT_HZ = 0.6
HIGHCUT_HZ = 3.3
POS_WINDOW_SECONDS = 1.6
SPECTRAL_WINDOW_HZ = 0.1
ROI_WIDTH = 300
ROI_HEIGHT = 180


def _process_video(frames):
	"""Calcula o valor medio dos canais RGB de cada quadro."""
	rgb_means = []
	for frame in frames:
		if frame.ndim != 3 or frame.shape[2] != 3:
			raise ValueError("Cada quadro deve ter tres canais de cor.")
		rgb_means.append(np.mean(frame, axis=(0, 1)))
	return np.asarray(rgb_means, dtype=np.float64)


def POS_WANG(frames, fs):
	"""Extrai o BVP usando a implementacao POS fornecida pelo usuario."""
	if fs <= 0:
		raise ValueError("A taxa de quadros deve ser maior que zero.")

	win_sec = 1.6
	rgb = _process_video(frames)
	sample_count = rgb.shape[0]
	window_length = math.ceil(win_sec * fs)
	if sample_count < window_length:
		raise ValueError(
			f"O video precisa de pelo menos {win_sec:.1f} s para executar o POS."
		)

	h_accumulated = np.zeros((1, sample_count), dtype=np.float64)
	projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

	for end in range(sample_count):
		start = end - window_length
		if start < 0:
			continue

		window_rgb = np.true_divide(rgb[start:end], np.mean(rgb[start:end], axis=0))
		projected = projection @ window_rgb.T
		second_std = np.std(projected[1])
		if second_std <= np.finfo(float).eps:
			continue
		projected_signal = projected[0] + (np.std(projected[0]) / second_std) * projected[1]
		projected_signal -= np.mean(projected_signal)
		h_accumulated[0, start:end] += projected_signal

	bvp = signal.detrend(h_accumulated.T, type="linear")
	bvp = np.asarray(bvp).reshape(-1)
	butter_b, butter_a = signal.butter(1, [0.75 / fs * 2, 3.0 / fs * 2], btype="bandpass")
	return bvp.astype(np.double)#signal.filtfilt(butter_b, butter_a, bvp.astype(np.double))


def select_roi(video_path):
	"""Seleciona uma ROI e oferece a ultima ROI salva como sugestao."""
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

	ok, first_frame = capture.read()
	capture.release()
	if not ok:
		raise RuntimeError("Nao foi possivel ler o primeiro frame do video.")
	frame_height, frame_width = first_frame.shape[:2]
	if frame_width < ROI_WIDTH or frame_height < ROI_HEIGHT:
		raise ValueError(
			f"O video precisa ter pelo menos {ROI_WIDTH}x{ROI_HEIGHT} pixels."
		)

	default_path = Path(__file__).resolve().parent / "roi_default.json"
	default_roi = None
	if default_path.exists():
		try:
			with default_path.open("r", encoding="utf-8") as file:
				points = json.load(file)
			x1, y1 = map(int, points["top_left"])
			x1 = min(max(x1, 0), frame_width - ROI_WIDTH)
			y1 = min(max(y1, 0), frame_height - ROI_HEIGHT)
			default_roi = (x1, y1, ROI_WIDTH, ROI_HEIGHT)
		except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
			default_roi = None

	if default_roi is None:
		default_roi = (
			(frame_width - ROI_WIDTH) // 2,
			(frame_height - ROI_HEIGHT) // 2,
			ROI_WIDTH,
			ROI_HEIGHT,
		)

	selection = {"roi": default_roi, "start": None, "dragging": False}
	window_name = "ROI 300x180 - arraste | Enter aceita | R centraliza | Esc cancela"

	def draw_selection(frame):
		preview = frame.copy()
		if selection["roi"] is not None:
			x, y, width, height = selection["roi"]
			cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
		cv2.putText(
			preview,
			"ROI 300x180 | Arraste: posicionar | Enter: aceitar | R: centralizar | Esc: cancelar",
			(10, 25),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)
		return preview

	def on_mouse(event, x, y, _flags, _param):
		if event == cv2.EVENT_LBUTTONDOWN:
			selection["start"] = (x, y)
			selection["dragging"] = True
			selection["roi"] = (
				min(max(x, 0), frame_width - ROI_WIDTH),
				min(max(y, 0), frame_height - ROI_HEIGHT),
				ROI_WIDTH,
				ROI_HEIGHT,
			)
		elif event == cv2.EVENT_MOUSEMOVE and selection["dragging"]:
			x_position = min(max(x, 0), frame_width - ROI_WIDTH)
			y_position = min(max(y, 0), frame_height - ROI_HEIGHT)
			selection["roi"] = (x_position, y_position, ROI_WIDTH, ROI_HEIGHT)
		elif event == cv2.EVENT_LBUTTONUP:
			selection["dragging"] = False

	if default_roi is not None:
		print(f"ROI padrao carregada de: {default_path}")
	else:
		print("Nenhuma ROI padrao encontrada; desenhe uma nova regiao.")

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.setMouseCallback(window_name, on_mouse)
	while True:
		cv2.imshow(window_name, draw_selection(first_frame))
		key = cv2.waitKey(20) & 0xFF
		if key in (13, 32):
			if selection["roi"] is not None and selection["roi"][2] > 0 and selection["roi"][3] > 0:
				break
		elif key == ord("r"):
			selection["roi"] = (
				(frame_width - ROI_WIDTH) // 2,
				(frame_height - ROI_HEIGHT) // 2,
				ROI_WIDTH,
				ROI_HEIGHT,
			)
		elif key == 27:
			cv2.destroyWindow(window_name)
			raise ValueError("Selecao da regiao de interesse cancelada.")
	cv2.destroyWindow(window_name)

	x, y, width, height = selection["roi"]
	points = {
		"top_left": [x, y],
		"bottom_right": [x + width, y + height],
	}
	with default_path.open("w", encoding="utf-8") as file:
		json.dump(points, file, indent=2)
	print(f"Pontos da ROI salvos em: {default_path}")

	roi_frame = first_frame.copy()
	cv2.rectangle(
		roi_frame,
		(x, y),
		(x + width, y + height),
		(0, 255, 0),
		2,
	)
	output_path = Path(__file__).resolve().parent / f"roi_{Path(video_path).stem}.png"
	if not cv2.imwrite(str(output_path), roi_frame):
		raise RuntimeError(f"Nao foi possivel salvar a imagem da ROI: {output_path}")
	print(f"Imagem da ROI salva em: {output_path}")
	return x, y, width, height


def read_video_rgb(video_path, roi):
	"""Le o video, recorta a ROI e retorna quadros RGB e a taxa de quadros."""
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")

	fps = capture.get(cv2.CAP_PROP_FPS)
	if not np.isfinite(fps) or fps <= 0:
		capture.release()
		raise RuntimeError("Nao foi possivel determinar o FPS do video.")

	x, y, width, height = roi
	frames = []
	try:
		while True:
			ok, frame_bgr = capture.read()
			if not ok:
				break
			roi_bgr = frame_bgr[y:y + height, x:x + width]
			if roi_bgr.size == 0:
				raise RuntimeError("A regiao selecionada nao e valida para todos os frames.")
			# OpenCV fornece BGR; o POS trabalha com canais RGB.
			frames.append(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
	finally:
		capture.release()

	if not frames:
		raise RuntimeError("O video nao contem quadros legiveis.")
	return np.asarray(frames), float(fps)


def bandpass_filter(bvp, fs, lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ):
	"""Aplica um Butterworth passa-banda ao BVP."""
	nyquist = fs / 2.0
	if not 0 < lowcut < highcut < nyquist:
		raise ValueError(
			f"A banda precisa estar entre 0 e Nyquist ({nyquist:.2f} Hz); "
			f"recebida: {lowcut}-{highcut} Hz."
		)

	b, a = signal.butter(3, [lowcut / nyquist, highcut / nyquist], btype="bandpass")
	pad_length = 3 * max(len(a), len(b))
	if len(bvp) <= pad_length:
		raise ValueError("O video e curto demais para filtrar o sinal com seguranca.")
	return signal.filtfilt(b, a, bvp)


def plot_results(filtered_bvp, raw_bvp, fs, video_path):
	"""Mostra tempo, frequencia e espectrograma em uma unica figura."""
	sample_count = len(filtered_bvp)
	time = np.arange(sample_count) / fs
	frequencies, spectrum = signal.periodogram(filtered_bvp, fs=fs, scaling="spectrum")
	frequency_mask = (frequencies >= LOWCUT_HZ) & (frequencies <= HIGHCUT_HZ)
	band_frequencies = frequencies[frequency_mask]
	band_spectrum = spectrum[frequency_mask]
	if band_frequencies.size == 0:
		raise ValueError("Nao ha bins espectrais dentro da banda configurada.")
	peak_index = int(np.argmax(band_spectrum))
	peak_frequency = float(band_frequencies[peak_index])
	peak_bpm = 60.0 * peak_frequency

	peak_window = np.abs(band_frequencies - peak_frequency) <= SPECTRAL_WINDOW_HZ
	harmonic_frequency_target = 2.0 * peak_frequency
	harmonic_window = (
		(np.abs(band_frequencies - harmonic_frequency_target) <= SPECTRAL_WINDOW_HZ)
		& ~peak_window
	)
	if not np.any(harmonic_window):
		harmonic_frequency = harmonic_frequency_target
		harmonic_power = 0.0
	else:
		harmonic_index = int(np.argmax(np.where(harmonic_window, band_spectrum, -np.inf)))
		harmonic_frequency = float(band_frequencies[harmonic_index])
		harmonic_power = float(np.sum(band_spectrum[harmonic_window]))

	peak_power = float(np.sum(band_spectrum[peak_window]))
	noise_window = ~(peak_window | harmonic_window)
	noise_spectrum = band_spectrum[noise_window]
	if noise_spectrum.size == 0:
		raise ValueError("Nao ha bins suficientes para estimar o piso de ruido espectral.")

	# Estima o piso de ruido pela mediana dos bins fora do pico e do harmonico.
	# A potencia e escalada para a mesma largura espectral da janela do pico.
	noise_floor_density = float(np.median(noise_spectrum))
	peak_bin_count = int(np.count_nonzero(peak_window))
	peak_density = peak_power / max(peak_bin_count, 1)
	minimum_noise_density = peak_density * 1e-12
	noise_floor_density = max(
		noise_floor_density,
		minimum_noise_density,
		np.finfo(float).tiny,
	)
	noise_power = noise_floor_density * peak_bin_count
	snr_linear = peak_power / noise_power
	snr_db = 10.0 * np.log10(max(snr_linear, np.finfo(float).tiny))
	harmonic_ratio_db = 10.0 * np.log10(
		max(peak_power, np.finfo(float).tiny)
		/ max(harmonic_power, np.finfo(float).tiny)
	)

	nperseg = min(sample_count, max(16, int(round(8 * fs))))
	noverlap = min(nperseg - 1, nperseg // 2)
	nfft = 2 ** math.ceil(math.log2(4 * nperseg))
	spec_freq, spec_time, power = signal.spectrogram(
		filtered_bvp,
		fs=fs,
		window="hann",
		nperseg=nperseg,
		nfft=nfft,
		noverlap=noverlap,
		scaling="density",
	)
	spec_mask = (spec_freq >= LOWCUT_HZ) & (spec_freq <= HIGHCUT_HZ)
	spec_power = np.maximum(power[spec_mask], np.finfo(float).tiny)
	spec_power_db = 10 * np.log10(spec_power)
	vmin = np.percentile(spec_power_db, 2)
	vmax = np.percentile(spec_power_db, 98)
	if vmax <= vmin:
		vmin = float(np.min(spec_power_db))
		vmax = float(np.max(spec_power_db))
	if vmax <= vmin:
		vmax = vmin + 1.0

	figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
	figure.suptitle(f"POS - {Path(video_path).name}", fontsize=15)

	axes[0, 0].plot(time, raw_bvp, color="0.35", linewidth=0.8)
	axes[0, 0].set_title("BVP extraido pelo POS")
	axes[0, 0].set_ylabel("Amplitude")

	axes[0, 1].plot(time, filtered_bvp, color="tab:blue", linewidth=0.9)
	axes[0, 1].set_title(f"BVP filtrado ({LOWCUT_HZ}-{HIGHCUT_HZ} Hz)")
	axes[0, 1].set_ylabel("Amplitude")

	axes[1, 0].plot(band_frequencies, band_spectrum, color="tab:orange")
	axes[1, 0].plot(peak_frequency, band_spectrum[peak_index], "o", color="crimson", markersize=7)
	axes[1, 0].axvline(peak_frequency, color="crimson", linestyle="--", linewidth=1)
	if harmonic_window.any():
		harmonic_index = int(np.argmax(np.where(harmonic_window, band_spectrum, -np.inf)))
		axes[1, 0].plot(
			harmonic_frequency,
			band_spectrum[harmonic_index],
			"o",
			color="darkgreen",
			markersize=6,
		)
		axes[1, 0].axvline(harmonic_frequency, color="darkgreen", linestyle=":", linewidth=1)
	axes[1, 0].annotate(
		f"Pico: {peak_frequency:.3f} Hz\n{peak_bpm:.1f} BPM\n"
		f"1o harmonico: {harmonic_frequency:.3f} Hz\n"
		f"Pico/harmonico: {harmonic_ratio_db:.2f} dB\n"
		f"SNR: {snr_db:.2f} dB",
		xy=(peak_frequency, band_spectrum[peak_index]),
		xytext=(10, 12),
		textcoords="offset points",
		color="crimson",
		fontweight="bold",
	)
	axes[1, 0].set_title(
		f"Espectro - SNR pico/piso: {snr_linear:.2f} ({snr_db:.2f} dB)"
	)
	axes[1, 0].set_xlabel("Frequencia (Hz)")
	axes[1, 0].set_ylabel("Potencia")
	axes[1, 0].set_xlim(LOWCUT_HZ, HIGHCUT_HZ)

	mesh = axes[1, 1].pcolormesh(
		spec_time,
		spec_freq[spec_mask],
		spec_power_db,
		shading="auto",
		cmap="magma",
		vmin=vmin,
		vmax=vmax,
	)
	axes[1, 1].set_title("Espectrograma")
	axes[1, 1].set_xlabel("Tempo (s)")
	axes[1, 1].set_ylabel("Frequencia (Hz)")
	axes[1, 1].set_ylim(LOWCUT_HZ, HIGHCUT_HZ)
	figure.colorbar(mesh, ax=axes[1, 1], label="dB")

	for axis in axes.flat:
		axis.grid(True, alpha=0.25)
	plt.show()


def select_video():
	"""Abre o seletor de arquivo e retorna um video AVI."""
	root = tk.Tk()
	root.withdraw()
	root.update()
	video_path = filedialog.askopenfilename(
		title="Selecione um video AVI",
		filetypes=[("Videos AVI", "*.avi"), ("Todos os arquivos", "*.*")],
	)
	root.destroy()
	return video_path


def main():
	video_path = select_video()
	if not video_path:
		return

	try:
		roi = select_roi(video_path)
		print(f"ROI selecionada: x={roi[0]}, y={roi[1]}, largura={roi[2]}, altura={roi[3]}")
		frames, fs = read_video_rgb(video_path, roi)
		print(f"Quadros lidos: {len(frames)} | FPS: {fs:.3f}")
		raw_bvp = POS_WANG(frames, fs)
		filtered_bvp = bandpass_filter(raw_bvp, fs)
		plot_results(filtered_bvp, raw_bvp, fs, video_path)
	except (RuntimeError, ValueError) as error:
		messagebox.showerror("Erro no processamento", str(error))
		raise


if __name__ == "__main__":
	main()
