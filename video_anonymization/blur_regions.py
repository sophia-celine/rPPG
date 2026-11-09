from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass
class BlurRegion:
    start_frame: int
    end_frame: int
    polygon: np.ndarray


def _blur_polygon(frame: np.ndarray, polygons: list[np.ndarray], sigma: float) -> np.ndarray:
    if not polygons:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)
    return np.where(mask[:, :, None] != 0, blurred, frame)


def export_video(
    input_path: Path,
    output_path: Path,
    regions: list[BlurRegion],
    ffmpeg_path: str,
    blur_sigma: float,
    progress: Callable[[int, int], None],
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("O arquivo de saída não pode ser o próprio vídeo original.")
    if output_path.suffix.lower() != ".avi":
        raise ValueError("O formato de saída deve ser AVI.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("O vídeo não possui informações válidas de duração ou resolução.")
    encoded_width = width + width % 2
    encoded_height = height + height % 2
    for region in regions:
        if (
            region.start_frame < 0
            or region.end_frame < region.start_frame
            or region.end_frame >= frame_count
        ):
            capture.release()
            raise ValueError("Uma região possui um intervalo fora da duração do vídeo.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        capture.release()
        raise

    frame_index = 0
    writer: cv2.VideoWriter | None = None
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_", dir=output_path.parent) as temp_dir:
        temporary_video = Path(temp_dir) / "processed_video.avi"
        temporary_output = Path(temp_dir) / "video_with_audio.avi"
        try:
            writer = cv2.VideoWriter(
                str(temporary_video),
                cv2.VideoWriter_fourcc(*"I420"),
                fps,
                (encoded_width, encoded_height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    "O OpenCV não conseguiu criar o vídeo AVI com o codec I420."
                )

            while True:
                success, frame = capture.read()
                if not success:
                    break

                polygons_to_blur = [
                    region.polygon
                    for region in regions
                    if region.start_frame <= frame_index <= region.end_frame
                ]
                frame = _blur_polygon(frame, polygons_to_blur, blur_sigma)
                if encoded_width != width or encoded_height != height:
                    frame = cv2.copyMakeBorder(
                        frame,
                        0,
                        encoded_height - height,
                        0,
                        encoded_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                writer.write(frame)
                frame_index += 1
                progress(frame_index, frame_count)

            if frame_index < frame_count:
                raise RuntimeError(
                    f"O vídeo terminou no frame {frame_index}, antes do esperado ({frame_count})."
                )
        finally:
            if writer is not None:
                writer.release()
            capture.release()

        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
            raise RuntimeError("O OpenCV não gerou um arquivo AVI válido.")
        written_video = cv2.VideoCapture(str(temporary_video))
        if not written_video.isOpened():
            written_video.release()
            raise RuntimeError("O OpenCV não conseguiu validar o AVI gerado.")
        written_frames = int(written_video.get(cv2.CAP_PROP_FRAME_COUNT))
        written_size = (
            int(written_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(written_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        written_video.release()
        if written_frames != frame_count or written_size != (encoded_width, encoded_height):
            raise RuntimeError(
                "O AVI gerado pelo OpenCV não corresponde à duração ou resolução "
                f"esperada ({written_frames}/{frame_count} frames, {written_size})."
            )

        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temporary_video),
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-f",
                "avi",
                str(temporary_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or f"código de saída {result.returncode}"
            raise RuntimeError(f"FFmpeg não conseguiu preservar o áudio do AVI: {details}")
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("FFmpeg não gerou o arquivo AVI final.")
        os.replace(temporary_output, output_path)


class BlurRegionsApp:
    BG = "#10131a"
    PANEL = "#191e28"
    PANEL_LIGHT = "#222a37"
    TEXT = "#eef2f8"
    MUTED = "#a3adbd"
    ACCENT = "#6c8cff"
    GREEN = "#44c08a"
    RED = "#e16b76"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FrameLab | Borrar regiões em vídeos")
        self.root.geometry("1360x860")
        self.root.minsize(1060, 700)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.fps = 0.0
        self.video_width = 0
        self.video_height = 0
        self.current_frame = 0
        self.current_image: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.image_rect = (0, 0, 0, 0)
        self.regions: list[BlurRegion] = []
        self.current_polygon: list[tuple[int, int]] = []
        self.polygon_finished = False
        self.start_frame: int | None = None
        self.end_frame: int | None = None
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.exporting = False

        self._configure_style()
        self._build_layout()
        self.root.after(100, self._poll_messages)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT)
        style.configure(
            "Title.TLabel", background=self.BG, foreground=self.TEXT,
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "Subtitle.TLabel", background=self.BG, foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel", background=self.PANEL, foreground=self.TEXT,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Accent.TButton", background=self.ACCENT, foreground="#ffffff",
            padding=(12, 9), borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#829cff"), ("disabled", "#46516d")])
        style.configure(
            "Secondary.TButton", background=self.PANEL_LIGHT, foreground=self.TEXT,
            padding=(10, 8), borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#303a4b")])
        style.configure(
            "Horizontal.TScale", background=self.PANEL, troughcolor="#303747",
        )
        style.configure(
            "Horizontal.TProgressbar", background=self.GREEN, troughcolor="#303747",
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(24, 18, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="FrameLab", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Marque polígonos fixos e desfoque-os no intervalo de frames escolhido.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.open_button = ttk.Button(
            header, text="Abrir vídeo", style="Accent.TButton", command=self._open_video
        )
        self.open_button.grid(row=0, column=1, rowspan=2, sticky="e")

        content = ttk.Frame(self.root, padding=(24, 6, 24, 12))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        viewer = ttk.Frame(content, style="Panel.TFrame", padding=12)
        viewer.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            viewer, background="#080a0f", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._render_frame)

        sidebar = ttk.Frame(content, style="Panel.TFrame", padding=16, width=300)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="1  |  Marcar início", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.start_button = ttk.Button(
            sidebar, text="Usar frame atual como início", style="Secondary.TButton",
            command=self._set_start,
        )
        self.start_button.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        self.start_status = ttk.Label(
            sidebar, text="Nenhum início definido", style="Panel.TLabel", foreground=self.MUTED
        )
        self.start_status.grid(row=2, column=0, sticky="w", pady=(0, 15))

        ttk.Label(sidebar, text="2  |  Desenhar região", style="Section.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            text="Clique nos vértices no frame inicial. A posição e o formato ficam fixos.",
            style="Panel.TLabel", wraplength=250, justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(7, 9))
        polygon_actions = ttk.Frame(sidebar, style="Panel.TFrame")
        polygon_actions.grid(row=5, column=0, sticky="ew")
        polygon_actions.columnconfigure(0, weight=1)
        polygon_actions.columnconfigure(1, weight=1)
        self.finish_polygon_button = ttk.Button(
            polygon_actions, text="Concluir polígono", style="Secondary.TButton",
            command=self._finish_polygon,
        )
        self.finish_polygon_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.undo_button = ttk.Button(
            polygon_actions, text="Desfazer ponto", style="Secondary.TButton",
            command=self._undo_point,
        )
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(sidebar, text="3  |  Marcar fim do intervalo", style="Section.TLabel").grid(
            row=6, column=0, sticky="w", pady=(18, 0)
        )
        self.end_button = ttk.Button(
            sidebar, text="Usar frame atual como fim", style="Secondary.TButton",
            command=self._set_end,
        )
        self.end_button.grid(row=7, column=0, sticky="ew", pady=(10, 4))
        self.end_status = ttk.Label(
            sidebar, text="Nenhum fim definido", style="Panel.TLabel", foreground=self.MUTED
        )
        self.end_status.grid(row=8, column=0, sticky="w")
        self.add_region_button = ttk.Button(
            sidebar, text="Adicionar região", style="Accent.TButton",
            command=self._add_region,
        )
        self.add_region_button.grid(row=9, column=0, sticky="ew", pady=(10, 18))

        ttk.Label(sidebar, text="Regiões configuradas", style="Section.TLabel").grid(
            row=10, column=0, sticky="w"
        )
        self.region_list = tk.Listbox(
            sidebar,
            height=7,
            background=self.PANEL_LIGHT,
            foreground=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        self.region_list.grid(row=11, column=0, sticky="nsew", pady=(8, 6))
        sidebar.rowconfigure(11, weight=1)
        ttk.Button(
            sidebar, text="Remover região selecionada", style="Secondary.TButton",
            command=self._remove_region,
        ).grid(row=12, column=0, sticky="ew")

        footer = ttk.Frame(self.root, padding=(24, 8, 24, 18))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        navigation = ttk.Frame(footer)
        navigation.grid(row=0, column=0, sticky="w")
        self.previous_button = ttk.Button(
            navigation, text="◀", width=4, style="Secondary.TButton",
            command=lambda: self._seek(self.current_frame - 1),
        )
        self.previous_button.grid(row=0, column=0, padx=(0, 5))
        self.frame_entry = ttk.Entry(navigation, width=9, justify="center")
        self.frame_entry.grid(row=0, column=1, ipady=5)
        self.frame_entry.bind("<Return>", self._seek_from_entry)
        self.frame_total_label = ttk.Label(navigation, text="/ —", style="Subtitle.TLabel")
        self.frame_total_label.grid(row=0, column=2, padx=(5, 5))
        self.next_button = ttk.Button(
            navigation, text="▶", width=4, style="Secondary.TButton",
            command=lambda: self._seek(self.current_frame + 1),
        )
        self.next_button.grid(row=0, column=3, padx=(0, 8))
        self.time_label = ttk.Label(navigation, text="00:00.000", style="Subtitle.TLabel")
        self.time_label.grid(row=0, column=4)

        self.timeline = ttk.Scale(
            footer, from_=0, to=1, orient="horizontal", command=self._on_timeline
        )
        self.timeline.grid(row=0, column=1, sticky="ew", padx=18)

        export_area = ttk.Frame(footer)
        export_area.grid(row=0, column=2, sticky="e")
        self.status_label = ttk.Label(
            export_area, text="Abra um vídeo para começar", style="Subtitle.TLabel"
        )
        self.status_label.grid(row=0, column=0, sticky="e", padx=(0, 12))
        self.progress = ttk.Progressbar(export_area, length=130, mode="determinate")
        self.progress.grid(row=0, column=1, padx=(0, 10))
        self.export_button = ttk.Button(
            export_area, text="Exportar vídeo", style="Accent.TButton",
            command=self._export,
        )
        self.export_button.grid(row=0, column=2)

    def _open_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Selecione um vídeo",
            filetypes=[
                ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not selected:
            return
        capture = cv2.VideoCapture(selected)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Erro ao abrir vídeo", "O OpenCV não conseguiu abrir esse arquivo.")
            return

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            messagebox.showerror(
                "Vídeo inválido", "Não foi possível ler a duração, FPS ou resolução do vídeo."
            )
            return

        if self.capture is not None:
            self.capture.release()
        self.capture = capture
        self.video_path = Path(selected)
        self.frame_count = frame_count
        self.fps = fps
        self.video_width = width
        self.video_height = height
        self.current_frame = 0
        self.regions.clear()
        self.current_polygon.clear()
        self.polygon_finished = False
        self.start_frame = None
        self.end_frame = None
        self.region_list.delete(0, tk.END)
        self.start_status.configure(text="Nenhum início definido")
        self.end_status.configure(text="Nenhum fim definido")
        self.frame_total_label.configure(text=f"/ {frame_count - 1}")
        self.timeline.configure(to=max(frame_count - 1, 1))
        self.status_label.configure(text=f"{self.video_path.name}  ·  {fps:.2f} FPS")
        self._seek(0)

    def _seek(self, frame_index: int) -> None:
        if self.capture is None:
            return
        self.current_frame = min(max(int(frame_index), 0), self.frame_count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        success, frame = self.capture.read()
        if not success:
            self.status_label.configure(text=f"Não foi possível ler o frame {self.current_frame}")
            return
        self.current_image = frame
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(self.current_frame))
        self.timeline.set(self.current_frame)
        milliseconds = self.current_frame / self.fps * 1000
        minutes, remainder = divmod(milliseconds / 1000, 60)
        self.time_label.configure(text=f"{int(minutes):02d}:{remainder:06.3f}")
        self._render_frame()

    def _seek_from_entry(self, _event: tk.Event[tk.Misc]) -> None:
        del _event
        try:
            self._seek(int(self.frame_entry.get()))
        except ValueError:
            self.status_label.configure(text="Digite um número de frame válido.")

    def _on_timeline(self, value: str) -> None:
        if self.capture is not None:
            frame_index = int(float(value))
            if frame_index != self.current_frame:
                self._seek(frame_index)

    def _render_frame(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        del _event
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 2)
        canvas_height = max(self.canvas.winfo_height(), 2)
        if self.current_image is None:
            self.canvas.create_text(
                canvas_width // 2, canvas_height // 2,
                text="Abra um vídeo para visualizar os frames",
                fill="#7e899b", font=("Segoe UI", 14),
            )
            return

        height, width = self.current_image.shape[:2]
        scale = min((canvas_width - 12) / width, (canvas_height - 12) / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))
        left = (canvas_width - display_width) // 2
        top = (canvas_height - display_height) // 2
        self.image_rect = (left, top, display_width, display_height)
        resized = cv2.resize(
            self.current_image, (display_width, display_height), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")

        for region in self.regions:
            if region.start_frame <= self.current_frame <= region.end_frame:
                self._draw_polygon(region.polygon, "#44c08a", close=True)
        if self.current_polygon:
            self._draw_polygon(np.asarray(self.current_polygon), self.ACCENT, close=False)

    def _draw_polygon(self, polygon: np.ndarray, color: str, close: bool) -> None:
        left, top, display_width, display_height = self.image_rect
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        scale_x = display_width / self.video_width
        scale_y = display_height / self.video_height
        screen_points = [
            (left + int(x * scale_x), top + int(y * scale_y)) for x, y in points
        ]
        if len(screen_points) >= 2:
            flattened = [coordinate for point in screen_points for coordinate in point]
            if close or len(screen_points) > 2:
                flattened.extend(screen_points[0])
            self.canvas.create_line(*flattened, fill=color, width=2, smooth=False)
        for x, y in screen_points:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="#ffffff")

    def _on_canvas_click(self, event: tk.Event[tk.Canvas]) -> None:
        if self.start_frame is None or self.end_frame is not None:
            self.status_label.configure(text="Defina um início antes de desenhar a região.")
            return
        if self.polygon_finished:
            self.status_label.configure(text="Polígono concluído; desfaça um ponto para editá-lo.")
            return
        left, top, display_width, display_height = self.image_rect
        if display_width <= 0 or not (
            left <= event.x <= left + display_width and top <= event.y <= top + display_height
        ):
            return
        x = int((event.x - left) * self.video_width / display_width)
        y = int((event.y - top) * self.video_height / display_height)
        self.current_polygon.append(
            (min(max(x, 0), self.video_width - 1), min(max(y, 0), self.video_height - 1))
        )
        self._render_frame()

    def _set_start(self) -> None:
        if self.capture is None:
            messagebox.showinfo("Abra um vídeo", "Selecione um vídeo antes de marcar regiões.")
            return
        self.start_frame = self.current_frame
        self.end_frame = None
        self.current_polygon.clear()
        self.polygon_finished = False
        self.start_status.configure(text=f"Frame {self.start_frame}")
        self.end_status.configure(text="Nenhum fim definido")
        self.status_label.configure(text="Clique nos vértices da região fixa no vídeo.")
        self._render_frame()

    def _finish_polygon(self) -> None:
        if len(self.current_polygon) < 3:
            messagebox.showinfo("Polígono incompleto", "Adicione pelo menos 3 pontos ao polígono.")
            return
        self.polygon_finished = True
        self.status_label.configure(text="Polígono fixo pronto. Navegue até o último frame.")
        self._render_frame()

    def _undo_point(self) -> None:
        if self.current_polygon:
            self.current_polygon.pop()
            self.polygon_finished = False
            self._render_frame()

    def _set_end(self) -> None:
        if (
            self.start_frame is None
            or len(self.current_polygon) < 3
            or not self.polygon_finished
        ):
            messagebox.showinfo(
                "Região incompleta",
                "Marque o início, desenhe e conclua o polígono antes do frame final.",
            )
            return
        if self.current_frame < self.start_frame:
            messagebox.showwarning(
                "Frame final inválido", "O frame final deve ser igual ou posterior ao início."
            )
            return
        self.end_frame = self.current_frame
        self.end_status.configure(text=f"Frame {self.end_frame}")
        self.status_label.configure(text="Confirme para adicionar o intervalo.")

    def _add_region(self) -> None:
        if (
            self.start_frame is None
            or self.end_frame is None
            or len(self.current_polygon) < 3
            or not self.polygon_finished
        ):
            messagebox.showinfo(
                "Região incompleta", "Defina início, polígono e fim antes de adicionar a região."
            )
            return
        polygon = np.asarray(self.current_polygon, dtype=np.int32)
        self.regions.append(BlurRegion(self.start_frame, self.end_frame, polygon))
        index = len(self.regions)
        self.region_list.insert(
            tk.END,
            f"Região {index}  ·  frames {self.start_frame}–{self.end_frame}",
        )
        self.start_frame = None
        self.end_frame = None
        self.current_polygon.clear()
        self.polygon_finished = False
        self.start_status.configure(text="Nenhum início definido")
        self.end_status.configure(text="Nenhum fim definido")
        self.status_label.configure(text=f"Região {index} adicionada.")
        self._render_frame()

    def _remove_region(self) -> None:
        selection = self.region_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.region_list.delete(index)
        del self.regions[index]
        self.region_list.delete(0, tk.END)
        for region_index, region in enumerate(self.regions, start=1):
            self.region_list.insert(
                tk.END,
                f"Região {region_index}  ·  frames {region.start_frame}–{region.end_frame}",
            )
        self._render_frame()

    def _export(self) -> None:
        if self.exporting:
            return
        if self.video_path is None or not self.regions:
            messagebox.showinfo(
                "Nada para exportar", "Abra um vídeo e adicione pelo menos uma região."
            )
            return
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            messagebox.showerror(
                "FFmpeg não encontrado",
                "Instale o FFmpeg e adicione a pasta bin ao PATH do Windows.",
            )
            return
        output_name = filedialog.asksaveasfilename(
            title="Salvar vídeo borrado sem perda",
            initialfile=f"{self.video_path.stem}_borrado.avi",
            defaultextension=".avi",
            filetypes=[("Vídeo AVI sem compressão (I420)", "*.avi")],
        )
        if not output_name:
            return
        output_path = Path(output_name)
        if output_path.suffix.lower() != ".avi":
            output_path = output_path.with_suffix(".avi")

        self.exporting = True
        self.export_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.progress.configure(maximum=self.frame_count, value=0)
        self.status_label.configure(text="Preparando exportação…")
        regions = [
            BlurRegion(region.start_frame, region.end_frame, region.polygon.copy())
            for region in self.regions
        ]
        worker = threading.Thread(
            target=self._export_worker,
            args=(self.video_path, output_path, regions, ffmpeg_path),
            daemon=True,
        )
        worker.start()

    def _export_worker(
        self,
        input_path: Path,
        output_path: Path,
        regions: list[BlurRegion],
        ffmpeg_path: str,
    ) -> None:
        try:
            export_video(
                input_path,
                output_path,
                regions,
                ffmpeg_path,
                blur_sigma=22.0,
                progress=lambda done, total: self.messages.put(("progress", (done, total))),
            )
            self.messages.put(("done", output_path))
        except Exception as error:
            self.messages.put(("error", str(error)))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "progress":
                    done, total = payload  # type: ignore[misc]
                    self.progress.configure(maximum=total, value=done)
                    self.status_label.configure(text=f"Processando frames: {done}/{total}")
                elif kind == "done":
                    self.exporting = False
                    self.export_button.configure(state="normal")
                    self.open_button.configure(state="normal")
                    self.status_label.configure(text="Exportação concluída.")
                    messagebox.showinfo("Vídeo salvo", f"Arquivo criado:\n{payload}")
                elif kind == "error":
                    self.exporting = False
                    self.export_button.configure(state="normal")
                    self.open_button.configure(state="normal")
                    self.status_label.configure(text="A exportação falhou.")
                    messagebox.showerror("Erro na exportação", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _close(self) -> None:
        if self.exporting:
            messagebox.showwarning(
                "Exportação em andamento",
                "Aguarde a conclusão da exportação antes de fechar o aplicativo.",
            )
            return
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    BlurRegionsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()