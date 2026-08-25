import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import platform

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import heartpy as hp

SIGNAL_TIME_OFFSET = timedelta(hours=-5)


@dataclass(frozen=True)
class AppSettings:
    remote_params_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi0TexCrRMbHHODBHKEWmoA8ipixOkFQqgVdHiznKbn19cBa6VignR47r90AweuomdhyQFCBInDE9y/pub?output=csv"


    if platform.system() == "Windows":
        uti_data_path = Path(r"\\10.8.0.1\uti\Data")
        dataset_raw_file = Path.home() / "Documents" / "rPPG_data" / "ground_truth" / "dataset_raw.csv"
        patient_data_root = Path(r"A:\dataset_raw")

    else:
        uti_data_path = "/mnt/10.8.0.1/uti/Data"
        dataset_raw_file = Path.home() / "rppg" / "rPPG_data" / "ground_truth" / "dataset_raw.csv"
        patient_data_root = Path.home() / "ssd" / "dataset_raw"

    default_video_filename: str = "video_cropped.avi"
    signal_filename_templates: dict[str, str] = field(default_factory=lambda: {
        "ecg": "ecg_{date}_{bed}_{start}_{end}.csv",
        "ppg": "ppg_{date}_{bed}_{start}_{end}.txt",
        "respiration": "respiration_{date}_{bed}_{start}_{end}.txt",
    })
    reference_filename_template: str = "{signal}_{patient_id}.txt"
    video_filename_template: str = "{bed}-{date}-{hour}.avi"


APP_SETTINGS = AppSettings()


def signal_timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp) + SIGNAL_TIME_OFFSET

# =============================================================================
# CONFIGURAÇÕES DA EXTRAÇÃO
# =============================================================================

@dataclass
class Config:
    file_path: str = ""
    date: str = ""
    start_time: str = "00:00:00"
    end_time: str = "00:00:00"
    bed: str = ""
    output_dir: str = ""
    save_ecg: bool = True
    save_spo2_wave: bool = True
    save_rr: bool = True
    show_plots: bool = True
    data_pack_head: bytes = b"\x02\x0B\x00\x00"
    data_add: int = 36
    ecg_id: int = 65796
    spo2_id: int = 458768
    resp_id: int = 327688
    interactive_select_time: bool = True
    duration_seconds: float = 120.0  # Aceita casas decimais da duração do vídeo
    selected_start_ts: float | None = None
    selected_end_ts: float | None = None

    def __post_init__(self):
        self.hora_inicio = self.start_time.replace(':', '-')
        self.hora_fim = self.end_time.replace(':', '-')
        self.output_path = Path(self.output_dir) if self.output_dir else Path(APP_SETTINGS.patient_data_root)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.ecg_dir = self.output_path
        self.spo2_dir = self.output_path
        self.rr_dir = self.output_path
        for folder in (self.ecg_dir, self.spo2_dir, self.rr_dir):
            folder.mkdir(parents=True, exist_ok=True)


def human_filename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def write_cut_video(input_path, start_frame, output_path=None, codec='I420'):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open input: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if output_path is None:
        dirname = os.path.dirname(input_path)
        name = human_filename(input_path)
        output_path = os.path.join(dirname, f"{name}_cut_{start_frame:06d}.avi")
    elif not str(output_path).lower().endswith(".avi"):
        output_path = str(output_path).rsplit(".", 1)[0] + ".avi"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Writing from frame {start_frame} to end into: {output_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Saved cut video: {output_path}")
    return output_path


# =============================================================================
# FUNÇÕES DE PROCESSAMENTO DE SINAL
# =============================================================================

def estimate_hr_heartpy(segment, fs):
    try:
        _, metrics = hp.process(segment, sample_rate=fs)
        return metrics["bpm"]
    except Exception:
        return np.nan

def load_hdf5_packets(file_path, data_pack_head, data_add):
    with h5py.File(file_path, "r") as hdf:
        raw_packets = hdf["data"][:]
        timestamps = hdf["data_timestamps"][:]

    datas, ids, seqs, seqsts = [], [], [], []
    for raw_data, ts in zip(raw_packets, timestamps):
        pack_id = bytes(raw_data[0:4])
        if pack_id != data_pack_head:
            continue

        frame_seq = int.from_bytes(raw_data[24:26], byteorder="big")
        local_data_add = data_add
        process_next = True

        while process_next:
            if local_data_add + 4 > len(raw_data):
                process_next = False
                continue

            data_head = raw_data[local_data_add:local_data_add + 4]
            data_len = int(raw_data[local_data_add + 3]) * 2
            if data_len <= 0:
                process_next = False
                continue

            if local_data_add + 4 + data_len <= len(raw_data):
                payload = raw_data[local_data_add + 4:local_data_add + 4 + data_len]
            else:
                payload = raw_data[local_data_add + 4:]

            signal_data = np.frombuffer(payload, dtype=">i2")
            datas.append(signal_data)
            ids.append(int.from_bytes(data_head[0:3], byteorder="big"))
            seqs.append(frame_seq)
            seqsts.append(ts)
            local_data_add = int(local_data_add) + 4 + data_len

    return datas, np.array(ids), np.array(seqs), np.array(seqsts)

def build_time_vectors(ts, datas_chunk, fs):
    dt = 1 / fs
    time_vector = []
    for t, data_chunk in zip(ts, datas_chunk):
        time_vector.append(t + np.arange(len(data_chunk)) * dt)
    return np.concatenate(time_vector)

def get_window_mask(dates_np, start_time, end_time):
    start_dt = datetime.combine(dates_np[0].date(), datetime.strptime(start_time, "%H:%M:%S").time())
    end_dt = datetime.combine(dates_np[0].date(), datetime.strptime(end_time, "%H:%M:%S").time())
    return (dates_np >= start_dt) & (dates_np <= end_dt)

def process_ecg(config, datas, ids, seqsts):
    if config.ecg_id not in np.unique(ids):
        return None, None

    indices = np.where(ids == config.ecg_id)[0]
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))
    
    sig = np.concatenate([datas[i] for i in indices])
    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    
    def select_start_point_interactive(dates, signal, fs):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(dates, signal, color='tab:blue', alpha=0.8)
        ax.set_xlabel('Horário')
        ax.set_ylabel('Amplitude')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.grid(True)
        plt.title(f'Clique no INÍCIO dos sinais. Duração do corte: {config.duration_seconds:.1f}s. "y" p/ salvar, "n" p/ cancelar.')
        plt.tight_layout()

        dnums = mdates.date2num(dates)
        vline = ax.axvline(dnums[0], color='gray', linewidth=1, linestyle='--')
        hover_ann = ax.annotate('', xy=(0,0), xytext=(15,15), textcoords='offset points', bbox=dict(boxstyle='round', fc='w'), visible=False)
        sel_marker, = ax.plot([], [], 'o', color='red', markersize=10, markeredgecolor='black', visible=False)

        selected = {'index': None, 'confirmed': False}

        def on_move(event):
            if event.inaxes != ax or event.xdata is None: return
            x = event.xdata
            vline.set_xdata([x, x])
            idx = int(np.argmin(np.abs(dnums - x)))
            tstr = dates[idx].strftime('%H:%M:%S')
            hover_ann.set_text(f'{tstr} [{idx}]')
            hover_ann.xy = (dnums[idx], signal[idx])
            hover_ann.set_visible(True)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None: return
            x = event.xdata
            idx = int(np.argmin(np.abs(dnums - x)))
            selected['index'] = idx
            sel_marker.set_data([dnums[idx]], [signal[idx]])
            sel_marker.set_visible(True)
            fig.canvas.draw_idle()

        def on_key(event):
            if selected['index'] is None: return
            if event.key == 'y':
                selected['confirmed'] = True
                plt.close(fig)
            elif event.key == 'n':
                selected['index'] = None
                sel_marker.set_visible(False)
                fig.canvas.draw_idle()

        cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        try:
            fig.canvas.mpl_disconnect(cid_move)
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
        except Exception:
            pass

        if selected['confirmed'] and selected['index'] is not None:
            start_idx = selected['index']
            end_idx = min(start_idx + int(config.duration_seconds * fs), len(signal) - 1)
            start_time_dt = dates[start_idx]
            end_time_dt = dates[end_idx]
            return start_time_dt.strftime('%H:%M:%S'), end_time_dt.strftime('%H:%M:%S'), start_idx, end_idx
        return None, None, None, None

    selected_info = (None, None, None, None)
    if config.interactive_select_time and config.show_plots:
        s, e, si, ei = select_start_point_interactive(dates_np, sig, fs)
        if s is not None and e is not None:
            config.start_time = s
            config.end_time = e
            config.hora_inicio = config.start_time.replace(':', '-')
            config.hora_fim = config.end_time.replace(':', '-')
            config.selected_start_ts = time_vector[si]
            config.selected_end_ts = time_vector[ei]
            selected_info = (s, e, si, ei)

    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    output_file = ""
    if config.save_ecg:
        output_file = config.ecg_dir / APP_SETTINGS.signal_filename_templates["ecg"].format(
            date=config.date, bed=config.bed, start=config.hora_inicio, end=config.hora_fim
        )
        np.savetxt(output_file, sig[mask], delimiter=",", fmt="%d")

    return selected_info, str(output_file)

def process_spo2(config, datas, ids, seqsts):
    indices = np.where(ids == config.spo2_id)[0]
    if len(indices) == 0:
        return ""

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([signal_timestamp_to_datetime(ts_value) for ts_value in time_vector])
    
    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    output_file = ""
    if config.save_spo2_wave:
        sig_m = sig[mask].astype(float)
        output_file = config.spo2_dir / APP_SETTINGS.signal_filename_templates["ppg"].format(
            date=config.date, bed=config.bed, start=config.hora_inicio, end=config.hora_fim
        )
        np.savetxt(output_file, sig_m, fmt="%.7e")
    
    return str(output_file)

def process_rr(config, datas, ids, seqsts):
    indices = np.where(ids == config.resp_id)[0]
    if len(indices) == 0:
        return ""

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([signal_timestamp_to_datetime(ts_value) for ts_value in time_vector])
    
    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    sig_m = sig[mask].astype(float)
    output_file = config.rr_dir / APP_SETTINGS.signal_filename_templates["respiration"].format(
        date=config.date, bed=config.bed, start=config.hora_inicio, end=config.hora_fim
    )
    np.savetxt(output_file, sig_m, fmt="%.7e")
    
    return str(output_file)

def run_extraction_for_patient(h5_file, bed, date_str, duration_seconds=120.0, patient_output_dir=None):
    output_dir = str(patient_output_dir) if patient_output_dir is not None else str(Path(APP_SETTINGS.patient_data_root))
    config = Config(
        file_path=h5_file,
        bed=bed,
        date=date_str.replace("/", "-"),
        duration_seconds=duration_seconds,
        output_dir=output_dir
    )
    config.output_path = Path(output_dir)
    config.output_path.mkdir(parents=True, exist_ok=True)
    config.ecg_dir = config.output_path
    config.spo2_dir = config.output_path
    config.rr_dir = config.output_path
    
    try:
        datas, ids, _, seqsts = load_hdf5_packets(
            config.file_path,
            config.data_pack_head,
            config.data_add,
        )
    except Exception as e:
        return {"error": f"Erro ao ler H5: {e}"}

    if not datas:
        return {"error": "Nenhum pacote de dados encontrado no H5."}

    sel, ecg_path = process_ecg(config, datas, ids, seqsts)
    if not sel or sel[0] is None:
        return {"error": "Seleção cancelada."}
        
    start_time, end_time = sel[0], sel[1]
    spo2_path = process_spo2(config, datas, ids, seqsts)
    rr_path = process_rr(config, datas, ids, seqsts) if config.save_rr else ""

    return {
        "init_time": start_time,
        "end_time": end_time,
        "sinal_ECG": ecg_path,
        "sinal_PPG": spo2_path,
        "sinal_resp": rr_path,
        "error": None
    }


# =============================================================================
# INTERFACE GRÁFICA (GUI)
# =============================================================================

class SignalExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extrator de Sinais - UTI")
        self.root.geometry("1200x500")
        
        self.dataset_raw_file = APP_SETTINGS.dataset_raw_file
        self.df = None
        
        # Configurando Grid Principal
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        # Header
        header_frame = tk.Frame(self.root, pady=10)
        header_frame.grid(row=0, column=0, sticky="ew")
        
        tk.Label(header_frame, text="Selecione um paciente para Processar Sinais ou Cortar Vídeo", font=("Arial", 14, "bold")).pack()
        
        # Treeview (Tabela)
        columns = ("index", "id", "leito", "data", "hora", "status", "reference_status", "video_status")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("index", text="Índice")
        self.tree.heading("id", text="ID Paciente")
        self.tree.heading("leito", text="Leito")
        self.tree.heading("data", text="Data")
        self.tree.heading("hora", text="Hora")
        self.tree.heading("status", text="Status (Sinais)")
        self.tree.heading("reference_status", text="Sinais de referência carregados")
        self.tree.heading("video_status", text="Vídeo")
        
        self.tree.column("index", width=50, anchor="center")
        self.tree.column("id", width=120, anchor="center")
        self.tree.column("leito", width=80, anchor="center")
        self.tree.column("data", width=100, anchor="center")
        self.tree.column("hora", width=80, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.column("reference_status", width=220, anchor="center")
        self.tree.column("video_status", width=120, anchor="center")
        
        self.tree.grid(row=1, column=0, sticky="nsew", padx=20)
        
        # Scrollbar para a Tabela
        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky='ns', pady=0)
        
        # Estilos para o Status
        self.tree.tag_configure('pendente', foreground='red')
        self.tree.tag_configure('preenchido', foreground='green')
        
        # Botões
        btn_frame = tk.Frame(self.root, pady=15)
        btn_frame.grid(row=2, column=0, sticky="ew")
        
        btn_cut_video = tk.Button(btn_frame, text="1. Cortar Vídeo", command=self.cut_video_for_selected, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), width=15)
        btn_cut_video.pack(side=tk.LEFT, padx=15)
        
        btn_process_signals = tk.Button(btn_frame, text="2. Processar Sinais", command=self.process_signals_for_selected, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_process_signals.pack(side=tk.LEFT, padx=15)

        btn_visualize_signals = tk.Button(btn_frame, text="3. Visualizar Sinais", command=self.visualize_saved_signals_for_selected, bg="#FF9800", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_visualize_signals.pack(side=tk.LEFT, padx=15)

        btn_sync_remote = tk.Button(btn_frame, text="4. Sincronizar Planilha", command=self.sync_local_dataset_with_remote, bg="#9C27B0", fg="white", font=("Arial", 11, "bold"), width=20)
        btn_sync_remote.pack(side=tk.LEFT, padx=15)

        btn_load_reference = tk.Button(btn_frame, text="5. Carregar Sinais de Referência", command=self.load_reference_signals_for_selected, bg="#607D8B", fg="white", font=("Arial", 11, "bold"), width=28)
        btn_load_reference.pack(side=tk.LEFT, padx=15)

        btn_analyze_dataset = tk.Button(btn_frame, text="6. Analisar Planilha", command=self.show_dataset_analysis, bg="#795548", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_analyze_dataset.pack(side=tk.LEFT, padx=15)
        
        btn_refresh = tk.Button(btn_frame, text="Recarregar Planilha", command=self.load_data, font=("Arial", 10))
        btn_refresh.pack(side=tk.RIGHT, padx=20)
        
        # Carregar Dados
        self.load_data()

    def _normalize_sheet_like_get_h5(self, df):
        if df is None or df.empty:
            return pd.DataFrame()

        df_norm = df.copy()

        try:

            df_norm = df_norm.iloc[:, 1:]
            df_norm = df_norm.T.reset_index()
            df_norm.columns = df_norm.iloc[0]
            df_norm = df_norm.iloc[1:].reset_index(drop=True)
            df_norm.columns = [
                "" if pd.isna(column) else str(column).strip()
                for column in df_norm.columns
            ]
            df_norm = df_norm.loc[:, [
                column and not column.lower().startswith("unnamed")
                for column in df_norm.columns
            ]]
            print('df norm', df_norm)
            return df_norm
        except Exception:
            return df_norm

    def _normalize_remote_sheet(self, df):
        """Aplica o mesmo tratamento usado em get_h5.py:
        remove a coluna de índice, transpõe o DataFrame, usa a primeira linha como cabeçalho
        e reinicia o índice.
        """
        return self._normalize_sheet_like_get_h5(df)

    @staticmethod
    def _coerce_remote_value(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return pd.NA

        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return pd.NA
            normalized = value.lower()
            if normalized in {"true", "false"}:
                return normalized == "true"
            if normalized in {"nan", "none", "null"}:
                return pd.NA

            try:
                if value.startswith("0") and value[1:].isdigit() and len(value) > 1:
                    return int(value)
            except Exception:
                pass

            try:
                return int(value)
            except (TypeError, ValueError):
                pass

            try:
                return float(value)
            except (TypeError, ValueError):
                pass

            return value

        return value

    @staticmethod
    def _is_filled(value):
        if value is None or pd.isna(value):
            return False
        normalized = str(value).strip().lower()
        return (
            normalized not in {"", "nan", "none", "null", "<na>"}
            and not normalized.startswith("unnamed")
        )

    @staticmethod
    def _normalize_patient_id(value):
        normalized = str(value).strip().lower()
        if normalized.endswith(".0") and normalized[:-2].isdigit():
            return normalized[:-2]
        return normalized

    @staticmethod
    def _patient_folder_name(value):
        patient_id = str(value).strip().lower()
        if "." in patient_id:
            base_id, index = patient_id.rsplit(".", 1)
            if base_id and index.isdigit():
                patient_id = f"{base_id}_{index}"
        return "".join(char if char.isalnum() or char == "_" else "_" for char in patient_id)

    def _patient_data_dir(self, patient_id):
        return Path(APP_SETTINGS.patient_data_root) / self._patient_folder_name(patient_id)

    @staticmethod
    def _is_reference_signal_file(file_path):
        return "_gt_" in Path(str(file_path)).name or "_referencia_" in Path(str(file_path)).name

    def _delete_previous_processed_signals(self, previous_paths, current_paths):
        current_paths = {os.path.normcase(os.path.abspath(str(path))) for path in current_paths if self._is_filled(path)}
        for path in previous_paths:
            if not self._is_filled(path) or self._is_reference_signal_file(path):
                continue
            normalized_path = os.path.normcase(os.path.abspath(str(path)))
            if normalized_path in current_paths:
                continue
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                print(f"Aviso: não foi possível remover o sinal anterior {path}: {exc}")

    @staticmethod
    def _normalize_hour(value):
        if not SignalExtractorApp._is_filled(value):
            return ""

        parts = str(value).strip().split(":")
        if len(parts) >= 2:
            try:
                return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
            except ValueError:
                pass
        return str(value).strip().lower()

    def _get_bed_ip(self, bed, ip_ids_file):
        bed_norm = str(bed).strip()
        if bed_norm.isdigit():
            bed_norm = bed_norm.zfill(2)

        with open(ip_ids_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if f"LEITO {bed_norm}" in line:
                    return line.split(',')[1].strip()
        raise ValueError(f"Bed {bed} not found in {ip_ids_file}")

    def _build_h5_path_from_row(self, row):
        date = str(row.get("Dia", "")).strip()
        bed = str(row.get("Leito", "")).strip()
        time_value = str(row.get("Hora", "")).strip()
        if not date or not bed or not time_value:
            return ""

        if "/" in date:
            date_parts = [p.strip() for p in date.split("/")]
        elif "-" in date:
            date_parts = [p.strip() for p in date.split("-")]
        else:
            return ""

        if len(date_parts) != 3:
            return ""

        day, month, year = date_parts
        day_folder = f"{year}{month}{day}"

        try:
            time = str(int(float(str(time_value).split(":")[0]))).strip()
        except Exception:
            return ""

        try:
            bed_norm = str(int(float(bed))).zfill(2)
        except Exception:
            bed_norm = bed

        ip_ids_file = f"{APP_SETTINGS.uti_data_path}/{day_folder}/{day_folder}_{int(time) + 1}_onLineDevices.log"

        try:
            bed_ip = self._get_bed_ip(bed_norm, ip_ids_file)
        except Exception:
            return ""

        return f"{APP_SETTINGS.uti_data_path}/{day_folder}/{bed_ip}_{day_folder}_{time}.h5"

    def sync_local_dataset_with_remote(self):
        try:
            remote_df = pd.read_csv(APP_SETTINGS.remote_params_url)
        except Exception as exc:
            messagebox.showerror("Erro de sincronização", f"Não foi possível ler a planilha remota:\n{exc}")
            return

        remote_df = self._normalize_remote_sheet(remote_df)
        if remote_df.empty:
            messagebox.showwarning("Sincronização", "A planilha remota está vazia.")
            return

        if "Id do paciente" not in remote_df.columns:
            messagebox.showerror("Erro de sincronização", "A planilha remota não possui a coluna 'Id do paciente'.")
            return

        remote_df = remote_df[remote_df["Id do paciente"].apply(self._is_filled)].copy()
        if remote_df.empty:
            messagebox.showwarning("Sincronização", "A planilha remota não possui pacientes com ID preenchido.")
            return

        if not os.path.exists(self.dataset_raw_file) or os.path.getsize(self.dataset_raw_file) == 0:
            base_df = remote_df.copy()
        else:
            base_df = pd.read_csv(self.dataset_raw_file)


        local_specific_cols = [
            "init_time", "end_time", "sinal_ECG", "sinal_PPG", "sinal_resp",
            "video_path", "video_duration", "h5_file"
        ]
        for col in local_specific_cols:
            if col not in base_df.columns:
                base_df[col] = pd.NA

        if "Id do paciente" not in base_df.columns:
            messagebox.showerror("Erro de sincronização", "A planilha local não possui a coluna 'Id do paciente'.")
            return
        base_df = base_df[base_df["Id do paciente"].apply(self._is_filled)].copy()
        if "Hora" not in base_df.columns:
            base_df["Hora"] = pd.NA

        shared_cols = [c for c in remote_df.columns if c in base_df.columns and c != "h5_file"]
        added_count = 0
        updated_count = 0

        for idx, row in remote_df.iterrows():
            patient_id = self._normalize_patient_id(row["Id do paciente"])
            patient_hour = self._normalize_hour(row.get("Hora", ""))
            normalized_ids = base_df["Id do paciente"].apply(self._normalize_patient_id)
            normalized_hours = base_df["Hora"].apply(self._normalize_hour)
            match_mask = (normalized_ids == patient_id) & (normalized_hours == patient_hour)
            if match_mask.any():
                target_idx = base_df.index[match_mask][0]
                patient_updated = False
                for col in shared_cols:
                    if col not in row.index:
                        continue
                    value = self._coerce_remote_value(row[col])
                    if pd.isna(value):
                        continue

                    current_value = base_df.at[target_idx, col]
                    current_text = "" if pd.isna(current_value) else str(current_value).strip()
                    remote_text = str(value).strip()
                    if current_text != remote_text:
                        base_df.at[target_idx, col] = value
                        patient_updated = True

                if "h5_file" not in base_df.columns:
                    base_df["h5_file"] = pd.NA
                existing_h5 = str(base_df.at[target_idx, "h5_file"]).strip() if pd.notna(base_df.at[target_idx, "h5_file"]) else ""
                if not existing_h5 or not os.path.exists(existing_h5):
                    built_h5 = self._build_h5_path_from_row(base_df.loc[target_idx].to_dict())
                    if built_h5:
                        base_df.at[target_idx, "h5_file"] = built_h5

                if patient_updated:
                    updated_count += 1
            else:
                new_row = {col: pd.NA for col in local_specific_cols}
                for col in remote_df.columns:
                    if col in base_df.columns:
                        new_row[col] = self._coerce_remote_value(row[col])
                if "h5_file" in new_row:
                    new_row["h5_file"] = self._build_h5_path_from_row(new_row)
                base_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)
                added_count += 1

        if "Id do paciente" in base_df.columns:
            base_df["Id do paciente"] = base_df["Id do paciente"].astype(str)

        base_df.to_csv(self.dataset_raw_file, index=False)
        self.df = base_df
        self.update_treeview()

        message = (
            "Sincronização concluída.\n\n"
            f"Pacientes adicionados: {added_count}\n"
            f"Pacientes com informações atualizadas: {updated_count}\n\n"
            f"Arquivo salvo em: {self.dataset_raw_file}"
        )
        messagebox.showinfo("Sincronização concluída", message)

    def load_data(self):
        if not os.path.exists(self.dataset_raw_file):
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{self.dataset_raw_file}\n\nRode o get_h5.py primeiro.")
            return
            
        try:
            if os.path.getsize(self.dataset_raw_file) == 0:
                self.df = pd.DataFrame()
                novas_colunas = [
                    "init_time", "end_time", "sinal_ECG", "sinal_PPG",
                    "sinal_resp", "video_path", "video_duration",
                ]
                for col in novas_colunas:
                    self.df[col] = pd.NA
                self.update_treeview()
                messagebox.showwarning(
                    "Planilha vazia",
                    "A planilha local está vazia. Clique em 'Sincronizar Planilha' para recuperá-la da planilha remota.",
                )
                return
            self.df = pd.read_csv(self.dataset_raw_file)
            
            # Garante a existência das colunas, incluindo video_duration
            novas_colunas = ["init_time", "end_time", "sinal_ECG", "sinal_PPG", "sinal_resp", "video_path", "video_duration"]
            for col in novas_colunas:
                if col not in self.df.columns:
                    self.df[col] = pd.NA
                    
            self.update_treeview()
        except Exception as e:
            messagebox.showerror("Erro de Leitura", f"Erro ao ler o CSV:\n{e}\n{self.dataset_raw_file}")

    @staticmethod
    def _format_analysis_value(value):
        if isinstance(value, (bool, np.bool_)):
            return "Verdadeiro" if value else "Falso"
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _analysis_distribution(series):
        valid = series.dropna()
        valid = valid[valid.astype(str).str.strip() != ""]
        if valid.empty:
            return [], 0, int(series.size)

        counts = valid.value_counts(dropna=False)
        total = int(valid.size)
        distribution = [
            (SignalExtractorApp._format_analysis_value(value), int(count), count / total * 100)
            for value, count in counts.items()
        ]
        return distribution, total, int(series.size - total)

    def show_dataset_analysis(self):
        if self.df is None:
            messagebox.showwarning("Análise", "A planilha local ainda não foi carregada.")
            return

        analysis_columns = [
            "Distância horizontal da câmera ao rosto (cm)",
            "Iluminação (lux)",
            "Luz direta natural",
            "Luz indireta natural",
            "Luz direta artificial",
            "Luz indireta artificial",
            "Sexo biológico",
            "Fototipo",
        ]
        signal_columns = {
            "ECG": "ECG",
            "PPG": "PPG",
            "Sinal respiratório": "Sinal respiratório",
        }
        report_lines = [
            "ANÁLISE DA PLANILHA LOCAL",
            f"Arquivo: {self.dataset_raw_file}",
            f"Registros analisados: {len(self.df)}",
            "",
        ]

        figure = Figure(figsize=(16, 12), dpi=100)
        figure.patch.set_facecolor("#F5F7FA")
        figure.suptitle("Panorama dos dados da planilha", fontsize=16, fontweight="bold", color="#263238")
        axes = figure.subplots(3, 4).ravel()
        for axis in axes:
            axis.set_facecolor("#F5F7FA")
        distance_column = analysis_columns[0]
        distance_values = pd.to_numeric(
            self.df[distance_column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).dropna() if distance_column in self.df.columns else pd.Series(dtype=float)
        distance_axis = axes[0]
        if distance_values.empty:
            distance_axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        else:
            distance_summary = [distance_values.min(), distance_values.mean(), distance_values.max()]
            distance_axis.bar(["Mínimo", "Média", "Máximo"], distance_summary, color=["#4C78A8", "#59A14F", "#E15759"])
            distance_axis.set_ylabel("cm")
            for position, value in enumerate(distance_summary):
                distance_axis.text(position, value, f"{value:.2f}", ha="center", va="bottom")
        distance_axis.set_title("Distância da câmera")
        distance_axis.grid(axis="y", alpha=0.3)
        report_lines.extend([
            distance_column.upper(),
            f"  Válidos: {int(distance_values.size)} | Ausentes: {int(len(self.df) - distance_values.size)}",
        ])
        if distance_values.empty:
            report_lines.append("  Nenhum valor disponível.")
        else:
            report_lines.append(f"  Mínimo: {distance_values.min():.2f} cm")
            report_lines.append(f"  Média: {distance_values.mean():.2f} cm")
            report_lines.append(f"  Máximo: {distance_values.max():.2f} cm")
        report_lines.append("")

        for chart_index, column in enumerate(analysis_columns[1:], start=1):
            report_lines.append(column.upper())
            if column not in self.df.columns:
                report_lines.append("  Coluna não encontrada.")
                report_lines.append("")
                axes[chart_index].set_title(column)
                axes[chart_index].text(0.5, 0.5, "Coluna não encontrada", ha="center", va="center")
                continue

            axis = axes[chart_index]
            if column == "Iluminação (lux)":
                illumination_values = pd.to_numeric(
                    self.df[column].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                ).dropna()
                if illumination_values.empty:
                    axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
                    report_lines.append("  Nenhum valor disponível.")
                else:
                    illumination_summary = [
                        illumination_values.min(),
                        illumination_values.mean(),
                        illumination_values.max(),
                    ]
                    axis.bar(
                        ["Mínimo", "Média", "Máximo"],
                        illumination_summary,
                        color=["#4C78A8", "#59A14F", "#E15759"],
                    )
                    axis.set_ylabel("lux")
                    axis.grid(axis="y", alpha=0.3)
                    for position, value in enumerate(illumination_summary):
                        axis.text(position, value, f"{value:.2f}", ha="center", va="bottom")
                    report_lines.append(
                        f"  Válidos: {int(illumination_values.size)} | Ausentes: {int(len(self.df) - illumination_values.size)}"
                    )
                    report_lines.append(f"  Mínimo: {illumination_values.min():.2f} lux")
                    report_lines.append(f"  Média: {illumination_values.mean():.2f} lux")
                    report_lines.append(f"  Máximo: {illumination_values.max():.2f} lux")
                axis.set_title("Iluminação")
                report_lines.append("")
                continue

            distribution, valid_count, missing_count = self._analysis_distribution(self.df[column])
            if distribution:
                labels = [item[0] for item in distribution]
                percentages = [item[2] for item in distribution]
                axis.pie(
                    percentages,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 8},
                )
            else:
                axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            axis.set_title(column.replace(" (cm)", "").replace(" (lux)", ""))
            report_lines.append(f"  Válidos: {valid_count} | Ausentes: {missing_count}")
            if not distribution:
                report_lines.append("  Nenhum valor disponível.")
            else:
                for value, count, percentage in distribution:
                    report_lines.append(f"  {value}: {count} ({percentage:.2f}%)")
            report_lines.append("")

        report_lines.append("CAPTURA DOS SINAIS")
        for signal_index, (label, column) in enumerate(signal_columns.items(), start=8):
            if column not in self.df.columns:
                report_lines.append(f"{label}: coluna não encontrada.")
                axes[signal_index].set_title(label)
                axes[signal_index].text(0.5, 0.5, "Coluna não encontrada", ha="center", va="center")
                continue

            normalized = self.df[column].map(self._normalize_boolean_value)
            valid = normalized.dropna()
            total = int(valid.size)
            missing = int(len(self.df) - total)
            report_lines.append(f"{label}")
            if total == 0:
                report_lines.append("  Nenhum valor booleano disponível.")
                axes[signal_index].set_title(label)
                axes[signal_index].text(0.5, 0.5, "Sem dados", ha="center", va="center")
            else:
                captured = int((valid == True).sum())
                not_captured = int((valid == False).sum())
                axes[signal_index].pie(
                    [captured, not_captured],
                    labels=["Capturado", "Não capturado"],
                    autopct="%1.1f%%",
                    startangle=90,
                    counterclock=False,
                    colors=["#59A14F", "#E15759"],
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 8},
                )
                axes[signal_index].set_title(label)
                report_lines.append(f"  Capturado (Verdadeiro): {captured} ({captured / total * 100:.2f}%)")
                report_lines.append(f"  Não capturado (Falso): {not_captured} ({not_captured / total * 100:.2f}%)")
                report_lines.append(f"  Ausentes ou inválidos: {missing}")
        figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0, w_pad=1.5)
        self._open_analysis_window("Análise da Planilha", "\n".join(report_lines), figure)

    @staticmethod
    def _normalize_boolean_value(value):
        if value is None or pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "verdadeiro", "sim", "1"}:
            return True
        if normalized in {"false", "falso", "não", "nao", "0"}:
            return False
        return pd.NA

    def _open_analysis_window(self, title, content, figure=None):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("1200x900")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=3)
        window.rowconfigure(1, weight=1)

        if figure is not None:
            chart_frame = tk.Frame(window)
            chart_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
            chart_frame.columnconfigure(0, weight=1)
            chart_frame.rowconfigure(0, weight=1)
            canvas = FigureCanvasTkAgg(figure, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        text = tk.Text(window, wrap=tk.WORD, font=("Consolas", 10))
        text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)

    def load_reference_signals_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        idx = int(self.tree.item(selected_item[0], "values")[0])
        paciente = self.df.loc[idx]
        patient_id = paciente.get("Id do paciente", "")
        if not self._is_filled(patient_id):
            messagebox.showwarning("Aviso", "O paciente selecionado não possui ID preenchido.")
            return

        h5_file = str(paciente.get("h5_file", "")).strip()
        if not self._is_filled(h5_file) or not os.path.exists(h5_file):
            h5_file = self._build_h5_path_from_row(paciente.to_dict())

        if not h5_file or not os.path.exists(h5_file):
            messagebox.showerror("Erro", f"Arquivo H5 do paciente não encontrado:\n{h5_file or 'caminho não calculado'}")
            return

        try:
            datas, ids, _, _ = load_hdf5_packets(h5_file, b"\x02\x0B\x00\x00", 36)
        except Exception as exc:
            messagebox.showerror("Erro", f"Não foi possível ler o H5:\n{exc}")
            return

        signal_info = {
            "sinal_ECG": (Config.ecg_id, "ecg"),
            "sinal_PPG": (Config.spo2_id, "ppg"),
            "sinal_resp": (Config.resp_id, "respiration"),
        }
        patient_folder = self._patient_folder_name(patient_id)
        output_dir = self._patient_data_dir(patient_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(char if char.isalnum() else "_" for char in patient_folder)
        saved_paths = {}
        missing_signals = []

        for column, (signal_id, filename_prefix) in signal_info.items():
            signal_indices = np.where(ids == signal_id)[0]
            if len(signal_indices) == 0:
                missing_signals.append(filename_prefix)
                continue

            signal = np.concatenate([datas[i] for i in signal_indices])
            output_file = output_dir / APP_SETTINGS.reference_filename_template.format(
                signal=filename_prefix,
                patient_id=safe_id,
            )
            np.savetxt(output_file, signal, fmt="%.7e")
            self.df.at[idx, column] = str(output_file)
            saved_paths[column] = str(output_file)

        self.df.at[idx, "h5_file"] = h5_file
        if not saved_paths:
            messagebox.showwarning("Aviso", "Nenhum sinal de referência foi encontrado no H5.")
            return

        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
            self.update_treeview()
            self._select_row_by_idx(idx)
        except Exception as exc:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível atualizar o CSV:\n{exc}")
            return

        message = f"Sinais de referência salvos em:\n{output_dir}"
        if missing_signals:
            message += f"\n\nNão encontrados: {', '.join(missing_signals)}"
        messagebox.showinfo("Sucesso", message)

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for idx, row in self.df.iterrows():
            id_pac = row.get("Id do paciente", "N/A")
            leito = row.get("Leito", "N/A")
            data = row.get("Dia", "N/A")
            hora = row.get("Hora", "N/A")
            reference_paths = [row.get(column) for column in ("sinal_ECG", "sinal_PPG", "sinal_resp")]
            loaded_reference_count = sum(self._is_filled(path) for path in reference_paths)
            if loaded_reference_count == len(reference_paths):
                reference_status = "Carregados"
            elif loaded_reference_count > 0:
                reference_status = "Parcial"
            else:
                reference_status = "Pendente"
            
            # Verifica Sinais
            if pd.notna(row["init_time"]) and str(row["init_time"]).strip() != "":
                status = "Preenchido"
                tag = "preenchido"
            else:
                status = "Pendente"
                tag = "pendente"
                
            # Verifica Vídeo
            video_status = "Cortado" if pd.notna(row.get("video_path")) and str(row.get("video_path")).strip() != "" else "Sem vídeo"
                
            self.tree.insert("", tk.END, values=(idx, id_pac, leito, data, hora, status, reference_status, video_status), tags=(tag,))

    def _select_row_by_idx(self, target_idx):
        """Função auxiliar para restaurar a seleção do paciente na tabela"""
        for item in self.tree.get_children():
            if int(self.tree.item(item, "values")[0]) == target_idx:
                self.tree.selection_set(item)
                self.tree.focus(item)
                self.tree.see(item)
                break

    def _build_video_output_path(self, patient_row=None, start_time=None):
        """Monta o caminho do vídeo cortado usando o horário da planilha."""
        base_dir = Path(APP_SETTINGS.patient_data_root)
        base_dir.mkdir(parents=True, exist_ok=True)

        if patient_row is not None:
            date_str = str(patient_row.get("Dia", "")).strip()
            bed_str = str(patient_row.get("Leito", "")).strip()
            patient_id = patient_row.get("Id do paciente", "")
            sheet_hour = str(patient_row.get("Hora", "") if start_time is None else start_time).strip()

            if date_str and bed_str and self._is_filled(patient_id):
                date_str = date_str.replace("/", "-")
                bed_str = bed_str.replace(" ", "")
                if bed_str.isdigit():
                    bed_str = f"L{int(bed_str):02d}"
                elif not bed_str.upper().startswith("L"):
                    bed_str = f"L{bed_str}"

                hour_str = self._normalize_hour(sheet_hour).replace(":", "-")
                if not hour_str:
                    hour_str = "00-00"
                patient_dir = self._patient_data_dir(patient_id)
                patient_dir.mkdir(parents=True, exist_ok=True)
                return patient_dir / APP_SETTINGS.video_filename_template.format(
                    bed=bed_str,
                    date=date_str,
                    hour=hour_str,
                )

            return base_dir / APP_SETTINGS.default_video_filename

        return base_dir / APP_SETTINGS.default_video_filename

    def cut_video_dialog(self, patient_row=None):
        """Abre janela interativa para navegar pelo vídeo e escolher o frame inicial."""
        input_path = filedialog.askopenfilename(
            title="Selecione o vídeo original (.avi, .mp4)",
            filetypes=[("Video files", "*.avi *.mp4 *.mkv")]
        )
        if not input_path:
            return None, 120.0

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo selecionado.")
            return None, 120.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        current = 0
        start_frame = None
        win = "Cut Video"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        instructions = [
            "Controls: a - prev, d - next, s - select start",
            "w - write cut video, q - quit"
        ]

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            ret, frame = cap.read()
            if not ret:
                print("End of video or read error")
                break

            display = frame.copy()
            lines = [f"Frame: {current + 1}/{total_frames}  FPS:{fps:.2f}"]
            if start_frame is not None:
                lines.append(f"Selected start: {start_frame + 1}")
            else:
                lines.append("Selected start: -")
            lines.extend(instructions)

            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if start_frame is not None and current >= start_frame:
                cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)

            cv2.imshow(win, display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('d'), 83):
                if current < total_frames - 1:
                    current += 1
            elif key in (ord('a'), 81):
                if current > 0:
                    current -= 1
            elif key == ord('s'):
                start_frame = current
                print(f"Selected start frame: {start_frame}")
            elif key == ord('w'):
                if start_frame is None:
                    print("No start selected; using current frame as start")
                    start_frame = current

                output_path = self._build_video_output_path(
                    patient_row=patient_row,
                    start_time=(patient_row.get("Hora") if patient_row is not None else None),
                )
                try:
                    output_path = write_cut_video(input_path, start_frame, str(output_path), codec='I420')
                    duration_seconds = (total_frames - start_frame) / fps if fps else 0.0
                    messagebox.showinfo(
                        "Sucesso",
                        f"Vídeo cortado e salvo com sucesso!\nDuração final: {duration_seconds:.2f} segundos."
                    )
                    cap.release()
                    cv2.destroyAllWindows()
                    return output_path, duration_seconds
                except Exception as exc:
                    messagebox.showerror("Erro", f"Não foi possível cortar o vídeo:\n{exc}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None, 120.0
            elif key == ord('q'):
                break
            else:
                if key != 255:
                    pass

        cap.release()
        cv2.destroyAllWindows()
        return None, 120.0

    def cut_video_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return
            
        item_values = self.tree.item(selected_item[0], "values")
        idx = int(item_values[0])
        paciente = self.df.loc[idx]
        
        # Verifica se já existe vídeo salvo no df
        video_existente = str(paciente.get("video_path", ""))
        
        if video_existente and video_existente != "nan" and os.path.exists(video_existente):
            sobrescrever = messagebox.askyesno("Vídeo já existente", f"Já existe um vídeo cortado para este paciente:\n{video_existente}\n\nDeseja sobrescrever com um novo corte?")
            if not sobrescrever:
                return

        # Roda o Dialog de corte
        res_path, res_dur = self.cut_video_dialog(patient_row=paciente)
        
        # Salva se concluiu
        if res_path:
            self.df.at[idx, "video_path"] = res_path
            self.df.at[idx, "video_duration"] = res_dur
            try:
                self.df.to_csv(self.dataset_raw_file, index=False)
                self.update_treeview()
                
                # Restaura a seleção para que a função de sinais saiba quem usar
                self._select_row_by_idx(idx)
                
                # VERIFICAÇÃO DE SINCRONIA: Se já existiam sinais, sugere reprocessar
                if pd.notna(paciente.get("init_time")) and str(paciente.get("init_time")).strip() != "":
                    reprocessar = messagebox.askyesno(
                        "Sincronia Recomendada", 
                        "Este paciente já possuía sinais extraídos anteriormente.\n\n"
                        f"Como você acabou de salvar um novo vídeo (com duração de {res_dur:.1f}s), "
                        "é altamente recomendável reprocessar os sinais para que a duração deles "
                        "seja perfeitamente igual ao vídeo cortado.\n\n"
                        "Deseja reprocessar os sinais agora?"
                    )
                    
                    if reprocessar:
                        # Chama o processamento de sinais ignorando o aviso comum de sobrescrita, 
                        # pois o usuário acabou de aceitar a sincronia.
                        self.process_signals_for_selected(skip_overwrite_warning=True)
                        
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o CSV:\n{e}")

    def _read_signal_file(self, file_path):
        file_path = str(file_path)
        if not file_path or file_path == "nan" or not os.path.exists(file_path):
            return None

        try:
            data = np.loadtxt(file_path, delimiter=",", ndmin=1)
            if data.size == 0:
                return None
            return np.asarray(data).ravel()
        except Exception:
            try:
                data = np.loadtxt(file_path, ndmin=1)
                if data.size == 0:
                    return None
                return np.asarray(data).ravel()
            except Exception:
                return None

    def _build_saved_interval_mask(self, dates_np, date_str, local_init_time, local_end_time):
        try:
            signal_date = datetime.strptime(date_str, "%d/%m/%Y").date()
            start_dt = datetime.combine(
                signal_date,
                datetime.strptime(local_init_time, "%H:%M:%S").time(),
            )
            end_dt = datetime.combine(
                signal_date,
                datetime.strptime(local_end_time, "%H:%M:%S").time(),
            )
        except (TypeError, ValueError):
            return np.zeros(len(dates_np), dtype=bool)

        return (dates_np >= start_dt) & (dates_np <= end_dt)

    def _load_full_signal_by_id(self, h5_file, signal_id, date_str, local_init_time, local_end_time):
        try:
            datas, ids, _, seqsts = load_hdf5_packets(h5_file, b"\x02\x0B\x00\x00", 36)
        except Exception:
            return None, None, None, None

        signal_indices = np.where(ids == signal_id)[0]
        if len(signal_indices) == 0:
            return None, None, None, None

        sig = np.concatenate([datas[i] for i in signal_indices])
        ts = seqsts[signal_indices]
        fs = len(datas[signal_indices[0]]) / np.median(np.diff(ts))
        time_vector = build_time_vectors(ts, [datas[i] for i in signal_indices], fs)
        dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])

        mask = self._build_saved_interval_mask(
            dates_np,
            date_str,
            local_init_time,
            local_end_time,
        )

        return dates_np, sig, mask, time_vector

    def visualize_saved_signals_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        idx = int(self.tree.item(selected_item[0], "values")[0])
        paciente = self.df.loc[idx]

        h5_file = paciente.get("h5_file", "")
        date_str = str(paciente.get("Dia", "")).strip()
        init_time = str(paciente.get("init_time", "")).strip()
        end_time = str(paciente.get("end_time", "")).strip()

        if not h5_file or str(h5_file) == "nan" or not os.path.exists(str(h5_file)):
            print(h5_file)
            messagebox.showwarning("Aviso", "Arquivo H5 do paciente não encontrado para visualizar os sinais.")
            return

        if not self._is_filled(date_str) or not self._is_filled(init_time) or not self._is_filled(end_time):
            messagebox.showwarning("Aviso", "O paciente ainda não possui intervalo de sinal selecionado.")
            return

        signal_info = {
            "ECG": Config.ecg_id,
            "PPG": Config.spo2_id,
            "Respiração": Config.resp_id,
        }

        fig, axes = plt.subplots(len(signal_info), 1, figsize=(14, 4 * len(signal_info)), sharex=True)
        if len(signal_info) == 1:
            axes = [axes]

        for ax, (name, signal_id) in zip(axes, signal_info.items()):
            dates_np, sig, mask, _ = self._load_full_signal_by_id(
                str(h5_file),
                signal_id,
                date_str,
                init_time,
                end_time,
            )
            if dates_np is None or sig is None:
                continue

            ax.plot(dates_np, sig, color='tab:blue', linewidth=1.5, label='sinal completo')
            if np.any(mask):
                ax.plot(dates_np[mask], sig[mask], color='tab:red', linewidth=2.5, label='intervalo salvo')
            ax.set_title(f"{name} - sinal completo e intervalo salvo")
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Horário')
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    def process_signals_for_selected(self, skip_overwrite_warning=False):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return
            
        item_values = self.tree.item(selected_item[0], "values")
        idx = int(item_values[0])
        status = item_values[5]
        
        # Adicionei a variável "skip_overwrite_warning" para evitar perguntar duas vezes no caso de sincronia
        if status == "Preenchido" and not skip_overwrite_warning:
            if not messagebox.askyesno("Sobrescrever Sinais", "Este paciente já possui dados extraídos. Deseja sobrescrever os dados existentes?"):
                return
                
        paciente = self.df.loc[idx]
        previous_signal_paths = [paciente.get(column, "") for column in ("sinal_ECG", "sinal_PPG", "sinal_resp")]
        h5_file = paciente.get("h5_file", "")
        
        if not h5_file or pd.isna(h5_file) or not os.path.exists(str(h5_file)):
            messagebox.showerror("Erro", f"Arquivo H5 não encontrado:\n{h5_file}")
            return
            
        # Pega a duração salva ou usa 120
        video_dur = paciente.get("video_duration", pd.NA)
        if pd.notna(video_dur) and str(video_dur).strip() != "" and float(video_dur) > 0:
            duration_for_signals = float(video_dur)
            messagebox.showinfo("Duração Sincronizada", f"Sinais serão cortados para {duration_for_signals:.2f}s, acompanhando a duração do vídeo salvo.")
        else:
            duration_for_signals = 120.0
            messagebox.showinfo("Duração Padrão", "Nenhum vídeo cortado encontrado para este paciente. Sinais usarão o tempo padrão de 120s.")

        messagebox.showinfo("Instruções de Sinais", f"A janela do gráfico do ECG vai abrir.\n\n1. Passe o mouse para ver o momento.\n2. Clique para marcar o Início (O final será calculado para {duration_for_signals:.1f}s automaticamente).\n3. Pressione 'y' para confirmar ou 'n' para cancelar.")
        
        patient_output_dir = self._patient_data_dir(paciente.get("Id do paciente", ""))
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        self.root.iconify()
        
        resultados = run_extraction_for_patient(
            h5_file=str(h5_file),
            bed=str(paciente.get("Leito", "")).strip(),
            date_str=str(paciente.get("Dia", "")).strip(),
            duration_seconds=duration_for_signals,
            patient_output_dir=patient_output_dir
        )
        
        self.root.deiconify()
        
        if resultados.get("error"):
            messagebox.showwarning("Extração Interrompida", resultados["error"])
            return
            
        self.df.at[idx, "init_time"] = resultados["init_time"]
        self.df.at[idx, "end_time"] = resultados["end_time"]
        self.df.at[idx, "sinal_ECG"] = resultados["sinal_ECG"]
        self.df.at[idx, "sinal_PPG"] = resultados["sinal_PPG"]
        self.df.at[idx, "sinal_resp"] = resultados["sinal_resp"]
        self._delete_previous_processed_signals(
            previous_signal_paths,
            [resultados["sinal_ECG"], resultados["sinal_PPG"], resultados["sinal_resp"]],
        )
        
        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
            messagebox.showinfo("Sucesso", "Sinais extraídos e salvos na planilha com sucesso!")
            self.update_treeview()
            self._select_row_by_idx(idx) # Restaura seleção visual

            visualizar = messagebox.askyesno(
                "Visualizar sinais salvos",
                "Deseja abrir as janelas com os sinais ECG, PPG e respiração salvos?"
            )
            if visualizar:
                self.visualize_saved_signals_for_selected()
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o CSV:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalExtractorApp(root)
    root.mainloop()