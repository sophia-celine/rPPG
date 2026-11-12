import importlib.util
import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import APP_SETTINGS, Config
from clinical_data import ClinicalDataApp
from signal_processing import (
    build_time_vectors,
    load_hdf5_packets,
    load_hdf5_numeric_packets,
    run_extraction_for_patient,
    write_cut_video,
)

ANONYMIZATION_COLUMNS = ("Anonimização padrão", "Anonimização completa")
CHECKBOX_MARKS = {True: "☑", False: "☐"}
CLINICAL_DATA_COLUMNS = (
    "Diabetes",
    "Hipertensão",
    "Problema cardiovascular",
    "Outras comorbidades",
    "MUC",
    "Alergia",
    "Ultima prescrição",
)


def classify_cid(cid):
    cid = cid.upper().strip()

    if cid.startswith(("A", "B")):
        return "Doença infecciosa"

    elif cid.startswith("I"):
        return "Doença cardiovascular"

    elif cid.startswith("J"):
        return "Doença respiratória"

    elif cid.startswith("G"):
        return "Doença neurológica"
    
    elif cid.startswith("K"):
        return "Doença gastrointestinal"
    
    elif cid.startswith(("S", "T", "V")):
        return "Trauma"

    else:
        return "Outro"


class SignalExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extrator de Sinais - UTI")
        self.root.geometry("1600x950")
        self.root.minsize(1120, 720)

        self.dataset_raw_file = APP_SETTINGS.dataset_raw_file
        self.df = None
        self.video_batch_messages = queue.Queue()
        self.video_batch_running = False
        self.video_batch_errors = []
        self.video_batch_results = []
        self.last_video_processing_duration = None
        self.pending_video_patients = None
        self.action_buttons = []
        self.clinical_data_window = None

        self._configure_ui_style()
        self.root.configure(bg="#10131a")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)

        header_frame = ttk.Frame(self.root, style="App.TFrame", padding=(24, 20, 24, 14))
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(
            header_frame,
            text="Extrator de Sinais",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header_frame,
            text="Selecione um paciente para processar sinais ou cortar vídeo.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        columns = (
            "index",
            "id",
            "leito",
            "data",
            "hora",
            "status",
            "video_status",
            "anon_padrao",
            "anon_completa",
        )
        table_frame = ttk.Frame(self.root, style="Panel.TFrame", padding=12)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=8)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended"
        )
        self.tree.heading("index", text="Índice")
        self.tree.heading("id", text="ID Paciente")
        self.tree.heading("leito", text="Leito")
        self.tree.heading("data", text="Data")
        self.tree.heading("hora", text="Hora")
        self.tree.heading("status", text="Status (Sinais)")
        self.tree.heading("video_status", text="Vídeo")
        self.tree.heading("anon_padrao", text="Anonimização padrão")
        self.tree.heading("anon_completa", text="Anonimização completa")

        self.tree.column("index", width=50, anchor="center")
        self.tree.column("id", width=120, anchor="center")
        self.tree.column("leito", width=80, anchor="center")
        self.tree.column("data", width=100, anchor="center")
        self.tree.column("hora", width=80, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.column("video_status", width=120, anchor="center")
        self.tree.column("anon_padrao", width=190, anchor="center")
        self.tree.column("anon_completa", width=210, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<ButtonRelease-1>", self._toggle_anonymization_from_click)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree.tag_configure("pendente", foreground="#f08a8a")
        self.tree.tag_configure("preenchido", foreground="#62d6a5")

        btn_frame = ttk.Frame(self.root, style="App.TFrame", padding=(18, 12, 18, 20))
        btn_frame.grid(row=2, column=0, sticky="ew")

        btn_sync_remote = ttk.Button(btn_frame, text="Sincronizar Planilha", command=self.sync_local_dataset_with_remote, style="Purple.TButton", width=20)
        btn_sync_remote.pack(side=tk.LEFT, padx=6)

        btn_cut_video = ttk.Button(btn_frame, text="Cortar Vídeo", command=self.cut_video_for_selected, style="Secondary.TButton", width=15)
        btn_cut_video.pack(side=tk.LEFT, padx=6)

        btn_process_signals = ttk.Button(btn_frame, text="Processar Sinais", command=self.process_signals_for_selected, style="Success.TButton", width=18)
        btn_process_signals.pack(side=tk.LEFT, padx=6)

        btn_visualize_signals = ttk.Button(btn_frame, text="Visualizar Sinais", command=self.visualize_saved_signals_for_selected, style="Secondary.TButton", width=18)
        btn_visualize_signals.pack(side=tk.LEFT, padx=6)

        btn_add_signals = ttk.Button(btn_frame, text="Adicionar sinais", command=self.add_additional_signals_for_selected, style="Secondary.TButton", width=18)
        btn_add_signals.pack(side=tk.LEFT, padx=6)

        btn_load_clinical = ttk.Button(btn_frame, text="Carregar dados clínicos", command=self.load_clinical_data_for_selected, style="Secondary.TButton", width=22)
        btn_load_clinical.pack(side=tk.LEFT, padx=6)

        btn_clinical_data = ttk.Button(
            btn_frame,
            text="Preencher dados clínicos",
            command=self.open_clinical_data_window,
            style="Secondary.TButton",
            width=20,
        )
        btn_clinical_data.pack(side=tk.LEFT, padx=6)

        btn_analyze_dataset = ttk.Button(btn_frame, text="Analisar Planilha", command=self.show_dataset_analysis, style="Accent.TButton", width=18)
        btn_analyze_dataset.pack(side=tk.LEFT, padx=6)

        btn_process_videos = ttk.Button(
            btn_frame,
            text="Processar vídeos",
            command=self.process_selected_videos,
            style="Process.TButton",
            width=18,
        )
        btn_process_videos.pack(side=tk.LEFT, padx=6)

        btn_start_video_processing = ttk.Button(
            btn_frame,
            text="Começar",
            command=self.start_video_batch,
            style="Success.TButton",
            width=12,
            state="disabled",
        )
        btn_start_video_processing.pack(side=tk.LEFT, padx=6)

        self.action_buttons = [
            btn_sync_remote,
            btn_cut_video,
            btn_process_signals,
            btn_visualize_signals,
            btn_add_signals,
            btn_load_clinical,
            btn_analyze_dataset,
        ]
        self.process_videos_button = btn_process_videos
        self.start_video_processing_button = btn_start_video_processing
        self.clinical_data_button = btn_clinical_data

        batch_frame = ttk.Frame(self.root, style="Panel.TFrame", padding=(14, 9))
        batch_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 14))
        batch_frame.columnconfigure(0, weight=1)
        self.video_batch_status = ttk.Label(
            batch_frame,
            text="Selecione os pacientes, clique em “Processar vídeos” e depois em “Começar”.",
            style="Panel.TLabel",
        )
        self.video_batch_status.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.video_batch_progress = ttk.Progressbar(
            batch_frame,
            mode="determinate",
            maximum=1,
            value=0,
            style="Horizontal.TProgressbar",
        )
        self.video_batch_progress.grid(row=1, column=0, sticky="ew")

        self.load_data()

    def _configure_ui_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#10131a")
        style.configure("Panel.TFrame", background="#191e28")
        style.configure("Title.TLabel", background="#10131a", foreground="#eef2f8", font=("Segoe UI Semibold", 21))
        style.configure("Subtitle.TLabel", background="#10131a", foreground="#a3adbd", font=("Segoe UI", 10))
        style.configure(
            "Treeview",
            background="#191e28",
            fieldbackground="#191e28",
            foreground="#e7ebf2",
            rowheight=32,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#222a37",
            foreground="#cbd4e2",
            relief="flat",
            padding=(10, 10),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", "#394d82")],
            foreground=[("selected", "#ffffff")],
        )
        style.map("Treeview.Heading", background=[("active", "#303a4b")])
        style.configure(
            "Accent.TButton",
            background="#6c8cff",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Accent.TButton", background=[("active", "#829cff")])
        style.configure(
            "Secondary.TButton",
            background="#222a37",
            foreground="#e7ebf2",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Secondary.TButton", background=[("active", "#303a4b")])
        style.configure(
            "Success.TButton",
            background="#21875f",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Success.TButton", background=[("active", "#2ba675")])
        style.configure(
            "Purple.TButton",
            background="#6147a8",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Purple.TButton", background=[("active", "#795fbd")])
        style.configure(
            "Process.TButton",
            background="#287f91",
            foreground="#ffffff",
            padding=(12, 10),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map("Process.TButton", background=[("active", "#3298aa")])
        style.configure(
            "Panel.TLabel",
            background="#191e28",
            foreground="#a3adbd",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Horizontal.TProgressbar",
            background="#44c08a",
            troughcolor="#303747",
            borderwidth=0,
        )

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
            return df_norm
        except Exception:
            return df_norm

    def _normalize_remote_sheet(self, df):
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
        if len(normalized.split('.')) > 1 and normalized[:-2].isdigit():
            normalized = normalized[:-2]
        if len(normalized) < 7:
            normalized = (7-len(normalized)) * "0" + normalized
        return normalized

    @staticmethod
    def _patient_folder_name(value):
        patient_id = str(value).strip().lower()
        if "." in patient_id:
            base_id, index = patient_id.rsplit(".", 1)
            if base_id and index.isdigit():
                patient_id = f"{base_id}_{index}"
        return "".join(char if char.isalnum() or char == "_" else "_" for char in patient_id)

    @staticmethod
    def _extract_patient_id_from_evolution_line(line):
        if line is None:
            return None

        match = re.search(r"PACIENTE\s*:\s*([^\n\r]+)", line, flags=re.IGNORECASE)
        if match is None:
            return None

        candidate = match.group(1).strip()
        cleaned = re.sub(r"[^0-9]", "", candidate)
        if not cleaned:
            return None

        return cleaned[3:10] if len(cleaned) > 10 else cleaned

    def _patient_data_dir(self, patient_id):
        return Path(APP_SETTINGS.patient_data_root) / self._patient_folder_name(patient_id)

    @staticmethod
    def _extract_patient_age_from_evolution(line):
        if line is None:
            return None

        complete_age = line.split('|')[-1].strip()
        age_in_years = complete_age.split('a')[0]

        return int(age_in_years) if age_in_years.isdigit() else None

    @staticmethod
    def _extract_cid_from_evolution(line):
        if line is None:
            return None

        cid = line.split(':')[-1].strip()

        return cid if cid else None


    def _delete_previous_processed_signals(self, previous_paths, current_paths):
        current_paths = {os.path.normcase(os.path.abspath(str(path))) for path in current_paths if self._is_filled(path)}
        for path in previous_paths:
            if not self._is_filled(path):
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

    def _get_absolute_path(self, row, column_name):
        filename = str(row.get(column_name, "")).strip()

        if not self._is_filled(filename) or filename == "nan":
            return ""

        if os.path.isabs(filename):
            filename = filename.split("/")[-1]

        if column_name == "h5_file":
            date = str(row.get("Dia", "")).strip()
            if not date:
                return ""
            date_parts = [p.strip() for p in date.replace("-", "/").split("/")]
            if len(date_parts) == 3:
                day, month, year = date_parts
                day_folder = f"{year}{month}{day}"
                return f"{APP_SETTINGS.uti_data_path}/{day_folder}/{filename}"
            return ""

        patient_id = row.get("Id do paciente", "")
        if self._is_filled(patient_id):
            return str(self._patient_data_dir(patient_id) / filename)
        return ""

    def _resolve_h5_path(self, row, row_index):
        h5_file = self._get_absolute_path(row, "h5_file")
        if self._is_filled(row.get("h5_file", "")):
            return h5_file

        built_h5_file = self._build_h5_path_from_row(row.to_dict())
        if not built_h5_file:
            return ""
        self.df.at[row_index, "h5_file"] = Path(built_h5_file).name
        self.df.to_csv(self.dataset_raw_file, index=False)
        return built_h5_file

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
            "sinal_P1", "sinal_PVC", "sinal_ART", "sinal_CO2", "sinal_FC", "sinal_SpO2",
            "sinal_FR_impedancia", "sinal_FR_CO2",
            "video_path", "video_duration", "init_frame", "end_frame", "h5_file", "evolucoes_path",
            "Movement index", "Detection rate",
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
                existing_h5_name = str(base_df.at[target_idx, "h5_file"]).strip() if pd.notna(base_df.at[target_idx, "h5_file"]) else ""
                existing_h5 = self._get_absolute_path(base_df.loc[target_idx], "h5_file") if existing_h5_name else ""
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

        self._ensure_anonymization_columns(base_df)
        self._ensure_clinical_columns(base_df)
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
                    "sinal_resp", "sinal_P1", "sinal_PVC", "sinal_ART", "sinal_CO2",
                    "sinal_FC", "sinal_SpO2", "sinal_FR_impedancia", "sinal_FR_CO2",
                    "video_path", "video_duration", "init_frame", "end_frame", "evolucoes_path",
                    "Movement index", "Detection rate",
                ]
                for col in novas_colunas:
                    self.df[col] = pd.NA
                self._ensure_anonymization_columns(self.df)
                self._ensure_clinical_columns(self.df)
                self.df.to_csv(self.dataset_raw_file, index=False)
                self.update_treeview()
                messagebox.showwarning(
                    "Planilha vazia",
                    "A planilha local está vazia. Clique em 'Sincronizar Planilha' para recuperá-la da planilha remota.",
                )
                return
            self.df = pd.read_csv(self.dataset_raw_file)

            novas_colunas = [
                "init_time", "end_time", "sinal_ECG", "sinal_PPG", "sinal_resp",
                "sinal_P1", "sinal_PVC", "sinal_ART", "sinal_CO2", "sinal_FC", "sinal_SpO2",
                "sinal_FR_impedancia", "sinal_FR_CO2",
                "video_path", "video_duration", "init_frame", "end_frame", "evolucoes_path",
                "Movement index", "Detection rate",
            ]
            for col in novas_colunas:
                if col not in self.df.columns:
                    self.df[col] = pd.NA

            anonymization_columns_added = self._ensure_anonymization_columns(self.df)
            clinical_columns_added = self._ensure_clinical_columns(self.df)
            if anonymization_columns_added or clinical_columns_added:
                self.df.to_csv(self.dataset_raw_file, index=False)
            self.update_treeview()
        except Exception as e:
            messagebox.showerror("Erro de Leitura", f"Erro ao ler o CSV:\n{e}\n{self.dataset_raw_file}")

    @staticmethod
    def _parse_boolean_value(value):
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return bool(value)

        normalized = str(value).strip().lower()
        if normalized in ("true", "1", "sim", "verdadeiro", "yes"):
            return True
        if normalized in ("false", "0", "não", "nao", "falso", "no", ""):
            return False
        raise ValueError(f"Valor booleano inválido na planilha: {value!r}")

    @classmethod
    def _ensure_anonymization_columns(cls, dataframe):
        changed = False
        for column in ANONYMIZATION_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = False
                changed = True
            else:
                normalized = dataframe[column].map(cls._parse_boolean_value)
                if not dataframe[column].equals(normalized):
                    changed = True
                dataframe[column] = normalized.astype(bool)
        return changed

    @staticmethod
    def _ensure_clinical_columns(dataframe):
        changed = False
        for column in CLINICAL_DATA_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA
                changed = True
        return changed

    def _toggle_anonymization_from_click(self, event):
        item = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        column_by_id = {
            "#8": (ANONYMIZATION_COLUMNS[0], "anon_padrao"),
            "#9": (ANONYMIZATION_COLUMNS[1], "anon_completa"),
        }
        target = column_by_id.get(column_id)
        if not item or target is None or self.df is None:
            return

        column, tree_column = target
        row_index = int(self.tree.item(item, "values")[0])
        previous_value = bool(self.df.at[row_index, column])
        self.df.at[row_index, column] = not previous_value
        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
        except (OSError, PermissionError, ValueError) as exc:
            self.df.at[row_index, column] = previous_value
            self.update_treeview()
            messagebox.showerror(
                "Erro ao salvar",
                f"Não foi possível atualizar a planilha:\n{exc}",
                parent=self.root,
            )
            return

        self.tree.selection_set(item)
        self.tree.focus(item)
        self.tree.set(item, tree_column, CHECKBOX_MARKS[not previous_value])

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
        cid_column = "CID"
        signal_columns = {
            "ECG": "ECG",
            "PPG": "PPG",
            "Sinal respiratório": "Sinal respiratório",
        }
        patient_id_column = "Id do paciente"
        age_column = "Idade"
        report_lines = [
            "ANÁLISE DA PLANILHA LOCAL",
            f"Arquivo: {self.dataset_raw_file}",
            f"Registros analisados: {len(self.df)}",
            "",
        ]

        figure = Figure(figsize=(16, 14), dpi=100)
        figure.patch.set_facecolor("#10131a")
        figure.suptitle(
            "Panorama dos dados da planilha",
            fontsize=18,
            fontweight="bold",
            color="#eef2f8",
        )
        axes = figure.subplots(4, 4).ravel()
        chart_colors = [
            "#6c8cff",
            "#44c08a",
            "#f0a35e",
            "#d879d8",
            "#4fc3c7",
            "#e16b76",
            "#9d8df1",
            "#e7c96a",
        ]
        for axis in axes:
            axis.set_facecolor("#191e28")
            axis.set_prop_cycle(color=chart_colors)
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
            distance_axis.bar(
                ["Mínimo", "Média", "Máximo"],
                distance_summary,
                color=["#6c8cff", "#44c08a", "#e16b76"],
            )
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
                        color=["#6c8cff", "#44c08a", "#e16b76"],
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
                    wedgeprops={"edgecolor": "#191e28", "linewidth": 1.5},
                    textprops={"fontsize": 8, "color": "#d9e0eb"},
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

        age_axis = axes[8]
        report_lines.append("IDADE DOS PACIENTES")
        if age_column not in self.df.columns or patient_id_column not in self.df.columns:
            age_axis.set_title("Média de idade")
            age_axis.text(0.5, 0.5, "Coluna não encontrada", ha="center", va="center")
            report_lines.append("  Coluna 'Idade' ou 'Id do paciente' não encontrada.")
        else:
            age_data = self.df[[patient_id_column, age_column]].copy()
            age_data = age_data[age_data[patient_id_column].apply(self._is_filled)]
            age_data["_patient_id"] = (
                age_data[patient_id_column]
                .astype(str)
                .str.strip()
                .str.split(".", n=1)
                .str[0]
            )
            age_data[age_column] = pd.to_numeric(
                age_data[age_column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            unique_ages = age_data.groupby("_patient_id")[age_column].first().dropna()
            if unique_ages.empty:
                age_axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
                report_lines.append("  Nenhuma idade válida disponível.")
            else:
                average_age = unique_ages.mean()
                max_age = unique_ages.max()
                min_age = unique_ages.min()
                age_values = [min_age, average_age, max_age]
                age_labels = ["Mínima", "Média", "Máxima"]
                age_axis.bar(
                    range(len(age_values)),
                    age_values,
                    color=["#59A14F", "#A18E4F", "#4F62A1"],
                    width=0.6,
                )
                age_axis.set_xticks(range(len(age_labels)), age_labels)
                age_axis.set_ylabel("anos")
                age_axis.set_ylim(0, max(max_age * 1.25, 1))
                for position, value in enumerate(age_values):
                    age_axis.text(position, value, f"{value:.2f}", ha="center", va="bottom")
                age_axis.grid(axis="y", alpha=0.3)
                report_lines.append(f"  Pacientes únicos com idade válida: {len(unique_ages)}")
                report_lines.append(f"  Média: {average_age:.2f} anos")
                report_lines.append(f"  Máxima: {max_age:.2f} anos")
                report_lines.append(f"  Mínima: {min_age:.2f} anos")
            age_axis.set_title("Média de idade")
        report_lines.append("")

        cid_axis = axes[9]
        report_lines.append("GRUPO DE ADMISSÃO (CID)")
        if cid_column not in self.df.columns:
            cid_axis.set_title("Grupo de admissão (CID)")
            cid_axis.text(0.5, 0.5, "Coluna não encontrada", ha="center", va="center")
            report_lines.append("  Coluna não encontrada.")
        else:
            cid_data = self.df[[patient_id_column, cid_column]].copy()
            cid_data = cid_data[cid_data[patient_id_column].apply(self._is_filled)]
            cid_data["_patient_id"] = (
                cid_data[patient_id_column]
                .astype(str)
                .str.strip()
                .str.split(".", n=1)
                .str[0]
            )
            classified_cids = (
                cid_data[cid_column]
                .where(cid_data[cid_column].apply(self._is_filled))
                .astype("string")
                .str.split("-", n=1)
                .str[0]
                .map(classify_cid, na_action="ignore")
            )
            classified_cids.index = cid_data["_patient_id"]
            classified_cids = classified_cids.groupby(level=0).first()

            admission_group_column = "Grupo de admissão"
            if admission_group_column not in self.df.columns:
                self.df[admission_group_column] = pd.NA
            patient_ids = (
                self.df[patient_id_column]
                .astype(str)
                .str.strip()
                .str.split(".", n=1)
                .str[0]
            )
            classified_for_rows = patient_ids.map(classified_cids)
            rows_to_fill = (
                ~self.df[admission_group_column].apply(self._is_filled)
                & classified_for_rows.notna()
            )
            if rows_to_fill.any():
                self.df.loc[rows_to_fill, admission_group_column] = classified_for_rows.loc[rows_to_fill]
                try:
                    self.df.to_csv(self.dataset_raw_file, index=False)
                except Exception as e:
                    messagebox.showerror(
                        "Erro ao Salvar",
                        f"Não foi possível salvar os grupos de admissão no CSV:\n{e}",
                    )

            distribution, valid_count, missing_count = self._analysis_distribution(classified_cids)
            if distribution:
                labels = [item[0] for item in distribution]
                percentages = [item[2] for item in distribution]
                cid_axis.pie(
                    percentages,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"edgecolor": "#191e28", "linewidth": 1.5},
                    textprops={"fontsize": 8, "color": "#d9e0eb"},
                )
            else:
                cid_axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            cid_axis.set_title("Grupo de admissão (CID)")
            report_lines.append(f"  Pacientes únicos: {valid_count} | Ausentes: {missing_count}")
            for value, count, percentage in distribution:
                report_lines.append(f"  {value}: {count} ({percentage:.2f}%)")
        report_lines.append("")

        report_lines.append("CAPTURA DOS SINAIS")
        for signal_index, (label, column) in enumerate(signal_columns.items(), start=10):
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
                    colors=["#44c08a", "#e16b76"],
                    wedgeprops={"edgecolor": "#191e28", "linewidth": 1.5},
                    textprops={"fontsize": 8, "color": "#d9e0eb"},
                )
                axes[signal_index].set_title(label)
                report_lines.append(f"  Capturado (Verdadeiro): {captured} ({captured / total * 100:.2f}%)")
                report_lines.append(f"  Não capturado (Falso): {not_captured} ({not_captured / total * 100:.2f}%)")
                report_lines.append(f"  Ausentes ou inválidos: {missing}")

        oxygen_axis = axes[13]
        oxygen_column = "Oxigenoterapia"
        report_lines.append("OXIGENOTERAPIA")
        if oxygen_column not in self.df.columns:
            oxygen_axis.set_title(oxygen_column)
            oxygen_axis.text(0.5, 0.5, "Coluna não encontrada", ha="center", va="center")
            report_lines.append("  Coluna não encontrada.")
        else:
            distribution, valid_count, missing_count = self._analysis_distribution(self.df[oxygen_column])
            if distribution:
                labels = [item[0] for item in distribution]
                percentages = [item[2] for item in distribution]
                oxygen_axis.pie(
                    percentages,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"edgecolor": "#191e28", "linewidth": 1.5},
                    textprops={"fontsize": 8, "color": "#d9e0eb"},
                )
            else:
                oxygen_axis.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            oxygen_axis.set_title(oxygen_column)
            report_lines.append(f"  Válidos: {valid_count} | Ausentes: {missing_count}")
            if not distribution:
                report_lines.append("  Nenhum valor disponível.")
            else:
                for value, count, percentage in distribution:
                    report_lines.append(f"  {value}: {count} ({percentage:.2f}%)")
        report_lines.append("")

        for axis in axes:
            axis.title.set_color("#eef2f8")
            axis.title.set_fontsize(10)
            axis.tick_params(colors="#a3adbd", labelsize=8)
            axis.yaxis.label.set_color("#a3adbd")
            axis.xaxis.label.set_color("#a3adbd")
            axis.grid(color="#384152", alpha=0.45)
            for spine in axis.spines.values():
                spine.set_color("#384152")
            for chart_text in axis.texts:
                chart_text.set_color("#d9e0eb")
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
        window.geometry("1440x1000")
        window.minsize(1000, 700)
        window.configure(bg="#10131a")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=4)
        window.rowconfigure(2, weight=1)

        header = ttk.Frame(window, style="App.TFrame", padding=(22, 18, 22, 14))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Panorama, distribuição e qualidade dos dados locais.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        if figure is not None:
            chart_frame = ttk.Frame(window, style="Panel.TFrame", padding=10)
            chart_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
            chart_frame.columnconfigure(0, weight=1)
            chart_frame.rowconfigure(0, weight=1)
            canvas = FigureCanvasTkAgg(figure, master=chart_frame)
            canvas.draw()
            chart_widget = canvas.get_tk_widget()
            chart_widget.configure(
                background="#10131a",
                highlightthickness=0,
                borderwidth=0,
            )
            chart_widget.grid(row=0, column=0, sticky="nsew")

        report_frame = ttk.Frame(window, style="Panel.TFrame", padding=10)
        report_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 18))
        report_frame.columnconfigure(0, weight=1)
        report_frame.rowconfigure(0, weight=1)
        text = tk.Text(
            report_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            background="#191e28",
            foreground="#d9e0eb",
            insertbackground="#eef2f8",
            selectbackground="#394d82",
            relief=tk.FLAT,
            borderwidth=0,
            padx=14,
            pady=12,
            spacing1=2,
            spacing3=2,
        )
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)

    def open_clinical_data_window(self):
        if self.df is None:
            messagebox.showwarning(
                "Dados clínicos", "A planilha local ainda não foi carregada."
            )
            return

        if (
            self.clinical_data_window is not None
            and self.clinical_data_window.winfo_exists()
        ):
            self.clinical_data_window.lift()
            self.clinical_data_window.focus_set()
            return

        window = tk.Toplevel(self.root)
        self.clinical_data_window = window
        self.clinical_data_app = ClinicalDataApp(
            window,
            dataframe=self.df,
            save_callback=self._save_clinical_data,
        )
        if window.winfo_exists():
            window.protocol("WM_DELETE_WINDOW", self._close_clinical_data_window)

    def _close_clinical_data_window(self):
        if (
            self.clinical_data_window is not None
            and self.clinical_data_window.winfo_exists()
        ):
            self.clinical_data_window.destroy()
        self.clinical_data_window = None
        self.clinical_data_app = None

    def _save_clinical_data(self, row_index, values):
        previous_values = {
            column: self.df.at[row_index, column] for column in values
        }
        for column, value in values.items():
            self.df.at[row_index, column] = value

        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
        except (OSError, PermissionError, ValueError):
            for column, value in previous_values.items():
                self.df.at[row_index, column] = value
            raise

    def load_clinical_data_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        idx = int(self.tree.item(selected_item[0], "values")[0])
        paciente = self.df.loc[idx]
        patient_id = self._normalize_patient_id(paciente.get("Id do paciente", ""))
        patient_id = patient_id.split('.')[0]
        if not self._is_filled(patient_id):
            messagebox.showwarning("Aviso", "O paciente selecionado não possui ID preenchido.")
            return

        dia = str(paciente.get("Dia", "")).strip()
        if not dia or "/" not in dia:
            messagebox.showerror("Erro", "A linha selecionada não possui a data no formato esperado para buscar as evoluções.")
            return

        try:
            day, month, year = [part.strip() for part in dia.split("/")]
            folder_name = f"{year}{month}{day}"
        except ValueError:
            messagebox.showerror("Erro", f"Data inválida para busca de evolução: {dia}")
            return

        evolution_root = Path(APP_SETTINGS.evolucoes_path) / folder_name
        found_file = None
        registered_evolution = paciente.get("evolucoes_path", "")
        if self._is_filled(registered_evolution):
            registered_path = Path(str(registered_evolution).strip())
            found_file = registered_path if registered_path.is_absolute() else evolution_root / registered_path
            if not found_file.is_file():
                messagebox.showerror("Erro", f"Arquivo de evolução registrado não encontrado:\n{found_file}")
                return
        else:
            if not evolution_root.exists():
                messagebox.showerror("Erro", f"Diretório de evoluções não encontrado:\n{evolution_root}")
                return

            txt_files = sorted(evolution_root.glob("*.txt"))
            if not txt_files:
                messagebox.showerror("Erro", f"Nenhum arquivo .txt encontrado em:\n{evolution_root}")
                return

            for txt_file in txt_files:
                try:
                    with open(txt_file, "r", encoding="utf-8", errors="ignore") as handle:
                        lines = handle.readlines()
                except Exception:
                    continue

                if len(lines) < 2:
                    continue

                extracted_id = self._extract_patient_id_from_evolution_line(lines[1])
                if extracted_id is None:
                    continue

                if self._normalize_patient_id(extracted_id) == patient_id:
                    found_file = txt_file
                    break

        if found_file is None:
            messagebox.showerror("Erro", f"Nenhum arquivo de evolução foi encontrado para o paciente {patient_id} na pasta {folder_name}.")
            return

        try:
            with open(found_file, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
        except Exception as exc:
            messagebox.showerror("Erro", f"Não foi possível ler o arquivo de evolução:\n{found_file}\n\n{exc}")
            return

        if len(lines) <= 9:
            messagebox.showerror("Erro", f"O arquivo de evolução não possui dados clínicos suficientes:\n{found_file}")
            return

        self.df.at[idx, "evolucoes_path"] = found_file.name

        self.df.at[idx, "Idade"] = self._extract_patient_age_from_evolution(lines[1])
        self.df.at[idx, "CID"] = self._extract_cid_from_evolution(lines[9])

        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
            self.update_treeview()
            self._select_row_by_idx(idx)
            messagebox.showinfo("Sucesso", f"Arquivo de evolução encontrado e salvo na planilha:\n{found_file.name}\nInformações clínicas atualizadas\n")

        except Exception as exc:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível atualizar o CSV:\n{exc}")

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, row in self.df.iterrows():
            id_pac = row.get("Id do paciente", "N/A")
            leito = row.get("Leito", "N/A")
            data = row.get("Dia", "N/A")
            hora = row.get("Hora", "N/A")
            if pd.notna(row["init_time"]) and str(row["init_time"]).strip() != "":
                status = "Preenchido"
                tag = "preenchido"
            else:
                status = "Pendente"
                tag = "pendente"

            video_status = "Cortado" if pd.notna(row.get("video_path")) and str(row.get("video_path")).strip() != "" else "Sem vídeo"
            values = (
                idx,
                id_pac,
                leito,
                data,
                hora,
                status,
                video_status,
                CHECKBOX_MARKS[bool(row[ANONYMIZATION_COLUMNS[0]])],
                CHECKBOX_MARKS[bool(row[ANONYMIZATION_COLUMNS[1]])],
            )
            self.tree.insert("", tk.END, values=values, tags=(tag,))

    def _select_row_by_idx(self, target_idx):
        for item in self.tree.get_children():
            if int(self.tree.item(item, "values")[0]) == target_idx:
                self.tree.selection_set(item)
                self.tree.focus(item)
                self.tree.see(item)
                break

    def process_selected_videos(self):
        if self.video_batch_running:
            return
        if self.df is None or self.df.empty:
            messagebox.showwarning("Processar vídeos", "A planilha local ainda não foi carregada.")
            return

        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning(
                "Processar vídeos", "Selecione um ou mais pacientes na tabela."
            )
            return

        patients = []
        for item in selected_items:
            values = self.tree.item(item, "values")
            row_index = int(values[0])
            patient = self.df.loc[row_index].copy()
            patient_id = str(patient.get("Id do paciente", "")).strip()
            patient_label = patient_id or f"linha {row_index}"
            video_value = patient.get("video_path", "")

            if not self._is_filled(video_value):
                video_path = None
                validation_error = "A coluna video_path está vazia."
            else:
                video_filename = Path(str(video_value).strip().replace("\\", "/")).name
                if not video_filename:
                    video_path = None
                    validation_error = "O nome do vídeo em video_path está vazio."
                elif not self._is_filled(patient.get("Id do paciente", "")):
                    video_path = None
                    validation_error = "O ID do paciente está vazio."
                else:
                    video_path = self._patient_data_dir(patient_id) / video_filename
                    validation_error = (
                        "" if video_path.is_file()
                        else f"Vídeo original não encontrado: {video_path}"
                    )

            patients.append(
                (row_index, patient_label, video_path, validation_error)
            )

        self.pending_video_patients = patients
        self.video_batch_progress.configure(maximum=len(patients), value=0)
        self.video_batch_status.configure(
            text=(
                f"{len(patients)} paciente(s) selecionado(s). "
                "Clique em “Começar” para iniciar o processamento."
            )
        )
        self.start_video_processing_button.configure(state="normal")

    def start_video_batch(self):
        if self.video_batch_running:
            return
        if not self.pending_video_patients:
            messagebox.showwarning(
                "Processar vídeos",
                "Selecione os pacientes e clique em “Processar vídeos” antes de começar.",
            )
            return

        selected_row_indices = {
            int(self.tree.item(item, "values")[0])
            for item in self.tree.selection()
        }
        pending_row_indices = {
            patient[0] for patient in self.pending_video_patients
        }
        if selected_row_indices != pending_row_indices:
            self.pending_video_patients = None
            self.start_video_processing_button.configure(state="disabled")
            self.video_batch_status.configure(
                text="A seleção mudou. Clique em “Processar vídeos” para preparar os pacientes novamente."
            )
            messagebox.showwarning(
                "Seleção alterada",
                "A seleção de pacientes mudou. Clique em “Processar vídeos” novamente antes de começar.",
            )
            return

        patients = self.pending_video_patients
        new_metric_columns = [
            column
            for column in ("Movement index", "Detection rate")
            if column not in self.df.columns
        ]
        for column in new_metric_columns:
            self.df[column] = pd.NA
        if new_metric_columns:
            try:
                self.df.to_csv(self.dataset_raw_file, index=False)
            except Exception as exc:
                self.df.drop(columns=new_metric_columns, inplace=True)
                messagebox.showerror(
                    "Erro ao Salvar",
                    f"Não foi possível criar as colunas de métricas na planilha:\n{exc}",
                )
                return

        self.video_batch_running = True
        self.pending_video_patients = None
        self.video_batch_errors = []
        self.video_batch_results = []
        self.last_video_processing_duration = None
        total = len(patients)
        self.video_batch_progress.configure(maximum=total, value=0)
        self.video_batch_status.configure(
            text=f"Preparando processamento de {total} paciente(s)..."
        )
        self.process_videos_button.configure(state="disabled")
        self.start_video_processing_button.configure(state="disabled")
        for button in self.action_buttons:
            button.configure(state="disabled")
        self.tree.state(["disabled"])

        threading.Thread(
            target=self._process_videos_worker,
            args=(patients,),
            daemon=True,
        ).start()
        self.root.after(100, self._poll_video_batch_messages)

    @staticmethod
    def _load_process_video_function():
        process_video_path = (
            Path(__file__).resolve().parents[2]
            / "face_detection"
            / "process_video_optimized.py"
        )
        if not process_video_path.is_file():
            raise FileNotFoundError(f"Arquivo de processamento não encontrado: {process_video_path}")

        spec = importlib.util.spec_from_file_location(
            "_rppg_face_detection_process_video",
            process_video_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Não foi possível carregar o módulo: {process_video_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.MODEL_DIR = process_video_path.parent / "mediapipe_models"
        module.FACE_MODEL_PATH = module.MODEL_DIR / "face_landmarker.task"
        return module.process_video

    def _process_videos_worker(self, patients):
        self.video_batch_messages.put(("status", "Carregando o processador de vídeos..."))
        try:
            process_video = self._load_process_video_function()
        except Exception as exc:
            self.video_batch_messages.put(("fatal_error", str(exc)))
            self.video_batch_messages.put(("finished", None))
            return

        total = len(patients)
        for position, (row_index, patient_label, video_path, validation_error) in enumerate(
            patients, start=1
        ):
            self.video_batch_messages.put(
                ("patient_started", (position, total, patient_label))
            )
            if validation_error:
                self.video_batch_messages.put(
                    (
                        "patient_finished",
                        (position, row_index, patient_label, None, validation_error, None),
                    )
                )
                continue

            started_at = time.perf_counter()
            try:
                result = process_video(video_path, show=False, ids_csv_path=APP_SETTINGS.ids_file)
                elapsed_seconds = time.perf_counter() - started_at
                if result is None:
                    error = "Nenhum rosto detectado no primeiro frame; não foram retornadas métricas."
                    self.video_batch_messages.put(
                        (
                            "patient_finished",
                            (position, row_index, patient_label, None, error, elapsed_seconds),
                        )
                    )
                    continue
                movement_index, detection_rate = result
                self.video_batch_messages.put(
                    (
                        "patient_finished",
                        (
                            position,
                            row_index,
                            patient_label,
                            (movement_index, detection_rate),
                            "",
                            elapsed_seconds,
                        ),
                    )
                )
            except (Exception, SystemExit) as exc:
                elapsed_seconds = time.perf_counter() - started_at
                self.video_batch_messages.put(
                    (
                        "patient_finished",
                        (position, row_index, patient_label, None, str(exc), elapsed_seconds),
                    )
                )

        self.video_batch_messages.put(("finished", None))

    def _poll_video_batch_messages(self):
        try:
            while True:
                kind, payload = self.video_batch_messages.get_nowait()
                if kind == "status":
                    self.video_batch_status.configure(text=payload)
                elif kind == "patient_started":
                    position, total, patient_label = payload
                    self.video_batch_progress.configure(value=position - 1)
                    self.video_batch_status.configure(
                        text=f"Vídeo {position}/{total} sendo processado: {patient_label}"
                    )
                elif kind == "patient_finished":
                    (
                        position,
                        row_index,
                        patient_label,
                        metrics,
                        error,
                        elapsed_seconds,
                    ) = payload
                    duration_text = ""
                    if elapsed_seconds is not None:
                        total_seconds = int(elapsed_seconds)
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        self.last_video_processing_duration = (
                            f"{hours:02}:{minutes:02}:{seconds:02}"
                        )
                        duration_text = (
                            f" | Tempo do último vídeo: "
                            f"{self.last_video_processing_duration}"
                        )
                    if error:
                        self.video_batch_errors.append(f"{patient_label}: {error}")
                        self.video_batch_results.append(False)
                    else:
                        self.df.at[row_index, "Movement index"] = metrics[0]
                        self.df.at[row_index, "Detection rate"] = metrics[1]
                        try:
                            self.df.to_csv(self.dataset_raw_file, index=False)
                            self.video_batch_results.append(True)
                        except Exception as exc:
                            self.video_batch_errors.append(
                                f"{patient_label}: métricas calculadas, mas não foi possível salvar "
                                f"a planilha: {exc}"
                            )
                            self.video_batch_results.append(False)
                    self.video_batch_progress.configure(value=position)
                    self.video_batch_status.configure(
                        text=f"Vídeo {position}/{self.video_batch_progress.cget('maximum')} concluído: "
                        f"{patient_label}{duration_text}"
                    )
                elif kind == "fatal_error":
                    self.video_batch_errors.append(
                        f"Falha ao carregar o processador de vídeos: {payload}"
                    )
                elif kind == "finished":
                    self.video_batch_running = False
                    self.process_videos_button.configure(state="normal")
                    self.start_video_processing_button.configure(state="disabled")
                    for button in self.action_buttons:
                        button.configure(state="normal")
                    self.tree.state(["!disabled"])
                    succeeded = sum(self.video_batch_results)
                    failed = len(self.video_batch_errors)
                    completion_status = (
                        f"Processamento concluído: {succeeded} sucesso(s), {failed} falha(s)."
                    )
                    if self.last_video_processing_duration is not None:
                        completion_status += (
                            f" Tempo do último vídeo processado: "
                            f"{self.last_video_processing_duration}."
                        )
                    self.video_batch_status.configure(
                        text=completion_status
                    )
                    if self.video_batch_errors:
                        details = "\n".join(f"• {error}" for error in self.video_batch_errors[:8])
                        if len(self.video_batch_errors) > 8:
                            details += f"\n... e mais {len(self.video_batch_errors) - 8} erro(s)."
                        messagebox.showwarning(
                            "Processamento concluído com falhas",
                            f"Sucesso: {succeeded}\nFalhas: {failed}\n\n{details}",
                        )
                    else:
                        messagebox.showinfo(
                            "Processamento concluído",
                            f"{succeeded} paciente(s) processado(s) e salvo(s) na planilha.",
                        )
                    return
        except queue.Empty:
            pass

        if self.video_batch_running:
            self.root.after(100, self._poll_video_batch_messages)

    def _close_window(self):
        if self.video_batch_running:
            messagebox.showwarning(
                "Processamento em andamento",
                "Aguarde o processamento dos vídeos terminar antes de fechar o programa.",
            )
            return
        self.root.destroy()

    def _build_video_output_path(self, patient_row=None, start_time=None):
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
        input_path = filedialog.askopenfilename(
            title="Selecione o vídeo original (.avi, .mp4)",
            filetypes=[("Video files", "*.avi *.mp4 *.mkv")]
        )
        if not input_path:
            return None, 120.0, None

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo selecionado.")
            return None, 120.0, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        current = 0
        start_frame = None
        end_frame = None
        win = "Cut Video"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        instructions = [
            "Controls: a - prev, d - next, s - select start, e - select end",
            "c - clear end",
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
            lines.append(f"Selected end: {end_frame + 1 if end_frame is not None else '-'}")
            lines.extend(instructions)

            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if (
                start_frame is not None
                and current >= start_frame
                and (end_frame is None or current <= end_frame)
            ):
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
                if end_frame is not None and end_frame < start_frame:
                    end_frame = None
                print(f"Selected start frame: {start_frame}")
            elif key == ord('e'):
                if start_frame is None:
                    messagebox.showwarning("Aviso", "Selecione o frame inicial antes do frame final.")
                elif current < start_frame:
                    messagebox.showwarning("Aviso", "O frame final deve ser igual ou posterior ao inicial.")
                else:
                    end_frame = current
                    print(f"Selected end frame: {end_frame}")
            elif key == ord('c'):
                end_frame = None
            elif key == ord('w'):
                if start_frame is None:
                    messagebox.showwarning("Aviso", "Selecione o frame inicial usando a tecla 's' antes de salvar.")
                    continue

                output_path = self._build_video_output_path(
                    patient_row=patient_row,
                    start_time=(patient_row.get("Hora") if patient_row is not None else None),
                )
                try:
                    output_path = write_cut_video(
                        input_path,
                        start_frame,
                        str(output_path),
                        codec='I420',
                        end_frame=end_frame,
                    )
                    frame_count = (
                        end_frame - start_frame + 1
                        if end_frame is not None
                        else total_frames - start_frame
                    )
                    duration_seconds = frame_count / fps if fps else 0.0
                    messagebox.showinfo(
                        "Sucesso",
                        f"Vídeo cortado e salvo com sucesso!\nDuração final: {duration_seconds:.2f} segundos."
                    )
                    cap.release()
                    cv2.destroyAllWindows()
                    return Path(output_path).name, duration_seconds, start_frame, end_frame
                except Exception as exc:
                    messagebox.showerror("Erro", f"Não foi possível cortar o vídeo:\n{exc}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None, 120.0, None, None
            elif key == ord('q'):
                break
            else:
                if key != 255:
                    pass

        cap.release()
        cv2.destroyAllWindows()
        return None, 120.0, None, None

    def cut_video_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        item_values = self.tree.item(selected_item[0], "values")
        idx = int(item_values[0])
        paciente = self.df.loc[idx]

        video_existente = self._get_absolute_path(paciente, "video_path")

        if video_existente and video_existente != "nan" and os.path.exists(video_existente):
            sobrescrever = messagebox.askyesno("Vídeo já existente", f"Já existe um vídeo cortado para este paciente:\n{video_existente}\n\nDeseja sobrescrever com um novo corte?")
            if not sobrescrever:
                return

        res_path, res_dur, init_frame, end_frame = self.cut_video_dialog(patient_row=paciente)

        if res_path:
            if "end_frame" not in self.df.columns:
                self.df["end_frame"] = pd.NA
            self.df.at[idx, "video_path"] = res_path
            self.df.at[idx, "video_duration"] = res_dur
            self.df.at[idx, "init_frame"] = init_frame
            self.df.at[idx, "end_frame"] = end_frame if end_frame is not None else pd.NA
            try:
                self.df.to_csv(self.dataset_raw_file, index=False)
                self.update_treeview()
                self._select_row_by_idx(idx)

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

    def _load_full_signals_by_id(
        self,
        paciente,
        h5_file,
        signal_ids,
        date_str,
        local_init_time,
        local_end_time,
        numeric_signal_ids=(),
    ):
        h5_files = [h5_file]
        time_parts = str(paciente.get("Hora", "")).strip().split(":")
        if len(time_parts) >= 2:
            try:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
            except ValueError:
                hour = minute = -1
            if 0 <= minute <= 59 and 0 < 60 - minute <= 3:
                next_row = paciente.to_dict()
                next_row["Hora"] = f"{hour + 1}:00"
                next_h5_file = self._build_h5_path_from_row(next_row)
                if next_h5_file and os.path.exists(next_h5_file):
                    h5_files.append(next_h5_file)

        datas = []
        ids = []
        seqsts = []
        numeric_records = {signal_id: [] for signal_id in numeric_signal_ids}
        for current_h5 in h5_files:
            try:
                current_datas, current_ids, _, current_seqsts = load_hdf5_packets(
                    current_h5,
                    b"\x02\x0B\x00\x00",
                    36,
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise OSError(f"Não foi possível ler o arquivo H5 {current_h5}:\n{exc}") from exc

            datas.extend(current_datas)
            ids.extend(current_ids.tolist())
            seqsts.extend(current_seqsts.tolist())

            if numeric_records:
                try:
                    current_numeric_records = load_hdf5_numeric_packets(
                        current_h5,
                        numeric_records,
                    )
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise OSError(
                        f"Não foi possível decodificar tendências numéricas de {current_h5}:\n{exc}"
                    ) from exc

                for numeric_id, (numeric_times, numeric_values) in current_numeric_records.items():
                    numeric_records[numeric_id].extend(
                        zip(numeric_times.tolist(), numeric_values.tolist())
                    )

        if not datas and not any(numeric_records.values()):
            raise ValueError("Nenhum pacote de dados foi encontrado nos arquivos H5.")

        ids = np.asarray(ids)
        seqsts = np.asarray(seqsts)
        loaded_signals = {}
        invalid_signal_ids = []

        for signal_id in signal_ids:
            indices = np.flatnonzero(ids == signal_id)
            if indices.size == 0:
                continue

            chunks = [np.asarray(datas[index]) for index in indices]
            timestamps = np.asarray(seqsts[indices], dtype=float)
            if (
                any(
                    chunk.ndim != 1
                    or chunk.size == 0
                    or not np.issubdtype(chunk.dtype, np.number)
                    or not np.isfinite(chunk).all()
                    for chunk in chunks
                )
                or timestamps.size != len(chunks)
                or not np.isfinite(timestamps).all()
            ):
                invalid_signal_ids.append(signal_id)
                continue

            positive_diffs = np.diff(timestamps)
            positive_diffs = positive_diffs[positive_diffs > 0]
            if positive_diffs.size == 0:
                invalid_signal_ids.append(signal_id)
                continue

            sample_interval = float(np.median(positive_diffs))
            sampling_rate = len(chunks[0]) / sample_interval
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                invalid_signal_ids.append(signal_id)
                continue

            try:
                signal = np.concatenate(chunks)
                time_vector = build_time_vectors(timestamps, chunks, sampling_rate)
                if (
                    signal.size == 0
                    or signal.size != time_vector.size
                    or not np.isfinite(time_vector).all()
                ):
                    invalid_signal_ids.append(signal_id)
                    continue
                dates = np.array([datetime.fromtimestamp(value) for value in time_vector])
            except (TypeError, ValueError, OverflowError):
                invalid_signal_ids.append(signal_id)
                continue

            mask = self._build_saved_interval_mask(
                dates,
                date_str,
                local_init_time,
                local_end_time,
            )
            loaded_signals[signal_id] = (dates, signal, mask, time_vector)

        for signal_id, records in numeric_records.items():
            if not records:
                continue

            records.sort(key=lambda record: record[0])
            timestamps = np.asarray([record[0] for record in records], dtype=float)
            signal = np.asarray(
                [np.nan if record[1] == -32768 else record[1] for record in records],
                dtype=float,
            )
            positive_diffs = np.diff(timestamps)
            positive_diffs = positive_diffs[
                np.isfinite(positive_diffs) & (positive_diffs > 0)
            ]
            if (
                not np.isfinite(timestamps).all()
                or positive_diffs.size == 0
                or not np.isfinite(signal).any()
            ):
                invalid_signal_ids.append(signal_id)
                continue

            sampling_rate = 1.0 / float(np.median(positive_diffs))
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                invalid_signal_ids.append(signal_id)
                continue

            try:
                dates = np.array([datetime.fromtimestamp(value) for value in timestamps])
            except (TypeError, ValueError, OverflowError, OSError):
                invalid_signal_ids.append(signal_id)
                continue

            mask = self._build_saved_interval_mask(
                dates,
                date_str,
                local_init_time,
                local_end_time,
            )
            loaded_signals[signal_id] = (dates, signal, mask, timestamps)

        return loaded_signals, invalid_signal_ids

    def visualize_saved_signals_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        idx = int(self.tree.item(selected_item[0], "values")[0])
        paciente = self.df.loc[idx]
        h5_file = self._resolve_h5_path(paciente, idx)
        date_str = str(paciente.get("Dia", "")).strip()
        init_time = str(paciente.get("init_time", "")).strip()
        end_time = str(paciente.get("end_time", "")).strip()

        if not h5_file or str(h5_file) == "nan" or not os.path.exists(str(h5_file)):
            messagebox.showwarning("Aviso", f"Arquivo H5 {h5_file} do paciente não encontrado para visualizar os sinais.")
            return

        if not self._is_filled(date_str) or not self._is_filled(init_time) or not self._is_filled(end_time):
            messagebox.showwarning("Aviso", "O paciente ainda não possui intervalo de sinal selecionado.")
            return

        signal_info = {
            "ECG": Config.ecg_id,
            "PPG": Config.spo2_id,
            "Respiração": Config.resp_id,
        }

        try:
            loaded_signals, _ = self._load_full_signals_by_id(
                paciente,
                str(h5_file),
                signal_info.values(),
                date_str,
                init_time,
                end_time,
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Erro ao ler sinais", str(exc))
            return

        if not loaded_signals:
            messagebox.showwarning("Sinais indisponíveis", "Nenhum dos sinais ECG, PPG ou respiração foi encontrado como vetor válido.")
            return

        loaded_signal_info = [
            (name, signal_id)
            for name, signal_id in signal_info.items()
            if signal_id in loaded_signals
        ]
        fig, axes = plt.subplots(len(loaded_signals), 1, figsize=(14, 4 * len(loaded_signals)), sharex=True)
        if len(loaded_signals) == 1:
            axes = [axes]

        for ax, (name, signal_id) in zip(axes, loaded_signal_info):
            dates_np, sig, mask, _ = loaded_signals[signal_id]

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

    def _choose_additional_signals_to_save(self, signal_names):
        dialog = tk.Toplevel(self.root)
        dialog.title("Selecionar sinais para salvar")
        dialog.transient(self.root)
        dialog.resizable(False, False)

        tk.Label(dialog, text="Selecione os sinais que deseja salvar como TXT:").pack(
            padx=20,
            pady=(15, 8),
            anchor="w",
        )
        selections = {
            name: tk.BooleanVar(master=dialog, value=False)
            for name in signal_names
        }
        for name, selected in selections.items():
            ttk.Checkbutton(dialog, text=name, variable=selected).pack(
                padx=20,
                anchor="w",
            )

        result = {"selected": None}

        def finish_selection():
            result["selected"] = [
                name for name, selected in selections.items() if selected.get()
            ]
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=15)
        ttk.Button(button_frame, text="Salvar selecionados", command=finish_selection).pack(
            side=tk.LEFT,
            padx=5,
        )
        ttk.Button(button_frame, text="Cancelar", command=dialog.destroy).pack(
            side=tk.LEFT,
            padx=5,
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        self.root.wait_window(dialog)
        return result["selected"]

    def add_additional_signals_for_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        idx = int(self.tree.item(selected_item[0], "values")[0])
        paciente = self.df.loc[idx]
        h5_file = self._resolve_h5_path(paciente, idx)
        if not h5_file or not os.path.exists(str(h5_file)):
            messagebox.showerror("Erro", f"Arquivo H5 não encontrado:\n{h5_file}")
            return

        signal_info = {
            "P1": {
                "id": Config.p1_id, "template": "p1", "column": "sinal_P1",
                "scale": 250.0 / 2048.0, "unit": "mmHg",
            },
            "PVC": {
                "id": Config.pvc_id, "template": "pvc", "column": "sinal_PVC",
                "scale": 1.0, "unit": "contagens",
            },
            "ART": {
                "id": Config.art_id, "template": "art", "column": "sinal_ART",
                "scale": 250.0 / 2048.0, "unit": "mmHg",
            },
            "CO2": {
                "id": Config.co2_id, "template": "co2", "column": "sinal_CO2",
                "scale": 0.1, "unit": "mmHg",
            },
            "Frequência cardíaca": {
                "id": Config.heart_rate_id, "template": "heart_rate", "column": "sinal_FC",
                "scale": 1.0, "unit": "bpm",
            },
            "Saturação de oxigênio": {
                "id": Config.spo2_trend_id, "template": "spo2_trend", "column": "sinal_SpO2",
                "scale": 1.0, "unit": "%",
            },
            "Frequência respiratória (impedância, provável)": {
                "id": Config.respiratory_rate_impedance_id,
                "template": "respiratory_rate_impedance",
                "column": "sinal_FR_impedancia",
                "scale": 1.0,
                "unit": "rpm",
            },
            "Frequência respiratória (CO₂, provável)": {
                "id": Config.respiratory_rate_co2_id,
                "template": "respiratory_rate_co2",
                "column": "sinal_FR_CO2",
                "scale": 1.0,
                "unit": "rpm",
            },
        }
        date_str = str(paciente.get("Dia", "")).strip()
        init_time = str(paciente.get("init_time", "")).strip()
        end_time = str(paciente.get("end_time", "")).strip()

        try:
            loaded_signals, invalid_signal_ids = self._load_full_signals_by_id(
                paciente,
                str(h5_file),
                [details["id"] for details in signal_info.values()],
                date_str,
                init_time,
                end_time,
                numeric_signal_ids=(
                    Config.heart_rate_id,
                    Config.spo2_trend_id,
                    Config.respiratory_rate_impedance_id,
                    Config.respiratory_rate_co2_id,
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Erro ao ler sinais", str(exc))
            return

        if not loaded_signals:
            invalid_names = [
                name for name, details in signal_info.items()
                if details["id"] in invalid_signal_ids
            ]
            detail = f"\n\nVetor inválido: {', '.join(invalid_names)}." if invalid_names else ""
            messagebox.showwarning(
                "Sinais indisponíveis",
                f"Nenhum dos sinais solicitados foi encontrado como vetor válido no H5.{detail}",
            )
            return

        if invalid_signal_ids:
            invalid_names = [
                name for name, details in signal_info.items()
                if details["id"] in invalid_signal_ids
            ]
            messagebox.showwarning(
                "Sinais ignorados",
                f"Estes sinais foram encontrados, mas não formam vetores válidos e não serão exibidos: {', '.join(invalid_names)}.",
            )

        valid_names = [
            name for name, details in signal_info.items()
            if details["id"] in loaded_signals
        ]
        fig, axes = plt.subplots(len(valid_names), 1, figsize=(14, 4 * len(valid_names)), sharex=True)
        if len(valid_names) == 1:
            axes = [axes]

        for ax, name in zip(axes, valid_names):
            dates, raw_signal, mask, _ = loaded_signals[signal_info[name]["id"]]
            signal = raw_signal.astype(float) * signal_info[name]["scale"]
            ax.plot(dates, signal, color="tab:blue", linewidth=1.5, label="sinal completo")
            if np.any(mask):
                ax.plot(dates[mask], signal[mask], color="tab:red", linewidth=2.5, label="intervalo salvo")
            ax.set_title(f"{name} - sinal completo e intervalo salvo")
            ax.set_ylabel(signal_info[name]["unit"])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Horário")
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

        if not messagebox.askyesno(
            "Salvar sinais adicionais",
            "Deseja salvar algum dos sinais exibidos como TXT?",
        ):
            return

        if not all(self._is_filled(value) for value in (date_str, init_time, end_time)):
            messagebox.showwarning(
                "Intervalo não selecionado",
                "O paciente não possui um intervalo salvo. Os sinais foram exibidos completos, mas não há um trecho para salvar.",
            )
            return

        signals_with_interval = [
            name for name in valid_names
            if np.any(loaded_signals[signal_info[name]["id"]][2])
        ]
        if not signals_with_interval:
            messagebox.showwarning(
                "Intervalo sem amostras",
                "Nenhum sinal possui amostras dentro do intervalo salvo deste paciente.",
            )
            return

        selected_names = self._choose_additional_signals_to_save(signals_with_interval)
        if not selected_names:
            return

        date_for_filename = date_str.replace("/", "-")
        start_for_filename = init_time.replace(":", "-")
        end_for_filename = end_time.replace(":", "-")
        bed = str(paciente.get("Leito", "")).strip()
        patient_output_dir = self._patient_data_dir(paciente.get("Id do paciente", ""))
        output_paths = {}
        for name in selected_names:
            details = signal_info[name]
            filename = APP_SETTINGS.signal_filename_templates[details["template"]].format(
                date=date_for_filename,
                bed=bed,
                start=start_for_filename,
                end=end_for_filename,
            )
            output_paths[name] = patient_output_dir / filename

        existing_paths = [path for path in output_paths.values() if path.exists()]
        if existing_paths and not messagebox.askyesno(
            "Sobrescrever sinais",
            "Já existem arquivos para um ou mais sinais selecionados. Deseja sobrescrevê-los?",
        ):
            return

        try:
            patient_output_dir.mkdir(parents=True, exist_ok=True)
            for name in selected_names:
                details = signal_info[name]
                signal_id = details["id"]
                _, raw_signal, mask, _ = loaded_signals[signal_id]
                signal = raw_signal.astype(float) * details["scale"]
                np.savetxt(output_paths[name], signal[mask], fmt="%.7e")
                self.df.at[idx, signal_info[name]["column"]] = output_paths[name].name

            self.df.to_csv(self.dataset_raw_file, index=False)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Erro ao salvar sinais",
                f"Não foi possível salvar todos os sinais selecionados ou atualizar a planilha:\n{exc}",
            )
            return

        self.update_treeview()
        self._select_row_by_idx(idx)
        messagebox.showinfo(
            "Sinais salvos",
            "Sinais salvos com sucesso:\n" + "\n".join(str(path) for path in output_paths.values()),
        )

    def process_signals_for_selected(self, skip_overwrite_warning=False):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return

        item_values = self.tree.item(selected_item[0], "values")
        idx = int(item_values[0])
        status = item_values[5]

        if status == "Preenchido" and not skip_overwrite_warning:
            if not messagebox.askyesno("Sobrescrever Sinais", "Este paciente já possui dados extraídos. Deseja sobrescrever os dados existentes?"):
                return

        paciente = self.df.loc[idx]
        previous_signal_paths = [self._get_absolute_path(paciente, column) for column in ("sinal_ECG", "sinal_PPG", "sinal_resp")]
        
        h5_file = self._resolve_h5_path(paciente, idx)

        if not h5_file or pd.isna(h5_file) or not os.path.exists(str(h5_file)):
            messagebox.showerror("Erro", f"Arquivo H5 não encontrado:\n{h5_file}")
            return

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
            patient_output_dir=patient_output_dir,
            patient_time=str(paciente.get("Hora", "")).strip(),
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
        current_absolute_paths = [
            str(patient_output_dir / res) for res in [resultados["sinal_ECG"], resultados["sinal_PPG"], resultados["sinal_resp"]] if res
        ]
        self._delete_previous_processed_signals(
            previous_signal_paths,
            current_absolute_paths,
        )

        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
            messagebox.showinfo("Sucesso", "Sinais extraídos e salvos na planilha com sucesso!")
            self.update_treeview()
            self._select_row_by_idx(idx)

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
