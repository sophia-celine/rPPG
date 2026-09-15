import os
import re
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
from signal_processing import (
    build_time_vectors,
    load_hdf5_packets,
    run_extraction_for_patient,
    write_cut_video,
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

    elif cid.startswith(("S", "T", "V")):
        return "Trauma"

    else:
        return "Outro"


class SignalExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Extrator de Sinais - UTI")
        self.root.geometry("1920x1080")

        self.dataset_raw_file = APP_SETTINGS.dataset_raw_file
        self.df = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header_frame = tk.Frame(self.root, pady=10)
        header_frame.grid(row=0, column=0, sticky="ew")

        tk.Label(header_frame, text="Selecione um paciente para Processar Sinais ou Cortar Vídeo", font=("Arial", 14, "bold")).pack()

        columns = ("index", "id", "leito", "data", "hora", "status", "video_status")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("index", text="Índice")
        self.tree.heading("id", text="ID Paciente")
        self.tree.heading("leito", text="Leito")
        self.tree.heading("data", text="Data")
        self.tree.heading("hora", text="Hora")
        self.tree.heading("status", text="Status (Sinais)")
        self.tree.heading("video_status", text="Vídeo")

        self.tree.column("index", width=50, anchor="center")
        self.tree.column("id", width=120, anchor="center")
        self.tree.column("leito", width=80, anchor="center")
        self.tree.column("data", width=100, anchor="center")
        self.tree.column("hora", width=80, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.column("video_status", width=120, anchor="center")

        self.tree.grid(row=1, column=0, sticky="nsew", padx=20)

        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky='ns', pady=0)

        self.tree.tag_configure('pendente', foreground='red')
        self.tree.tag_configure('preenchido', foreground='green')

        btn_frame = tk.Frame(self.root, pady=15)
        btn_frame.grid(row=2, column=0, sticky="ew")

        btn_sync_remote = tk.Button(btn_frame, text="Sincronizar Planilha", command=self.sync_local_dataset_with_remote, bg="#9C27B0", fg="white", font=("Arial", 11, "bold"), width=20)
        btn_sync_remote.pack(side=tk.LEFT, padx=15)

        btn_cut_video = tk.Button(btn_frame, text="Cortar Vídeo", command=self.cut_video_for_selected, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), width=15)
        btn_cut_video.pack(side=tk.LEFT, padx=15)

        btn_process_signals = tk.Button(btn_frame, text="Processar Sinais", command=self.process_signals_for_selected, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_process_signals.pack(side=tk.LEFT, padx=15)

        btn_visualize_signals = tk.Button(btn_frame, text="Visualizar Sinais", command=self.visualize_saved_signals_for_selected, bg="#FF9800", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_visualize_signals.pack(side=tk.LEFT, padx=15)

        btn_load_clinical = tk.Button(btn_frame, text="Carregar dados clínicos", command=self.load_clinical_data_for_selected, bg="#E91E63", fg="white", font=("Arial", 11, "bold"), width=22)
        btn_load_clinical.pack(side=tk.LEFT, padx=15)

        btn_analyze_dataset = tk.Button(btn_frame, text="Analisar Planilha", command=self.show_dataset_analysis, bg="#795548", fg="white", font=("Arial", 11, "bold"), width=18)
        btn_analyze_dataset.pack(side=tk.LEFT, padx=15)

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
            return filename

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
            "video_path", "video_duration", "h5_file", "evolucoes_path"
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
                    "sinal_resp", "video_path", "video_duration", "evolucoes_path",
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

            novas_colunas = ["init_time", "end_time", "sinal_ECG", "sinal_PPG", "sinal_resp", "video_path", "video_duration", "evolucoes_path"]
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
        figure.patch.set_facecolor("#F5F7FA")
        figure.suptitle("Panorama dos dados da planilha", fontsize=16, fontweight="bold", color="#263238")
        axes = figure.subplots(4, 4).ravel()
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
                .fillna("")
                .astype(str)
                .str.split("-", n=1)
                .str[0]
                .map(classify_cid)
            )
            classified_cids.index = cid_data["_patient_id"]
            classified_cids = classified_cids.groupby(level=0).first()
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
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 8},
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
                    colors=["#59A14F", "#E15759"],
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 8},
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
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    textprops={"fontsize": 8},
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
            self.tree.insert("", tk.END, values=(idx, id_pac, leito, data, hora, status, video_status), tags=(tag,))

    def _select_row_by_idx(self, target_idx):
        for item in self.tree.get_children():
            if int(self.tree.item(item, "values")[0]) == target_idx:
                self.tree.selection_set(item)
                self.tree.focus(item)
                self.tree.see(item)
                break

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
                    return Path(output_path).name, duration_seconds
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

        video_existente = self._get_absolute_path(paciente, "video_path")

        if video_existente and video_existente != "nan" and os.path.exists(video_existente):
            sobrescrever = messagebox.askyesno("Vídeo já existente", f"Já existe um vídeo cortado para este paciente:\n{video_existente}\n\nDeseja sobrescrever com um novo corte?")
            if not sobrescrever:
                return

        res_path, res_dur = self.cut_video_dialog(patient_row=paciente)

        if res_path:
            self.df.at[idx, "video_path"] = res_path
            self.df.at[idx, "video_duration"] = res_dur
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

    def _load_full_signal_by_id(self, paciente, h5_file, signal_id, date_str, local_init_time, local_end_time):
        try:
            h5_files = [h5_file]
            time_value = str(paciente.get("Hora", "")).strip()
            time_parts = [part.strip() for part in time_value.split(":")]
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                minutes_to_next_hour = 60 - minute
                if 0 <= minute <= 59 and 0 < minutes_to_next_hour <= 3:
                    next_row = paciente.to_dict()
                    next_row["Hora"] = f"{hour + 1}:00"
                    next_h5_file = self._build_h5_path_from_row(next_row)
                    if next_h5_file and os.path.exists(next_h5_file):
                        h5_files.append(next_h5_file)

            combined_datas = []
            combined_ids = []
            combined_seqsts = []

            for current_h5 in h5_files:
                try:
                    datas, ids, _, seqsts = load_hdf5_packets(
                        current_h5,
                        b"\x02\x0B\x00\x00",
                        36,
                    )
                except Exception as e:
                    if current_h5 == h5_file:
                        return {"error": f"Erro ao ler H5: {e}"}
                    continue

                if not datas:
                    continue

                combined_datas.extend(datas)
                combined_ids.extend(ids.tolist())
                combined_seqsts.extend(seqsts.tolist())

            if not combined_datas:
                return {"error": "Nenhum pacote de dados encontrado no H5."}

            datas = combined_datas
            ids = np.asarray(combined_ids)
            seqsts = np.asarray(combined_seqsts)
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

        h5_file = self._resolve_h5_path(paciente, idx)
        date_str = str(paciente.get("Dia", "")).strip()
        init_time = str(paciente.get("init_time", "")).strip()
        end_time = str(paciente.get("end_time", "")).strip()

        if not h5_file or str(h5_file) == "nan" or not os.path.exists(str(h5_file)):
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
                paciente,
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
