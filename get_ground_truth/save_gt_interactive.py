import os
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.interpolate import interp1d
from scipy.signal import resample

# =============================================================================
# CONFIGURAÇÕES DA EXTRAÇÃO
# =============================================================================

@dataclass
class Config:
    file_path: str = ""
    date: str = ""
    start_time: str = "16:00:00"
    end_time: str = "16:02:00"
    bed: str = ""
    output_dir: str = "../../rPPG_data/ground_truth"
    video_source_path: str = ""
    n_points: int = 2997
    save_ecg: bool = True
    save_spo2_wave: bool = True
    resample_spo2: bool = False
    save3lines: bool = False
    save_rr: bool = True
    show_plots: bool = True
    data_pack_head: bytes = b"\x02\x0B\x00\x00"
    data_add: int = 36
    ecg_id: int = 65796
    spo2_id: int = 458768
    resp_id: int = 327688
    interactive_select_time: bool = True
    duration_seconds: int = 120
    selected_start_index: Optional[int] = None
    selected_end_index: Optional[int] = None
    selected_start_ts: Optional[float] = None
    selected_end_ts: Optional[float] = None

    def __post_init__(self):
        self.hora_inicio = self.start_time.replace(':', '-')
        self.hora_fim = self.end_time.replace(':', '-')
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.ecg_dir = self.output_path / "ECG"
        self.spo2_dir = self.output_path / "spo2"
        self.rr_dir = self.output_path / "thoracic_impedance"
        for folder in (self.ecg_dir, self.spo2_dir, self.rr_dir):
            folder.mkdir(parents=True, exist_ok=True)


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

def process_ecg(config, datas, ids, seqs, seqsts):
    if config.ecg_id not in np.unique(ids):
        return None, None

    indices = np.where(ids == config.ecg_id)[0]
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))
    
    sig = np.concatenate([datas[i] for i in indices])
    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    
    def select_start_point_interactive(dates, signal, time_vector, fs):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(dates, signal, color='tab:blue')
        ax.set_xlabel('Horário')
        ax.set_ylabel('Amplitude')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.grid(True)
        plt.title('Passe o mouse para preview. Clique para selecionar INÍCIO. Pressione y para confirmar, n para cancelar.')
        plt.tight_layout()

        dnums = mdates.date2num(dates)
        vline = ax.axvline(dnums[0], color='gray', linewidth=1, linestyle='--')
        hover_ann = ax.annotate('', xy=(0,0), xytext=(15,15), textcoords='offset points', bbox=dict(boxstyle='round', fc='w'), visible=False)
        sel_marker, = ax.plot([], [], 'ro', markersize=12, visible=False)

        selected = {'index': None, 'confirmed': False}

        def on_move(event):
            if event.inaxes != ax or event.xdata is None: return
            x = event.xdata
            vline.set_xdata([x, x])
            idx = int(np.argmin(np.abs(dnums - x)))
            tstr = datetime.fromtimestamp(time_vector[idx]).strftime('%H:%M:%S')
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
            start_time_dt = datetime.fromtimestamp(time_vector[start_idx])
            end_time_dt = datetime.fromtimestamp(time_vector[end_idx])
            return start_time_dt.strftime('%H:%M:%S'), end_time_dt.strftime('%H:%M:%S'), start_idx, end_idx
        return None, None, None, None

    selected_info = (None, None, None, None)
    if config.interactive_select_time and config.show_plots:
        s, e, si, ei = select_start_point_interactive(dates_np, sig, time_vector, fs)
        if s is not None and e is not None:
            config.start_time = s
            config.end_time = e
            config.hora_inicio = config.start_time.replace(':', '-')
            config.hora_fim = config.end_time.replace(':', '-')
            config.selected_start_index = si
            config.selected_end_index = ei
            config.selected_start_ts = time_vector[si]
            config.selected_end_ts = time_vector[ei]
            selected_info = (s, e, si, ei)

    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    output_file = ""
    if config.save_ecg:
        output_file = config.ecg_dir / f"ecg_signal_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.csv"
        np.savetxt(output_file, sig[mask], delimiter=",", fmt="%d")

    return selected_info, str(output_file)

def process_spo2(config, datas, ids, seqs, seqsts):
    indices = np.where(ids == config.spo2_id)[0]
    if len(indices) == 0:
        return ""

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    
    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    output_file = ""
    if config.save_spo2_wave:
        sig_m = sig[mask].astype(float)
        output_file = config.spo2_dir / f"original_spo2_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt"
        np.savetxt(output_file, sig_m, fmt="%.7e")
    
    return str(output_file)

def process_rr(config, datas, ids, seqs, seqsts):
    indices = np.where(ids == config.resp_id)[0]
    if len(indices) == 0:
        return ""

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    
    if config.selected_start_ts is not None and config.selected_end_ts is not None:
        mask = (time_vector >= config.selected_start_ts) & (time_vector <= config.selected_end_ts)
    else:
        mask = get_window_mask(dates_np, config.start_time, config.end_time)

    sig_m = sig[mask].astype(float)
    output_file = config.rr_dir / f"{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt"
    np.savetxt(output_file, sig_m, fmt="%.7e")
    
    return str(output_file)

def run_extraction_for_patient(h5_file, bed, date_str):
    config = Config(
        file_path=h5_file,
        bed=bed,
        date=date_str.replace("/", "-")
    )
    
    try:
        datas, ids, seqs, seqsts = load_hdf5_packets(
            config.file_path,
            config.data_pack_head,
            config.data_add,
        )
    except Exception as e:
        return {"error": f"Erro ao ler H5: {e}"}

    if not datas:
        return {"error": "Nenhum pacote de dados encontrado no H5."}

    sel, ecg_path = process_ecg(config, datas, ids, seqs, seqsts)
    if not sel or sel[0] is None:
        return {"error": "Seleção cancelada."}
        
    start_time, end_time = sel[0], sel[1]
    spo2_path = process_spo2(config, datas, ids, seqs, seqsts)
    rr_path = process_rr(config, datas, ids, seqs, seqsts) if config.save_rr else ""

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
        self.root.geometry("800x500")
        
        self.dataset_raw_file = r"C:\Users\Sophia\Documents\rPPG_data\ground_truth\dataset_raw.csv"
        self.df = None
        
        # Configurando Grid Principal
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        # Header
        header_frame = tk.Frame(self.root, pady=10)
        header_frame.grid(row=0, column=0, sticky="ew")
        
        tk.Label(header_frame, text="Selecione um paciente para extrair os sinais", font=("Arial", 14, "bold")).pack()
        
        # Treeview (Tabela)
        columns = ("index", "id", "leito", "data", "status")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("index", text="Índice")
        self.tree.heading("id", text="ID Paciente")
        self.tree.heading("leito", text="Leito")
        self.tree.heading("data", text="Data")
        self.tree.heading("status", text="Status")
        
        self.tree.column("index", width=50, anchor="center")
        self.tree.column("id", width=120, anchor="center")
        self.tree.column("leito", width=100, anchor="center")
        self.tree.column("data", width=100, anchor="center")
        self.tree.column("status", width=150, anchor="center")
        
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
        
        btn_process = tk.Button(btn_frame, text="Processar Selecionado", command=self.process_selected, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=20)
        btn_process.pack(side=tk.LEFT, padx=20)
        
        btn_refresh = tk.Button(btn_frame, text="Recarregar Planilha", command=self.load_data, font=("Arial", 10))
        btn_refresh.pack(side=tk.LEFT, padx=10)
        
        # Carregar Dados
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.dataset_raw_file):
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{self.dataset_raw_file}\n\nRode o get_h5.py primeiro.")
            return
            
        try:
            self.df = pd.read_csv(self.dataset_raw_file)
            
            # Garante colunas
            novas_colunas = ["init_time", "end_time", "sinal_ECG", "sinal_PPG", "sinal_resp"]
            for col in novas_colunas:
                if col not in self.df.columns:
                    self.df[col] = pd.NA
                    
            self.update_treeview()
        except Exception as e:
            messagebox.showerror("Erro de Leitura", f"Erro ao ler o CSV:\n{e}")

    def update_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for idx, row in self.df.iterrows():
            id_pac = row.get("Id do paciente", "N/A")
            leito = row.get("Leito", "N/A")
            data = row.get("Dia", "N/A")
            
            if pd.notna(row["init_time"]) and str(row["init_time"]).strip() != "":
                status = "Preenchido"
                tag = "preenchido"
            else:
                status = "Pendente"
                tag = "pendente"
                
            self.tree.insert("", tk.END, values=(idx, id_pac, leito, data, status), tags=(tag,))

    def process_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione um paciente na lista primeiro.")
            return
            
        item_values = self.tree.item(selected_item[0], "values")
        idx = int(item_values[0])
        status = item_values[4]
        
        if status == "Preenchido":
            if not messagebox.askyesno("Sobrescrever", "Este paciente já possui dados extraídos. Deseja sobrescrever os dados existentes?"):
                return
                
        paciente = self.df.loc[idx]
        h5_file = paciente.get("h5_file", "")
        
        if not h5_file or pd.isna(h5_file) or not os.path.exists(h5_file):
            messagebox.showerror("Erro", f"Arquivo H5 não encontrado:\n{h5_file}")
            return
            
        messagebox.showinfo("Instruções", "A janela do gráfico vai abrir.\n\n1. Passe o mouse para ver o momento.\n2. Clique para marcar o Início.\n3. Pressione a tecla 'y' para confirmar ou 'n' para cancelar.")
        
        # Esconde a janela principal para focar no matplotlib (opcional)
        self.root.iconify()
        
        # Roda extração
        resultados = run_extraction_for_patient(
            h5_file=str(h5_file),
            bed=str(paciente.get("Leito", "")).strip(),
            date_str=str(paciente.get("Dia", "")).strip()
        )
        
        # Restaura a janela
        self.root.deiconify()
        
        if resultados.get("error"):
            messagebox.showwarning("Extração Interrompida", resultados["error"])
            return
            
        # Salva dados no DataFrame
        self.df.at[idx, "init_time"] = resultados["init_time"]
        self.df.at[idx, "end_time"] = resultados["end_time"]
        self.df.at[idx, "sinal_ECG"] = resultados["sinal_ECG"]
        self.df.at[idx, "sinal_PPG"] = resultados["sinal_PPG"]
        self.df.at[idx, "sinal_resp"] = resultados["sinal_resp"]
        
        try:
            self.df.to_csv(self.dataset_raw_file, index=False)
            messagebox.showinfo("Sucesso", "Dados extraídos e salvos na planilha com sucesso!")
            self.update_treeview() # Atualiza interface
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o CSV:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalExtractorApp(root)
    root.mainloop()