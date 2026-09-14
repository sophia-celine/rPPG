import os
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import heartpy as hp

from config import APP_SETTINGS, Config, signal_timestamp_to_datetime


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
            if event.inaxes != ax or event.xdata is None:
                return
            x = event.xdata
            vline.set_xdata([x, x])
            idx = int(np.argmin(np.abs(dnums - x)))
            tstr = dates[idx].strftime('%H:%M:%S')
            hover_ann.set_text(f'{tstr} [{idx}]')
            hover_ann.xy = (dnums[idx], signal[idx])
            hover_ann.set_visible(True)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            x = event.xdata
            idx = int(np.argmin(np.abs(dnums - x)))
            selected['index'] = idx
            sel_marker.set_data([dnums[idx]], [signal[idx]])
            sel_marker.set_visible(True)
            fig.canvas.draw_idle()

        def on_key(event):
            if selected['index'] is None:
                return
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

    return selected_info, output_file.name if output_file else ""


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

    return output_file.name if output_file else ""


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

    return output_file.name if output_file else ""


def run_extraction_for_patient(h5_file, bed, date_str, duration_seconds=120.0, patient_output_dir=None, patient_time=None):
    output_dir = str(patient_output_dir) if patient_output_dir is not None else str(Path(APP_SETTINGS.patient_data_root))
    config = Config(
        file_path=h5_file,
        bed=bed,
        date=date_str.replace("/", "-"),
        duration_seconds=duration_seconds,
        output_dir=output_dir,
    )
    config.output_path = Path(output_dir)
    config.output_path.mkdir(parents=True, exist_ok=True)
    config.ecg_dir = config.output_path
    config.spo2_dir = config.output_path
    config.rr_dir = config.output_path

    h5_files = [h5_file]
    if patient_time is not None:
        try:
            time_parts = [part.strip() for part in str(patient_time).split(":")]
            print(time_parts)
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                print(f"{0 <= minute <= 59 and 60 - minute <= 3}")
                if 0 <= minute <= 59 and 60 - minute <= 3:
                    print("Adding next hour H5 file for extraction...")
                    date_parts = [p.strip() for p in str(date_str).replace("-", "/").split("/")]
                    print(date_parts)
                    if len(date_parts) == 3:
                        day, month, year = date_parts
                        day_folder = f"{year}{month}{day}"
                        file_stem = os.path.splitext(os.path.basename(h5_file))[0]
                        if f"_{day_folder}_" in file_stem:
                            bed_ip = file_stem.rsplit(f"_{day_folder}_", 1)[0]
                        elif f"{day_folder}" in file_stem:
                            bed_ip = file_stem.rsplit(f"{day_folder}", 1)[0]
                        else:
                            bed_ip = file_stem.rsplit("_", 1)[0]
                        next_h5_file = f"{APP_SETTINGS.uti_data_path}/{day_folder}/{bed_ip}_{day_folder}_{hour + 1}.h5"
                        print(next_h5_file)
                        if os.path.exists(next_h5_file):
                            h5_files.append(next_h5_file)
        except Exception:
            pass

    combined_datas = []
    combined_ids = []
    combined_seqsts = []

    for current_h5 in h5_files:
        try:
            datas, ids, _, seqsts = load_hdf5_packets(
                current_h5,
                config.data_pack_head,
                config.data_add,
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
        "error": None,
    }
