from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import h5py
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import heartpy as hp
from scipy.interpolate import interp1d
from scipy.signal import resample


@dataclass
class Config:
    file_path: str = "../../rppG_data/pilot/ground_truth/10.10.10.129_20251211_16.h5"
    date: str = '11-12-2025'
    start_time: str = "16:45:38"
    end_time: str = "16:47:38"
    bed: str = "L8"
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
        self._update_n_points_from_video_source()

    def _update_n_points_from_video_source(self):
        if not self.video_source_path:
            return

        video_path = Path(self.video_source_path)
        if not video_path.exists():
            print(f"Warning: video source path does not exist: {video_path}. Using n_points={self.n_points}")
            return

        frame_count = get_video_frame_count(video_path)
        if frame_count is not None and frame_count > 0:
            self.n_points = frame_count
        else:
            print(f"Warning: unable to determine frame count for video {video_path}. Using n_points={self.n_points}")


def estimate_hr_heartpy(segment, fs):
    try:
        _, metrics = hp.process(segment, sample_rate=fs)
        return metrics["bpm"]
    except Exception:
        return np.nan


def get_video_frame_count(video_path: Path) -> int | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count


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


def show_or_close(fig=None, show_plots=True):
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def process_ecg(config, datas, ids, seqs, seqsts):
    if config.ecg_id not in np.unique(ids):
        return

    indices = np.where(ids == config.ecg_id)[0]
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))
    print("fs ecg:", fs)

    sig = np.concatenate([datas[i] for i in indices])
    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    mask = get_window_mask(dates_np, config.start_time, config.end_time)

    if config.save_ecg:
        output_file = config.ecg_dir / f"ecg_signal_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.csv"
        np.savetxt(output_file, sig[mask], delimiter=",", fmt="%d")

    seq = [seqs[i] for i in indices]
    print(np.sum(np.diff(seq) > 1) / len(seq))

    plt.plot(dates_np, sig)
    plt.xlabel("Horário")
    plt.ylabel("Amplitude")
    show_or_close(show_plots=config.show_plots)

    plt.plot(dates_np[mask], sig[mask])
    plt.xlabel("Horário")
    plt.ylabel("Amplitude")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    show_or_close(show_plots=config.show_plots)


def process_spo2(config, datas, ids, seqs, seqsts):
    indices = np.where(ids == config.spo2_id)[0]
    if len(indices) == 0:
        return

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))
    print("fs spo2:", fs)

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    mask = get_window_mask(dates_np, config.start_time, config.end_time)

    if config.save_spo2_wave:
        sig_m = sig[mask].astype(float)
        t_m = time_vector[mask]
        t_m_rel = t_m - t_m[0]
        np.savetxt(config.spo2_dir / f"original_spo2_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt", sig_m, fmt="%.7e")

        if config.resample_spo2:
            spo2wave = resample(sig_m, config.n_points)
            if not config.save3lines:
                spo2wave = np.reshape(spo2wave, (1, spo2wave.shape[0]))
                np.savetxt(config.spo2_dir / f"sp02wave_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt", spo2wave, fmt="%.7e")
                return

            t_new = np.linspace(t_m_rel[0], t_m_rel[-1], config.n_points)
            window_sec = 15
            step_sec = 1
            window_len = int(window_sec * fs)
            step_len = int(step_sec * fs)

            hr_times, hr_values = [], []
            for i in range(0, len(sig_m) - window_len, step_len):
                segment = sig_m[i:i + window_len]
                hr_values.append(estimate_hr_heartpy(segment, fs))
                hr_times.append(t_m_rel[i + window_len // 2])

            hr_times = np.array(hr_times)
            hr_values = np.array(hr_values)
            valid_mask = ~np.isnan(hr_values)
            if np.sum(valid_mask) > 1:
                f_hr = interp1d(hr_times[valid_mask], hr_values[valid_mask], kind="linear", fill_value="extrapolate")
                hr_resampled = f_hr(t_new)
            else:
                hr_resampled = np.full(config.n_points, np.nan)

            data_save = np.vstack((spo2wave, hr_resampled, t_new))
            np.savetxt(config.spo2_dir / f"sp02wave_{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt", data_save, fmt="%.7e")

        plt.plot(dates_np[mask], sig[mask])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        show_or_close(show_plots=config.show_plots)


def process_rr(config, datas, ids, seqs, seqsts):
    indices = np.where(ids == config.resp_id)[0]
    if len(indices) == 0:
        return

    sig = np.concatenate([datas[i] for i in indices])
    ts = seqsts[indices]
    fs = len(datas[indices[0]]) / np.median(np.diff(ts))
    print("rr fs:", fs)

    time_vector = build_time_vectors(ts, [datas[i] for i in indices], fs)
    dates_np = np.array([datetime.fromtimestamp(ts_value) for ts_value in time_vector])
    mask = get_window_mask(dates_np, config.start_time, config.end_time)

    sig_m = sig[mask].astype(float)
    np.savetxt(config.rr_dir / f"{config.date}_{config.bed}_{config.hora_inicio}_{config.hora_fim}.txt", sig_m, fmt="%.7e")

    plt.figure(figsize=(12, 4))
    plt.plot(dates_np[mask], sig_m)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xlabel("Horário")
    plt.ylabel("Amplitude")
    plt.title(f"Impedância Torácica - {config.bed} ({config.start_time} - {config.end_time})")
    plt.grid(True)
    plt.tight_layout()
    show_or_close(show_plots=config.show_plots)


def save_gt_data(config=None):
    config = config or Config()
    datas, ids, seqs, seqsts = load_hdf5_packets(
        config.file_path,
        config.data_pack_head,
        config.data_add,
    )

    if not datas:
        raise ValueError("No data packets were parsed from the HDF5 file.")

    process_ecg(config, datas, ids, seqs, seqsts)
    process_spo2(config, datas, ids, seqs, seqsts)
    if config.save_rr:
        process_rr(config, datas, ids, seqs, seqsts)


if __name__ == "__main__":
    save_gt_data()

