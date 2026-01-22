import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d


# Substitua pelo caminho do seu arquivo HDF5
file_path = "10.10.10.129_20251211_16.h5"

# Definições
DATA_PACK_HEAD = b"\x02\x0B\x00\x00"  # Definido como bytes
data_add = 36

# Abre o arquivo HDF5 no modo leitura
hdf = h5py.File(file_path, 'r')

data = hdf['data'][:]
data_timestamps = hdf['data_timestamps'][:]

datas = []
ids = []
seqs = [] 
seqsts = []

# Processamento dos dados
for raw_data, ts in zip(data, data_timestamps):
    pack_id = bytes(raw_data[0:4])
    if pack_id == DATA_PACK_HEAD: 
        frame_len = int.from_bytes(raw_data[4:6], byteorder='big')
        frame_seq = int.from_bytes(raw_data[24:26], byteorder='big')
        npk = 0
        process_next = True
        local_data_add = data_add  # Use uma variável local para evitar sobrescrever a original
        while process_next:
            if local_data_add + 4 <= len(raw_data):
                data_head = raw_data[local_data_add:local_data_add+4]
                data_len = int(raw_data[local_data_add+3]) * 2
                if data_len > 0:
                    if local_data_add + 4 + data_len <= len(raw_data):
                        data = raw_data[local_data_add+4:local_data_add+4+data_len]
                    else:
                        data = raw_data[local_data_add+4:]
                    data = np.frombuffer(data, dtype='>i2')  # Converte para np.array de inteiros
                    datas.append(data)
                    ids.append(int.from_bytes(data_head[0:3], byteorder='big'))
                    seqs.append(frame_seq)
                    seqsts.append(ts)
                    npk += 1
                    local_data_add = int(local_data_add) + 4 + data_len
                else:
                    process_next = False
            else:
                process_next = False


seqs = np.array(seqs)
ids = np.array(ids)
uniqIDs = np.unique(ids)
uniqseqs = np.unique(seqs)
seqsts = np.array(seqsts)

# idECGs = [65540]
ECGsID = 65540

if ECGsID in uniqIDs:
    indices = np.where(ids == ECGsID)[0]
    ts = seqsts[indices]
    Fs = len(datas[indices[0]])/np.median(np.diff(ts))
    print(Fs)
    dt = 1/Fs
    dtSeq = dt * len(datas[indices[0]])

    time_vector = []
    for i, t in enumerate(ts):
        time_vector.append(t + np.arange(len(datas[indices[i]])) * dt)
    time_vector = np.concatenate(time_vector)
    dates = [datetime.fromtimestamp(ts) for ts in time_vector]
    dates_np = np.array(dates)

    start_dt = datetime.combine(dates_np[0].date(), datetime.strptime("16:35", "%H:%M").time())
    end_dt = datetime.combine(dates_np[0].date(), datetime.strptime("16:45", "%H:%M").time())

    mask = (dates_np >= start_dt) & (dates_np <= end_dt)

    sig = [datas[i] for i in indices]
    sig = np.concatenate(sig)
    print(sig)
    seq = [seqs[i] for i in indices]
    print(np.sum(np.diff(seq) > 1)/len(seq))
    # plt.subplot(4,3,nplot)+
    # plt.plot(dates, sig)
    plt.plot(dates_np[mask], sig[mask])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()

idSpO2 = 458768
idResp = 327688
idP1 = 4063240
idPVC = 3473416
idART = 2883592
idCO2 = 4784136

indices = np.where(ids == idSpO2)[0]
sig = [datas[i] for i in indices]
sig = np.concatenate(sig)
seq = [seqs[i] for i in indices]
ts = seqsts[indices]

Fs = len(datas[indices[0]])/np.median(np.diff(ts))
dt = 1/Fs
dtSeq = dt * len(datas[indices[0]])

time_vector = []
for i, t in enumerate(ts):
    time_vector.append(t + np.arange(len(datas[indices[i]])) * dt)
time_vector = np.concatenate(time_vector)

dates = [datetime.fromtimestamp(ts) for ts in time_vector]

dates_np = np.array(dates)

start_dt = datetime.combine(dates_np[0].date(), datetime.strptime("16:17", "%H:%M").time())
end_dt = datetime.combine(dates_np[0].date(), datetime.strptime("16:18", "%H:%M").time())

mask = (dates_np >= start_dt) & (dates_np <= end_dt)

plt.plot(dates_np[mask], sig[mask])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Process and save data (Amplitude, HR, Timestamp)
sig_m = sig[mask].astype(float)
t_m = time_vector[mask]
t_m = t_m - t_m[0]  # Start timestamp from 0

# 1. Bandpass Filter (0.7 - 4.0 Hz) to get AC Amplitude
sos = signal.butter(2, [0.7, 4.0], btype='bandpass', fs=Fs, output='sos')
sig_filt = signal.sosfiltfilt(sos, sig_m)

# 2. Calculate Pulse Rate (Sliding Window FFT)
window_sec = 15
window_samples = int(window_sec * Fs)
step_samples = int(1 * Fs)  # 1 second step
hr_t = []
hr_bpm = []

# Zero padding for higher frequency resolution (0.1 BPM)
n_pad = int(60 * Fs / 0.1)

for i in range(0, len(sig_filt) - window_samples, step_samples):
    segment = sig_filt[i:i+window_samples]
    segment = segment * np.hanning(len(segment))
    fft_out = np.fft.rfft(segment, n=n_pad)
    freqs = np.fft.rfftfreq(n_pad, 1/Fs)
    
    valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
    if len(valid_idx) > 0:
        peak_idx = valid_idx[np.argmax(np.abs(fft_out[valid_idx]))]
        hr_bpm.append(freqs[peak_idx] * 60)
        hr_t.append(t_m[i + window_samples//2])

f_interp = interp1d(hr_t, hr_bpm, kind='linear', fill_value="extrapolate")
hr_interp = f_interp(t_m)

data_save = np.vstack((sig_filt, hr_interp, t_m))
np.savetxt("sp02_processed.txt", data_save, fmt='%.7e')

plt.show()
