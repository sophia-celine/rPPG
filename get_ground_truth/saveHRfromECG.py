import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import heartpy as hp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import resample


sample_rate = 250


# Substitua pelo caminho do seu arquivo HDF5
file_path = "../pilot/ground_truth/10.10.10.138_20251211_16.h5"
HORA_INICIO = "16:04:13"
HORA_FIM = "16:06:12"
LEITO = "L9"
n_points = 2997
hora_inicio = HORA_INICIO.replace(':', '-')
hora_fim = HORA_FIM.replace(':', '-')

# Definições

DATA_PACK_HEAD = b"\x02\x0B\x00\x00"  # Definido como bytes
data_add = 36

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
        local_data_add = data_add 
        while process_next:
            if local_data_add + 4 <= len(raw_data):
                data_head = raw_data[local_data_add:local_data_add+4]
                data_len = int(raw_data[local_data_add+3]) * 2
                if data_len > 0:
                    if local_data_add + 4 + data_len <= len(raw_data):
                        data = raw_data[local_data_add+4:local_data_add+4+data_len]
                    else:
                        data = raw_data[local_data_add+4:]
                    data = np.frombuffer(data, dtype='>i2')  
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

################## IDENTIFICAÇÃO DA FREQUÊNCIA CARDÍACA PELO ECG ##################

# idECGs = [65540]
ECGsID = 65796
# idECGs = [65540, 65796, 66052, 66308, 66564, 66820, 67076, 67332, 67588, 67844, 68100, 68356]

# if ECGsID in uniqIDs:
#     indices = np.where(ids == ECGsID)[0]
#     ts = seqsts[indices]
#     Fs = len(datas[indices[0]])/np.median(np.diff(ts))
#     dt = 1/Fs
#     dtSeq = dt * len(datas[indices[0]])

#     time_vector = []
#     for i, t in enumerate(ts):
#         time_vector.append(t + np.arange(len(datas[indices[i]])) * dt)
#     time_vector = np.concatenate(time_vector)
#     dates = [datetime.fromtimestamp(ts) for ts in time_vector]
#     dates_np = np.array(dates)

#     start_dt = datetime.combine(dates_np[0].date(), datetime.strptime(HORA_INICIO, "%H:%M:%S").time())
#     end_dt = datetime.combine(dates_np[0].date(), datetime.strptime(HORA_FIM, "%H:%M:%S").time())

#     mask = (dates_np >= start_dt) & (dates_np <= end_dt)

#     sig = [datas[i] for i in indices]
#     sig = np.concatenate(sig)
#     np.savetxt(f"ecg_signal_{LEITO}_{hora_inicio}_{hora_fim}.csv", sig[mask], delimiter=",", fmt='%d')
#     print(sig)

#     # Calculate HR from ECG
#     ecg_m = sig[mask]
#     segment_width = 15
#     segment_overlap = 0
#     wd, m = hp.process_segmentwise(ecg_m, sample_rate=250, segment_width=segment_width, segment_overlap=segment_overlap)
#     hr_ecg = m['bpm']
#     print('hr ecg initial len', len(hr_ecg))
#     step = segment_width * (1 - segment_overlap)
#     hr_times = np.arange(len(hr_ecg)) * step + (segment_width / 2)

#     seq = [seqs[i] for i in indices]
#     print(np.sum(np.diff(seq) > 1)/len(seq))
#     # plt.subplot(4,3,nplot)
#     # plt.plot(dates, sig)
#     plt.plot(dates_np[mask], sig[mask])
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     # nplot+=1
# plt.show()


################## VETOR ##################

idSpO2 = 458768
# idResp = 327688
# idP1 = 4063240
# idPVC = 3473416
# idART = 2883592
# idCO2 = 4784136

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

start_dt = datetime.combine(dates_np[0].date(), datetime.strptime(HORA_INICIO, "%H:%M:%S").time())
end_dt = datetime.combine(dates_np[0].date(), datetime.strptime(HORA_FIM, "%H:%M:%S").time())

mask = (dates_np >= start_dt) & (dates_np <= end_dt)

plt.plot(dates_np[mask], sig[mask])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.savefig(f"sp02_{LEITO}_{hora_inicio}_{hora_fim}.png")

print(sig[mask].shape)
spo2wave = resample(sig[mask], n_points)
spo2wave = np.reshape(spo2wave, (1, spo2wave.shape[0]))
print(spo2wave.shape)


np.savetxt(f"sp02wave_{LEITO}_{hora_inicio}_{hora_fim}.txt", spo2wave, fmt='%.7e')

plt.show()
