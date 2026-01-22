import h5py
import matplotlib.pyplot as plt
import numpy as np

file = "10.10.10.129_20251211_16.h5"
hdf = h5py.File(file, 'r')

# Definitions
DATA_PACK_HEAD = b"\x02\x0B\x00\x00"  # Defined as bytes
data_add = 36

data = hdf['data'][:]
data_timestamps = hdf['data_timestamps'][:]

datas = []
ids = []
seqs = []
seqsts = []

# Data processing
for raw_data, ts in zip(data, data_timestamps):
    pack_id = bytes(raw_data[0:4])
    if pack_id == DATA_PACK_HEAD:
        frame_len = int.from_bytes(raw_data[4:6], byteorder='big')
        frame_seq = int.from_bytes(raw_data[24:26], byteorder='big')
        npk = 0
        process_next = True
        local_data_add = data_add  # Use a local variable to avoid overwriting the original
        while process_next:
            if local_data_add + 4 <= len(raw_data):
                data_head = raw_data[local_data_add:local_data_add+4]
                data_len = raw_data[local_data_add+3] * 2
                if data_len > 0:
                    if local_data_add + 4 + data_len <= len(raw_data):
                        data_bytes = raw_data[local_data_add+4:local_data_add+4+data_len]
                    else:
                        data_bytes = raw_data[local_data_add+4:]
                    data_array = np.frombuffer(data_bytes, dtype='>i2')  # Convert to np.array of integers
                    datas.append(data_array)
                    ids.append(int.from_bytes(data_head[0:3], byteorder='big'))
                    seqs.append(frame_seq)
                    seqsts.append(ts)
                    npk += 1
                    local_data_add = local_data_add + 4 + data_len
                else:
                    process_next = False
            else:
                process_next = False
                

seqs = np.array(seqs)
ids = np.array(ids)
seqsts = np.array(seqsts)

######

idECGs =  65540               
idSpO2 = 458768
idResp = 327688
idP1 = 4063240
idPVC = 3473416
idART = 2883592
idCO2 = 4784136


indices = np.where(ids == idECGs)[0]
Fs = np.round(len(datas[indices[0]])/np.median(np.diff(seqsts)))
dt = 1/Fs
dtSeq = dt * len(datas[indices[0]])
print(f"Fs = {Fs}")



plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(np.diff(timestamp))
plt.show()


