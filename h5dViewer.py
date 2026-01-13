import h5py
import numpy as np
import matplotlib.pyplot as plt
import time


# Substitua pelo caminho do seu arquivo HDF5
file_path = "10.10.10.138_20251211_16.h5"

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
                data_len = raw_data[local_data_add+3] * 2
                if data_len > 0:
                    if local_data_add + 4 + data_len <= len(raw_data):
                        data = raw_data[local_data_add+4:local_data_add+4+data_len]
                    else:
                        data = raw_data[local_data_add+4:]
                    data = np.frombuffer(data, dtype='>i2')  # Converte para np.array de inteiros
                    datas.append(data)
                    ids.append(int.from_bytes(data_head[0:3], byteorder='big'))
                    seqs.append(frame_seq)
                    npk += 1
                    local_data_add = local_data_add + 4 + data_len
                else:
                    process_next = False
            else:
                process_next = False


seqs = np.array(seqs)
ids = np.array(ids)
uniqIDs = np.unique(ids)
uniqseqs = np.unique(seqs)



idECGs = [65540, 65796, 66052, 66308, 66564, 66820, 67076, 67332, 67588, 67844, 68100, 68356]

nplot = 1;
for ECGsID in idECGs:
    if ECGsID in uniqIDs:
        indices = np.where(ids == ECGsID)[0]
        sig = [datas[i] for i in indices]
        sig = np.concatenate(sig)
        seq = [seqs[i] for i in indices]
        print(np.sum(np.diff(seq) > 1)/len(seq))
        plt.subplot(4,3,nplot)
        plt.plot(sig)
    nplot+=1
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

plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(seq)
plt.show()


A = np.diff(seq)
np.sum(A > 1)
