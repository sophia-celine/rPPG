"""
Opens a .mat file containing data from the ECG/EEG device and saves a csv with the ecg point data
Crops the first 5 seconds of data to eliminate the ECG sincronization noise
fs = 1000 Hz
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = loadmat(r"C:\Users\Sophia\Documents\20260423_Coleta Vinicius\20260423_Coleta Vinicius\Coleta01.mat")
print(data.keys())
ch1 = data['Data1__Stopped__Ch1']
ch1_struct = ch1[0, 0]
ecg_data = ch1_struct['values'][1500000:1700000]
save_ecg = True
noise_up = False
video_duration = 120 # seconds

fs = 1000  
dt = 1 / fs 

time = np.arange(len(ecg_data)) * dt

plt.plot(ecg_data)
plt.xlabel('índice da amostra')
plt.ylabel('Amplitude')
plt.show()

max_val_idx = np.argmax(ecg_data)
min_val_idx = np.argmin(ecg_data)

if noise_up:
    init_idx = max_val_idx+5000
else:
    init_idx = min_val_idx+5000


end_idx = max_val_idx + video_duration*fs

plot_data = ecg_data[init_idx:end_idx]

print(plot_data)

df = pd.DataFrame(plot_data)
if save_ecg: df.to_csv(r'C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video023_ecg.csv', index=False)

plot_time = time[init_idx:end_idx]

plt.plot(plot_time, plot_data)
plt.title('ECG')
plt.xlabel('Tempo (s)')
plt.show()