"""
Opens a .mat file containing data from the ECG/EEG device and saves a csv with the ecg point data
Crops the first 5 seconds of data to eliminate the ECG sincronization noise
fs = 1000 Hz
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = loadmat(r"C:\Users\Sophia\Documents\20260309_Coleta Vinicius\20260309_Coleta Vinicius\Coleta02.mat")
ch1 = data['Coleta02_Ch1']
ch1_struct = ch1[0, 0]
ecg_data = ch1_struct['values']#[740000:]

fs = 1000  
dt = 1 / fs 

time = np.arange(len(ecg_data)) * dt

plt.plot(ecg_data)
plt.xlabel('índice da amostra')
plt.ylabel('Amplitude')
plt.show()

max_val_idx = np.argmax(ecg_data)

init_idx = max_val_idx+5000

end_idx = max_val_idx + 122000

plot_data = ecg_data[init_idx:end_idx]

print(plot_data)

df = pd.DataFrame(plot_data)
df.to_csv(r'C:\Users\Sophia\Documents\rPPG\get_ground_truth\ECG\vinicius_video017_ecg.csv', index=False)

plot_time = time[init_idx:end_idx]

plt.plot(plot_time, plot_data)
plt.title('ECG')
plt.xlabel('Tempo (s)')
plt.show()