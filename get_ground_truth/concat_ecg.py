import pandas as pd
import matplotlib.pyplot as plt

arquivo1 = r"C:\Users\Sophia\Documents\rPPG_data\ground_truth\ECG\ecg_signal_20-08-2026_L6_15-50-00_16-00-00.csv"
arquivo2 = r"C:\Users\Sophia\Documents\rPPG_data\ground_truth\ECG\ecg_signal_20-08-2026_L6_16-00-00_16-10-00.csv"

df1 = pd.read_csv(arquivo1, header=None)
df2 = pd.read_csv(arquivo2, header=None)

ecg1 = df1.iloc[:, 0].to_numpy()
ecg2 = df2.iloc[:, 0].to_numpy()

ecg_total = pd.concat(
    [pd.Series(ecg1), pd.Series(ecg2)],
    ignore_index=True
).to_numpy()

ponto_concatenacao = len(ecg1)

plt.figure(figsize=(15, 5))

plt.plot(ecg_total, linewidth=0.8)
# plt.axvline(
#     ponto_concatenacao,
#     linestyle="--",
#     linewidth=1.5,
#     label="Início do segundo arquivo"
# )

plt.xlabel("Amostra")
plt.ylabel("Amplitude")
plt.title("ECG concatenado")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()