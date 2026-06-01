import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Aplica um filtro passa-banda Butterworth."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def _next_power_of_2(x):
    """Calcula a próxima potência de 2 para otimizar a FFT e aumentar a resolução."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def load_signal(file_path):
    """Carrega o sinal de um arquivo txt, tratando casos 1D ou 2D."""
    data = np.loadtxt(file_path)
    if data.ndim > 1:
        # Se for matriz, assume que a dimensão maior é o tempo
        return data[0, :] if data.shape[0] < data.shape[1] else data[:, 0]
    return data

def plot_spectral_analysis(folder_paths, fs=25.0, lowcut=0.6, highcut=3.3):
    """
    Gera duas figuras: uma com espectrogramas e outra com espectros de potência.
    folder_paths: pode ser uma string (caminho único) ou uma lista de strings.
    """
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    all_files = []
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"Erro: A pasta {folder} não existe.")
            continue
        
        # Listar e ordenar arquivos de cada pasta
        folder_files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        for f in folder_files:
            all_files.append((folder, f))

    if not all_files:
        print(f"Nenhum arquivo .txt encontrado nas pastas fornecidas.")
        return

    num_files = len(all_files)
    cols = 2
    rows = int(np.ceil(num_files / cols))

    # --- FIGURA 1: Espectrogramas ---
    fig_spec, axes_spec = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes_spec = axes_spec.flatten()

    # --- FIGURA 2: Espectro de Potência (PSD) ---
    fig_psd, axes_psd = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes_psd = axes_psd.flatten()

    print(f"Processando {num_files} arquivos...")

    for i, (folder, filename) in enumerate(all_files):
        file_path = os.path.join(folder, filename)
        
        # Identificar o método e a pasta (se houver mais de uma)
        method_name = filename.split('_')[1] if '_' in filename else filename
        folder_label = os.path.basename(folder.rstrip(os.sep)) if len(folder_paths) > 1 else ""
        display_name = f"{folder_label}: {method_name}" if folder_label else method_name

        try:
            sig = load_signal(file_path)
            # Normalização e Filtragem
            sig = sig - np.mean(sig)
            sig_filt = bandpass_filter(sig, lowcut, highcut, fs)
            
            # Definir NFFT para alta resolução (Zero-padding)
            nfft_val = _next_power_of_2(len(sig_filt)) * 2

            # 1. Cálculo do Espectrograma
            # Usamos nfft para suavizar a imagem do espectrograma
            f, t, Sxx = signal.spectrogram(sig_filt, fs=fs, nperseg=min(len(sig_filt), 256), 
                                           noverlap=128, nfft=nfft_val)
            
            ax_s = axes_spec[i]
            im = ax_s.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='jet')
            ax_s.set_title(f"{display_name}")
            ax_s.set_xlabel("Tempo (s)")
            ax_s.set_ylabel("Frequência (Hz)")
            ax_s.set_ylim(lowcut, highcut)
            fig_spec.colorbar(im, ax=ax_s, label='dB')

            # 2. Cálculo do Espectro de Potência (Welch ou Periodograma)
            # O Welch minimiza o ruído através da média de segmentos com janelamento
            freqs, psd = signal.welch(sig_filt, fs=fs, nperseg=min(len(sig_filt), 256), 
                                      nfft=nfft_val, window='hamming')
            
            ax_p = axes_psd[i]
            ax_p.plot(freqs, psd, color='tab:green', linewidth=2)
            
            # Encontrar e marcar o pico principal (Frequência Cardíaca estimada)
            mask = (freqs >= lowcut) & (freqs <= highcut)
            if np.any(mask):
                peak_idx = np.argmax(psd[mask])
                peak_f = freqs[mask][peak_idx]
                ax_p.axvline(peak_f, color='red', linestyle='--', alpha=0.6)
                ax_p.text(peak_f + 0.1, np.max(psd)*0.8, f"{peak_f*60:.1f} BPM", color='red')

            ax_p.set_title(f"{display_name}")
            ax_p.set_xlabel("Frequência (Hz)")
            ax_p.set_ylabel("Densidade Espectral")
            ax_p.set_xlim(0, highcut + 1)
            ax_p.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            axes_spec[i].text(0.5, 0.5, "Erro", ha='center')
            axes_psd[i].text(0.5, 0.5, "Erro", ha='center')

    # Remover eixos extras
    for j in range(num_files, len(axes_spec)):
        fig_spec.delaxes(axes_spec[j])
        fig_psd.delaxes(axes_psd[j])

    fig_spec.tight_layout()
    fig_psd.tight_layout()
    
    print("Exibindo gráficos...")
    plt.show()

if __name__ == "__main__":
    # =========================
    # CONFIGURAÇÃO
    # =========================
    # Defina aqui uma ou mais pastas para análise
    # Para usar apenas uma pasta: FOLDERS = ["/caminho/pasta1"]
    # Para usar duas pastas: FOLDERS = ["/caminho/pasta1", "/caminho/pasta2"]
    
    FOLDERS = [
        "../preliminary_results/L9/bvp",
        # "/home/soph/rppg/rPPG/preliminary_results/L9/bvp"
        # "/home/soph/rppg/rPPG/preliminary_results/L9/outra_pasta" # Exemplo de segunda pasta
    ]
    
    # Frequência de amostragem da câmera
    SAMPLING_RATE = 25 
    
    # Banda de interesse (0.6 a 3.3 Hz corresponde a ~36 a 198 BPM)
    LOW_HZ = 0.6
    HIGH_HZ = 3.3

    plot_spectral_analysis(
        FOLDERS, 
        fs=SAMPLING_RATE, 
        lowcut=LOW_HZ, 
        highcut=HIGH_HZ
    )