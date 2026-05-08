import os
import numpy as np
import matplotlib.pyplot as plt

def plot_first_n_samples(folder_path, ref_path=None, n_samples=500, fs=25, ref_fs=62.5, show_ref=False):
    """
    Lê todos os arquivos .txt de uma pasta e plota as primeiras n amostras em subplots.
    """
    # 1. Listar e ordenar arquivos .txt
    if not os.path.exists(folder_path):
        print(f"Erro: A pasta {folder_path} não existe.")
        return

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    if not files:
        print(f"Nenhum arquivo .txt encontrado em: {folder_path}")
        return

    num_plots = len(files)
    if ref_path and show_ref:
        num_plots += 1

    cols = 2
    rows = int(np.ceil(num_plots / cols))

    # 2. Configurar a figura
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0

    # Plotar o sinal de referência primeiro (ex: PPG real)
    if ref_path and os.path.exists(ref_path) and show_ref:
        ax = axes[plot_idx]
        try:
            data = np.loadtxt(ref_path)
            if data.ndim > 1:
                signal = data[0, :] if data.shape[0] < data.shape[1] else data[:, 0]
            else:
                signal = data

            # Calcular amostras da referência para bater com o tempo do rPPG
            print(len(signal))
            ref_n_samples = int((n_samples / fs) * ref_fs)
            plot_data = signal[:ref_n_samples]
            time_axis = np.arange(len(plot_data)) / ref_fs

            ax.plot(time_axis, plot_data, color='tab:red', linewidth=1.5)
            ax.set_title(f"Sinal de referência (PPG)", fontsize=10, fontweight='bold')
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle='--', alpha=0.5)
            plot_idx += 1
        except Exception as e:
            print(f"Erro ao carregar referência: {e}")

    print(f"Processando {len(files)} arquivos rPPG...")

    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        ax = axes[plot_idx]
        
        try:
            # Carregar dados usando numpy
            data = np.loadtxt(file_path)

            # Tratamento para arquivos com múltiplas dimensões (garantir sinal 1D)
            # Se for matriz, pega a dimensão mais longa como o sinal
            if data.ndim > 1:
                if data.shape[0] < data.shape[1]:
                    signal = data[0, :]
                else:
                    signal = data[:, 0]
            else:
                signal = data

            # Limitar às primeiras n amostras
            plot_data = signal[:n_samples]
            
            # Criar vetor de tempo em segundos
            time_axis = np.arange(len(plot_data)) / fs
            
            # Plotagem
            ax.plot(time_axis, plot_data, color='tab:blue', linewidth=1.5)
            ax.set_title(f"{filename.split('_')[1]}", fontsize=10)
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle='--', alpha=0.5)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Erro ao carregar:\n{filename}", 
                    ha='center', va='center', color='red')
            print(f"Erro no arquivo {filename}: {e}")
        
        plot_idx += 1

    # 3. Remover eixos de subplots vazios (se houver)
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # =========================
    # CONFIGURAÇÃO
    # =========================
    # Caminho para a pasta com seus resultados de BVP
    TARGET_FOLDER = r"../preliminary_results/examples"
    # Caminho para o sinal de referência (ex: Ground Truth de outra pasta)
    # REFERENCE_PATH = r"C:\Users\Sophia\Documents\rPPG\get_ground_truth\spo2\original_spo2_L9_16-05-26_16-07-25.txt"
    REFERENCE_PATH = r"..\get_ground_truth\spo2\original_spo2_L9_16-05-26_16-07-25"
    N_SAMPLES = 1000  # Quantidade de amostras para visualizar 
    FS = 25          # Frequência de amostragem (FPS da câmera)
    REF_FS = 62.5    # Frequência do sinal de referência (ex: SpO2)
    SHOW_REF = False

    plot_first_n_samples(TARGET_FOLDER, ref_path=REFERENCE_PATH, n_samples=N_SAMPLES, fs=FS, ref_fs=REF_FS, show_ref=SHOW_REF)