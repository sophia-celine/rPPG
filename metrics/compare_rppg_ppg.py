import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import os

def normalize_signal(s):
    """Normalização Z-score (média 0, desvio padrão 1)."""
    return (s - np.mean(s)) / np.std(s)

def sync_and_correlate(rppg_path, ppg_path, fs_rppg=25, fs_ppg=62.5):
    """
    Analisa a correlação entre rPPG e PPG com sincronização por cross-correlation.
    Retorna o coeficiente de Pearson e o melhor lag.
    """
    # 1. Carregamento dos dados
    try:
        raw_rppg = np.loadtxt(rppg_path)
        raw_ppg = np.loadtxt(ppg_path)

        method_name = os.path.basename(rppg_path).split('_')[1]

        # Tratamento para arquivos com múltiplas linhas (ex: Sinal, HR, Time)
        if raw_ppg.ndim > 1:
            raw_ppg = raw_ppg[0, :] if raw_ppg.shape[0] < raw_ppg.shape[1] else raw_ppg[:, 0]
        if raw_rppg.ndim > 1:
            raw_rppg = raw_rppg[0, :] if raw_rppg.shape[0] < raw_rppg.shape[1] else raw_rppg[:, 0]
        
        print(f"Pontos originais - rPPG: {len(raw_rppg)}, PPG: {len(raw_ppg)}")

        # Mostrar sinais originais sobrepostos no tempo real (sem sincronização)
        # plt.figure(figsize=(12, 4))
        # plt.plot(np.arange(len(raw_rppg))/fs_rppg, normalize_signal(raw_rppg), label='rPPG (Raw)', color='red', alpha=0.6)
        # plt.plot(np.arange(len(raw_ppg))/fs_ppg, normalize_signal(raw_ppg), label='PPG (Raw)', color='blue', alpha=0.6)
        # plt.title(f"Sinais Originais Sobrepostos - {method_name}\n(Eixo X em segundos, Amplitudes Normalizadas)")
        # plt.xlabel("Tempo (s)")
        # plt.legend()
        # plt.show()
            
    except Exception as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None, None, None, None

    # 2. Reamostragem baseada no tempo real (considerando frequências diferentes)
    fs_common = max(fs_rppg, fs_ppg)
    
    # Duração em segundos baseada na frequência de cada um
    dur_rppg = (len(raw_rppg) - 1) / fs_rppg
    dur_ppg = (len(raw_ppg) - 1) / fs_ppg
    
    t_rppg_orig = np.linspace(0, dur_rppg, len(raw_rppg))
    t_ppg_orig = np.linspace(0, dur_ppg, len(raw_ppg))
    
    # Novos eixos de tempo na frequência comum
    t_new_rppg = np.linspace(0, dur_rppg, int(dur_rppg * fs_common))
    t_new_ppg = np.linspace(0, dur_ppg, int(dur_ppg * fs_common))
    
    rppg_resampled = interp1d(t_rppg_orig, raw_rppg, kind='cubic')(t_new_rppg)
    ppg_resampled = interp1d(t_ppg_orig, raw_ppg, kind='cubic')(t_new_ppg)

    # 3. Normalização (Essencial para Cross-Correlation e Pearson)
    rppg_norm = normalize_signal(rppg_resampled)
    ppg_norm = normalize_signal(ppg_resampled)

    # 4. Cross-Correlation para Sincronização
    # A função correlate retorna a correlação para todos os possíveis deslocamentos (lags)
    correlation = signal.correlate(ppg_norm, rppg_norm, mode='full')
    lags = signal.correlation_lags(len(ppg_norm), len(rppg_norm), mode='full')
    
    # O lag ideal é onde a correlação é máxima
    best_lag = lags[np.argmax(correlation)]
    
    # 5. Alinhamento dos sinais com base no lag
    if best_lag > 0:
        # PPG está atrasado em relação ao rPPG
        synced_ppg = ppg_norm[best_lag:]
        synced_rppg = rppg_norm[:len(synced_ppg)]
    elif best_lag < 0:
        # rPPG está atrasado em relação ao PPG
        synced_rppg = rppg_norm[abs(best_lag):]
        synced_ppg = ppg_norm[:len(synced_rppg)]
    else:
        synced_rppg = rppg_norm
        synced_ppg = ppg_norm

    # Garantir que terminem com o mesmo tamanho após o corte
    min_len = min(len(synced_ppg), len(synced_rppg))
    synced_ppg = synced_ppg[:min_len]
    synced_rppg = synced_rppg[:min_len]

    # 6. Cálculo do Coeficiente de Pearson
    coef_pearson, p_valor = pearsonr(synced_ppg, synced_rppg)

    # =========================
    # Visualização
    # =========================
    # fig = plt.figure(figsize=(12, 10))
    
    # # Subplot 1: Sinais Originais (Normalizados para comparação visual)
    # plt.subplot(4, 1, 1)
    # plt.plot(rppg_norm, label='rPPG (Original)', color='red', alpha=0.6)
    # plt.plot(ppg_norm, label='PPG (Original)', color='blue', alpha=0.6)
    # plt.title("Etapa 1: Sinais Originais (Normalizados e Reamostrados)")
    # plt.legend()
    # plt.grid(True)

    # # Subplot 2: Resultado da Correlação Cruzada
    # plt.subplot(4, 1, 2)
    # plt.plot(lags, correlation, color='black')
    # plt.axvline(x=best_lag, color='green', linestyle='--', label=f'Melhor Lag: {best_lag}')
    # plt.title("Etapa 2: Correlação Cruzada (Busca pelo atraso ideal)")
    # plt.legend()
    # plt.grid(True)

    # # Subplot 3: Sinais Sincronizados
    # plt.subplot(4, 1, 3)
    # plt.plot(synced_rppg, label='rPPG Alinhado', color='red')
    # plt.plot(synced_ppg, label='PPG Alinhado', color='blue', linestyle='--')
    # plt.title(f"Etapa 3: Sinais Sincronizados (Lag aplicado: {best_lag})")
    # plt.legend()
    # plt.grid(True)

    # # Subplot 4: Gráfico de Dispersão (Scatter Plot) para Pearson
    # plt.subplot(4, 1, 4)
    # plt.scatter(synced_ppg, synced_rppg, alpha=0.3, s=10, color='purple')
    # plt.xlabel("PPG (Contact)")
    # plt.ylabel("rPPG (Remote)")
    # plt.title(f"Resultado Final: Coeficiente de Pearson = {coef_pearson:.4f}")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    print(f"\n--- Resultados da Análise ({os.path.basename(rppg_path)}) ---")
    print(f"Melhor Lag encontrado: {best_lag} pontos (@ {fs_common} Hz -> {best_lag/fs_common:.3f} s)")
    print(f"Coeficiente de Pearson: {coef_pearson:.4f}")
    print(f"P-valor: {p_valor:.4e}")
    
    return coef_pearson, best_lag, synced_rppg, synced_ppg

if __name__ == "__main__":
    # =========================
    # CONFIGURAÇÃO
    # =========================
    # Caminho para o seu arquivo PPG fixo (Ground Truth)
    PPG_PATH = "../get_ground_truth/spo2/original_spo2_L7_16-22-48_16-24-47.txt" 
    # PPG_PATH = r"../get_ground_truth/spo2/original_spo2_L9_16-05-26_16-07-25.txt" 
    # PPG_PATH = "/home/soph/rppg/rPPG/get_ground_truth/spo2/original_spo2_L9_16-05-26_16-07-25.txt"
    
    
    # Pasta contendo os arquivos rPPG (.txt)
    RPPG_FOLDER = "/home/soph/rppg/rPPG/preliminary_results/examples" 
    
    # Frequências de amostragem originais
    FS_RPPG = 25      # Câmera (rPPG)
    FS_PPG = 62.5     # Sensor (Ground Truth - ex: SpO2)

    # Geração de dados sintéticos para demonstração (caso os arquivos não existam)
    if not os.path.exists(PPG_PATH):
        print("Arquivos não encontrados. ")
    

    # =========================
    # PROCESSAMENTO EM LOTE
    # =========================
    if not os.path.exists(RPPG_FOLDER):
        print(f"Erro: Pasta {RPPG_FOLDER} não encontrada.")
    else:
        # Lista e ordena os arquivos para processamento organizado
        rppg_files = sorted([f for f in os.listdir(RPPG_FOLDER) if f.endswith('.txt')])
        summary_results = []
        
        for filename in rppg_files:
            rppg_full_path = os.path.join(RPPG_FOLDER, filename)
            print(f"\n" + "="*60)
            print(f"ANALISANDO: {filename}")
            
            pearson, lag, s_rppg, s_ppg = sync_and_correlate(rppg_full_path, PPG_PATH, fs_rppg=FS_RPPG, fs_ppg=FS_PPG)
            if pearson is not None:
                summary_results.append({
                    'name': filename.split('_')[1],
                    'pearson': pearson,
                    'lag': lag,
                    'rppg': s_rppg,
                    'ppg': s_ppg
                })

        # =========================
        # PLOT ADICIONAL (RESUMO DE SOBREPOSIÇÃO)
        # =========================
        if summary_results:
            num_files = len(summary_results)
            cols = 2
            rows = int(np.ceil(num_files / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
            axes = axes.flatten()
            
            for i, res in enumerate(summary_results):
                ax = axes[i]
                # Plotamos um trecho (ex: primeiros 10 segundos na fs_common) para melhor visibilidade
                fs_common = max(FS_RPPG, FS_PPG)
                limit = int(10 * fs_common)

                # Criar vetor de tempo para o trecho selecionado
                actual_limit = min(limit, len(res['ppg']))
                time_axis = np.arange(actual_limit) / fs_common
                
                ax.plot(time_axis, res['ppg'][:actual_limit], label='PPG', color='blue', alpha=0.6)
                ax.plot(time_axis, res['rppg'][:actual_limit], label='rPPG', color='red', alpha=0.8)
                
                ax.set_title(f"{res['name']}\nPearson: {res['pearson']:.4f}")
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Amplitude normalizada (u.a.)")
                # ax.tick_params(axis='both', labelsize=16)
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.5)

            # Remove subplots vazios se houver
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
                
            plt.tight_layout()
            plt.show()

        # Exibição do resumo comparativo final
        if summary_results:
            print("\n\n" + "="*45)
            print(f"{'ARQUIVO':<30} | {'PEARSON':<10} | {'LAG'}")
            print("-" * 50)
            for res in summary_results:
                print(f"{res['name']:<30} | {res['pearson']:.4f}     | {res['lag']}")