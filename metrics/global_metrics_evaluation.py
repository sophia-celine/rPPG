import os
import numpy as np
import heartpy as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


def estimate_hr_heartpy(segment, fs):
    """Estima a frequência cardíaca usando HeartPy."""
    try:
        wd, m = hp.process(segment, sample_rate=fs)
        return m['bpm']
    except Exception:
        return np.nan

def calculate_mape(y_true, y_pred):
    """Calcula o Erro Médio Absoluto Percentual."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_global_evaluation(patients_config, window_sec=15):
    """
    Calcula métricas globais agregando dados de múltiplos pacientes e métodos.
    
    patients_config: Lista de dicionários contendo 'gt_path', 'pred_folder' e 'fs'.
    """
    # Dicionário para armazenar listas de (y_true, y_pred) por método
    # Estrutura: { "METODO_NOME": {"true": [], "pred": []} }
    # Esta estrutura é para as métricas GLOBAIS (agregadas de todos os pacientes)
    global_data = {}

    for p_idx, config in enumerate(patients_config):
        gt_path = config['gt_path']
        pred_folder = config['pred_folder']
        fs_ecg = config['fs']
        patient_name = config.get('patient_name', f"P{p_idx + 1}")

        print(f"\n>>> Processando Paciente {patient_name}...")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_folder):
            print(f"Erro: Caminhos não encontrados para o paciente {patient_name}. Pulando...")
            continue

        # 1. Gerar Ground Truth do ECG
        sig = hp.get_data(gt_path)
        sig = hp.remove_baseline_wander(sig, fs_ecg)
        
        window_len = int(window_sec * fs_ecg)
        n_windows = len(sig) // window_len
        
        ecg_hr_values = []
        for i in range(n_windows):
            segment = sig[i * window_len : (i + 1) * window_len]
            ecg_hr_values.append(estimate_hr_heartpy(segment, fs_ecg))
        ecg_hr_values = np.array(ecg_hr_values)

        # 2. Ler arquivos de predição rPPG
        pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt')]
        
        for file_name in pred_files:
            # Identificar o método (ex: "POS" de "hr_POS_pred.txt")
            # A lógica aqui assume que o nome do método está após o primeiro '_'
            # Se seus arquivos forem nomeados de forma diferente (ex: "POS.txt"),
            # você pode precisar ajustar esta linha.
            # Ex: "POS.txt" -> method_name = "POS"
            method_name = os.path.splitext(file_name)[0].split('_')[1] if '_' in file_name else os.path.splitext(file_name)[0]
            
            if method_name not in global_data:
                global_data[method_name] = {"true": [], "pred": []}
            
            try:
                predictions = np.loadtxt(os.path.join(pred_folder, file_name))
                if predictions.ndim == 0: predictions = np.array([predictions])
                
                # Alinhar tamanhos
                min_len = min(len(ecg_hr_values), len(predictions))
                y_true = ecg_hr_values[:min_len]
                y_pred = predictions[:min_len]
                
                # Filtrar NaNs
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                
                # Adicionar ao acumulador global do método
                # Note: Aqui estamos estendendo as listas, o que significa que os dados
                # de diferentes pacientes para o mesmo método são concatenados.
                # Isso é o que queremos para as métricas globais.
                # Para o gráfico por paciente, precisaremos de uma estrutura diferente
                # que mantenha os dados separados por paciente.
                global_data[method_name]["true"].extend(y_true[mask])
                global_data[method_name]["pred"].extend(y_pred[mask])
                
            except Exception as e:
                print(f"Erro ao ler {file_name} do paciente {p_idx + 1}: {e}")

    # 3. Calcular Métricas Finais
    final_stats = []
    
    print("\n" + "="*70)
    print(f"{'MÉTODO':<15} | {'MAE (BPM)':<10} | {'RMSE (BPM)':<10} | {'MAPE (%)':<10} | {'N (Janelas)'}")
    print("-" * 70)

    for method, data in global_data.items():
        y_true_all = np.array(data["true"])
        y_pred_all = np.array(data["pred"])
        
        if len(y_true_all) == 0:
            continue
            
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mape = calculate_mape(y_true_all, y_pred_all)
        n_samples = len(y_true_all)
        
        final_stats.append({
            'Método': method,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'N': n_samples
        })
        
        print(f"{method:<15} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<10.2f} | {n_samples}")

    # # 4. Opcional: Salvar resultado em CSV
    # df_results = pd.DataFrame(final_stats)
    # df_results.to_csv("/home/soph/rppg/rPPG/metrics/global_results_summary.csv", index=False)
    # print("\nResultados salvos em: /home/soph/rppg/rPPG/metrics/global_results_summary.csv")

    # =========================================================
    # 4. Gerar Gráficos de Erro Absoluto por Janela para Cada Paciente
    # =========================================================
    print("\nGerando gráficos de Erro Absoluto por Janela para cada paciente...")

    # Estrutura: {p_idx: {method_name: (valid_window_indices_array, abs_errors_array)}}
    patient_errors_data = {}

    for p_idx, config in enumerate(patients_config):
        gt_path = config['gt_path']
        pred_folder = config['pred_folder']
        fs_ecg = config['fs']

        if not os.path.exists(gt_path) or not os.path.exists(pred_folder):
            continue # Já foi avisado antes

        sig = hp.get_data(gt_path)
        sig = hp.remove_baseline_wander(sig, fs_ecg)
        
        window_len = int(window_sec * fs_ecg)
        n_windows = len(sig) // window_len
        
        ecg_hr_values = []
        for i in range(n_windows):
            segment = sig[i * window_len : (i + 1) * window_len]
            ecg_hr_values.append(estimate_hr_heartpy(segment, fs_ecg))
        ecg_hr_values = np.array(ecg_hr_values)

        # Extrair identificador do paciente para o título do gráfico
        patient_id = config.get('patient_name', f"P{p_idx + 1}")
        # Capturar o tipo de método da configuração (default 'Geral' se não especificado)
        method_type = config.get('type', 'Geral')
        patient_errors_data[p_idx] = {'id': patient_id, 'type': method_type, 'methods': {}}
        
        pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.txt')]
        for file_name in pred_files:
            method_name = os.path.splitext(file_name)[0].split('_')[1] if '_' in file_name else os.path.splitext(file_name)[0]
            
            try:
                predictions = np.loadtxt(os.path.join(pred_folder, file_name))
                if predictions.ndim == 0:
                    predictions = np.array([predictions])
                
                min_len = min(len(ecg_hr_values), len(predictions))
                y_true = ecg_hr_values[:min_len]
                y_pred = predictions[:min_len]
                
                original_window_indices = np.arange(min_len)
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                
                valid_window_indices = original_window_indices[mask]
                abs_errors_for_valid_windows = np.abs(y_true[mask] - y_pred[mask])
                
                patient_errors_data[p_idx]['methods'][method_name] = (valid_window_indices, abs_errors_for_valid_windows)
                
            except Exception as e:
                print(f"Erro ao coletar erros por janela para {file_name} do paciente {p_idx + 1}: {e}")

    if patient_errors_data:
        num_patients = len(patients_config)
        cols_plot = 3
        rows_plot = int(np.ceil(num_patients / cols_plot))

        fig_mae_per_window, axes_mae_per_window = plt.subplots(
            rows_plot, cols_plot, figsize=(15, 5 * rows_plot), squeeze=False
        )
        axes_mae_per_window = axes_mae_per_window.flatten()

        for p_idx, patient_data_entry in patient_errors_data.items():
            ax = axes_mae_per_window[p_idx]
            patient_label = f"Paciente {patient_data_entry['id']} (Métodos {patient_data_entry['type']})"
            patient_methods_data = patient_data_entry['methods']

            max_window_idx_for_patient = -1
            has_valid_data_for_patient = False
            for method_name, (indices, errors) in patient_methods_data.items():
                if len(indices) > 0:
                    max_window_idx_for_patient = max(max_window_idx_for_patient, np.max(indices))
                    has_valid_data_for_patient = True
            
            if has_valid_data_for_patient and max_window_idx_for_patient >= 0:
                all_method_errors_matrix = np.full((len(patient_methods_data), max_window_idx_for_patient + 1), np.nan)

                method_row_counter = 0
                for method_name, (indices, errors) in patient_methods_data.items():
                    for i, window_idx in enumerate(indices):
                        all_method_errors_matrix[method_row_counter, window_idx] = errors[i]
                    method_row_counter += 1

                mean_abs_error_per_window = np.nanmean(all_method_errors_matrix, axis=0)

                valid_mean_error_windows_mask = ~np.isnan(mean_abs_error_per_window)
                plot_windows = np.arange(len(mean_abs_error_per_window))[valid_mean_error_windows_mask]
                plot_mean_errors = mean_abs_error_per_window[valid_mean_error_windows_mask]

                if len(plot_windows) > 0:
                    # ax.plot(plot_windows, plot_mean_errors, label=f"Média dos métodos {patient_data_entry['type']}", color='black', linewidth=2)
                    ax.plot(plot_windows, plot_mean_errors, color='black', linewidth=2)
                else:
                    ax.text(0.5, 0.5, "Nenhum dado válido para média", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Nenhum dado de método para este paciente", ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"{patient_label}")
            ax.set_xlabel("Janela de Tempo")
            ax.set_ylabel("Erro Absoluto Médio (bpm)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, linestyle='--', alpha=0.7)

        for j in range(num_patients, len(axes_mae_per_window)):
            fig_mae_per_window.delaxes(axes_mae_per_window[j])

        fig_mae_per_window.tight_layout()
        plt.show()

if __name__ == "__main__":
    # =========================================================
    # CONFIGURAÇÃO DE MULTI-PACIENTES
    # Adicione aqui os caminhos para cada "caso" de teste
    # =========================================================
    DATA_CONFIG = [
        {
            "patient_name": "3",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L8_16-45-38_16-47-38.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L8/dl_hr_preds",
            "fs": 250,
            "type": "supervisionados"
        },
        {
            "patient_name": "2",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L9_16-05-26_16-07-25.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L9/dl_hr_preds",
            "fs": 250,
            "type": "supervisionados"
        },
        {
            "patient_name": "1",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L7_16-22-48_16-24-47.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L7/dl_hr_preds",
            "fs": 250,
            "type": "supervisionados"
        },
        {
            "patient_name": "3",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L8_16-45-38_16-47-38.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L8/hr_preds",
            "fs": 250,
            "type": "não supervisionados"
        },
        {
            "patient_name": "2",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L9_16-05-26_16-07-25.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L9/hr_preds",
            "fs": 250,
            "type": "não supervisionados"
        },
        {
            "patient_name": "1",
            "gt_path": "/home/soph/rppg/rPPG/get_ground_truth/ECG/ecg_signal_L7_16-22-48_16-24-47.csv",
            "pred_folder": "/home/soph/rppg/rPPG/preliminary_results/L7/hr_preds",
            "fs": 250,
            "type": "não supervisionados"
        },
    ]
    
    # Tamanho da janela (deve ser o mesmo usado na geração das predições)
    WINDOW_SIZE = 15

    run_global_evaluation(DATA_CONFIG, window_sec=WINDOW_SIZE)