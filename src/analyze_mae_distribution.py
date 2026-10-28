
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================
# Configuração
# ============================================================

arquivo = Path(r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp_mov_obs.csv")
nome_metrica = "mae_hr_POS"

# ============================================================
# 1. Ler a planilha e localizar a linha do MAE do POS
# ============================================================

df = pd.read_csv(arquivo, header=0)

# A primeira coluna contém os nomes das métricas
coluna_metrica = df.columns[0]

df[coluna_metrica] = df[coluna_metrica].astype(str).str.strip()

linha = df.loc[
    df[coluna_metrica] == nome_metrica
]

if linha.empty:
    raise ValueError(
        f"Métrica '{nome_metrica}' não encontrada na planilha."
    )

if len(linha) > 1:
    raise ValueError(
        f"Foi encontrada mais de uma linha '{nome_metrica}'."
    )

# Cada coluna, exceto a primeira, representa uma coleta
valores_originais = linha.iloc[0, 1:]

# O CSV usa vírgula como separador decimal, por exemplo "0,4726".
valores_normalizados = (
    valores_originais.astype("string")
    .str.strip()
    .str.replace(",", ".", regex=False)
)
valores = pd.to_numeric(valores_normalizados, errors="coerce")

# Excluir valores ausentes, preservando o identificador da coleta
mae = valores.dropna()
mae.name = "MAE_POS"

# ============================================================
# 2. Conferir os dados extraídos
# ============================================================

print("\n=== MAE do algoritmo POS por coleta ===")
print(mae.to_string())

print("\n=== Valores ausentes ou não numéricos ===")
print(valores.isna().sum())

if len(mae) < 3:
    raise ValueError(
        "São necessárias pelo menos 3 observações válidas "
        "para esta análise."
    )

if (mae < 0).any():
    raise ValueError("Foram encontrados MAEs negativos.")

# ============================================================
# 3. Estatísticas descritivas
# ============================================================

print("\n=== Estatísticas descritivas ===")
print(f"Número de coletas: {len(mae)}")
print(f"Média:              {mae.mean():.4f} bpm")
print(f"Mediana:            {mae.median():.4f} bpm")
print(f"Desvio-padrão:      {mae.std():.4f} bpm")
print(f"Mínimo:             {mae.min():.4f} bpm")
print(f"1º quartil (Q1):    {mae.quantile(0.25):.4f} bpm")
print(f"3º quartil (Q3):    {mae.quantile(0.75):.4f} bpm")
print(f"Máximo:             {mae.max():.4f} bpm")
print(f"Assimetria:         {stats.skew(mae):.4f}")
print(f"Curtose excedente:  {stats.kurtosis(mae):.4f}")

# ============================================================
# 4. Teste de normalidade de Shapiro-Wilk
# ============================================================

if len(mae) <= 5000:
    estatistica, p_valor = stats.shapiro(mae)

    print("\n=== Teste de Shapiro-Wilk ===")
    print(f"Estatística W: {estatistica:.4f}")
    print(f"Valor-p:       {p_valor:.6f}")

    if p_valor < 0.05:
        print(
            "Resultado: há evidência contra a normalidade "
            "dos valores brutos (nível de 5%)."
        )
    else:
        print(
            "Resultado: não foi possível rejeitar a normalidade "
            "dos valores brutos (nível de 5%)."
        )

# ============================================================
# 5. Gráficos para avaliar a distribuição
# ============================================================

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Histograma com curva de densidade
sns.histplot(
    mae,
    kde=True,
    stat="density",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Distribuição do MAE do POS")
axes[0, 0].set_xlabel("MAE (bpm)")
axes[0, 0].set_ylabel("Densidade")

# Boxplot
sns.boxplot(x=mae, ax=axes[0, 1])
axes[0, 1].set_title("Boxplot do MAE do POS")
axes[0, 1].set_xlabel("MAE (bpm)")

# Gráfico Q-Q para uma distribuição normal
stats.probplot(mae, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Gráfico Q-Q: comparação com a normal")

# Valores por coleta, identificados pelo cabeçalho da coluna
axes[1, 1].scatter(
    range(len(mae)),
    mae.values
)
axes[1, 1].set_title("MAE por coleta")
axes[1, 1].set_xlabel("Coleta (ordem na planilha)")
axes[1, 1].set_ylabel("MAE (bpm)")

fig.tight_layout()
plt.show()
