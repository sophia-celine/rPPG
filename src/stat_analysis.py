
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

# arquivo = Path(r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp_mov_obs.csv")
arquivo = Path(r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp_mov_NoOutliers.csv")

# Nomes das métricas na primeira coluna do CSV
METRICAS = {
    "mae": "mae_hr_POS",
    "movimento": "movement_index",
    "iluminacao": "Iluminação (lux)",
    "fototipo": "Fototipo",
}

# ============================================================
# 2. LEITURA DO CSV
# ============================================================

df = pd.read_csv(
    arquivo,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

# Remove espaços dos nomes das colunas e das métricas
df.columns = df.columns.astype(str).str.strip()

coluna_metrica = df.columns[0]
df[coluna_metrica] = df[coluna_metrica].astype(str).str.strip()

# Busca os nomes das métricas sem diferenciar maiúsculas/minúsculas
# ou acentos. Isso permite reconhecer "Iluminação (lux)".
def normalizar(texto):
    import unicodedata

    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(
        c for c in texto if not unicodedata.combining(c)
    )
    return texto.strip().lower()


nomes_normalizados = {
    normalizar(nome): nome
    for nome in df[coluna_metrica]
}

linhas = {}

for chave, nome_esperado in METRICAS.items():
    chave_normalizada = normalizar(nome_esperado)

    if chave_normalizada not in nomes_normalizados:
        raise ValueError(
            f"Métrica '{nome_esperado}' não encontrada.\n"
            f"Métricas disponíveis: "
            f"{df[coluna_metrica].tolist()}"
        )

    nome_real = nomes_normalizados[chave_normalizada]
    linha = df.loc[df[coluna_metrica] == nome_real]

    if len(linha) != 1:
        raise ValueError(
            f"A métrica '{nome_real}' aparece mais de uma vez."
        )

    linhas[chave] = linha.iloc[0, 1:]

# ============================================================
# 3. CONSTRUIR UMA LINHA POR COLETA
# ============================================================

dados = pd.DataFrame({
    "coleta": linhas["mae"].index.astype(str),
    "mae_hr_POS": linhas["mae"].values,
    "movement_index": linhas["movimento"].values,
    "iluminacao": linhas["iluminacao"].values,
    "fototipo": linhas["fototipo"].values,
})

# Converter números, aceitando decimais com vírgula
def converter_numero(serie):
    serie = serie.astype(str).str.strip()
    serie = serie.str.replace(",", ".", regex=False)
    return pd.to_numeric(serie, errors="coerce")


for coluna in ["mae_hr_POS", "movement_index", "iluminacao"]:
    dados[coluna] = converter_numero(dados[coluna])

dados["fototipo"] = (
    dados["fototipo"]
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({"NAN": np.nan, "": np.nan})
)

# ============================================================
# 4. IDENTIFICAR O PACIENTE
# ============================================================

# Exemplo:
# 2357983_0 -> paciente 2357983
# 2357983_1 -> paciente 2357983
#
# Considera que o último sufixo numérico representa
# o identificador da coleta dentro do paciente.

def extrair_paciente(coleta):
    match = re.match(r"^(.+)_\d+$", coleta)

    if match:
        return match.group(1)

    warnings.warn(
        f"Não foi possível extrair o paciente de '{coleta}'. "
        "O identificador completo será usado como paciente."
    )
    return coleta


dados["paciente"] = dados["coleta"].map(extrair_paciente)

# Remover coletas sem dados completos nas variáveis do modelo
colunas_modelo = [
    "paciente",
    "mae_hr_POS",
    "movement_index",
    "iluminacao",
    "fototipo",
]

n_original = len(dados)
dados = dados.dropna(subset=colunas_modelo).copy()

if (dados["mae_hr_POS"] < 0).any():
    raise ValueError("Foram encontrados MAEs negativos.")

if (dados["movement_index"] < 0).any():
    raise ValueError("Foram encontrados índices de movimento negativos.")

if (dados["iluminacao"] <= 0).any():
    raise ValueError("Foram encontradas iluminações não positivas.")

print("\n=== Dados preparados ===")
print(dados.to_string(index=False))
print(f"\nColetas inicialmente encontradas: {n_original}")
print(f"Coletas completas utilizadas: {len(dados)}")
print(f"Pacientes distintos: {dados['paciente'].nunique()}")

print("\nColetas por paciente:")
print(dados.groupby("paciente").size().to_string())

print("\nPacientes por fototipo:")
print(
    dados.groupby("fototipo", observed=True)["paciente"]
    .nunique()
    .to_string()
)

# ============================================================
# 5. PREPARAR AS VARIÁVEIS EXPLICATIVAS
# ============================================================

# Padronizar movimento e iluminação:
# coeficientes passam a representar a variação de MAE
# associada a um desvio-padrão da variável explicativa.

for coluna in ["movement_index", "iluminacao"]:
    desvio = dados[coluna].std()

    if desvio == 0 or pd.isna(desvio):
        raise ValueError(
            f"A variável '{coluna}' não apresenta variabilidade."
        )

    dados[coluna + "_z"] = (
        dados[coluna] - dados[coluna].mean()
    ) / desvio

# Usa somente os fototipos observados.
# Categorias sem nenhuma observação não podem ter seu efeito estimado.
fototipos_presentes = sorted(dados["fototipo"].unique())

if len(fototipos_presentes) < 2:
    raise ValueError(
        "São necessários pelo menos dois fototipos observados."
    )

# Escolher referência presente nos dados.
# Prioriza IV; se não houver IV, usa o fototipo mais frequente.
if "IV" in fototipos_presentes:
    referencia = "IV"
else:
    referencia = dados["fototipo"].value_counts().idxmax()

print(f"\nFototipos observados: {fototipos_presentes}")
print(f"Fototipo de referência: {referencia}")

if "II" not in fototipos_presentes:
    print(
        "ATENÇÃO: não há coletas do fototipo II. "
        "Seu efeito não poderá ser estimado neste ajuste."
    )

# ============================================================
# 6. AJUSTAR O MODELO LINEAR MISTO
# ============================================================

formula = (
    "mae_hr_POS ~ movement_index_z + iluminacao_z "
    f'+ C(fototipo, Treatment(reference="{referencia}"))'
)

modelo = smf.mixedlm(
    formula=formula,
    data=dados,
    groups=dados["paciente"],
    re_formula="1",
)

resultado = modelo.fit(
    reml=True,
    method="lbfgs",
)

print("\n=== Resultado do modelo linear misto ===")
print(resultado.summary())

print("\nConvergiu:", resultado.converged)
print(
    "Variância estimada entre pacientes:",
    float(resultado.cov_re.iloc[0, 0])
)
print("Variância residual:", resultado.scale)


# ============================================================
# 7. DIAGNÓSTICO DOS RESÍDUOS
# ============================================================

# Fitted values e resíduos condicionais:
# incluem o efeito aleatório estimado de cada paciente.

dados["ajustado"] = np.asarray(resultado.fittedvalues)
dados["residuo"] = np.asarray(resultado.resid)

# Resíduos padronizados para facilitar a visualização
desvio_residual = np.sqrt(resultado.scale)
dados["residuo_padronizado"] = (
    dados["residuo"] / desvio_residual
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Resíduos versus valores ajustados
axes[0, 0].scatter(
    dados["ajustado"],
    dados["residuo_padronizado"],
)
axes[0, 0].axhline(0, color="red", linestyle="--")
axes[0, 0].set_xlabel("Valores ajustados (bpm)")
axes[0, 0].set_ylabel("Resíduos padronizados")
axes[0, 0].set_title("Resíduos versus valores ajustados")

# Q-Q plot dos resíduos
stats.probplot(
    dados["residuo_padronizado"],
    dist="norm",
    plot=axes[0, 1],
)
axes[0, 1].set_title("Q-Q plot dos resíduos")

# Histograma dos resíduos
axes[1, 0].hist(
    dados["residuo_padronizado"],
    bins="auto",
    edgecolor="black",
)
axes[1, 0].set_xlabel("Resíduos padronizados")
axes[1, 0].set_ylabel("Frequência")
axes[1, 0].set_title("Distribuição dos resíduos")

# Resíduos por paciente
pacientes = dados["paciente"].astype("category")
codigos = pacientes.cat.codes

axes[1, 1].scatter(
    codigos,
    dados["residuo_padronizado"],
)
axes[1, 1].axhline(0, color="red", linestyle="--")
axes[1, 1].set_xlabel("Paciente (código)")
axes[1, 1].set_ylabel("Resíduos padronizados")
axes[1, 1].set_title("Resíduos por paciente")

fig.tight_layout()
plt.show()

# ============================================================
# MODELO LOG
# ============================================================
# Todos os valores devem ser estritamente positivos
if (dados["mae_hr_POS"] <= 0).any():
    raise ValueError(
        "O log(MAE) exige MAEs estritamente positivos."
    )

dados["log_mae"] = np.log(dados["mae_hr_POS"])

formula_log = (
    "log_mae ~ movement_index_z + iluminacao_z "
    f'+ C(fototipo, Treatment(reference="{referencia}"))'
)

modelo_log = smf.mixedlm(
    formula=formula_log,
    data=dados,
    groups=dados["paciente"],
    re_formula="1",
)

resultado_log = modelo_log.fit(
    reml=True,
    method="lbfgs",
)

print(resultado_log.summary())
print("\nConvergiu:", resultado_log.converged)

# Diagnósticos dos resíduos condicionais
dados["ajustado_log"] = resultado_log.fittedvalues
dados["residuo_log"] = resultado_log.resid
dados["residuo_log_pad"] = (
    dados["residuo_log"] / np.sqrt(resultado_log.scale)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    dados["ajustado_log"],
    dados["residuo_log_pad"]
)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Valores ajustados de log(MAE)")
axes[0].set_ylabel("Resíduos padronizados")
axes[0].set_title("Resíduos versus valores ajustados")

stats.probplot(
    dados["residuo_log_pad"],
    dist="norm",
    plot=axes[1]
)
axes[1].set_title("Q-Q plot dos resíduos")

plt.tight_layout()
plt.show()

# ============================================================
# 8. EXPORTAR DADOS E RESÍDUOS
# ============================================================

dados.to_csv(
    "diagnostico_modelo_POS.csv",
    index=False,
    decimal=",",
)

print(
    "\nTabela com dados preparados, valores ajustados e resíduos "
    "salva em 'diagnostico_modelo_POS.csv'."
)
