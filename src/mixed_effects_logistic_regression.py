
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ARQUIVO_CSV = Path(r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp_mov_obs.csv")
PASTA_SAIDA = Path(r"C:\Users\Sophia\Documents\rPPG_data\results\resultados_comOutliers")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

LIMIAR_MAE = 5.0

# Este arquivo usa vírgula como separador de colunas. Os campos com
# decimais em vírgula estão entre aspas no CSV.
SEPARADOR_CSV = ","
SEPARADOR_DECIMAL = ","

# Nomes possíveis para as linhas da planilha original.
# O script procura os nomes após normalizar maiúsculas,
# acentos e espaços.
ALIASES = {
    "paciente": [
        "paciente",
        "id_paciente",
        "patient",
        "patient_id",
        "codigo_paciente",
        "id_patient",
    ],
    "mae_hr_POS": [
        "mae_hr_POS",
        "mae_hr_pos",
        "mae_pos",
        "mae_hr",
    ],
    "movement_index": [
        "movement_index",
        "indice_movimento",
        "índice_movimento",
        "movement",
    ],
    "iluminacao": [
        "iluminacao",
        "iluminação",
        "Iluminação (lux)",
        "iluminacao_lux",
        "lux",
        "illuminance",
        "illumination",
    ],
    "fototipo": [
        "fototipo",
        "foto_tipo",
        "skin_type",
        "skin_phototype",
    ],
}

# Fototipo de referência da regressão.
FOTOTIPO_REFERENCIA = "IV"

# Use uma lista de identificadores de coleta apenas se quiser
# fazer uma análise de sensibilidade excluindo coletas cuja
# referência de ECG foi considerada não confiável.
#
# Os identificadores são os nomes das colunas do CSV original.
COLETAS_REFERENCIA_NAO_CONFIAVEL = []

# IMPORTANTE:
# Não coloque aqui coletas apenas por apresentarem MAE elevado.
# A exclusão deve ser baseada na qualidade da referência,
# avaliada independentemente do resultado do POS.


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def normalizar_nome(valor):
    """Normaliza nomes para facilitar a correspondência."""
    texto = str(valor).strip().lower()

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def localizar_linha(tabela, aliases, nome_variavel):
    """Localiza uma linha da planilha usando aliases."""

    indice_normalizado = {
        normalizar_nome(indice): indice
        for indice in tabela.index
    }

    for alias in aliases:
        chave = normalizar_nome(alias)

        if chave in indice_normalizado:
            return indice_normalizado[chave]

    raise KeyError(
        f"Não encontrei a linha de '{nome_variavel}'.\n"
        f"Aliases procurados: {aliases}\n"
        f"Linhas disponíveis no CSV: "
        f"{list(tabela.index)}"
    )


def converter_numerico(serie):
    """Converte valores numéricos e trata strings vazias."""
    valores = (
        serie.astype("string")
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
        })
    )
    if SEPARADOR_DECIMAL != ".":
        valores = valores.str.replace(
            SEPARADOR_DECIMAL,
            ".",
            regex=False,
        )

    return pd.to_numeric(
        valores,
        errors="coerce",
    )


# ============================================================
# 3. LEITURA DA PLANILHA ORIGINAL
# ============================================================

if not ARQUIVO_CSV.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_CSV.resolve()}\n"
        "Altere ARQUIVO_CSV para o caminho correto."
    )

tabela = pd.read_csv(
    ARQUIVO_CSV,
    sep=SEPARADOR_CSV,
    decimal=SEPARADOR_DECIMAL,
    index_col=0,
    dtype=str,
    encoding="utf-8-sig",
)

tabela.index = tabela.index.astype(str).str.strip()
tabela.columns = tabela.columns.astype(str).str.strip()

print("\n=== Estrutura da planilha original ===")
print("Número de linhas:", len(tabela))
print("Número de coletas/colunas:", len(tabela.columns))
print("Linhas encontradas:")
print(tabela.index.tolist())


# ============================================================
# 4. LOCALIZAR AS VARIÁVEIS
# ============================================================

linhas = {}

for variavel, aliases in ALIASES.items():
    if variavel == "paciente":
        try:
            linhas[variavel] = localizar_linha(
                tabela,
                aliases,
                variavel,
            )
        except KeyError:
            linhas[variavel] = None
    else:
        linhas[variavel] = localizar_linha(
            tabela,
            aliases,
            variavel,
        )

print("\n=== Correspondência das variáveis ===")
for variavel, linha in linhas.items():
    if variavel == "paciente" and linha is None:
        print("paciente: derivado do identificador de cada coleta")
    else:
        print(f"{variavel}: linha '{linha}'")


# ============================================================
# 5. TRANSFORMAR PARA FORMATO LONGO
# ============================================================

# Cada coluna original passa a representar uma coleta.
# O nome da coluna original é preservado como ID da coleta.

coletas = tabela.columns.astype(str)
if linhas["paciente"] is None:
    pacientes = []
    for coleta in coletas:
        correspondencia = re.fullmatch(r"(.+)_\d+", coleta)
        if correspondencia is None:
            raise ValueError(
                "A linha de paciente não existe no CSV e o identificador "
                f"da coleta {coleta!r} não segue o formato "
                "'id_paciente_numero_da_observacao'."
            )
        pacientes.append(correspondencia.group(1))
else:
    pacientes = tabela.loc[linhas["paciente"]].values

dados = pd.DataFrame({
    "coleta": coletas,
    "paciente": pacientes,
    "mae_hr_POS": tabela.loc[linhas["mae_hr_POS"]].values,
    "movement_index": tabela.loc[
        linhas["movement_index"]
    ].values,
    "iluminacao": tabela.loc[
        linhas["iluminacao"]
    ].values,
    "fototipo": tabela.loc[linhas["fototipo"]].values,
})

# Limpeza de texto
for coluna in ["paciente", "fototipo"]:
    dados[coluna] = (
        dados[coluna]
        .astype(str)
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NA": np.nan,
        })
    )

# Conversão numérica
for coluna in [
    "mae_hr_POS",
    "movement_index",
    "iluminacao",
]:
    dados[coluna] = converter_numerico(dados[coluna])

# Normalização dos fototipos
dados["fototipo"] = (
    dados["fototipo"]
    .astype("string")
    .str.upper()
    .str.strip()
    .str.replace("FOTOTIPO", "", regex=False)
    .str.strip()
)

# Não converter IDs de pacientes em números:
# identificadores como "001" devem permanecer distintos de "1".

dados.to_csv(
    PASTA_SAIDA / "dados_transformados_brutos.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. VALIDAR E PREPARAR OS DADOS
# ============================================================

# Uma coleta precisa ter todas as variáveis necessárias
# para entrar no modelo principal.
colunas_modelo = [
    "coleta",
    "paciente",
    "mae_hr_POS",
    "movement_index",
    "iluminacao",
    "fototipo",
]

dados["elegivel_modelo"] = (
    dados[colunas_modelo]
    .notna()
    .all(axis=1)
)

dados["motivo_exclusao"] = ""

dados.loc[
    dados["paciente"].isna(),
    "motivo_exclusao"
] += "ID do paciente ausente; "

dados.loc[
    dados["mae_hr_POS"].isna(),
    "motivo_exclusao"
] += "MAE ausente/inválido; "

dados.loc[
    dados["movement_index"].isna(),
    "motivo_exclusao"
] += "Movimento ausente/inválido; "

dados.loc[
    dados["iluminacao"].isna(),
    "motivo_exclusao"
] += "Iluminação ausente/inválida; "

dados.loc[
    dados["fototipo"].isna(),
    "motivo_exclusao"
] += "Fototipo ausente; "

dados.loc[
    dados["mae_hr_POS"] < 0,
    "motivo_exclusao"
] += "MAE negativo; "

dados.loc[
    dados["movement_index"] < 0,
    "motivo_exclusao"
] += "Movimento negativo; "

dados.loc[
    dados["iluminacao"] <= 0,
    "motivo_exclusao"
] += "Iluminação não positiva; "

dados.loc[
    ~dados["fototipo"].isin(
        ["I", "II", "III", "IV", "V", "VI"]
    ),
    "motivo_exclusao"
] += "Fototipo inválido; "

# Excluir valores incompatíveis com as definições das variáveis.
dados.loc[
    dados["mae_hr_POS"] < 0,
    "elegivel_modelo"
] = False

dados.loc[
    dados["movement_index"] < 0,
    "elegivel_modelo"
] = False

dados.loc[
    dados["iluminacao"] <= 0,
    "elegivel_modelo"
] = False

dados.loc[
    ~dados["fototipo"].isin(
        ["I", "II", "III", "IV", "V", "VI"]
    ),
    "elegivel_modelo"
] = False

# Identificar eventuais erros de digitação nos IDs.
if dados["coleta"].duplicated().any():
    raise ValueError(
        "Há identificadores de coleta duplicados."
    )

dados_validos = dados.loc[
    dados["elegivel_modelo"]
].copy()

if dados_validos.empty:
    raise ValueError(
        "Nenhuma coleta válida. Verifique o CSV e os aliases."
    )

if (dados_validos["mae_hr_POS"] < 0).any():
    raise ValueError("Há MAEs negativos nos dados válidos.")

if FOTOTIPO_REFERENCIA not in dados_validos["fototipo"].values:
    raise ValueError(
        f"O fototipo de referência {FOTOTIPO_REFERENCIA} "
        "não está presente nos dados válidos."
    )

# Classificação binária:
# 1 = sucesso; 0 = falha.
dados_validos["sucesso_POS"] = (
    dados_validos["mae_hr_POS"] < LIMIAR_MAE
).astype(int)

# Categorias possíveis, incluindo o fototipo II.
# Uma categoria ausente na amostra não terá coeficiente estimável.
dados_validos["fototipo"] = pd.Categorical(
    dados_validos["fototipo"],
    categories=["I", "II", "III", "IV", "V", "VI"],
)

# Remover apenas as coletas explicitamente identificadas
# como referência não confiável, na análise de sensibilidade.
dados_validos["referencia_confiavel"] = (
    ~dados_validos["coleta"].isin(
        COLETAS_REFERENCIA_NAO_CONFIAVEL
    )
)

# Padronizar os preditores quantitativos.
# Os parâmetros são calculados uma vez para a amostra principal.
for coluna, sufixo in [
    ("movement_index", "movement_index_z"),
    ("iluminacao", "iluminacao_z"),
]:
    media = dados_validos[coluna].mean()
    desvio = dados_validos[coluna].std(ddof=1)

    if not np.isfinite(desvio) or desvio == 0:
        raise ValueError(
            f"A variável {coluna} tem desvio-padrão zero "
            "ou inválido."
        )

    dados_validos[sufixo] = (
        dados_validos[coluna] - media
    ) / desvio

# Exportar também as observações não utilizadas e seus motivos.
dados.to_csv(
    PASTA_SAIDA / "auditoria_dados.csv",
    index=False,
    encoding="utf-8-sig",
)

dados_validos.to_csv(
    PASTA_SAIDA / "dados_modelo.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 7. ESTATÍSTICAS DESCRITIVAS
# ============================================================

print("\n=== Qualidade e elegibilidade ===")
print("Coletas originais:", len(dados))
print("Coletas válidas:", len(dados_validos))
print("Coletas excluídas:", len(dados) - len(dados_validos))
print("Pacientes:", dados_validos["paciente"].nunique())

print("\n=== Sucesso e falha ===")
contagem = (
    dados_validos["sucesso_POS"]
    .value_counts()
    .reindex([1, 0], fill_value=0)
)

for classe, nome in [(1, "Sucesso"), (0, "Falha")]:
    n = int(contagem.loc[classe])
    percentual = 100 * n / len(dados_validos)
    print(f"{nome}: {n} ({percentual:.1f}%)")

print("\n=== Por fototipo ===")
resumo_fototipo = (
    dados_validos
    .groupby("fototipo", observed=False)
    .agg(
        coletas=("sucesso_POS", "size"),
        sucessos=("sucesso_POS", "sum"),
        taxa_sucesso=("sucesso_POS", "mean"),
    )
)

resumo_fototipo["taxa_sucesso_pct"] = (
    100 * resumo_fototipo["taxa_sucesso"]
)

print(resumo_fototipo)

print("\n=== Por paciente ===")
resumo_paciente = (
    dados_validos
    .groupby("paciente")
    .agg(
        coletas=("sucesso_POS", "size"),
        sucessos=("sucesso_POS", "sum"),
        taxa_sucesso=("sucesso_POS", "mean"),
    )
)

print(resumo_paciente)

resumo_fototipo.to_csv(
    PASTA_SAIDA / "resumo_fototipo.csv",
    encoding="utf-8-sig",
)

resumo_paciente.to_csv(
    PASTA_SAIDA / "resumo_paciente.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 8. VERIFICAR SE HÁ EVENTOS SUFICIENTES
# ============================================================

n_sucessos = int(dados_validos["sucesso_POS"].sum())
n_falhas = len(dados_validos) - n_sucessos
n_pacientes = dados_validos["paciente"].nunique()

print("\n=== Verificação para modelagem ===")
print("Sucessos:", n_sucessos)
print("Falhas:", n_falhas)
print("Pacientes:", n_pacientes)

if min(n_sucessos, n_falhas) < 5:
    warnings.warn(
        "Há menos de 5 observações em uma das classes. "
        "A regressão pode ser instável; interprete os resultados "
        "com extrema cautela."
    )

if n_pacientes < 20:
    warnings.warn(
        "Há poucos pacientes para confiar plenamente na "
        "aproximação assintótica dos erros-padrão robustos do GEE."
    )

if dados_validos["fototipo"].nunique() < 2:
    raise ValueError(
        "Há apenas um fototipo observado; não é possível "
        "estimar seu efeito."
    )

if dados_validos["sucesso_POS"].nunique() < 2:
    raise ValueError(
        "Todos os casos têm a mesma classificação de sucesso/falha. "
        "Não é possível ajustar a regressão logística."
    )


# ============================================================
# 9. AJUSTAR A REGRESSÃO LOGÍSTICA GEE
# ============================================================

formula = (
    "sucesso_POS ~ movement_index_z + iluminacao_z + "
    f'C(fototipo, Treatment(reference="{FOTOTIPO_REFERENCIA}"))'
)

def ajustar_gee(df):
    """Ajusta GEE binomial com dependência agrupada por paciente."""

    modelo = smf.gee(
        formula=formula,
        groups="paciente",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )

    # Erros-padrão robustos agrupados por paciente.
    return modelo.fit()


resultado = ajustar_gee(dados_validos)

print("\n=== Resultado da regressão logística GEE ===")
print(resultado.summary())


# ============================================================
# 10. ODDS RATIOS, INTERVALOS DE CONFIANÇA E VALORES-P
# ============================================================

coef = resultado.params
ic = resultado.conf_int()
p_valores = resultado.pvalues

tabela_or = pd.DataFrame({
    "coef_log_odds": coef,
    "OR": np.exp(coef),
    "IC95_inferior_OR": np.exp(ic[0]),
    "IC95_superior_OR": np.exp(ic[1]),
    "p_valor": p_valores,
})

print("\n=== Odds ratios ===")
print(tabela_or)

tabela_or.to_csv(
    PASTA_SAIDA / "odds_ratios.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 11. TESTES GLOBAIS DOS PREDITORES
# ============================================================

# Wald conjunto: cada variável categórica é testada como um todo.
# Para movimento e iluminação, há um coeficiente por variável.
# Para fototipo, o teste considera conjuntamente os contrastes
# estimáveis em relação ao fototipo de referência.

print("\n=== Testes globais de Wald ===")

try:
    testes_globais = resultado.wald_test_terms()
    print(testes_globais.table)

    testes_globais.table.to_csv(
        PASTA_SAIDA / "testes_globais.csv",
        encoding="utf-8-sig",
    )

except Exception as erro:
    print(
        "Não foi possível calcular os testes globais "
        "automaticamente:",
        erro,
    )
    print(
        "Inspecione os coeficientes individuais e considere "
        "um teste conjunto por restrições lineares."
    )


# ============================================================
# 12. ANÁLISE DE SENSIBILIDADE
# ============================================================

if COLETAS_REFERENCIA_NAO_CONFIAVEL:
    sensibilidade = dados_validos.loc[
        dados_validos["referencia_confiavel"]
    ].copy()

    print("\n=== Análise de sensibilidade ===")
    print("Coletas após exclusão:", len(sensibilidade))
    print("Pacientes:", sensibilidade["paciente"].nunique())
    print(
        "Sucessos:",
        int(sensibilidade["sucesso_POS"].sum()),
    )
    print(
        "Falhas:",
        int((sensibilidade["sucesso_POS"] == 0).sum()),
    )

    if (
        len(sensibilidade) > 0
        and sensibilidade["sucesso_POS"].nunique() == 2
        and sensibilidade["paciente"].nunique() >= 2
    ):
        try:
            resultado_sensibilidade = ajustar_gee(sensibilidade)

            print(
                resultado_sensibilidade.summary()
            )

            tabela_sens = pd.DataFrame({
                "coef_log_odds": resultado_sensibilidade.params,
                "OR": np.exp(resultado_sensibilidade.params),
                "p_valor": resultado_sensibilidade.pvalues,
            })

            tabela_sens.to_csv(
                PASTA_SAIDA / "odds_ratios_sensibilidade.csv",
                encoding="utf-8-sig",
            )

        except Exception as erro:
            print(
                "Não foi possível ajustar o modelo de sensibilidade:",
                erro,
            )
    else:
        print(
            "A amostra de sensibilidade não permite estimar "
            "o modelo com segurança."
        )


# ============================================================
# 13. GRÁFICOS DESCRITIVOS
# ============================================================

# Taxa observada de sucesso por fototipo.
taxas = (
    dados_validos
    .groupby("fototipo", observed=False)["sucesso_POS"]
    .mean()
)

contagens = (
    dados_validos
    .groupby("fototipo", observed=False)["sucesso_POS"]
    .count()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    taxas.index.astype(str),
    taxas.values,
)

ax.set_ylim(0, 1)
ax.set_xlabel("Fototipo")
ax.set_ylabel("Proporção observada de sucesso")
ax.set_title(
    f"POS: proporção com MAE < {LIMIAR_MAE:g} bpm por fototipo"
)

for i, fototipo in enumerate(taxas.index):
    if contagens.iloc[i] > 0:
        ax.text(
            i,
            taxas.iloc[i] + 0.025,
            f"n={contagens.iloc[i]}",
            ha="center",
            fontsize=9,
        )

fig.tight_layout()
fig.savefig(
    PASTA_SAIDA / "taxa_sucesso_fototipo.png",
    dpi=200,
)
plt.show()


# MAE e movimento: visualização descritiva por classe.
fig, ax = plt.subplots(figsize=(8, 5))

for classe, nome in [(1, "Sucesso"), (0, "Falha")]:
    subconjunto = dados_validos[
        dados_validos["sucesso_POS"] == classe
    ]

    ax.scatter(
        subconjunto["movement_index"],
        subconjunto["mae_hr_POS"],
        label=nome,
        alpha=0.8,
    )

ax.axhline(
    LIMIAR_MAE,
    linestyle="--",
    label=f"Limiar = {LIMIAR_MAE:g} bpm",
)

ax.set_xlabel("Índice de movimento")
ax.set_ylabel("MAE do POS (bpm)")
ax.set_title("MAE do POS em função do índice de movimento")
ax.legend()

fig.tight_layout()
fig.savefig(
    PASTA_SAIDA / "mae_vs_movimento.png",
    dpi=200,
)
plt.show()


# ============================================================
# 14. SALVAR RESUMO FINAL
# ============================================================

with open(
    PASTA_SAIDA / "resumo_execucao.txt",
    "w",
    encoding="utf-8",
) as arquivo:
    arquivo.write("Análise binária do POS\n")
    arquivo.write(f"Limiar de sucesso: MAE < {LIMIAR_MAE} bpm\n")
    arquivo.write(f"Coletas originais: {len(dados)}\n")
    arquivo.write(f"Coletas válidas: {len(dados_validos)}\n")
    arquivo.write(f"Pacientes: {n_pacientes}\n")
    arquivo.write(f"Sucessos: {n_sucessos}\n")
    arquivo.write(f"Falhas: {n_falhas}\n")
    arquivo.write(f"Fórmula: {formula}\n")
    arquivo.write(
        "Modelo: GEE binomial com correlação exchangeable "
        "e agrupamento por paciente.\n"
    )

print("\nAnálise concluída.")
print("Resultados salvos em:", PASTA_SAIDA.resolve())
