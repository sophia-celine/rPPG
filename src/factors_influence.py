import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

SOURCE_CSV = r"C:\Users\Sophia\Documents\rPPG_data\results\results_mp_mov_obs.csv"


def correlate(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """
    Calculates the Pearson correlation between two paired one-dimensional arrays.

    Returns the correlation coefficient and its two-sided p-value. A positive
    coefficient means the variables tend to increase together; a negative
    coefficient means one tends to decrease as the other increases.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must have the same number of values.")
    if x_values.size < 3:
        raise ValueError("At least three paired observations are required.")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must contain only finite values.")
    if np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        raise ValueError("Correlation is undefined when either array is constant.")

    result = pearsonr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)


def _load_paired_movement_mae() -> pd.DataFrame:
    source_df = pd.read_csv(SOURCE_CSV)
    if "metric" not in source_df.columns:
        raise ValueError("The CSV must contain a 'metric' column.")

    metrics = source_df.set_index("metric")

    def get_metric_values(metric_name: str) -> pd.Series:
        if metric_name not in metrics.index:
            raise ValueError(f"Metric {metric_name!r} was not found in the CSV.")
        values = metrics.loc[metric_name]
        if isinstance(values, pd.DataFrame):
            raise ValueError(f"Metric {metric_name!r} appears more than once.")

        normalized_values = values.map(
            lambda value: value.replace(",", ".")
            if isinstance(value, str)
            else value
        )
        numeric_values = pd.to_numeric(normalized_values, errors="coerce")
        invalid_values = values.notna() & numeric_values.isna()
        if invalid_values.any():
            invalid_columns = values.index[invalid_values].tolist()
            raise ValueError(
                f"Metric {metric_name!r} contains non-numeric values in "
                f"observations: {invalid_columns}"
            )
        return numeric_values

    paired = pd.DataFrame(
        {
            "movement": get_metric_values("movement_index"),
            "mae": get_metric_values("mae_hr_POS"),
        }
    )
    paired["patient"] = [
        observation.rsplit("_", 1)[0]
        for observation in paired.index.astype(str)
    ]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["movement", "mae"]
    ).reset_index(drop=True)
    if len(paired) < 3:
        raise ValueError(
            "At least three complete movement-MAE observations are required."
        )
    if paired["patient"].nunique() < 2:
        raise ValueError(
            "At least two patients are required to account for clustering."
        )
    if paired["movement"].nunique() < 2:
        raise ValueError("Movement must vary across observations to fit the model.")
    return paired


def compare_movement_mae() -> dict[str, float | int]:
    """
    Relates movement to POS heart-rate MAE while accounting for repeated
    observations from the same patient with a Gaussian GEE model.

    Also returns Pearson's correlation across all paired observations as a
    descriptive statistic only; its usual independent-observation p-value is
    intentionally not reported.
    """
    paired = _load_paired_movement_mae()

    try:
        import statsmodels.api as sm
    except ImportError as error:
        raise ImportError(
            "compare_movement_mae requires statsmodels. "
            "Install it with 'python -m pip install statsmodels'."
        ) from error

    design = pd.DataFrame(
        {"Intercept": 1.0, "movement": paired["movement"].to_numpy()}
    )
    model = sm.GEE(
        endog=paired["mae"],
        exog=design,
        groups=paired["patient"],
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit(cov_type="robust")
    confidence_interval = result.conf_int().loc["movement"]

    pearson_result = pearsonr(paired["movement"], paired["mae"])

    return {
        "n_observations": int(len(paired)),
        "n_patients": int(paired["patient"].nunique()),
        "pearson_r_all_observations": float(pearson_result.statistic),
        "gee_intercept": float(result.params["Intercept"]),
        "gee_movement_slope": float(result.params["movement"]),
        "gee_movement_standard_error": float(result.bse["movement"]),
        "gee_movement_ci_lower": float(confidence_interval.iloc[0]),
        "gee_movement_ci_upper": float(confidence_interval.iloc[1]),
        "gee_movement_pvalue": float(result.pvalues["movement"]),
    }


def main() -> None:
    results = compare_movement_mae()
    print("Associação movimento–MAE POS")
    print(f"Observações usadas: {results['n_observations']}")
    print(f"Pacientes usados: {results['n_patients']}")
    print(
        "Inclinação GEE (MAE por unidade de movimento): "
        f"{results['gee_movement_slope']:.4f}"
    )
    print(
        "IC 95% da inclinação GEE: "
        f"[{results['gee_movement_ci_lower']:.4f}, "
        f"{results['gee_movement_ci_upper']:.4f}]"
    )
    print(f"p-valor GEE: {results['gee_movement_pvalue']:.4g}")
    print(
        "Pearson exploratório (todas as observações; sem p-valor): "
        f"r = {results['pearson_r_all_observations']:.4f}"
    )

    paired = _load_paired_movement_mae()
    plt.figure(figsize=(8, 6))
    plt.scatter(paired["movement"], paired["mae"], alpha=0.75)
    plt.xlabel("Índice de movimento")
    plt.ylabel("MAE de frequência cardíaca (POS)")
    plt.title("MAE POS por índice de movimento")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_dispersion(column_name1, column_name2):
    plt.figure(figsize=(8, 6))
    source_df = pd.read_csv(SOURCE_CSV)
    metrics = source_df.set_index("metric")

    column1 = np.array(metrics.loc[column_name1].astype(str).str.replace(',', '.').astype(float))
    column2 = np.array(metrics.loc[column_name2].astype(str).str.replace(',', '.').astype(float))

    plt.scatter(column1, column2)
    plt.xlabel(column_name1)
    plt.ylabel(column_name2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()



if __name__ == "__main__":
    plot_dispersion('movement_index', 'mae_hr_POS')
    plot_dispersion('Iluminação (lux)', 'mae_hr_POS')
    plt.show()