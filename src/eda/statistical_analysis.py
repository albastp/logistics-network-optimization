import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, iqr, f_oneway, chi2_contingency
from matplotlib import rcParams


rcParams["figure.dpi"] = 150
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.titlesize"] = 11
rcParams["axes.labelsize"] = 10
rcParams["legend.fontsize"] = 9

DISCRETE_VARS = [
    "cantidad",
    "dias_tardanza_aprobacion",
    "diferencia_dias",
    "tiempo_total_entrega",
    "dias_entre_aprobacion_y_envio",
    "pedidos_por_cliente_estado",
    "orden_compra_timestamp_dia",
]

CONTINUOUS_VARS = [
    "precio",
    "costo_de_flete",
    "peso_kg",
    "importe_a_pagar",
    "dias_entre_aprobacion_y_envio",
]

CATEGORICAL_VARS = ["categoria_nombre_producto", "estado_de_puntualidad"]

NUMERIC_VARS_FOR_BOXPLOT = [
    ("cantidad", "Product Quantity"),
    ("diferencia_dias", "Delivery Day Difference"),
    ("tiempo_total_entrega", "Total Delivery Time"),
    ("dias_entre_aprobacion_y_envio", "Days Between Approval and Shipment"),
    ("importe_a_pagar", "Amount Due"),
]


def plot_discrete_frequency(df: pd.DataFrame, variable: str, top_n: int = 5):
    freq = df[variable].value_counts().reset_index()
    freq.columns = [variable, "Frequency"]
    top = freq.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=variable, y="Frequency", data=top, ax=ax)
    ax.set_title(f"Frequency of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_histogram(df: pd.DataFrame, variable: str):
    data = df[variable].dropna()
    mean = data.mean()
    std = data.std()
    p5, p95 = np.percentile(data, 5), np.percentile(data, 95)
    x_min = min(p5, mean - std) - abs(std * 0.2)
    x_max = max(p95, mean + std) + abs(std * 0.2)

    bin_width = 2 * iqr(data) / np.cbrt(len(data))
    bins = int((x_max - x_min) / bin_width) if bin_width > 0 else 30
    bins = max(10, min(bins, 60))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts, bins_edges, _ = ax.hist(
        data, bins=bins, density=False, color="#A2D2FF",
        edgecolor="black", alpha=0.75, label="Data"
    )
    ax.axvline(mean, color="#e63946", linestyle="--", linewidth=1.2,
               label=f"Mean: {mean:.2f}")
    ax.axvline(mean - std, color="#1d3557", linestyle="--", linewidth=1.1,
               label=f"Lower bound: {mean - std:.2f}")
    ax.axvline(mean + std, color="#1d3557", linestyle="--", linewidth=1.1,
               label=f"Upper bound: {mean + std:.2f}")
    x = np.linspace(x_min, x_max, 500)
    y_pdf = norm.pdf(x, mean, std)
    y_scaled = y_pdf * len(data) * (bins_edges[1] - bins_edges[0])
    ax.plot(x, y_scaled, color="black", linewidth=2,
            label=f"Normal dist\nμ={mean:.2f}, σ={std:.2f}")
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"Histogram of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=True, facecolor="white",
               framealpha=0.95, edgecolor="gray")
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    return fig


def plot_boxplot_log(df: pd.DataFrame, variable: str):
    data_log = df[variable].apply(np.log1p)
    q1, q2, q3 = (data_log.quantile(0.25), data_log.quantile(0.50),
                  data_log.quantile(0.75))
    iqr_val = q3 - q1
    lower = q1 - 1.5 * iqr_val
    upper = q3 + 1.5 * iqr_val
    outliers = data_log[(data_log < lower) | (data_log > upper)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data_log, ax=ax)
    ax.set_title(f"Boxplot of {variable} (log scale)")
    plt.tight_layout()
    return fig, outliers


def _top_categories(df: pd.DataFrame, col: str, threshold: float = 0.7) -> list:
    proportions = df[col].value_counts(normalize=True)
    cumulative = proportions.cumsum()
    return cumulative[cumulative <= threshold].index.tolist()


def plot_boxplot_by_category(df: pd.DataFrame, num_var: str, cat_var: str,
                              percentile_cap: float = 0.90):
    top_cats = _top_categories(df, cat_var)
    filtered = df[df[cat_var].isin(top_cats)].copy()
    cap = filtered[num_var].quantile(percentile_cap)
    filtered = filtered[filtered[num_var] <= cap]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=cat_var, y=num_var, data=filtered,
                hue=cat_var, legend=False, ax=ax)
    ax.set_title(f"{num_var} by {cat_var}")
    ax.set_xlabel(cat_var)
    ax.set_ylabel(num_var)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list = None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap — Numeric Variables")
    plt.tight_layout()
    return fig


def run_anova(df: pd.DataFrame, cat_var: str, num_vars: list) -> pd.DataFrame:
    results = []
    for num_var in num_vars:
        groups = [
            group[num_var].dropna().values
            for _, group in df.groupby(cat_var)
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = f_oneway(*groups)
        results.append({"categorical_var": cat_var, "numeric_var": num_var,
                         "F_statistic": round(f_stat, 4), "p_value": round(p_val, 6),
                         "significant": p_val < 0.05})
    return pd.DataFrame(results)


def run_chi_square(df: pd.DataFrame, cat_vars: list) -> pd.DataFrame:
    results = []
    for i, v1 in enumerate(cat_vars):
        for v2 in cat_vars[i + 1:]:
            contingency = pd.crosstab(df[v1], df[v2])
            chi2, p, dof, _ = chi2_contingency(contingency)
            results.append({"var1": v1, "var2": v2, "chi2": round(chi2, 4),
                             "p_value": round(p, 6), "dof": dof,
                             "significant": p < 0.05})
    return pd.DataFrame(results)


def run_full_eda(df: pd.DataFrame, output_dir: str = "outputs/eda"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for var in DISCRETE_VARS:
        if var in df.columns:
            fig = plot_discrete_frequency(df, var)
            fig.savefig(f"{output_dir}/discrete_{var}.png")
            plt.close(fig)

    for var in CONTINUOUS_VARS:
        if var in df.columns:
            fig = plot_histogram(df, var)
            fig.savefig(f"{output_dir}/histogram_{var}.png")
            plt.close(fig)
            fig_box, _ = plot_boxplot_log(df, var)
            fig_box.savefig(f"{output_dir}/boxplot_log_{var}.png")
            plt.close(fig_box)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig_corr = plot_correlation_heatmap(df, numeric_cols)
    fig_corr.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close(fig_corr)

    num_vars_list = [v for v, _ in NUMERIC_VARS_FOR_BOXPLOT if v in df.columns]

    for cat_var in CATEGORICAL_VARS:
        if cat_var not in df.columns:
            continue
        anova_results = run_anova(df, cat_var, num_vars_list)
        anova_results.to_csv(f"{output_dir}/anova_{cat_var}.csv", index=False)

    chi_vars = [v for v in CATEGORICAL_VARS if v in df.columns]
    if len(chi_vars) >= 2:
        chi_results = run_chi_square(df, chi_vars)
        chi_results.to_csv(f"{output_dir}/chi_square.csv", index=False)

    print(f"EDA outputs saved to {output_dir}/")
