import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


N_CLUSTERS_NATIONAL = 7

STATE_COORDINATES = {
    "Ciudad de México": (19.4326, -99.1332),
    "Baja California": (30.8406, -115.2838),
    "Sonora": (29.2972, -110.3309),
    "Chihuahua": (28.6353, -106.0889),
    "Coahuila": (25.4381, -101.0053),
    "Nuevo León": (25.5922, -99.9962),
    "Tamaulipas": (23.7369, -99.1411),
    "Durango": (24.0277, -104.6532),
    "Estado de México": (19.3597, -99.7555),
    "Hidalgo": (20.0911, -98.7624),
    "Querétaro": (20.5888, -100.3899),
    "Puebla": (19.0414, -98.2063),
    "Morelos": (18.6813, -99.1013),
    "Tlaxcala": (19.3139, -98.2400),
    "Michoacán": (19.5665, -101.7068),
    "Guanajuato": (21.0190, -101.2574),
    "Oaxaca": (17.0732, -96.7266),
    "Chiapas": (16.7569, -93.1292),
    "Guerrero": (17.4392, -99.5451),
    "Tabasco": (17.9869, -92.9303),
    "Veracruz": (19.1738, -96.1342),
    "Campeche": (19.8301, -90.5349),
    "Yucatán": (20.7099, -89.0943),
    "Quintana Roo": (19.1817, -88.4791),
}


def prepare_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["latitud", "longitud"]).copy()
    return df


def run_kmeans_national(df: pd.DataFrame, n_clusters: int = N_CLUSTERS_NATIONAL,
                         random_state: int = 42) -> tuple:
    coords = prepare_coordinates(df)
    X = coords[["latitud", "longitud"]].values

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)

    coords = coords.copy()
    coords["cluster_nacional"] = model.labels_

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=["latitud_centro", "longitud_centro"]
    )
    centers.index.name = "cluster_nacional"
    centers = centers.reset_index()

    silhouette = silhouette_score(X, model.labels_)

    return coords, centers, model, silhouette


def elbow_analysis(df: pd.DataFrame, k_range: range = range(3, 12),
                    random_state: int = 42):
    coords = prepare_coordinates(df)
    X = coords[["latitud", "longitud"]].values

    inertias = []
    silhouettes = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, model.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), inertias, "o-", color="#1d3557")
    axes[0].set_title("Elbow Method — Inertia")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, "o-", color="#e63946")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    return fig, pd.DataFrame({"k": list(k_range), "inertia": inertias,
                               "silhouette": silhouettes})


def plot_national_clusters(df_clustered: pd.DataFrame, centers: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df_clustered["longitud"],
        df_clustered["latitud"],
        c=df_clustered["cluster_nacional"],
        cmap="tab10",
        s=10,
        alpha=0.5,
        label="Orders",
    )
    ax.scatter(
        centers["longitud_centro"],
        centers["latitud_centro"],
        c="black",
        marker="*",
        s=300,
        zorder=5,
        label="Distribution Centers",
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"National Distribution Center Proposal — {len(centers)} Centers")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    return fig


def build_national_centers_table(centers: pd.DataFrame,
                                  df_clustered: pd.DataFrame) -> pd.DataFrame:
    cluster_counts = (
        df_clustered.groupby("cluster_nacional")["order_id"]
        .count()
        .reset_index()
        .rename(columns={"order_id": "total_orders"})
    )
    centers = centers.merge(cluster_counts, on="cluster_nacional", how="left")
    centers["nombre"] = [f"Centro Nacional {i + 1}" for i in range(len(centers))]
    return centers


def run_national_clustering(df: pd.DataFrame, output_dir: str = "outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_clustered, centers, model, silhouette = run_kmeans_national(df)
    print(f"Silhouette score ({N_CLUSTERS_NATIONAL} clusters): {silhouette:.4f}")

    centers_final = build_national_centers_table(centers, df_clustered)

    fig = plot_national_clusters(df_clustered, centers)
    fig.savefig(f"{output_dir}/national_clusters.png")
    plt.close(fig)

    centers_final.to_csv(f"{output_dir}/national_distribution_centers.csv", index=False)
    df_clustered.to_csv(f"{output_dir}/orders_with_national_clusters.csv", index=False)

    return df_clustered, centers_final, model
