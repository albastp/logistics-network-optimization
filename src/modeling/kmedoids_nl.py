import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

try:
    import osmnx as ox
    from sklearn_extra.cluster import KMedoids
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("Warning: osmnx or sklearn_extra not installed. Road-network K-Medoids unavailable.")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


N_CLUSTERS_NL = 5
NL_BBOX = (25.3, 25.9, -100.6, -100.0)


def filter_nuevo_leon(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["estado_del_cliente"] == "Nuevo León"].dropna(
        subset=["latitud", "longitud"]
    ).copy()


def load_road_network(bbox: tuple = NL_BBOX):
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx is required for road-network clustering.")
    north, south, east, west = bbox
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    return G


def snap_to_nodes(df: pd.DataFrame, G) -> pd.DataFrame:
    df = df.copy()
    df["node"] = df.apply(
        lambda row: ox.nearest_nodes(G, row["longitud"], row["latitud"]), axis=1
    )
    return df


def build_distance_matrix(df: pd.DataFrame, G) -> np.ndarray:
    n = len(df)
    dist_matrix = np.zeros((n, n))
    nodes = df["node"].tolist()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                dist_matrix[i, j] = nx.shortest_path_length(
                    G, nodes[i], nodes[j], weight="length"
                )
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = np.inf
    return dist_matrix


def run_kmedoids_road_network(df: pd.DataFrame, G,
                               n_clusters: int = N_CLUSTERS_NL,
                               random_state: int = 42) -> tuple:
    df = snap_to_nodes(df, G)
    dist_matrix = build_distance_matrix(df, G)
    model = KMedoids(n_clusters=n_clusters, metric="precomputed",
                     random_state=random_state)
    model.fit(dist_matrix)
    df = df.copy()
    df["cluster_nl"] = model.labels_
    center_indices = model.medoid_indices_
    center_nodes = [df.iloc[idx]["node"] for idx in center_indices]
    centers = df.iloc[center_indices][["latitud", "longitud"]].copy()
    centers["cluster_nl"] = list(range(n_clusters))
    return df, centers, model, center_nodes


def run_kmeans_nl_fallback(df: pd.DataFrame,
                            n_clusters: int = N_CLUSTERS_NL,
                            random_state: int = 42) -> tuple:
    df = filter_nuevo_leon(df)
    X = df[["latitud", "longitud"]].values
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X)
    df = df.copy()
    df["cluster_nl"] = model.labels_
    centers = pd.DataFrame(
        model.cluster_centers_, columns=["latitud", "longitud"]
    )
    centers["cluster_nl"] = list(range(n_clusters))
    silhouette = silhouette_score(X, model.labels_)
    return df, centers, model, silhouette


def assign_zones_within_nl(df: pd.DataFrame,
                            n_zones: int = 5,
                            random_state: int = 42) -> pd.DataFrame:
    X = df[["latitud", "longitud"]].values
    zone_model = KMeans(n_clusters=n_zones, random_state=random_state, n_init=10)
    df = df.copy()
    df["zona_nl"] = zone_model.fit_predict(X)
    return df


def build_nl_center_names() -> dict:
    return {
        0: "Centro Monterrey",
        1: "Centro San Nicolás",
        2: "Centro Guadalupe",
        3: "Centro Apodaca",
        4: "Centro Santa Catarina",
    }


def plot_nl_clusters(df_clustered: pd.DataFrame, centers: pd.DataFrame,
                     center_names: dict = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_clustered["longitud"],
        df_clustered["latitud"],
        c=df_clustered["cluster_nl"],
        cmap="tab10",
        s=15,
        alpha=0.6,
        label="Orders",
    )
    for _, row in centers.iterrows():
        label = (
            center_names.get(int(row["cluster_nl"]), f"Center {int(row['cluster_nl'])}")
            if center_names
            else f"Center {int(row['cluster_nl'])}"
        )
        ax.scatter(row["longitud"], row["latitud"], c="black", marker="*",
                   s=400, zorder=5)
        ax.annotate(label, (row["longitud"], row["latitud"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"Nuevo León — {len(centers)} Distribution Centers (Metro Area)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig


def run_nl_clustering(df: pd.DataFrame, output_dir: str = "outputs",
                       use_road_network: bool = False):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_nl = filter_nuevo_leon(df)

    if use_road_network and OSMNX_AVAILABLE:
        G = load_road_network()
        df_clustered, centers, model, center_nodes = run_kmedoids_road_network(
            df_nl, G, n_clusters=N_CLUSTERS_NL
        )
    else:
        df_clustered, centers, model, silhouette = run_kmeans_nl_fallback(
            df_nl, n_clusters=N_CLUSTERS_NL
        )
        print(f"Nuevo León K-Means silhouette: {silhouette:.4f}")

    df_clustered = assign_zones_within_nl(df_clustered)
    center_names = build_nl_center_names()

    fig = plot_nl_clusters(df_clustered, centers, center_names)
    fig.savefig(f"{output_dir}/nl_clusters.png")
    plt.close(fig)

    centers["nombre"] = centers["cluster_nl"].map(center_names)
    centers.to_csv(f"{output_dir}/nl_distribution_centers.csv", index=False)
    df_clustered.to_csv(f"{output_dir}/nl_orders_clustered.csv", index=False)

    return df_clustered, centers, model
