import pandas as pd

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

ROUTE_COLORS = [
    "red", "blue", "green", "purple", "orange", "darkred", "cadetblue",
    "darkgreen", "pink", "darkblue", "darkpurple", "lightblue",
    "lightgreen", "gray", "black", "lightgray",
]

CENTER_ICON_COLORS = {0: "red", 1: "blue", 2: "green", 3: "purple", 4: "orange"}


def _require_folium():
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is required: pip install folium")


def plot_cluster_map(
    df_clustered: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: str = "outputs/cluster_map.html",
    zoom: int = 9,
):
    _require_folium()
    m = folium.Map(
        location=[df_clustered["latitud"].mean(), df_clustered["longitud"].mean()],
        zoom_start=zoom,
    )
    n_clusters = df_clustered["cluster_nl"].nunique()
    color_map = {i: ROUTE_COLORS[i % len(ROUTE_COLORS)] for i in range(n_clusters)}

    for _, row in df_clustered.iterrows():
        cid = int(row["cluster_nl"])
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=2, color=color_map.get(cid, "gray"),
            fill=True, fill_opacity=0.6,
        ).add_to(m)

    for _, row in centers.iterrows():
        cid = int(row["cluster_nl"])
        name = row.get("nombre", f"Center {cid}")
        folium.Marker(
            location=[row["latitud"], row["longitud"]],
            icon=folium.Icon(color=CENTER_ICON_COLORS.get(cid, "black"),
                             icon="star", prefix="fa"),
            tooltip=name,
            popup=folium.Popup(f"<b>{name}</b>", max_width=200),
        ).add_to(m)

    m.save(output_path)
    print(f"Cluster map saved to {output_path}")


def plot_comparison_map(
    df_baseline: pd.DataFrame,
    df_proposed: pd.DataFrame,
    centers: pd.DataFrame,
    output_path: str = "outputs/comparison_map.html",
):
    _require_folium()
    m = folium.Map(
        location=[df_proposed["latitud"].mean(), df_proposed["longitud"].mean()],
        zoom_start=10,
    )

    fg_baseline = folium.FeatureGroup(name="Baseline Routes", show=False)
    for _, df_ruta in df_baseline.groupby("grupo_ruta"):
        if "orden_visita" in df_ruta.columns:
            df_ruta = df_ruta.sort_values("orden_visita")
        coords = df_ruta[["latitud", "longitud"]].values.tolist()
        folium.PolyLine(coords, color="gray", weight=2, opacity=0.5).add_to(fg_baseline)
    fg_baseline.add_to(m)

    fg_proposed = folium.FeatureGroup(name="Proposed Routes", show=True)
    n_zones = df_proposed["zona_nl"].nunique() if "zona_nl" in df_proposed.columns else 1
    zone_colors = {i: ROUTE_COLORS[i % len(ROUTE_COLORS)] for i in range(n_zones)}
    group_col = "ruta_id" if "ruta_id" in df_proposed.columns else "grupo_ruta"

    for rid, df_ruta in df_proposed.groupby(group_col):
        if "orden_visita" in df_ruta.columns:
            df_ruta = df_ruta.sort_values("orden_visita")
        coords = df_ruta[["latitud", "longitud"]].values.tolist()
        zona = int(df_ruta.iloc[0].get("zona_nl", 0))
        color = zone_colors.get(zona % len(zone_colors), "blue")
        folium.PolyLine(coords, color=color, weight=3, opacity=0.75).add_to(fg_proposed)

        for _, row in df_ruta.iterrows():
            folium.CircleMarker(
                location=[row["latitud"], row["longitud"]],
                radius=4, color=color, fill=True, fill_opacity=0.7,
                tooltip=(
                    f"Stop: {row.get('orden_visita', '')}<br>"
                    f"Route: {rid}<br>"
                    f"Date: {row.get('fecha_entrega', '')}"
                ),
            ).add_to(fg_proposed)
    fg_proposed.add_to(m)

    fg_centers = folium.FeatureGroup(name="Distribution Centers", show=True)
    for _, row in centers.iterrows():
        cid = int(row["cluster_nl"])
        name = row.get("nombre", f"Center {cid}")
        folium.Marker(
            location=[row["latitud"], row["longitud"]],
            icon=folium.Icon(color=CENTER_ICON_COLORS.get(cid, "black"),
                             icon="star", prefix="fa"),
            tooltip=name,
            popup=folium.Popup(f"<b>{name}</b>", max_width=200),
        ).add_to(fg_centers)
    fg_centers.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Comparison map saved to {output_path}")


def generate_all_maps(
    df_clustered: pd.DataFrame,
    centers: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_proposed: pd.DataFrame,
    output_dir: str = "outputs",
):
    import os
    os.makedirs(output_dir, exist_ok=True)
    plot_cluster_map(df_clustered, centers, f"{output_dir}/cluster_map.html")
    plot_comparison_map(df_baseline, df_proposed, centers,
                        f"{output_dir}/comparison_map.html")
