import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import timedelta, date


ORDERS_PER_ROUTE = 40
FUEL_COST_PER_LITER = 22.5
FUEL_CONSUMPTION_L_PER_100KM = 10.0
CO2_EMISSION_FACTOR_KG_PER_KM = 0.27

CENTER_COORDINATES = {
    0: {"name": "Centro Monterrey", "lat": 25.6866, "lon": -100.3161},
    1: {"name": "Centro San Nicolás", "lat": 25.7453, "lon": -100.3065},
    2: {"name": "Centro Guadalupe", "lat": 25.6780, "lon": -100.2567},
    3: {"name": "Centro Apodaca", "lat": 25.7780, "lon": -100.1880},
    4: {"name": "Centro Santa Catarina", "lat": 25.6714, "lon": -100.4569},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def compute_route_distance(order_lats: list, order_lons: list,
                            center_lat: float, center_lon: float) -> tuple:
    total = 0.0
    prev_lat, prev_lon = center_lat, center_lon
    for lat, lon in zip(order_lats, order_lons):
        total += haversine_km(prev_lat, prev_lon, lat, lon)
        prev_lat, prev_lon = lat, lon
    route_km = total
    total_km = total + haversine_km(prev_lat, prev_lon, center_lat, center_lon)
    return route_km, total_km


def compute_costs(total_km: float) -> dict:
    liters = (total_km / 100) * FUEL_CONSUMPTION_L_PER_100KM
    fuel_cost = liters * FUEL_COST_PER_LITER
    co2 = total_km * CO2_EMISSION_FACTOR_KG_PER_KM
    return {"litros_consumidos": round(liters, 4),
            "gasto_gasolina": round(fuel_cost, 2),
            "co2_emitido_kg": round(co2, 4)}


def build_routes_for_cluster(df_cluster: pd.DataFrame, cluster_id: int,
                               center_info: dict,
                               orders_per_route: int = ORDERS_PER_ROUTE,
                               start_date: date = None) -> pd.DataFrame:
    if start_date is None:
        start_date = date(2018, 1, 1)

    center_lat = center_info["lat"]
    center_lon = center_info["lon"]
    center_name = center_info["name"]

    df_cluster = df_cluster.copy().reset_index(drop=True)
    routes = []
    n = len(df_cluster)
    route_num = 0
    delivery_date = start_date

    for start in range(0, n, orders_per_route):
        batch = df_cluster.iloc[start: start + orders_per_route]
        lats = batch["latitud"].tolist()
        lons = batch["longitud"].tolist()
        route_km, total_km = compute_route_distance(lats, lons, center_lat, center_lon)
        costs = compute_costs(total_km)

        for _, row in batch.iterrows():
            record = row.to_dict()
            record.update({
                "grupo_ruta": route_num,
                "zona": cluster_id,
                "centro": cluster_id,
                "nombre_centro": center_name,
                "fecha_entrega": delivery_date,
                "distancia_km": round(route_km, 2),
                "total_km": round(total_km, 2),
                **costs,
            })
            routes.append(record)

        route_num += 1
        delivery_date += timedelta(days=1)
        if delivery_date.weekday() == 6:
            delivery_date += timedelta(days=1)

    return pd.DataFrame(routes)


def build_all_routes(df_nl_clustered: pd.DataFrame,
                     center_coordinates: dict = None) -> pd.DataFrame:
    if center_coordinates is None:
        center_coordinates = CENTER_COORDINATES

    all_routes = []
    for cluster_id, center_info in center_coordinates.items():
        df_cluster = df_nl_clustered[
            df_nl_clustered["cluster_nl"] == cluster_id
        ].dropna(subset=["latitud", "longitud"])
        if df_cluster.empty:
            continue
        routes_df = build_routes_for_cluster(df_cluster, cluster_id, center_info)
        all_routes.append(routes_df)

    return pd.concat(all_routes, ignore_index=True)


def compute_baseline_costs(df: pd.DataFrame,
                             single_center_lat: float = 25.6866,
                             single_center_lon: float = -100.3161,
                             orders_per_route: int = ORDERS_PER_ROUTE) -> pd.DataFrame:
    df = df.dropna(subset=["latitud", "longitud"]).copy().reset_index(drop=True)
    records = []
    for start in range(0, len(df), orders_per_route):
        batch = df.iloc[start: start + orders_per_route]
        route_km, total_km = compute_route_distance(
            batch["latitud"].tolist(), batch["longitud"].tolist(),
            single_center_lat, single_center_lon
        )
        costs = compute_costs(total_km)
        records.append({
            "grupo_ruta": start // orders_per_route,
            "distancia_km": round(route_km, 2),
            "total_km": round(total_km, 2),
            **costs,
        })
    return pd.DataFrame(records)


def summarize_costs(df_routes: pd.DataFrame) -> dict:
    return {
        "total_km": round(df_routes["total_km"].sum() / len(df_routes["grupo_ruta"].unique()), 2),
        "km_reducidos": round(df_routes["distancia_km"].sum(), 2),
        "gasto_gasolina_total": round(df_routes["gasto_gasolina"].sum(), 2),
        "co2_total_kg": round(df_routes["co2_emitido_kg"].sum(), 2),
        "total_rutas": df_routes["grupo_ruta"].nunique(),
        "costo_promedio_ruta": round(
            df_routes["gasto_gasolina"].sum() / df_routes["grupo_ruta"].nunique(), 2
        ),
    }


def run_routing_pipeline(df_nl_clustered: pd.DataFrame,
                          output_dir: str = "outputs") -> dict:
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_nl = df_nl_clustered.dropna(subset=["latitud", "longitud"]).copy()

    baseline = compute_baseline_costs(df_nl)
    baseline["fecha_entrega"] = pd.date_range(
        start="2018-01-01", periods=len(baseline), freq="B"
    )
    baseline.to_csv(f"{output_dir}/baseline_costs_nl.csv", index=False)
    baseline_summary = summarize_costs(baseline)

    new_routes = build_all_routes(df_nl_clustered)
    new_routes.to_csv(f"{output_dir}/new_routes_nl.csv", index=False)
    new_summary = summarize_costs(new_routes)

    comparison = {
        "baseline": baseline_summary,
        "proposed": new_summary,
        "savings": {
            k: round(baseline_summary[k] - new_summary[k], 2)
            for k in baseline_summary
            if isinstance(baseline_summary[k], (int, float))
        },
    }

    print("\n=== Cost Comparison: Baseline vs Proposed ===")
    for metric in ["km_reducidos", "gasto_gasolina_total", "co2_total_kg",
                    "total_rutas", "costo_promedio_ruta"]:
        print(f"  {metric}: {baseline_summary.get(metric)} → {new_summary.get(metric)} "
              f"(savings: {comparison['savings'].get(metric, 'N/A')})")

    return comparison
