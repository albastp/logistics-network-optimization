import pandas as pd
import numpy as np


COLUMNS_TO_DROP = [
    "secuencia_de_pago",
    "cuotas_de_pago",
    "pago_total",
    "boleto",
    "tarjeta_de_credito",
    "tarjeta_de_debito",
    "voucher",
    "peso_producto_g",
    "producto_longitud_cm",
    "producto_altura_cm",
    "producto_ancho_cm",
    "densidad_g_cm3",
    "forma_producto",
    "tamaño",
    "horas_tardanza_aprobacion",
    "horas_de_procesamiento",
]

CATEGORY_CONSOLIDATION = {
    "electrodomesticos_2": "electrodomesticos",
    "horno_casero_portatil_y_cafetera": "electrodomesticos",
    "electrodomesticos_portatiles": "electrodomesticos",
    "telefonia_de_casa": "telefonia",
    "pc_gamer": "pcs",
    "bolsos_y_accesorios": "ropa_bolsas_y_accesorios",
    "cd_dvd_musica": "dvds_blu_ray",
    "alimentos": "alimentos_bebidas",
    "bebidas": "alimentos_bebidas",
    "pañales": "bebes",
    "regalos": "articulos_de_fiesta",
    "agricultura_industria_y_comercio": "industria_y_comercio",
}

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

GEOGRAPHIC_ZONES = {
    "Norte": [
        "Baja California", "Sonora", "Chihuahua", "Coahuila",
        "Nuevo León", "Tamaulipas", "Durango",
    ],
    "Centro": [
        "Ciudad de México", "Estado de México", "Hidalgo", "Querétaro",
        "Puebla", "Morelos", "Tlaxcala", "Michoacán", "Guanajuato",
    ],
    "Sur": [
        "Oaxaca", "Chiapas", "Guerrero", "Tabasco",
        "Veracruz", "Campeche", "Yucatán", "Quintana Roo",
    ],
}


def build_master_dataset(cleaned_datasets: dict) -> pd.DataFrame:
    df = cleaned_datasets["orders"].copy()

    df = df.merge(
        cleaned_datasets["clients"][["id_cliente", "id_único_de_cliente", "estado_del_cliente"]],
        left_on="cliente_id",
        right_on="id_cliente",
        how="left",
    )

    order_prod = cleaned_datasets["order_products"].copy()
    df = df.merge(
        order_prod[["order_id", "numero_de_producto_id", "producto_id", "vendedor_id",
                    "precio", "costo_de_flete"]],
        on="order_id",
        how="left",
    )

    products = cleaned_datasets["products"].copy()
    products["volumen_cm3"] = (
        products["producto_longitud_cm"]
        * products["producto_altura_cm"]
        * products["producto_ancho_cm"]
    ).astype(int)
    products["peso_kg"] = products["peso_producto_g"] / 1000

    df = df.merge(
        products[["id_producto", "categoria_nombre_producto", "volumen_cm3", "peso_kg"]],
        left_on="producto_id",
        right_on="id_producto",
        how="left",
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    for col in ["fecha_entrega_al_cliente", "fecha_de_entrega_estimada",
                "orden_compra_timestamp_date", "orden_pago_aprobado_date",
                "orden_entrega_transportista_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["tiempo_total_entrega"] = (
        df["fecha_entrega_al_cliente"] - df["orden_compra_timestamp_date"]
    ).dt.days

    df["dias_entre_aprobacion_y_envio"] = (
        df["orden_entrega_transportista_date"] - df["orden_pago_aprobado_date"]
    ).dt.days

    state_to_zone = {
        state: zone
        for zone, states in GEOGRAPHIC_ZONES.items()
        for state in states
    }
    df["zona_cliente"] = df["estado_del_cliente"].map(state_to_zone).fillna("Otro")

    df["pedidos_por_cliente_estado"] = df.groupby(
        ["id_unico_de_cliente", "estado_del_cliente"]
    )["order_id"].transform("count")

    df["peso_volumetrico"] = df["volumen_cm3"] / 6000

    df["categoria_nombre_producto"] = df["categoria_nombre_producto"].replace(
        CATEGORY_CONSOLIDATION
    )

    df["latitud"] = df["estado_del_cliente"].map(
        {k: v[0] for k, v in STATE_COORDINATES.items()}
    )
    df["longitud"] = df["estado_del_cliente"].map(
        {k: v[1] for k, v in STATE_COORDINATES.items()}
    )

    df["orden_compra_timestamp_dia"] = df["orden_compra_timestamp_date"].dt.day
    df["orden_compra_timestamp_date"] = df["orden_compra_timestamp_date"].dt.strftime(
        "%Y-%m"
    )

    return df


def load_and_build(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    return df
