import pandas as pd
import numpy as np


def load_raw_datasets(data_dir: str) -> dict:
    datasets = {
        "orders": pd.read_excel(f"{data_dir}/Ordenes.xlsx"),
        "products": pd.read_excel(f"{data_dir}/Productos.xlsx"),
        "order_products": pd.read_excel(f"{data_dir}/Orden de Productos.xlsx"),
        "clients": pd.read_excel(f"{data_dir}/Clientes.xlsx"),
        "order_payments": pd.read_excel(f"{data_dir}/Orden de Pagos.xlsx"),
    }
    return datasets


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(thresh=len(df.columns) - 1)

    date_cols = [
        "Orden compra timestamp",
        "Orden pago aprobado",
        "Orden entrega transportista",
        "Fecha entrega al cliente",
        "Fecha de entrega estimada",
    ]

    def impute_missing_date(row):
        dates = row[date_cols]
        if dates.isna().sum() == 1:
            col_missing = dates[dates.isna()].index[0]
            available = dates.dropna()
            if len(available) > 1:
                delta = (available.max() - available.min()) / (len(available) - 1)
                row[col_missing] = available.max() + delta
        return row

    df = df.apply(impute_missing_date, axis=1)
    df = df.reset_index(drop=True)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df["Categoría nombre producto"] = df["Categoría nombre producto"].fillna(
        "no_especificado"
    )
    df = df.dropna(subset=["peso_producto_g"])
    df = df.reset_index(drop=True)
    df = df.rename(columns={"Categoría nombre producto": "Categoria nombre producto"})
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    replacements = {
        "servicios domesticos": "servicios_domesticos",
        "instrumentos_musicais": "instrumentos_musicales",
        "camas, mesas y baños": "camas_mesas_y_baños",
        "herramientas de construcción": "herramientas_de_construcción",
        "accesorios informatica": "accesorios_informática",
        "belleza y salud": "belleza_y_salud",
        "bolsos y accesorios": "bolsos_y_accesorios",
        "ferramentas_jardim": "herramientas_jardin",
        "ropa, bolsas y accesorios": "ropa_bolsas_y_accesorios",
        "papelaria": "papeleria",
        "construcción casas": "construcción_casas",
        "electrodomesticos portatiles": "electrodomesticos_portatiles",
        "agricultura, industria y comercio": "agricultura_industria_y_comercio",
        "señalización y seguridad": "señalización_y_seguridad",
        "consolas de juegos": "consolas_de_juegos",
        "industria y comercio": "industria_y_comercio",
        "tefefonia de casa": "telefonia_de_casa",
        "artículos de fiesta": "artículos_de_fiesta",
        "confort hogar": "confort_hogar",
        "tabletas e impresoras": "tabletas_e_impresoras",
        "artículos de navidad": "artículos_de_navidad",
        "artes y manualidades": "artes_y_manualidades",
        "seguros y servicios": "seguros_y_servicios",
        "horno casero portátil y cafetera": "horno_casero_portatil_y_cafetera",
    }
    df.replace(replacements, inplace=True)
    return df


def clean_clients(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = df.drop_duplicates(subset="ID único de cliente", keep="first")
    df = df.reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


def clean_order_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["Fecha de entrega limite"], errors="ignore")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={"número_deproducto_id": "numero_de_producto_id"})
    return df


def run_cleaning_pipeline(data_dir: str) -> dict:
    raw = load_raw_datasets(data_dir)
    cleaned = {
        "orders": clean_orders(raw["orders"]),
        "products": clean_products(raw["products"]),
        "clients": clean_clients(raw["clients"]),
        "order_payments": clean_order_payments(raw["order_payments"]),
        "order_products": clean_order_products(raw["order_products"]),
    }
    return cleaned
