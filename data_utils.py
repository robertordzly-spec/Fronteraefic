"""
data_utils.py
--------------
Carga, valida y prepara la base de datos de clientes (Excel) para el modelo
de frontera eficiente. No contiene ninguna lógica de optimización: solo
lectura, limpieza y construcción del "índice de retorno combinado" que
alimenta al módulo markowitz.py.
"""
from __future__ import annotations

import io
import pandas as pd

# Columnas mínimas requeridas en cada hoja, según los "Inputs" del brief.
REQUIRED_CLIENTES_COLS = [
    "cliente_id", "nombre_cliente", "sector_industria", "tipo_empaque_principal",
    "region_geografica", "participacion_ventas_actual_pct", "antiguedad_relacion_anios",
    "tipo_contrato", "condicion_pago_acordada_dias", "cliente_unico_en_segmento",
]

REQUIRED_HIST_COLS = [
    "cliente_id", "nombre_cliente", "periodo", "volumen_vendido_ton",
    "precio_promedio_venta_mxn_kg", "ingresos_mxn", "margen_bruto_pct",
    "dias_pago_acordados", "dias_pago_reales", "cartera_vencida_mxn",
    "reclamaciones_calidad", "pedidos_cancelados",
]

REQUIRED_CONCENTRACION_COLS = [
    "cliente_id", "nombre_cliente", "participacion_ventas_actual_pct",
    "nivel_riesgo_concentracion",
]

# Variables numéricas del brief para la matriz/gráfica de correlaciones y el
# explorador de nubes de puntos.
CORRELACION_VARS_NUMERICAS = [
    "volumen_vendido_ton", "precio_promedio_venta_mxn_kg", "ingresos_mxn",
    "margen_bruto_pct", "dias_pago_acordados", "dias_pago_reales",
    "cartera_vencida_mxn", "reclamaciones_calidad", "pedidos_cancelados",
]

# Codificación ordinal para poder incluir la variable categórica
# nivel_riesgo_concentracion en una matriz de correlación de Pearson.
NIVEL_RIESGO_ORDINAL = {"Bajo": 1, "Medio": 2, "Alto": 3}


class SchemaError(Exception):
    """Se lanza cuando el archivo cargado no tiene las columnas esperadas."""


def _check_columns(df: pd.DataFrame, required: list[str], sheet_name: str) -> None:
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise SchemaError(
            f"A la hoja '{sheet_name}' le faltan columnas requeridas: {', '.join(faltantes)}"
        )


def load_workbook(file_like) -> dict[str, pd.DataFrame]:
    """
    Lee el Excel de origen y regresa un diccionario con los tres DataFrames
    esperados: clientes, historico y concentracion. `file_like` puede ser una
    ruta (str/Path) o un objeto tipo archivo (por ejemplo, el resultado de
    st.file_uploader).
    """
    xls = pd.ExcelFile(file_like)

    def _read(sheet):
        # Las hojas se generaron con un título y subtítulo en las primeras
        # filas (ver skill xlsx), por lo que se detecta automáticamente cuál
        # fila contiene los encabezados reales buscando 'cliente_id'.
        raw = xls.parse(sheet, header=None)
        header_row = None
        for i in range(min(5, len(raw))):
            if raw.iloc[i].astype(str).str.contains("cliente_id", case=False, na=False).any():
                header_row = i
                break
        if header_row is None:
            header_row = 0
        return xls.parse(sheet, header=header_row)

    clientes = _read("Clientes")
    historico = _read("Serie_Historica_Mensual")
    concentracion = _read("Resumen_Concentracion")

    _check_columns(clientes, REQUIRED_CLIENTES_COLS, "Clientes")
    _check_columns(historico, REQUIRED_HIST_COLS, "Serie_Historica_Mensual")
    _check_columns(concentracion, REQUIRED_CONCENTRACION_COLS, "Resumen_Concentracion")

    return {"clientes": clientes, "historico": historico, "concentracion": concentracion}


def construir_dataset_analisis(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Une la serie histórica con el nivel de riesgo de concentración de cada
    cliente (dato estático por cliente) para tener una sola tabla lista para
    correlaciones y para el explorador de variables.
    """
    hist = data["historico"].copy()
    conc = data["concentracion"][["cliente_id", "nivel_riesgo_concentracion"]].drop_duplicates()
    df = hist.merge(conc, on="cliente_id", how="left")
    df["nivel_riesgo_concentracion_ord"] = df["nivel_riesgo_concentracion"].map(NIVEL_RIESGO_ORDINAL)
    return df


def construir_indice_retorno(
    df_hist: pd.DataFrame,
    w_cumplimiento_pago: float = 0.3,
    w_cartera_vencida: float = 0.4,
    w_reclamaciones: float = 0.3,
    w_cancelaciones: float = 0.3,
) -> pd.DataFrame:
    """
    Construye, por cliente y periodo, el 'índice de retorno combinado' (%)
    sugerido en el diccionario de variables de la base:

        indice_retorno_pct = margen_bruto_pct
                              + w_pago        * cumplimiento_pago_pct
                              - w_cartera     * cartera_vencida_pct_ingresos
                              - w_reclamos    * reclamaciones_calidad
                              - w_cancelacion * pedidos_cancelados

    donde:
      cumplimiento_pago_pct = (dias_pago_acordados - dias_pago_reales)
                               / dias_pago_acordados * 100
        (positivo = el cliente pagó antes de lo acordado; negativo = pagó tarde)

      cartera_vencida_pct_ingresos = cartera_vencida_mxn / ingresos_mxn * 100

    Los pesos w_* son parámetros de negocio (ajustables en la app); el margen
    bruto entra con peso fijo 1 porque es la base del "retorno" del cliente.
    Esta función NO calcula retorno esperado, riesgo ni covarianza de
    portafolio — solo prepara la serie mensual que usará markowitz.py.
    """
    df = df_hist.copy()

    df["cumplimiento_pago_pct"] = (
        (df["dias_pago_acordados"] - df["dias_pago_reales"]) / df["dias_pago_acordados"] * 100
    )

    ingresos_seguro = df["ingresos_mxn"].replace(0, pd.NA)
    df["cartera_vencida_pct_ingresos"] = (df["cartera_vencida_mxn"] / ingresos_seguro * 100).fillna(0)

    df["indice_retorno_pct"] = (
        df["margen_bruto_pct"]
        + w_cumplimiento_pago * df["cumplimiento_pago_pct"]
        - w_cartera_vencida * df["cartera_vencida_pct_ingresos"]
        - w_reclamaciones * df["reclamaciones_calidad"]
        - w_cancelaciones * df["pedidos_cancelados"]
    )
    return df


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Serializa uno o varios DataFrames a bytes de un .xlsx (para descarga)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()
