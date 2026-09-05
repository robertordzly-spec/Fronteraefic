"""
app.py — Frontera Eficiente (Markowitz) para la Cartera de Clientes
====================================================================
Industria: empaque plástico industrial.

App de Streamlit para analizar la cartera de clientes como si fuera un
portafolio de inversión (enfoque de Harry Markowitz): cada cliente es un
"activo", su "retorno" es un índice combinado de rentabilidad y
comportamiento comercial, y su "riesgo" es la variabilidad histórica de
ese retorno. El objetivo es visualizar la frontera eficiente y comparar la
cartera actual contra carteras óptimas (menor riesgo para el mismo retorno).

Fuente de datos: Excel con las hojas "Clientes", "Serie_Historica_Mensual"
y "Resumen_Concentracion" (ver README.md para el diccionario de columnas).

Este archivo solo orquesta la interfaz; los cálculos viven en:
  - data_utils.py  -> carga, validación y construcción del índice de retorno
  - markowitz.py   -> media-varianza, frontera eficiente, portafolios óptimos
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    CORRELACION_VARS_NUMERICAS,
    SchemaError,
    construir_dataset_analisis,
    construir_indice_retorno,
    load_workbook,
    to_excel_bytes,
)
from markowitz import (
    calcular_mu_sigma,
    frontera_eficiente,
    pivot_retornos,
    portafolio_maxima_eficiencia,
    portafolio_stats,
    portafolio_varianza_minima,
    simular_portafolios_aleatorios,
)

# ---------------------------------------------------------------------------
# Paleta corporativa (consistente con el resto del set de materiales)
# ---------------------------------------------------------------------------
NAVY = "#1B2733"
TEAL = "#1F4E4A"
GOLD = "#D4A537"
GREEN_OK = "#5B8C5A"
AMBER = "#D9A441"
RED = "#C0574B"
BG_SOFT = "#F4F5F0"

RIESGO_COLOR = {"Bajo": GREEN_OK, "Medio": AMBER, "Alto": RED}

SAMPLE_PATH = Path(__file__).parent / "data" / "base_datos_segmentacion_clientes.xlsx"

st.set_page_config(
    page_title="Frontera Eficiente · Cartera de Clientes",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilos (look corporativo sobre el theme de .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 1.5rem; }}
        .header-banner {{
            background: {NAVY};
            color: white;
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            margin-bottom: 1.4rem;
        }}
        .header-banner .tag {{
            display: inline-block;
            background: {TEAL};
            color: white;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            padding: 0.15rem 0.6rem;
            border-radius: 4px;
            margin-bottom: 0.6rem;
        }}
        .header-banner h1 {{
            font-size: 1.6rem;
            margin: 0.2rem 0 0.3rem 0;
        }}
        .header-banner p {{
            color: #C9CFD3;
            margin: 0;
            font-size: 0.92rem;
        }}
        .header-banner hr {{
            border: none;
            border-top: 3px solid {GOLD};
            width: 60px;
            margin: 0.7rem 0;
        }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #E4E4DE;
            border-left: 4px solid {TEAL};
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="header-banner">
        <span class="tag">MARKOWITZ · CARTERA DE CLIENTES</span>
        <h1>Frontera Eficiente — Empaque Plástico Industrial</h1>
        <hr/>
        <p>¿Cómo diseñar una cartera de clientes óptima, reduciendo la vulnerabilidad
        por concentración de ventas en pocos clientes?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral — fuente de datos y parámetros del modelo
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Datos y parámetros")

uploaded = st.sidebar.file_uploader("Cargar base de datos (.xlsx)", type=["xlsx"])
usar_ejemplo = st.sidebar.checkbox(
    "Usar archivo de ejemplo incluido en el repo", value=uploaded is None
)

st.sidebar.markdown("---")
st.sidebar.subheader("Índice de retorno combinado")
st.sidebar.caption(
    "retorno = margen_bruto_pct + w₁·cumplimiento_pago − w₂·cartera_vencida% "
    "− w₃·reclamaciones − w₄·cancelaciones"
)
w_pago = st.sidebar.slider("w₁ · Cumplimiento de pago", 0.0, 1.5, 0.30, 0.05)
w_cartera = st.sidebar.slider("w₂ · Cartera vencida", 0.0, 1.5, 0.40, 0.05)
w_reclamos = st.sidebar.slider("w₃ · Reclamaciones de calidad", 0.0, 1.5, 0.30, 0.05)
w_cancel = st.sidebar.slider("w₄ · Pedidos cancelados", 0.0, 1.5, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Modelo de Markowitz")
anualizar = st.sidebar.checkbox("Anualizar retorno y riesgo (×12 / ×√12)", value=False)
w_min = st.sidebar.slider("Peso mínimo por cliente", 0.0, 0.30, 0.00, 0.01)
w_max = st.sidebar.slider("Peso máximo por cliente (tope de concentración)", 0.10, 1.00, 0.35, 0.05)
tasa_benchmark = st.sidebar.slider("Tasa mínima aceptable (benchmark, puntos %)", -10.0, 30.0, 0.0, 0.5)
n_puntos_frontera = st.sidebar.slider("Puntos de la frontera eficiente", 10, 100, 40, 5)
n_sim = st.sidebar.slider("Portafolios simulados (nube Monte Carlo)", 500, 8000, 3000, 500)

st.sidebar.markdown("---")
incluir_indice_en_correlacion = st.sidebar.checkbox(
    "Incluir índice de retorno combinado en correlaciones/explorador", value=True
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo y validando el Excel...")
def _cargar(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return load_workbook(io.BytesIO(file_bytes))


if uploaded is not None:
    file_bytes = uploaded.getvalue()
    fuente_label = uploaded.name
elif usar_ejemplo:
    if not SAMPLE_PATH.exists():
        st.error(
            "No se encontró el archivo de ejemplo en data/. Sube tu propio Excel "
            "en la barra lateral."
        )
        st.stop()
    file_bytes = SAMPLE_PATH.read_bytes()
    fuente_label = f"{SAMPLE_PATH.name} (ejemplo incluido en el repo)"
else:
    st.info(
        "Sube un archivo Excel con las hojas Clientes / Serie_Historica_Mensual / "
        "Resumen_Concentracion, o activa 'Usar archivo de ejemplo incluido'."
    )
    st.stop()

try:
    data = _cargar(file_bytes)
except SchemaError as e:
    st.error(f"El archivo no tiene el formato esperado: {e}")
    st.stop()

st.caption(f"Fuente de datos activa: **{fuente_label}**")

df_clientes = data["clientes"]
df_hist = construir_indice_retorno(
    data["historico"],
    w_cumplimiento_pago=w_pago,
    w_cartera_vencida=w_cartera,
    w_reclamaciones=w_reclamos,
    w_cancelaciones=w_cancel,
)
df_analisis = construir_dataset_analisis({**data, "historico": df_hist})

tab_resumen, tab_corr, tab_explorador, tab_frontera = st.tabs(
    ["📊 Resumen de cartera", "🔗 Correlaciones", "🧭 Explorador de variables", "🎯 Frontera eficiente"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Resumen de cartera
# ---------------------------------------------------------------------------
with tab_resumen:
    col1, col2, col3, col4 = st.columns(4)
    ingreso_total = df_hist["ingresos_mxn"].sum()
    top1_pct = df_clientes["participacion_ventas_actual_pct"].max()
    n_alto_riesgo = (df_clientes.merge(
        data["concentracion"][["cliente_id", "nivel_riesgo_concentracion"]], on="cliente_id"
    )["nivel_riesgo_concentracion"] == "Alto").sum()

    col1.metric("Clientes en la cartera", len(df_clientes))
    col2.metric("Ingresos totales (24 meses)", f"${ingreso_total:,.0f} MXN")
    col3.metric("Concentración del cliente top 1", f"{top1_pct:.0f}%")
    col4.metric("Clientes en riesgo 'Alto'", int(n_alto_riesgo))

    st.markdown("#### Participación de ventas por cliente")
    conc = data["concentracion"].sort_values("participacion_ventas_actual_pct", ascending=False)
    fig_conc = px.bar(
        conc,
        x="participacion_ventas_actual_pct",
        y="nombre_cliente",
        color="nivel_riesgo_concentracion",
        color_discrete_map=RIESGO_COLOR,
        orientation="h",
        labels={
            "participacion_ventas_actual_pct": "% de participación en ventas",
            "nombre_cliente": "",
            "nivel_riesgo_concentracion": "Riesgo de concentración",
        },
    )
    fig_conc.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_conc, use_container_width=True)

    st.markdown("#### Catálogo de clientes")
    st.dataframe(df_clientes, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2 — Correlaciones
# ---------------------------------------------------------------------------
with tab_corr:
    st.markdown(
        "Matriz de correlación (Pearson) entre las variables solicitadas. "
        "`nivel_riesgo_concentracion` se incluye codificada de forma ordinal "
        "(Bajo=1, Medio=2, Alto=3) para poder correlacionarla con el resto."
    )

    vars_corr = list(CORRELACION_VARS_NUMERICAS)
    rename_map = {"nivel_riesgo_concentracion_ord": "nivel_riesgo_concentracion"}
    cols_para_corr = vars_corr + ["nivel_riesgo_concentracion_ord"]
    if incluir_indice_en_correlacion:
        cols_para_corr = ["indice_retorno_pct"] + cols_para_corr
        rename_map["indice_retorno_pct"] = "indice_retorno_pct"

    df_corr = df_analisis[cols_para_corr].rename(columns=rename_map)
    corr_matrix = df_corr.corr(numeric_only=True)

    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale=[[0, RED], [0.5, "#FFFFFF"], [1, TEAL]],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_heat.update_layout(height=650, paper_bgcolor="white")
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Ver tabla de correlaciones"):
        st.dataframe(corr_matrix.style.format("{:.2f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Explorador de variables (nubes de puntos)
# ---------------------------------------------------------------------------
with tab_explorador:
    st.markdown("Elige dos variables para visualizar su nube de puntos y explorar relaciones.")

    variables_disponibles = list(CORRELACION_VARS_NUMERICAS) + [
        "cumplimiento_pago_pct",
        "cartera_vencida_pct_ingresos",
    ]
    if incluir_indice_en_correlacion:
        variables_disponibles = ["indice_retorno_pct"] + variables_disponibles

    c1, c2, c3 = st.columns(3)
    eje_x = c1.selectbox("Eje X", variables_disponibles, index=variables_disponibles.index("margen_bruto_pct"))
    eje_y = c2.selectbox(
        "Eje Y", variables_disponibles,
        index=variables_disponibles.index("cartera_vencida_mxn") if "cartera_vencida_mxn" in variables_disponibles else 0,
    )
    color_por = c3.selectbox(
        "Colorear por", ["sector_industria", "region_geografica", "nivel_riesgo_concentracion", "nombre_cliente"]
    )

    fig_scatter = px.scatter(
        df_analisis,
        x=eje_x,
        y=eje_y,
        color=color_por,
        hover_data=["nombre_cliente", "periodo"],
        opacity=0.75,
        color_discrete_map=RIESGO_COLOR if color_por == "nivel_riesgo_concentracion" else None,
    )
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=550)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Cada punto es una observación mensual de un cliente (10 clientes × 24 meses). "
        "Usa esta vista para detectar relaciones antes de interpretar la frontera eficiente."
    )

# ---------------------------------------------------------------------------
# TAB 4 — Frontera eficiente (Markowitz)
# ---------------------------------------------------------------------------
with tab_frontera:
    tabla_retornos = pivot_retornos(df_hist, value_col="indice_retorno_pct")
    mu, sigma = calcular_mu_sigma(tabla_retornos, anualizar=anualizar)

    orden = mu.index.tolist()
    nombres = df_clientes.set_index("cliente_id").reindex(orden)["nombre_cliente"]
    riesgo_individual = np.sqrt(np.diag(sigma.values))

    st.markdown("#### Retorno esperado y riesgo individual por cliente")
    st.caption(
        f"{'Cifras anualizadas' if anualizar else 'Cifras mensuales'} del índice de retorno "
        "combinado (puntos porcentuales). El 'riesgo' es la desviación estándar histórica "
        "de ese índice."
    )
    df_mu_sigma = pd.DataFrame({
        "cliente_id": orden,
        "nombre_cliente": nombres.values,
        "retorno_esperado_pct": mu.values,
        "riesgo_pct": riesgo_individual,
        "participacion_actual_pct": df_clientes.set_index("cliente_id").reindex(orden)["participacion_ventas_actual_pct"].values,
    }).sort_values("retorno_esperado_pct", ascending=False)
    st.dataframe(df_mu_sigma.style.format({
        "retorno_esperado_pct": "{:.2f}", "riesgo_pct": "{:.2f}", "participacion_actual_pct": "{:.0f}"
    }), use_container_width=True, hide_index=True)

    with st.expander("Ver matriz de covarianza entre clientes"):
        st.dataframe(sigma.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Ver matriz de correlación entre los retornos de los clientes"):
        st.dataframe(tabla_retornos.corr().style.format("{:.2f}"), use_container_width=True)

    # ---- Portafolios de referencia ----
    w_actual = df_clientes.set_index("cliente_id").reindex(orden)["participacion_ventas_actual_pct"].values / 100.0
    ret_actual, vol_actual = portafolio_stats(w_actual, mu.values, sigma.values)

    gmvp = portafolio_varianza_minima(mu, sigma, w_min, w_max)
    max_efi = portafolio_maxima_eficiencia(mu, sigma, tasa_benchmark, w_min, w_max)

    frontera_df = frontera_eficiente(mu, sigma, n_puntos_frontera, w_min, w_max)
    sim_df = simular_portafolios_aleatorios(mu, sigma, n_sim, w_max)

    st.markdown("#### Frontera eficiente")
    fig = go.Figure()

    if not sim_df.empty:
        fig.add_trace(go.Scatter(
            x=sim_df["riesgo"], y=sim_df["retorno"], mode="markers",
            marker=dict(size=4, color="#C9CFD3", opacity=0.5),
            name="Portafolios simulados (Monte Carlo)",
        ))

    if not frontera_df.empty:
        fig.add_trace(go.Scatter(
            x=frontera_df["riesgo"], y=frontera_df["retorno"], mode="lines+markers",
            line=dict(color=TEAL, width=3), marker=dict(size=5, color=TEAL),
            name="Frontera eficiente",
        ))

    fig.add_trace(go.Scatter(
        x=riesgo_individual, y=mu.values, mode="markers+text",
        marker=dict(size=10, color=NAVY, symbol="circle"),
        text=orden, textposition="top center",
        name="Clientes individuales",
    ))

    fig.add_trace(go.Scatter(
        x=[vol_actual], y=[ret_actual], mode="markers",
        marker=dict(size=16, color=RED, symbol="star"),
        name="Cartera actual",
    ))
    fig.add_trace(go.Scatter(
        x=[gmvp.vol], y=[gmvp.ret], mode="markers",
        marker=dict(size=14, color=GOLD, symbol="diamond"),
        name="Mínima varianza (GMVP)",
    ))
    fig.add_trace(go.Scatter(
        x=[max_efi.vol], y=[max_efi.ret], mode="markers",
        marker=dict(size=14, color=GREEN_OK, symbol="diamond"),
        name="Máxima eficiencia (retorno/riesgo)",
    ))

    fig.update_layout(
        xaxis_title="Riesgo (desviación estándar del retorno, %)",
        yaxis_title="Retorno esperado (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Comparación de carteras")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Cartera actual — Retorno", f"{ret_actual:.2f}%")
        st.metric("Cartera actual — Riesgo", f"{vol_actual:.2f}%")
    with colB:
        st.metric("Mínima varianza — Retorno", f"{gmvp.ret:.2f}%")
        st.metric("Mínima varianza — Riesgo", f"{gmvp.vol:.2f}%")
    with colC:
        st.metric("Máx. eficiencia — Retorno", f"{max_efi.ret:.2f}%")
        st.metric("Máx. eficiencia — Riesgo", f"{max_efi.vol:.2f}%")

    st.markdown("#### Pesos por cliente: actual vs. carteras óptimas")
    df_pesos = pd.DataFrame({
        "cliente_id": orden,
        "nombre_cliente": nombres.values,
        "Cartera actual": w_actual * 100,
        "Mínima varianza": gmvp.weights * 100,
        "Máxima eficiencia": max_efi.weights * 100,
    })
    df_pesos_melt = df_pesos.melt(
        id_vars=["cliente_id", "nombre_cliente"], var_name="Cartera", value_name="Peso (%)"
    )
    fig_pesos = px.bar(
        df_pesos_melt, x="nombre_cliente", y="Peso (%)", color="Cartera", barmode="group",
        color_discrete_map={"Cartera actual": RED, "Mínima varianza": GOLD, "Máxima eficiencia": GREEN_OK},
    )
    fig_pesos.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", height=500)
    st.plotly_chart(fig_pesos, use_container_width=True)

    st.markdown("#### Fórmulas usadas")
    st.latex(r"R_p(w) = w^\top \mu \qquad \sigma_p(w) = \sqrt{w^\top \Sigma w}")
    st.latex(
        r"\min_w \; w^\top \Sigma w \quad \text{s.a.} \quad w^\top \mu = R^{*},"
        r"\;\; \sum_i w_i = 1, \;\; w_{min} \le w_i \le w_{max}"
    )
    st.caption(
        "μ = retorno esperado por cliente (promedio histórico del índice combinado). "
        "Σ = matriz de covarianzas entre clientes. La frontera eficiente resuelve, para "
        "cada retorno objetivo R*, el portafolio de mínima varianza. Ver README.md para "
        "el detalle completo de fórmulas y supuestos."
    )

    excel_bytes = to_excel_bytes({
        "Retorno_Riesgo_Clientes": df_mu_sigma,
        "Frontera_Eficiente": frontera_df,
        "Pesos_Carteras": df_pesos,
    })
    st.download_button(
        "⬇️ Descargar resultados (Excel)",
        data=excel_bytes,
        file_name="resultados_frontera_eficiente.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
