# Frontera Eficiente — Cartera de Clientes (Empaque Plástico Industrial)

Aplicación en **Streamlit** para analizar la cartera de clientes de una empresa de
empaque plástico industrial usando el enfoque de **media-varianza de Markowitz**:
cada cliente se trata como un "activo", con un retorno esperado y un riesgo
(volatilidad) estimados a partir de su comportamiento histórico, y se calcula la
**frontera eficiente** de carteras (combinaciones de % de atención/ventas por
cliente) que minimizan el riesgo para cada nivel de retorno.

> ⚠️ **Alcance del proyecto**: esta app **sí incluye** el modelo de optimización
> de frontera eficiente (a diferencia de la base de datos entregada previamente,
> que era solo el insumo de datos). Los datos de ejemplo incluidos en `data/` son
> **sintéticos**, generados para pruebas — antes de usar resultados para decisiones
> reales, sustituye el archivo por datos verdaderos de tu ERP/CRM con la misma
> estructura de columnas.

## 1. Estructura del repositorio

```
.
├── app.py                # Interfaz de Streamlit (orquesta todo, sin lógica de negocio)
├── data_utils.py         # Carga/validación del Excel + construcción del índice de retorno
├── markowitz.py          # Media-varianza: mu, Sigma, frontera eficiente, portafolios óptimos
├── data/
│   └── base_datos_segmentacion_clientes.xlsx   # Datos de ejemplo (sintéticos)
├── .streamlit/
│   └── config.toml       # Tema visual corporativo
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Cómo correrlo en local

```bash
git clone <url-de-tu-repo>
cd frontera-eficiente-clientes
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá en `http://localhost:8501`. Por defecto usa el Excel de ejemplo en
`data/`; puedes cargar tu propio archivo desde la barra lateral.

## 3. Cómo subirlo a GitHub

```bash
git init
git add .
git commit -m "Frontera eficiente - cartera de clientes"
git branch -M main
git remote add origin <url-de-tu-repo-en-github>
git push -u origin main
```

## 4. Cómo desplegarlo en Streamlit Community Cloud

1. Sube el repo a GitHub (paso anterior).
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta.
3. "New app" → selecciona el repo y la rama `main`.
4. Main file path: `app.py`.
5. Deploy. La app queda disponible en una URL pública (puedes restringir acceso
   desde la configuración de la app si es información sensible).

## 5. Estructura de datos esperada (Excel)

El archivo debe tener estas tres hojas (los encabezados pueden ir precedidos de
un título/subtítulo — la app los detecta automáticamente):

**Clientes** — un renglón por cliente:
`cliente_id, nombre_cliente, sector_industria, tipo_empaque_principal,
region_geografica, participacion_ventas_actual_pct, antiguedad_relacion_anios,
tipo_contrato, condicion_pago_acordada_dias, cliente_unico_en_segmento`

**Serie_Historica_Mensual** — un renglón por cliente y mes:
`cliente_id, nombre_cliente, periodo, volumen_vendido_ton,
precio_promedio_venta_mxn_kg, ingresos_mxn, margen_bruto_pct,
dias_pago_acordados, dias_pago_reales, cartera_vencida_mxn,
reclamaciones_calidad, pedidos_cancelados`

**Resumen_Concentracion** — un renglón por cliente:
`cliente_id, nombre_cliente, participacion_ventas_actual_pct,
nivel_riesgo_concentracion` (Bajo / Medio / Alto)

## 6. Metodología y fórmulas

### 6.1 Índice de retorno combinado (por cliente y mes)

No existe un "retorno" financiero explícito para un cliente, así que se construye
un índice que combina rentabilidad y comportamiento comercial:

```
cumplimiento_pago_pct        = (dias_pago_acordados − dias_pago_reales) / dias_pago_acordados × 100
cartera_vencida_pct_ingresos = cartera_vencida_mxn / ingresos_mxn × 100

indice_retorno_pct = margen_bruto_pct
                      + w1 · cumplimiento_pago_pct
                      − w2 · cartera_vencida_pct_ingresos
                      − w3 · reclamaciones_calidad
                      − w4 · pedidos_cancelados
```

Los pesos `w1..w4` son ajustables desde la barra lateral de la app (valores por
defecto: 0.30, 0.40, 0.30, 0.30). `margen_bruto_pct` entra con peso fijo 1 porque
es la base del retorno; los demás términos son ajustes por comportamiento. Esta
fórmula es una **propuesta razonable, no una convención universal** — ajústala a
tu criterio de negocio.

### 6.2 Media-varianza de Markowitz

Con la serie mensual del índice de retorno por cliente (24 meses × 10 clientes
en el ejemplo):

- **Retorno esperado**: `μ_i = promedio histórico de indice_retorno_pct del cliente i`
- **Matriz de covarianzas**: `Σ_ij = covarianza histórica entre los clientes i, j`
- **Retorno de un portafolio**: `R_p(w) = wᵀμ`
- **Riesgo de un portafolio**: `σ_p(w) = √(wᵀΣw)`

Si "Anualizar" está activo, `μ` se multiplica por 12 y `Σ` por 12 (supuesto
simplificador estándar de independencia mes a mes).

### 6.3 Frontera eficiente

Para una malla de retornos objetivo `R*` entre el mínimo y el máximo `μ_i`
observado, se resuelve:

```
min_w   wᵀΣw
s.a.    wᵀμ = R*
        Σ wᵢ = 1
        w_min ≤ wᵢ ≤ w_max
```

`w_max` es el **tope de concentración por cliente** (evita que la solución
óptima vuelva a concentrarse en un solo cliente, que es justamente el problema
de negocio que se quiere resolver). Se resuelve con `scipy.optimize.minimize`
(SLSQP) en `markowitz.py`.

También se calculan:
- **GMVP** (Global Minimum Variance Portfolio): portafolio de menor riesgo posible.
- **Portafolio de máxima eficiencia**: maximiza `(R_p − tasa_benchmark) / σ_p`
  (análogo al ratio de Sharpe, usando una tasa mínima aceptable en vez de una
  tasa libre de riesgo de mercado).
- **Nube de portafolios simulados** (Monte Carlo, pesos aleatorios vía
  distribución de Dirichlet) — solo como referencia visual de la región factible.

## 7. Qué muestra cada pestaña de la app

1. **Resumen de cartera** — KPIs generales y participación de ventas por cliente.
2. **Correlaciones** — mapa de calor y tabla de correlación de Pearson entre las
   variables solicitadas (incluye `nivel_riesgo_concentracion` codificada de
   forma ordinal: Bajo=1, Medio=2, Alto=3).
3. **Explorador de variables** — nube de puntos interactiva entre cualquier par
   de variables numéricas, coloreada por sector/región/riesgo/cliente.
4. **Frontera eficiente** — retorno/riesgo por cliente, matriz de covarianza,
   gráfica de la frontera eficiente con la cartera actual, GMVP y portafolio de
   máxima eficiencia marcados, comparación de pesos, y descarga de resultados
   en Excel.

## 8. Limitaciones a tener en cuenta

- 24 meses de historia es una muestra pequeña para estimar covarianzas con
  precisión estadística; los resultados son ilustrativos.
- El índice de retorno combinado es una construcción de negocio, no un
  estándar contable ni financiero.
- El modelo asume que "reasignar ventas" entre clientes es operativamente
  posible, lo cual en la práctica está limitado por contratos, capacidad y
  relaciones comerciales — úsalo como herramienta de diagnóstico, no como
  regla mecánica de decisión.
