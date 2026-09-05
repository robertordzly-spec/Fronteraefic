"""
markowitz.py
------------
Implementación clásica de media-varianza (Markowitz) sobre la cartera de
clientes. Cada cliente es un "activo"; su serie mensual de indice_retorno_pct
(construida en data_utils.py) hace las veces de serie de retornos.

Fórmulas usadas (notación estándar de Markowitz):
    mu_i      = E[r_i]                         (retorno esperado del activo i)
    Sigma_ij  = Cov(r_i, r_j)                   (matriz de covarianzas)
    R_p(w)    = w' * mu                         (retorno esperado del portafolio)
    Var_p(w)  = w' * Sigma * w                  (varianza del portafolio)
    sigma_p(w)= sqrt(Var_p(w))                  (riesgo/volatilidad del portafolio)

Frontera eficiente: para cada retorno objetivo R*, se resuelve
    min_w   w' * Sigma * w
    s.a.    w' * mu = R*
            sum(w) = 1
            w_min <= w_i <= w_max   (cotas de concentración por cliente)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimResult:
    weights: np.ndarray
    ret: float
    vol: float
    success: bool


def pivot_retornos(df_hist: pd.DataFrame, value_col: str = "indice_retorno_pct") -> pd.DataFrame:
    """Convierte la serie larga (cliente, periodo, valor) a una matriz ancha
    periodo x cliente, requerida para calcular medias/covarianzas."""
    tabla = df_hist.pivot_table(index="periodo", columns="cliente_id", values=value_col)
    tabla = tabla.sort_index()
    return tabla


def calcular_mu_sigma(tabla_retornos: pd.DataFrame, anualizar: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calcula el vector de retornos esperados (mu) y la matriz de covarianzas
    (Sigma) a partir de la tabla ancha de retornos mensuales por cliente.
    Si `anualizar` es True, mu se multiplica por 12 y Sigma por 12 (supuesto
    estándar de independencia serial mes a mes; es una aproximación, no un
    ajuste econométrico riguroso).
    """
    mu = tabla_retornos.mean()
    sigma = tabla_retornos.cov()
    if anualizar:
        mu = mu * 12
        sigma = sigma * 12
    return mu, sigma


def portafolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Retorno y volatilidad (desv. estándar) de un portafolio con pesos w."""
    ret = float(w @ mu)
    var = float(w @ sigma @ w)
    return ret, float(np.sqrt(max(var, 0.0)))


def _bounds(n: int, w_min: float, w_max: float):
    return tuple((w_min, w_max) for _ in range(n))


def portafolio_varianza_minima(
    mu: pd.Series, sigma: pd.DataFrame, w_min: float = 0.0, w_max: float = 1.0
) -> OptimResult:
    """Global Minimum Variance Portfolio (GMVP): minimiza el riesgo sin fijar
    un retorno objetivo, solo sum(w) = 1 y cotas por cliente."""
    n = len(mu)
    w0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(
        lambda w: w @ sigma.values @ w,
        w0,
        method="SLSQP",
        bounds=_bounds(n, w_min, w_max),
        constraints=cons,
    )
    ret, vol = portafolio_stats(res.x, mu.values, sigma.values)
    return OptimResult(res.x, ret, vol, res.success)


def portafolio_retorno_objetivo(
    mu: pd.Series, sigma: pd.DataFrame, retorno_objetivo: float, w_min: float = 0.0, w_max: float = 1.0
) -> OptimResult:
    """Resuelve el portafolio de varianza mínima para un retorno objetivo dado
    (un punto de la frontera eficiente)."""
    n = len(mu)
    w0 = np.repeat(1 / n, n)
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu.values - retorno_objetivo},
    ]
    res = minimize(
        lambda w: w @ sigma.values @ w,
        w0,
        method="SLSQP",
        bounds=_bounds(n, w_min, w_max),
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    ret, vol = portafolio_stats(res.x, mu.values, sigma.values)
    return OptimResult(res.x, ret, vol, res.success)


def frontera_eficiente(
    mu: pd.Series, sigma: pd.DataFrame, n_puntos: int = 40, w_min: float = 0.0, w_max: float = 1.0
) -> pd.DataFrame:
    """
    Calcula portafolios de varianza mínima para retornos objetivo espaciados
    entre el retorno del GMVP y el retorno más alto alcanzable, dadas las
    cotas w_min/w_max, y regresa solo la rama **eficiente** (a mayor retorno,
    mayor o igual riesgo). La rama inferior de la curva de varianza mínima
    (retorno objetivo por debajo del GMVP) es matemáticamente válida pero no
    es "eficiente" -- para ese mismo riesgo existe un portafolio con más
    retorno -- así que se descarta aquí.
    """
    gmvp = portafolio_varianza_minima(mu, sigma, w_min, w_max)
    objetivos = np.linspace(gmvp.ret, mu.max(), n_puntos)
    filas = []
    for objetivo in objetivos:
        res = portafolio_retorno_objetivo(mu, sigma, objetivo, w_min, w_max)
        if res.success:
            fila = {"retorno": res.ret, "riesgo": res.vol}
            fila.update({f"peso_{cid}": w for cid, w in zip(mu.index, res.weights)})
            filas.append(fila)
    return pd.DataFrame(filas).sort_values("riesgo").reset_index(drop=True)


def portafolio_maxima_eficiencia(
    mu: pd.Series, sigma: pd.DataFrame, tasa_benchmark: float = 0.0, w_min: float = 0.0, w_max: float = 1.0
) -> OptimResult:
    """
    Portafolio que maximiza el ratio (retorno - tasa_benchmark) / riesgo
    (análogo al ratio de Sharpe, usando una tasa mínima aceptable de
    referencia en lugar de una tasa libre de riesgo de mercado).
    """
    n = len(mu)
    w0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_ratio(w):
        ret, vol = portafolio_stats(w, mu.values, sigma.values)
        if vol == 0:
            return 0.0
        return -(ret - tasa_benchmark) / vol

    res = minimize(
        neg_ratio, w0, method="SLSQP", bounds=_bounds(n, w_min, w_max), constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    ret, vol = portafolio_stats(res.x, mu.values, sigma.values)
    return OptimResult(res.x, ret, vol, res.success)


def simular_portafolios_aleatorios(
    mu: pd.Series, sigma: pd.DataFrame, n_sim: int = 3000, w_max: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """
    Genera una nube de portafolios aleatorios (pesos que suman 1, respetando
    el tope w_max por cliente) para visualizar la región factible alrededor
    de la frontera eficiente calculada analíticamente. Es solo un recurso
    visual (Monte Carlo), no reemplaza la optimización.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    resultados = []
    intentos = 0
    while len(resultados) < n_sim and intentos < n_sim * 30:
        intentos += 1
        w = rng.dirichlet(np.ones(n))
        if w.max() > w_max:
            continue
        ret, vol = portafolio_stats(w, mu.values, sigma.values)
        resultados.append((ret, vol))
    return pd.DataFrame(resultados, columns=["retorno", "riesgo"])
