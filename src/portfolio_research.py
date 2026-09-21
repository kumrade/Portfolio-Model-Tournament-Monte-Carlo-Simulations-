"""Core research logic for portfolio model comparison and Monte Carlo risk analysis."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
COVARIANCE_METHODS = ("Historical", "EWMA", "Ledoit-Wolf", "DCC-GARCH")
RETURN_METHODS = ("Historical mean", "EWMA mean", "CAPM", "Shrunk mean")
OBJECTIVES = ("Maximum Sharpe", "Minimum Variance", "Target Return")


def generate_demo_prices(symbols: list[str], observations: int = 504, seed: int = 7) -> pd.DataFrame:
    """Generate correlated, crash-aware synthetic prices for reproducible demonstrations."""
    rng = np.random.default_rng(seed)
    n = len(symbols)
    beta = rng.uniform(0.7, 1.3, n)
    idio = rng.uniform(0.008, 0.014, n)
    drift = rng.uniform(0.00015, 0.00055, n)
    market_vol = np.empty(observations)
    market_vol[0] = 0.010
    for t in range(1, observations):
        market_vol[t] = 0.97 * market_vol[t - 1] + 0.03 * 0.010
        market_vol[t] *= math.exp(0.10 * rng.standard_normal())
    market = 0.0002 + market_vol * rng.standard_normal(observations)
    start = int(observations * 0.58)
    market[start:start + 8] -= 0.012
    returns = np.column_stack([
        drift[i] + beta[i] * market + idio[i] * rng.standard_normal(observations)
        for i in range(n)
    ])
    prices = rng.uniform(300, 3000, n) * np.exp(np.cumsum(returns, axis=0))
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=observations)
    return pd.DataFrame(prices, index=index, columns=symbols)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index().ffill()
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 80 or returns.shape[1] < 2:
        raise ValueError("At least 80 observations and two assets are required.")
    return returns


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.clip(values, 1e-10, None)) @ vectors.T


def historical_covariance(returns: pd.DataFrame) -> np.ndarray:
    return nearest_psd(returns.cov().to_numpy() * TRADING_DAYS)


def ewma_covariance(returns: pd.DataFrame, decay: float = 0.94) -> np.ndarray:
    x = returns.to_numpy()
    covariance = np.cov(x, rowvar=False)
    for row in x:
        covariance = decay * covariance + (1 - decay) * np.outer(row, row)
    return nearest_psd(covariance * TRADING_DAYS)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    x = returns.to_numpy(dtype=float)
    x -= x.mean(axis=0)
    observations, assets = x.shape
    sample = x.T @ x / observations
    variances = np.diag(sample)
    std = np.sqrt(np.clip(variances, 1e-16, None))
    correlation = sample / np.outer(std, std)
    average_correlation = (correlation.sum() - assets) / (assets * (assets - 1))
    target = average_correlation * np.outer(std, std)
    np.fill_diagonal(target, variances)
    pi_matrix = (x ** 2).T @ (x ** 2) / observations - sample ** 2
    pi_hat = pi_matrix.sum()
    gamma_hat = np.sum((target - sample) ** 2)
    shrinkage = float(np.clip(pi_hat / max(gamma_hat * observations, 1e-16), 0, 1))
    return nearest_psd((shrinkage * target + (1 - shrinkage) * sample) * TRADING_DAYS)


@dataclass
class GarchFit:
    variance: np.ndarray
    residual: np.ndarray
    forecast: float


def _fit_garch(series: np.ndarray) -> GarchFit:
    residual = np.asarray(series, float) - np.mean(series)
    initial = max(float(np.var(residual)), 1e-10)

    def recursion(parameters):
        omega, alpha, beta = parameters
        variance = np.empty(len(residual)); variance[0] = initial
        for t in range(1, len(residual)):
            variance[t] = omega + alpha * residual[t - 1] ** 2 + beta * variance[t - 1]
        return variance

    def objective(parameters):
        omega, alpha, beta = parameters
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        variance = recursion(parameters)
        return 0.5 * np.sum(np.log(variance) + residual ** 2 / variance)

    fit = minimize(objective, [initial * 0.05, 0.05, 0.90], method="L-BFGS-B",
                   bounds=[(1e-12, None), (0, 0.999), (0, 0.999)])
    parameters = fit.x if fit.success and fit.x[1] + fit.x[2] < 0.999 else np.array([initial * 0.05, 0.05, 0.90])
    variance = recursion(parameters)
    forecast = parameters[0] + parameters[1] * residual[-1] ** 2 + parameters[2] * variance[-1]
    return GarchFit(variance, residual, float(forecast))


def dcc_garch_covariance(returns: pd.DataFrame) -> np.ndarray:
    x = returns.to_numpy(dtype=float)
    fits = [_fit_garch(x[:, i]) for i in range(x.shape[1])]
    standardized = np.column_stack([f.residual / np.sqrt(f.variance) for f in fits])
    long_run = nearest_psd(np.cov(standardized, rowvar=False))
    a, b = 0.02, 0.95
    q = long_run.copy()
    for row in standardized:
        q = (1 - a - b) * long_run + a * np.outer(row, row) + b * q
    scale = np.sqrt(np.clip(np.diag(q), 1e-12, None))
    correlation = q / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    volatility = np.sqrt([f.forecast for f in fits])
    return nearest_psd(np.diag(volatility) @ correlation @ np.diag(volatility) * TRADING_DAYS)


def covariance_matrix(returns: pd.DataFrame, method: str) -> np.ndarray:
    functions = {"Historical": historical_covariance, "EWMA": ewma_covariance,
                 "Ledoit-Wolf": ledoit_wolf_covariance, "DCC-GARCH": dcc_garch_covariance}
    if method not in functions:
        raise ValueError(f"Unknown covariance method: {method}")
    return functions[method](returns)


def expected_returns(returns: pd.DataFrame, method: str, risk_free: float = 0.065) -> np.ndarray:
    x = returns.to_numpy()
    historical = x.mean(axis=0) * TRADING_DAYS
    if method == "Historical mean":
        return historical
    if method == "EWMA mean":
        ages = np.arange(len(x) - 1, -1, -1)
        weights = 0.06 * 0.94 ** ages; weights /= weights.sum()
        return (weights[:, None] * x).sum(axis=0) * TRADING_DAYS
    if method == "CAPM":
        market = x.mean(axis=1); market_variance = max(float(np.var(market)), 1e-12)
        beta = np.array([np.cov(x[:, i], market)[0, 1] / market_variance for i in range(x.shape[1])])
        return risk_free + beta * (market.mean() * TRADING_DAYS - risk_free)
    if method == "Shrunk mean":
        return 0.5 * historical + 0.5 * historical.mean()
    raise ValueError(f"Unknown return method: {method}")


def optimize(mu: np.ndarray, covariance: np.ndarray, objective: str,
             max_weight: float = 0.40, risk_free: float = 0.065) -> np.ndarray:
    n = len(mu); max_weight = max(max_weight, 1 / n + 1e-9)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if objective == "Maximum Sharpe":
        loss = lambda w: -(w @ mu - risk_free) / math.sqrt(max(float(w @ covariance @ w), 1e-18))
    elif objective == "Minimum Variance":
        loss = lambda w: float(w @ covariance @ w)
    elif objective == "Target Return":
        target = float(np.median(mu))
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target})
        loss = lambda w: float(w @ covariance @ w)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    start = np.full(n, 1 / n)
    result = minimize(loss, start, method="SLSQP", bounds=[(0, max_weight)] * n,
                      constraints=constraints, options={"maxiter": 800, "ftol": 1e-10})
    weights = result.x if result.success else start
    weights = np.clip(weights, 0, max_weight); return weights / weights.sum()


def realised_statistics(daily_returns: np.ndarray, risk_free: float = 0.065) -> dict[str, float]:
    returns = np.asarray(daily_returns)
    annual_return = float(np.expm1(np.log1p(np.clip(returns, -0.999, None)).sum() * TRADING_DAYS / len(returns)))
    volatility = float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS))
    downside = returns[returns < 0]
    downside_volatility = float(np.std(downside, ddof=1) * math.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0
    wealth = np.cumprod(1 + np.clip(returns, -0.999, None))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    threshold = np.percentile(returns, 5); tail = returns[returns <= threshold]
    return {"oos_return": annual_return, "oos_volatility": volatility,
            "oos_sharpe": (annual_return - risk_free) / volatility if volatility else 0,
            "oos_sortino": (annual_return - risk_free) / downside_volatility if downside_volatility else 0,
            "max_drawdown": float(drawdown.min()), "tail_loss": float(-tail.mean())}


def _rank(values, higher=True):
    result = pd.Series(values).rank(pct=True, method="average")
    return result if higher else 1 - result + 1 / len(result)


def compare_covariances(returns: pd.DataFrame, train_window: int = 252,
                        test_window: int = 21) -> pd.DataFrame:
    starts = list(range(min(train_window, len(returns) - 42), len(returns) - 1, test_window))
    records = []
    for method in COVARIANCE_METHODS:
        losses, errors, conditions = [], [], []
        for start in starts:
            train = returns.iloc[max(0, start - train_window):start]
            test = returns.iloc[start:min(start + test_window, len(returns))]
            forecast = covariance_matrix(train, method) / TRADING_DAYS
            realised = np.atleast_2d(np.cov(test.to_numpy(), rowvar=False))
            sign, logdet = np.linalg.slogdet(forecast)
            inverse = np.linalg.pinv(forecast)
            quadratic = np.einsum("ij,jk,ik->i", test.to_numpy(), inverse, test.to_numpy())
            losses.append(0.5 * (logdet + quadratic.mean()))
            errors.append(np.linalg.norm(forecast - realised, "fro") / (np.linalg.norm(realised, "fro") + 1e-12))
            conditions.append(np.linalg.cond(forecast))
        records.append({"method": method, "gaussian_loss": np.mean(losses),
                        "forecast_error": np.mean(errors), "condition_number": np.median(conditions),
                        "folds": len(losses)})
    table = pd.DataFrame(records)
    table["score"] = 0.65 * _rank(table.gaussian_loss, False) + 0.25 * _rank(table.forecast_error, False) + 0.10 * _rank(table.condition_number, False)
    table = table.sort_values(["score", "gaussian_loss"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1)); return table


def compare_models(returns: pd.DataFrame, max_weight: float = 0.40,
                   train_window: int = 252, rebalance_window: int = 21,
                   cost_bps: float = 10) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    keys = list(itertools.product(RETURN_METHODS, COVARIANCE_METHODS, OBJECTIVES))
    paths = {key: [] for key in keys}; prior = {key: np.full(returns.shape[1], 1 / returns.shape[1]) for key in keys}
    turnover = {key: [] for key in keys}
    starts = range(min(train_window, len(returns) - 42), len(returns), rebalance_window)
    for start in starts:
        train = returns.iloc[max(0, start - train_window):start]
        test = returns.iloc[start:min(start + rebalance_window, len(returns))]
        mus = {name: expected_returns(train, name) for name in RETURN_METHODS}
        covariances = {name: covariance_matrix(train, name) for name in COVARIANCE_METHODS}
        for key in keys:
            weights = optimize(mus[key[0]], covariances[key[1]], key[2], max_weight)
            traded = float(np.abs(weights - prior[key]).sum())
            path = test.to_numpy() @ weights; path[0] -= traded * cost_bps / 10000
            paths[key].extend(path); turnover[key].append(traded); prior[key] = weights
    records, final_weights = [], {}
    for key in keys:
        statistics = realised_statistics(np.asarray(paths[key]))
        mu = expected_returns(returns, key[0]); covariance = covariance_matrix(returns, key[1])
        weights = optimize(mu, covariance, key[2], max_weight); final_weights[key] = weights
        records.append({"return_method": key[0], "covariance_method": key[1], "objective": key[2],
                        **statistics, "turnover": np.mean(turnover[key]),
                        "effective_holdings": 1 / np.sum(weights ** 2)})
    table = pd.DataFrame(records)
    table["score"] = (0.35 * _rank(table.oos_sharpe) + 0.20 * _rank(table.oos_sortino)
                      + 0.15 * _rank(table.oos_return) + 0.12 * _rank(table.max_drawdown)
                      + 0.08 * _rank(table.tail_loss, False) + 0.05 * _rank(table.turnover, False)
                      + 0.05 * _rank(table.effective_holdings))
    table = table.sort_values(["score", "oos_sharpe"], ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1)); return table, final_weights


def simulate_models(table: pd.DataFrame, weights: dict, returns: pd.DataFrame,
                    capital: float = 1_000_000, horizon: int = 252,
                    simulations: int = 10_000, seed: int = 42) -> pd.DataFrame:
    shocks = np.random.default_rng(seed).standard_normal((simulations, horizon))
    records = []
    for row in table.itertuples(index=False):
        key = (row.return_method, row.covariance_method, row.objective); w = weights[key]
        mu = expected_returns(returns, key[0]); covariance = covariance_matrix(returns, key[1])
        portfolio_return = float(w @ mu); volatility = math.sqrt(max(float(w @ covariance @ w), 1e-18))
        daily = (portfolio_return - 0.5 * volatility ** 2) / TRADING_DAYS + volatility / math.sqrt(TRADING_DAYS) * shocks
        final = capital * np.exp(daily.sum(axis=1)); total_return = final / capital - 1
        threshold = np.percentile(total_return, 5); tail = total_return[total_return <= threshold]
        records.append({"return_method": key[0], "covariance_method": key[1], "objective": key[2],
                        "simulated_return": total_return.mean(), "probability_of_loss": np.mean(total_return < 0),
                        "var_95": -threshold, "cvar_95": -tail.mean()})
    result = table.merge(pd.DataFrame(records), on=["return_method", "covariance_method", "objective"])
    result["combined_score"] = (0.60 * _rank(result.score) + 0.12 * _rank(result.simulated_return)
                                + 0.12 * _rank(result.cvar_95, False)
                                + 0.08 * _rank(result.var_95, False)
                                + 0.08 * _rank(result.probability_of_loss, False))
    result = result.sort_values(["combined_score", "oos_sharpe"], ascending=False).reset_index(drop=True)
    result.insert(0, "simulation_rank", np.arange(1, len(result) + 1)); return result


def run_research(prices: pd.DataFrame, **simulation_parameters):
    returns = log_returns(prices)
    covariance_ranking = compare_covariances(returns)
    model_ranking, weights = compare_models(returns)
    simulation_ranking = simulate_models(model_ranking, weights, returns, **simulation_parameters)
    winner = simulation_ranking.iloc[0]
    winner_key = (winner.return_method, winner.covariance_method, winner.objective)
    return {"returns": returns, "covariance_ranking": covariance_ranking,
            "model_ranking": model_ranking, "simulation_ranking": simulation_ranking,
            "winner_key": winner_key, "winner_weights": weights[winner_key]}
