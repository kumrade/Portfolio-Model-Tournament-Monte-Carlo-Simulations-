import numpy as np

from src.portfolio_research import (
    COVARIANCE_METHODS, OBJECTIVES, RETURN_METHODS,
    generate_demo_prices, log_returns, run_research,
)


def test_end_to_end_pipeline():
    prices = generate_demo_prices(["A", "B", "C", "D"], observations=320, seed=11)
    result = run_research(prices, capital=100_000, horizon=30, simulations=150, seed=42)
    assert len(result["covariance_ranking"]) == len(COVARIANCE_METHODS)
    assert len(result["model_ranking"]) == len(RETURN_METHODS) * len(COVARIANCE_METHODS) * len(OBJECTIVES)
    assert len(result["simulation_ranking"]) == len(result["model_ranking"])
    assert np.isclose(result["winner_weights"].sum(), 1.0)
    assert np.all(result["winner_weights"] >= 0)
    assert np.isfinite(result["simulation_ranking"]["combined_score"]).all()


def test_returns_are_finite():
    prices = generate_demo_prices(["A", "B", "C"], observations=100, seed=3)
    returns = log_returns(prices)
    assert np.isfinite(returns.to_numpy()).all()
