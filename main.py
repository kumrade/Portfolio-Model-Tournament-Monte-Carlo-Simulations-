"""Command-line entry point for the portfolio model tournament."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.portfolio_research import generate_demo_prices, run_research


def load_prices(path: str | None) -> pd.DataFrame:
    if path:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if frame.empty:
            raise ValueError("The supplied price CSV is empty.")
        return frame
    symbols = ["SBIN", "HDFCBANK", "RELIANCE", "LT", "ICICIBANK", "ITC", "TCS", "INFY", "BHARTIARTL"]
    return generate_demo_prices(symbols, observations=504, seed=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare portfolio models and run Monte Carlo risk analysis.")
    parser.add_argument("--prices", help="CSV containing a date index and one adjusted-close column per asset")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--horizon", type=int, default=252)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    prices = load_prices(args.prices)
    result = run_research(prices, capital=args.capital, horizon=args.horizon,
                          simulations=args.simulations, seed=args.seed)
    result["covariance_ranking"].to_csv(output / "covariance_ranking.csv", index=False)
    result["model_ranking"].to_csv(output / "model_ranking.csv", index=False)
    result["simulation_ranking"].to_csv(output / "simulation_ranking.csv", index=False)
    summary = {"winner": result["winner_key"],
               "weights": dict(zip(prices.columns, result["winner_weights"].round(6)))}
    (output / "winner.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
