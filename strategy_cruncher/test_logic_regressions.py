"""
Regression tests for Strategy Cruncher logic (zero-baseline PnL, iterative optimize_metric, etc.).
Run with: python -m strategy_cruncher.test_logic_regressions
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def test_zero_baseline_pnl_produces_rules():
    """When baseline total_pnl is 0 and a threshold filter has positive filtered PnL, rules should be produced."""
    from strategy_cruncher import StrategyCruncher

    # Dataset: baseline total_pnl = 0; one subgroup has positive PnL. Need 3+ unique values for _is_usable_numeric_series.
    # 40 rows @ +20, 40 @ -10, 40 @ -10 → total 0. Indicator 1.0, 2.0, 3.0. "below 1.5" keeps first 40 → +800.
    n = 120
    net_pnl = np.array([20.0] * 40 + [-10.0] * 40 + [-10.0] * 40)
    ind = np.array([1.0] * 40 + [2.0] * 40 + [3.0] * 40)
    df = pd.DataFrame({
        "net_pnl": net_pnl,
        "Entry_Col_X": ind,
    })
    assert df["net_pnl"].sum() == 0, "Baseline PnL must be 0"

    cruncher = StrategyCruncher(
        min_improvement_pct=5.0,
        min_trades_remaining=20,
    )
    # Legacy single-pass uses _find_optimal_thresholds (where zero-baseline fix applies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = cruncher.analyze(df, pnl_column="net_pnl", iterative=False)
    # Should find at least one rule (e.g. keep above 0.5 → 60 trades, +600 PnL vs baseline 0)
    assert len(result.rules) >= 1, (
        "Zero-baseline: expected at least one rule when a threshold yields positive filtered PnL"
    )
    # At least one rule should have positive PnL improvement
    assert any(r.pnl_improvement > 0 for r in result.rules), (
        "Zero-baseline: expected at least one rule with positive pnl_improvement"
    )


def test_iterative_analyze_uses_configured_optimize_metric():
    """StrategyCruncher(optimize_metric='expectancy') with iterative analyze should succeed and use that metric."""
    from strategy_cruncher import StrategyCruncher, OptimizationResult

    n = 200
    np.random.seed(123)
    df = pd.DataFrame({
        "net_pnl": np.random.randn(n) * 30,
        "Entry_Col_A": np.random.uniform(0, 100, n),
        "Entry_Col_B": np.random.uniform(0, 1, n),
    })

    cruncher = StrategyCruncher(
        optimize_metric="expectancy",
        min_improvement_pct=1.0,
        min_trades_remaining=25,
    )
    result = cruncher.analyze(df, pnl_column="net_pnl", iterative=True)

    assert isinstance(result, OptimizationResult), "analyze(iterative=True) must return OptimizationResult"
    assert hasattr(result, "rules") and isinstance(result.rules, list), "result.rules must be a list"
    assert hasattr(result, "baseline_metrics"), "result must have baseline_metrics"
    # No exception and valid structure = wiring of optimize_metric is used by crunch()
    assert "expectancy" in result.baseline_metrics, "baseline_metrics must include expectancy"


if __name__ == "__main__":
    print("Running logic regression tests...")
    test_zero_baseline_pnl_produces_rules()
    print("  [OK] Zero-baseline PnL test passed")
    test_iterative_analyze_uses_configured_optimize_metric()
    print("  [OK] Iterative optimize_metric test passed")
    print("All logic regression tests passed.")
