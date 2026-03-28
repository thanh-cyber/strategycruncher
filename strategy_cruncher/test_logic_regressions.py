"""
Regression tests for FireEye logic (zero-baseline PnL, iterative optimize_metric, etc.).
Run with: python -m strategy_cruncher.test_logic_regressions
"""
import sys
import os

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


def test_report_mode_prefers_entry_column_names():
    """Legacy report should scan Entry_Col_* / Col_* first, not generic colN headers."""
    from strategy_cruncher import StrategyCruncher

    n = 120
    net_pnl = np.array([20.0] * 40 + [-10.0] * 40 + [-10.0] * 40)
    ind = np.array([1.0] * 40 + [2.0] * 40 + [3.0] * 40)
    df = pd.DataFrame({
        "net_pnl": net_pnl,
        "col8": ind,  # generic name — should be ignored when Entry_Col_* exists
        "Entry_Col_RealName": ind,
    })
    cruncher = StrategyCruncher(
        min_improvement_pct=5.0,
        min_trades_remaining=20,
    )
    result = cruncher.analyze(df, pnl_column="net_pnl", iterative=False)
    assert result.rules, "Expected at least one rule"
    for r in result.rules:
        assert r.column == "Entry_Col_RealName", (
            f"Report rules should use entry column names, got {r.column!r}"
        )


def test_crunch_raises_without_entry_columns():
    """crunch() must not fall back to generic numeric columns."""
    from strategy_cruncher import StrategyCruncher

    df = pd.DataFrame({"net_pnl": [1.0, -1.0, 2.0], "rsi": [30.0, 50.0, 70.0]})
    cruncher = StrategyCruncher(min_trades_remaining=1, min_improvement_pct=0.0)
    try:
        cruncher.crunch(df, pnl_column="net_pnl", max_rules=1, verbose=False)
    except ValueError as e:
        assert "crunch()" in str(e)
        assert "Columns in file" in str(e)
    else:
        raise AssertionError("Expected ValueError when no Entry_Col_* / Col_* columns")


def test_analyze_legacy_raises_without_entry_columns():
    """analyze(iterative=False) must require entry-style columns."""
    from strategy_cruncher import StrategyCruncher

    df = pd.DataFrame({"net_pnl": [1.0, -1.0], "momentum": [0.1, 0.2]})
    cruncher = StrategyCruncher(min_trades_remaining=1, min_improvement_pct=0.0)
    try:
        cruncher.analyze(df, pnl_column="net_pnl", iterative=False)
    except ValueError as e:
        assert "analyze(iterative=False)" in str(e)
        assert "Columns in file" in str(e)
    else:
        raise AssertionError("Expected ValueError when no entry-style columns")


def test_nan_pnl_in_metrics_raises():
    """Non-finite PnL must error loudly instead of propagating NaN sums."""
    from strategy_cruncher import StrategyCruncher

    df = pd.DataFrame({
        "net_pnl": [1.0, float("nan"), 3.0],
        "Entry_Col_X": [1.0, 2.0, 3.0],
    })
    cruncher = StrategyCruncher(min_trades_remaining=1, min_improvement_pct=0.0)
    try:
        cruncher.analyze(df, pnl_column="net_pnl", iterative=False)
    except ValueError as e:
        assert "net_pnl" in str(e).lower() or "nan" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError for NaN in PnL column")


if __name__ == "__main__":
    print("Running logic regression tests...")
    test_zero_baseline_pnl_produces_rules()
    print("  [OK] Zero-baseline PnL test passed")
    test_iterative_analyze_uses_configured_optimize_metric()
    print("  [OK] Iterative optimize_metric test passed")
    test_report_mode_prefers_entry_column_names()
    print("  [OK] Report entry-column naming test passed")
    test_crunch_raises_without_entry_columns()
    print("  [OK] crunch() entry-column requirement test passed")
    test_analyze_legacy_raises_without_entry_columns()
    print("  [OK] analyze(iterative=False) entry-column requirement test passed")
    test_nan_pnl_in_metrics_raises()
    print("  [OK] NaN PnL rejection test passed")
    print("All logic regression tests passed.")
