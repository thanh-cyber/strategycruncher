"""Regression tests for iterative analyze()/crunch indicator selection controls."""

import pandas as pd
import pytest

from strategy_cruncher.cruncher import StrategyCruncher


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "net_pnl": [10, 9, 8, 7, 6, 0, -1, -2, -3, -4],
            "signal_a": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "signal_b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )


def test_iterative_analyze_respects_indicator_columns():
    c = StrategyCruncher(min_trades_remaining=4, min_improvement_pct=0.0)
    result = c.analyze(
        _sample_df(),
        pnl_column="net_pnl",
        iterative=True,
        max_rules=1,
        indicator_columns=["signal_b"],
    )
    assert result.rules, "expected at least one rule for signal_b"
    assert {r.column for r in result.rules} == {"signal_b"}


def test_iterative_analyze_applies_exclude_columns():
    c = StrategyCruncher(min_trades_remaining=4, min_improvement_pct=0.0)
    result = c.analyze(
        _sample_df(),
        pnl_column="net_pnl",
        iterative=True,
        max_rules=1,
        indicator_columns=["signal_a", "signal_b"],
        exclude_columns=["signal_a"],
    )
    assert result.rules, "expected at least one rule after excluding signal_a"
    assert {r.column for r in result.rules} == {"signal_b"}


def test_iterative_analyze_raises_on_missing_indicator_column():
    c = StrategyCruncher(min_trades_remaining=4, min_improvement_pct=0.0)
    with pytest.raises(ValueError, match="indicator_columns contains missing columns"):
        c.analyze(
            _sample_df(),
            pnl_column="net_pnl",
            iterative=True,
            indicator_columns=["missing_col"],
        )
