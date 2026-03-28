"""Second-sweep regression tests (pytest). Run: python -m pytest tests/"""

import pandas as pd
import pytest

from strategy_cruncher.enrichment import add_market_context
from strategy_cruncher.column_library_analyzer import ColumnLibraryAnalyzer


def test_add_market_context_preserves_trade_date_column():
    trades = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "net_pnl": [1.0, -1.0],
        }
    )
    spy = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
        }
    )
    out = add_market_context(trades, spy)
    assert pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert "spy_day_move" in out.columns


def test_add_market_context_requires_spy_date():
    trades = pd.DataFrame({"date": ["2024-01-01"], "x": [1.0]})
    spy_bad = pd.DataFrame({"open": [1.0]})
    with pytest.raises(ValueError, match="spy_data must include"):
        add_market_context(trades, spy_bad)


def test_add_market_context_requires_trade_date_column():
    trades = pd.DataFrame({"not_date": [1]})
    spy = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    with pytest.raises(ValueError, match="date column"):
        add_market_context(trades, spy, date_col="date")


def test_column_library_analyze_requires_pnl_column():
    analyzer = ColumnLibraryAnalyzer("dummy.xlsx")
    analyzer.library_data = {
        "Sheet1": pd.DataFrame({"column_name": ["Col_A"], "description": [""]})
    }
    df = pd.DataFrame({"Entry_Col_X": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="PnL column"):
        analyzer.analyze(df, pnl_column="missing_pnl")
