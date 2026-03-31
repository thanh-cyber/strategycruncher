"""Excel I/O engines and column library blocked / safe derivations."""

import io

import pandas as pd
import pytest

from strategy_cruncher.column_library_analyzer import ColumnLibraryAnalyzer
from strategy_cruncher.excel_io import excel_engine_for_path, read_excel_upload


def test_excel_engine_for_suffix():
    assert excel_engine_for_path("a.XLSX") == "openpyxl"
    assert excel_engine_for_path("b.xls") == "xlrd"
    with pytest.raises(ValueError, match="Unsupported Excel"):
        excel_engine_for_path("c.csv")


def test_read_excel_upload_xlsx_roundtrip():
    buf = io.BytesIO()
    df_in = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    df_in.to_excel(buf, index=False, engine="openpyxl")
    raw = buf.getvalue()
    out = read_excel_upload(raw, "t.xlsx", "")
    assert list(out.columns) == ["x", "y"]
    assert len(out) == 2


def test_distance_from_high_is_blocked_not_leaky():
    a = ColumnLibraryAnalyzer("dummy.xlsx")
    df = pd.DataFrame({"net_pnl": [1.0, -1.0], "entry_price": [10.0, 12.0]})
    ser, msg = a._try_calculate_column(df, "distance_from_high", "price")
    assert ser is None
    assert msg is not None and "Blocked" in msg


def test_price_percentile_uses_expanding_rank():
    a = ColumnLibraryAnalyzer("dummy.xlsx")
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "entry_price": [10.0, 20.0, 15.0],
        }
    )
    ser, msg = a._try_calculate_column(df, "price_percentile", "price")
    assert ser is not None
    assert len(ser) == 3
    assert bool(msg)
    assert (ser >= 0).all() and (ser <= 1).all()


def test_volume_relative_blocked_message():
    a = ColumnLibraryAnalyzer("dummy.xlsx")
    df = pd.DataFrame({"net_pnl": [1.0], "shares": [100], "entry_value": [5000.0]})
    ser, msg = a._try_calculate_column(df, "relative_volume", "volume")
    assert ser is None
    assert msg is not None and "Blocked" in msg


def test_volume_category_without_recognized_pattern_is_blocked():
    a = ColumnLibraryAnalyzer("dummy.xlsx")
    df = pd.DataFrame({"net_pnl": [1.0]})
    ser, msg = a._try_calculate_column(df, "obscure_metric", "Volume")
    assert ser is None
    assert msg is not None and "Blocked" in msg


def test_read_excel_upload_ole2_magic_selects_xlrd(monkeypatch):
    captured: dict = {}

    def spy(*args, **kwargs):
        captured["engine"] = kwargs.get("engine")
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "read_excel", spy)
    ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
    read_excel_upload(ole2, "", "application/vnd.ms-excel")
    assert captured.get("engine") == "xlrd"
