"""Third pass: upload parsing helpers (avoid full Streamlit run)."""

import io

import pandas as pd

from strategy_cruncher.app import _load_df_from_upload


def test_load_df_from_upload_csv():
    raw = b"a,b\n1,2\n3,4\n"
    df = _load_df_from_upload(raw, "trades.csv", "")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_df_from_upload_excel_by_extension():
    buf = io.BytesIO()
    df_in = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df_in.to_excel(buf, index=False, engine="openpyxl")
    raw = buf.getvalue()
    df = _load_df_from_upload(raw, "t.xlsx", "")
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2
