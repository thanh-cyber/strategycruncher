"""
Excel I/O with explicit engines: .xlsx → openpyxl, .xls → xlrd.
Avoids silent engine=None behavior across pandas versions.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

_OLE2_XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def excel_engine_for_path(path: Union[str, Path]) -> str:
    lower = str(path).lower()
    if lower.endswith(".xlsx"):
        return "openpyxl"
    if lower.endswith(".xls"):
        return "xlrd"
    raise ValueError(
        f"Unsupported Excel extension (expected .xlsx or .xls): {path!r}"
    )


def read_excel_path(path: Union[str, Path], *, sheet_name: int | str = 0) -> pd.DataFrame:
    engine = excel_engine_for_path(path)
    return pd.read_excel(path, sheet_name=sheet_name, engine=engine)


def read_excel_upload(
    file_bytes: bytes,
    fname_lower: str,
    mime_lower: str,
) -> pd.DataFrame:
    """
    Parse an uploaded workbook. Requires ``.xlsx``/``.xls`` name, xlsx MIME, or bytes
    starting with the OLE2 compound-document header (binary ``.xls``).

    Generic ``ms-excel`` MIME with a non-OLE2 payload defaults to openpyxl (modern ``.xlsx``).
    """
    bio = io.BytesIO(file_bytes)
    fn = (fname_lower or "").lower()
    mime = (mime_lower or "").lower()

    if fn.endswith(".xlsx") or "spreadsheetml.sheet" in mime:
        return pd.read_excel(bio, engine="openpyxl")
    if fn.endswith(".xls"):
        return pd.read_excel(bio, engine="xlrd")
    # Binary .xls often uploaded with generic MIME and no extension; OLE2 signature is reliable.
    if file_bytes.startswith(_OLE2_XLS_MAGIC):
        return pd.read_excel(io.BytesIO(file_bytes), engine="xlrd")
    if "spreadsheetml" in mime or "ms-excel" in mime:
        return pd.read_excel(bio, engine="openpyxl")
    raise ValueError(
        "Excel upload needs a filename ending in .xlsx or .xls "
        "(or a spreadsheet MIME type for .xlsx)."
    )
