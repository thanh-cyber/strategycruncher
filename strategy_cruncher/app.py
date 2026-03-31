"""
FireEye — Interactive Web Application

Backtest threshold discovery and rule discovery. Inspired by Dave Mabe's systematic trading approach.

Run with: streamlit run strategy_cruncher/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import base64
import io
import sys
import os
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Package import (Streamlit may run app as a script; second attempt must fail loud if still broken)
try:
    from .cruncher import StrategyCruncher, OptimizationResult, RuleCandidate
    from .excel_io import read_excel_upload
except ImportError as _e_rel:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from strategy_cruncher.cruncher import StrategyCruncher, OptimizationResult, RuleCandidate
        from strategy_cruncher.excel_io import read_excel_upload
    except ImportError as _e_abs:
        raise ImportError(
            "Cannot import strategy_cruncher. Install the package (pip install -e .) or run from the repo root."
        ) from _e_abs

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_BANNER_IMAGE = os.path.join(_APP_DIR, "assets", "fireeye_banner.gif")
_LEGACY_LOGO_PNG = os.path.join(_APP_DIR, "assets", "fireeye_logo.png")
# Primary banner / favicon: user’s animated GIF (falls back to bundled asset).
_USER_BANNER_GIF = r"C:\Users\johnn\dwhelper\Magnum Opus\Untitled (500 x 350 px) (1).gif"


def _banner_image_path() -> str:
    if os.path.isfile(_USER_BANNER_GIF):
        return _USER_BANNER_GIF
    if os.path.isfile(_BUNDLED_BANNER_IMAGE):
        return _BUNDLED_BANNER_IMAGE
    if os.path.isfile(_LEGACY_LOGO_PNG):
        return _LEGACY_LOGO_PNG
    return _USER_BANNER_GIF

# Dave Mabe crunch defaults (sidebar focuses on Optimize on + aggression factor)
_DEFAULT_MIN_IMPROVEMENT_PCT = 8.0
_DEFAULT_MAX_RULES = 8
_DEFAULT_N_THRESHOLD_BINS = 100
# Report tab: Dave Mabe–style tile grid (wide layout)
_REPORT_TOP_COLUMNS_GRID_COLS = 3

# Display label -> StrategyCruncher.optimize_metric / crunch target_metric
OPTIMIZE_ON_OPTIONS: list[tuple[str, str]] = [
    ("Profit (total P&L)", "total_profit"),
    ("Win rate", "win_rate"),
    ("Profit factor", "profit_factor"),
    ("Expectancy (avg per trade)", "expectancy"),
    ("Sharpe ratio", "sharpe_ratio"),
]

# UI theme: white background, black text, FireEye red accent
_JS_THEME: dict = {
    "bg": "#ffffff",
    "panel": "#f5f5f5",
    "card": "#f0f0f0",
    "border": "#d0d0d0",
    "text": "#000000",
    "muted": "#000000",
    "dim": "#333333",
    "accent": "#d21f2d",
    "positive": "#2d6a4f",
    "negative": "#9b2226",
    "neutral": "#555555",
    "plot_bg": "#fafafa",
    "grid": "#dddddd",
    "font_mono": "IBM Plex Mono, monospace",
    "font_sans": "IBM Plex Sans, system-ui, sans-serif",
}


def _plotly_base_font() -> dict:
    return dict(family=_JS_THEME["font_mono"], color=_JS_THEME["text"])


def _plotly_chart_layout(**extra) -> dict:
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor=_JS_THEME["plot_bg"],
        font=_plotly_base_font(),
    )
    base.update(extra)
    return base


def min_trades_from_aggression(n_trades: int, aggression_pct: int) -> int:
    """Minimum filtered trade count = aggression percent of backtest size (Dave Mabe: 1–49)."""
    ap = int(max(1, min(49, aggression_pct)))
    return max(1, int(np.ceil(n_trades * ap / 100.0)))


def _load_df_from_upload(file_bytes: bytes, fname_lower: str, mime_lower: str) -> pd.DataFrame:
    """Parse uploaded bytes as CSV or Excel (.xlsx → openpyxl, .xls → xlrd)."""
    is_excel = fname_lower.endswith((".xlsx", ".xls")) or (
        "spreadsheetml" in mime_lower or "ms-excel" in mime_lower
    )
    if is_excel:
        return read_excel_upload(file_bytes, fname_lower, mime_lower)
    return pd.read_csv(io.BytesIO(file_bytes))


def _demo_backtest_dataframe(n_rows: int = 320) -> pd.DataFrame:
    """
    Synthetic backtest for the in-app demo.

    Tuned so Report » Top columns usually fills **10 cards** (Mabe-style): latent ``score``
    drives ``net_pnl``, plus ``demo_signal_0``…``demo_signal_9`` as noisy views of that score
    so many columns pass min-improvement + min-trades with default sidebar aggression.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-02", periods=n_rows, freq="B")
    score = rng.uniform(0.0, 1.0, n_rows)
    net_pnl = (score - 0.48) * 420.0 + rng.normal(0.0, 22.0, n_rows)

    entry_price = rng.uniform(45.0, 210.0, n_rows)
    mins = 30 + (np.arange(n_rows) % 5) * 9
    entry_time = [f"{9 + (i % 5)}:{int(m):02d}" for i, m in enumerate(mins)]

    data = {
        "ticker": [f"D{i:03d}" for i in range(n_rows)],
        "date": dates.strftime("%Y-%m-%d"),
        "entry_time": entry_time,
        "entry_price": entry_price,
        "net_pnl": net_pnl,
    }
    for k in range(10):
        data[f"demo_signal_{k}"] = score * (1.06 + 0.035 * k) + rng.normal(
            0.0, 0.07, n_rows
        )
    data["gap_percent"] = (score - 0.5) * 6.0 + rng.normal(0.0, 1.2, n_rows)
    data["rsi"] = 38.0 + score * 34.0 + rng.normal(0.0, 3.5, n_rows)
    data["atr"] = rng.uniform(0.5, 4.5, n_rows)
    data["arval"] = score * 2.2 + rng.normal(0.0, 0.35, n_rows)
    data["position_in_range"] = np.clip(
        score + rng.normal(0.0, 0.12, n_rows), 0.05, 0.95
    )

    return pd.DataFrame(data)


def _demo_backtest_excel_bytes() -> bytes:
    buf = io.BytesIO()
    _demo_backtest_dataframe().to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _demo_column_library_xlsx_path() -> str:
    """
    Minimal column-library workbook for demo recommendations.
    Written once to the user temp dir (same shape as a real column_library.xlsx).
    """
    p = Path(tempfile.gettempdir()) / "fireeye_demo_column_library.xlsx"
    if not p.is_file():
        price = pd.DataFrame(
            {
                "Column Name": [
                    "gap_percent",
                    "position_in_range",
                    "price_percentile",
                    "distance_from_high",
                ],
                "Description": [
                    "Open gap vs prior close",
                    "Where price sits in range",
                    "Entry price percentile vs history",
                    "Distance from swing high",
                ],
            }
        )
        volume = pd.DataFrame(
            {
                "Column Name": ["relative_volume", "volume_surge", "obscure_volume_metric"],
                "Description": [
                    "Volume vs average",
                    "Spike in volume",
                    "Unmodeled volume idea",
                ],
            }
        )
        time_s = pd.DataFrame(
            {
                "Column Name": ["entry_hour", "minutes_since_open"],
                "Description": ["Hour of entry", "Minutes after open"],
            }
        )
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            price.to_excel(w, sheet_name="Price", index=False)
            volume.to_excel(w, sheet_name="Volume", index=False)
            time_s.to_excel(w, sheet_name="Time", index=False)
    return str(p)


@st.cache_data(show_spinner=False)
def _cached_loaded_df(
    file_bytes: bytes,
    fname_lower: str,
    mime_lower: str,
) -> pd.DataFrame:
    """Load upload/demo bytes into a DataFrame (cached per file bytes + type hints)."""
    return _load_df_from_upload(file_bytes, fname_lower, mime_lower)


@st.cache_data(show_spinner=False)
def _cached_report_analysis(
    file_bytes: bytes,
    fname_lower: str,
    mime_lower: str,
    pnl_column: str,
    aggression_pct: int,
    optimize_metric: str,
    analyze_library: bool,
    library_path: str,
):
    """
    Cached baseline + report-mode analysis (legacy top-column scan).
    Returns (df, report_result, baseline_metrics, min_trades).
    """
    df = _cached_loaded_df(file_bytes, fname_lower, mime_lower)
    if len(df) == 0:
        raise ValueError("Uploaded file has no data rows.")
    if pnl_column not in df.columns:
        raise ValueError(
            f"Column '{pnl_column}' not found in data. Available columns: {list(df.columns)}"
        )
    n_trades_bt = len(df)
    min_trades = min_trades_from_aggression(n_trades_bt, aggression_pct)
    cruncher = StrategyCruncher(
        min_trades_remaining=min_trades,
        min_improvement_pct=_DEFAULT_MIN_IMPROVEMENT_PCT,
        n_threshold_bins=_DEFAULT_N_THRESHOLD_BINS,
        optimize_metric=optimize_metric,
    )
    baseline_metrics = cruncher._calculate_metrics(df, pnl_column)
    report_result = cruncher.analyze(
        df,
        pnl_column=pnl_column,
        iterative=False,
        analyze_column_library=analyze_library,
        library_path=library_path,
    )
    return df, report_result, baseline_metrics, min_trades


@st.cache_data(show_spinner=False)
def _cached_crunch_analysis(
    file_bytes: bytes,
    fname_lower: str,
    mime_lower: str,
    pnl_column: str,
    aggression_pct: int,
    optimize_metric: str,
):
    """Cached iterative crunch analysis. Returns (crunch_rules, filtered_df, final_metrics)."""
    df = _cached_loaded_df(file_bytes, fname_lower, mime_lower)
    if len(df) == 0:
        raise ValueError("Uploaded file has no data rows.")
    if pnl_column not in df.columns:
        raise ValueError(
            f"Column '{pnl_column}' not found in data. Available columns: {list(df.columns)}"
        )
    min_trades = min_trades_from_aggression(len(df), aggression_pct)
    cruncher = StrategyCruncher(
        min_trades_remaining=min_trades,
        min_improvement_pct=_DEFAULT_MIN_IMPROVEMENT_PCT,
        n_threshold_bins=_DEFAULT_N_THRESHOLD_BINS,
        optimize_metric=optimize_metric,
    )
    crunch_rules, filtered_df, _, _ = cruncher.crunch(
        df,
        pnl_column=pnl_column,
        target_metric=cruncher.optimize_metric,
        min_trades=min_trades,
        min_improvement_pct=_DEFAULT_MIN_IMPROVEMENT_PCT,
        max_rules=_DEFAULT_MAX_RULES,
        verbose=False,
    )
    final_metrics = (
        cruncher._calculate_metrics(filtered_df, pnl_column)
        if len(filtered_df) > 0
        else cruncher._calculate_metrics(df, pnl_column)
    )
    return crunch_rules, filtered_df, final_metrics


def _equity_sort_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ("date", "entry_date", "entry_time", "EntryTime") if c in df.columns]


def rule_passes_series(df: pd.DataFrame, rule: RuleCandidate) -> pd.Series:
    """True = trade satisfies the rule (Better); NaN indicator values count as not passing."""
    col = df[rule.column]
    if rule.direction == "above":
        m = col >= rule.threshold
    else:
        m = col < rule.threshold
    return m.fillna(False)


def rule_calmar_from_candidate(rule: RuleCandidate) -> float:
    md = rule.max_drawdown or 0.0
    if md > 1e-12:
        return float(rule.total_pnl / md)
    return float("inf") if rule.total_pnl > 0 else 0.0


def create_better_worse_equity_figure(
    df: pd.DataFrame,
    pnl_column: str,
    rule: RuleCandidate,
    *,
    compact: bool = False,
    show_title: bool = True,
) -> go.Figure:
    """
    Green = cumulative P&L from trades that pass the rule (kept); red = cumulative from trades removed.
    Chronological order when date/time columns exist (Dave Mabe–style report).
    """
    sort_cols = _equity_sort_columns(df)
    plot_df = (
        df.sort_values(sort_cols).reset_index(drop=True)
        if sort_cols
        else df.reset_index(drop=True)
    )
    pnl = plot_df[pnl_column].astype(float).values
    passes = rule_passes_series(plot_df, rule).values
    cum_better = np.cumsum(np.where(passes, pnl, 0.0))
    cum_worse = np.cumsum(np.where(~passes, pnl, 0.0))
    x = np.arange(len(pnl))
    sym = "≥" if rule.direction == "above" else "<"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cum_better,
            mode="lines",
            name="Better (kept)",
            line=dict(color=_JS_THEME["positive"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cum_worse,
            mode="lines",
            name="Worse (removed)",
            line=dict(color=_JS_THEME["negative"], width=1.5),
        )
    )
    title_sz = 11 if compact else 15
    margins = (
        dict(l=48, r=12, t=36 if not show_title else 44, b=30)
        if compact
        else dict(l=60, r=20, t=60, b=40)
    )
    leg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    if compact:
        leg = {**leg, "font": dict(size=9)}
    layout_kw = dict(
        xaxis_title="Trade index (time-sorted)" if sort_cols else "Trade index",
        yaxis_title="Cumulative P&L ($)",
        legend=leg,
        hovermode="x unified",
        margin=margins,
    )
    if show_title:
        layout_kw["title"] = dict(
            text=f"{rule.column} {sym} {rule.threshold:.4g}",
            font=dict(size=title_sz, color=_JS_THEME["text"], family=_JS_THEME["font_mono"]),
        )
    fig.update_layout(**_plotly_chart_layout(**layout_kw))
    if compact:
        fig.update_layout(height=260)
    fig.update_xaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5)
    fig.update_yaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5, tickformat="$,.0f")
    return fig


def _fmt_pf(v) -> str:
    """Format profit factor; handles numpy scalars and +/- inf."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isinf(fv) and fv > 0:
        return "∞"
    if math.isnan(fv):
        return "N/A"
    return f"{fv:.2f}"


def _fmt_calmar(v) -> str:
    """Format Calmar ratio; handles numpy scalars and inf."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isinf(fv):
        return "∞" if fv > 0 else "-∞"
    if math.isnan(fv):
        return "N/A"
    return f"{fv:.2f}"


# Page configuration (tab icon follows same resolution order as top banner)
_page_icon = os.path.abspath(_banner_image_path())
st.set_page_config(
    page_title="FireEye",
    page_icon=_page_icon if os.path.isfile(_page_icon) else "🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light theme: white background, black typography, red accent
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --js-bg: #ffffff;
        --js-panel: #f5f5f5;
        --js-card: #f0f0f0;
        --js-border: #d0d0d0;
        --js-accent: #d21f2d;
        --js-text: #000000;
        --js-main-fg: #000000;
        --js-muted: #000000;
        --js-pos: #2d6a4f;
        --js-neg: #9b2226;
    }

    html, body, .stApp {
        color: var(--js-text);
    }

    .stApp {
        background-color: var(--js-bg);
    }

    /* Streamlit’s default top decoration + full-width header row = extra “empty” banner — hide it */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Collapse native header strip; pin toolbar (⋮) top-right so it doesn’t reserve a second bar */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        border-bottom: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        position: fixed !important;
        top: 0;
        right: 0;
        left: auto !important;
        width: auto !important;
        max-width: none !important;
        z-index: 1003;
        overflow: visible !important;
        pointer-events: none;
    }

    [data-testid="stHeader"] * {
        pointer-events: auto;
    }

    [data-testid="stToolbar"] {
        position: fixed !important;
        top: 0.35rem;
        right: 0.5rem;
        z-index: 1003;
        background: var(--js-bg);
        border-radius: 6px;
    }

    [data-testid="stHeader"] button,
    [data-testid="stHeader"] span {
        color: var(--js-main-fg) !important;
    }

    /* Centered wordmark: single top banner (logo + rule) */
    .fireeye-floating-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: var(--js-bg);
        padding: 0.45rem 1rem 0.35rem;
        border-bottom: 1px solid var(--js-border);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    .fireeye-banner-spacer {
        min-height: clamp(5.5rem, 18vh, 10rem);
        margin-bottom: 0.5rem;
        pointer-events: none;
    }

    @media (min-width: 768px) {
        .fireeye-banner-spacer {
            min-height: clamp(6rem, 20vh, 11rem);
        }
    }

    .fireeye-banner-logo {
        display: block;
        margin: 0 auto;
        max-width: min(500px, 92vw);
        max-height: min(350px, 22vh);
        width: auto;
        height: auto;
        object-fit: contain;
        /* Nudge right — optical center with Streamlit toolbar on the right */
        transform: translateX(clamp(1rem, 2.5vw, 2.25rem));
    }

    .fireeye-floating-banner--text .fireeye-wordmark {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: var(--js-text);
        display: inline-block;
        transform: translateX(clamp(1rem, 2.5vw, 2.25rem));
    }

    [data-testid="stSidebar"] {
        background-color: var(--js-panel) !important;
        border-right: 1px solid var(--js-border);
        color: var(--js-text) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaption"],
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: var(--js-text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: var(--js-muted) !important;
    }

    [data-testid="stMain"],
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6,
    [data-testid="stMain"] li,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stCaption"],
    [data-testid="stMain"] [data-baseweb="input"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] span,
    [data-testid="stMain"] .stRadio label,
    [data-testid="stMain"] .stCheckbox label,
    [data-testid="stMain"] .stSlider label,
    [data-testid="stMain"] [data-testid="stMetric"] label,
    [data-testid="stMain"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--js-main-fg) !important;
    }

    .sub-header {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.8125rem;
        font-weight: 400;
        color: var(--js-main-fg);
        text-align: left;
        margin: 0 0 1.5rem 0;
        line-height: 1.5;
        border-left: 3px solid var(--js-accent);
        padding-left: 0.65rem;
    }

    .metric-card {
        background-color: var(--js-card);
        border: 1px solid var(--js-border);
        border-top: 2px solid rgba(210, 31, 45, 0.35);
        border-radius: 2px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.6875rem;
        color: var(--js-main-fg);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--js-main-fg);
    }

    .metric-value.positive { color: var(--js-pos); }
    .metric-value.negative { color: var(--js-neg); }

    .rule-card {
        background-color: var(--js-card);
        border: 1px solid var(--js-border);
        border-left: 3px solid var(--js-accent);
        border-radius: 2px;
        padding: 0.85rem 1rem;
        margin: 0.6rem 0;
    }

    .rule-rank {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.8125rem;
        color: var(--js-accent);
        font-weight: 600;
    }

    .rule-expression {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.9375rem;
        color: var(--js-main-fg);
        margin: 0.4rem 0;
    }

    .rule-stats {
        display: flex;
        gap: 1.25rem;
        flex-wrap: wrap;
        margin-top: 0.45rem;
    }

    .rule-stat {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.75rem;
    }

    .rule-stat-label { color: var(--js-main-fg); }
    .rule-stat-value { color: var(--js-main-fg); font-weight: 500; }
    .rule-stat-value.positive { color: var(--js-pos); }
    .rule-stat-value.negative { color: var(--js-neg); }

    .edge-score {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--js-accent);
    }

    /* Report top-columns grid: tighter rule cards in tile cells */
    .rule-card.report-tile-card {
        margin: 0 0 0.5rem 0;
        padding: 0.65rem 0.75rem;
    }

    .rule-card.report-tile-card .rule-expression {
        font-size: 0.8125rem;
    }

    .rule-card.report-tile-card .edge-score {
        font-size: 1.05rem;
    }

    .section-title {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--js-main-fg);
        letter-spacing: -0.02em;
        border-bottom: 2px solid var(--js-accent);
        padding-bottom: 0.35rem;
        margin: 1.75rem 0 0.75rem 0;
    }

    .improvement-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 2px;
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.35rem;
    }

    .improvement-badge.positive {
        background: rgba(45, 106, 79, 0.12);
        color: var(--js-pos);
        border: 1px solid rgba(45, 106, 79, 0.35);
    }

    .improvement-badge.negative {
        background: rgba(155, 34, 38, 0.12);
        color: var(--js-neg);
        border: 1px solid rgba(155, 34, 38, 0.35);
    }

    .info-box {
        background-color: var(--js-card);
        border: 1px solid var(--js-border);
        border-left: 3px solid var(--js-accent);
        border-radius: 2px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
    }

    .info-box p {
        color: var(--js-main-fg);
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.8125rem;
        margin: 0;
        line-height: 1.55;
    }

    .stButton > button {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        background-color: var(--js-card);
        color: var(--js-text);
        border: 1px solid var(--js-accent);
        border-radius: 2px;
        padding: 0.45rem 1rem;
        font-weight: 500;
    }

    [data-testid="stMain"] .stButton > button {
        color: var(--js-main-fg) !important;
    }

    .stButton > button:hover {
        background-color: #e8e8e8;
        border-color: var(--js-accent);
        color: var(--js-text);
    }

    [data-testid="stMain"] .stButton > button:hover {
        color: var(--js-main-fg) !important;
    }

    [data-testid="stMain"] [data-baseweb="tab-list"] {
        background-color: var(--js-panel);
        border-bottom: 2px solid var(--js-accent);
        gap: 0;
    }

    [data-testid="stMain"] [data-baseweb="tab"] {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.8125rem;
        color: #000000 !important;
        border-radius: 0;
    }

    [data-testid="stMain"] [data-baseweb="tab"]:hover {
        color: #000000 !important;
    }

    [data-testid="stMain"] [aria-selected="true"][data-baseweb="tab"] {
        color: var(--js-accent) !important;
        border-bottom: 2px solid var(--js-accent);
    }

    [data-testid="stMain"] .stMarkdown h3 {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--js-main-fg) !important;
        letter-spacing: -0.02em;
        border-bottom: 2px solid var(--js-accent);
        padding-bottom: 0.3rem;
        margin-top: 0.25rem;
    }

    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--js-accent) !important;
        letter-spacing: -0.02em;
        border-left: 3px solid var(--js-accent);
        padding-left: 0.45rem;
        margin-top: 0.15rem;
    }

    [data-testid="stMain"] .stMarkdown h4 {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--js-accent) !important;
        letter-spacing: -0.02em;
        margin-top: 0.5rem;
    }

    div[data-testid="stMetric"] label {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.6875rem !important;
        color: var(--js-muted) !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
        color: var(--js-text);
    }

    [data-testid="stMain"] div[data-testid="stMetric"] label {
        color: var(--js-main-fg) !important;
    }

    [data-testid="stMain"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--js-main-fg) !important;
    }

    [data-testid="stMain"] [data-testid="stExpander"] summary,
    [data-testid="stMain"] [data-testid="stExpander"] summary span,
    [data-testid="stMain"] [data-testid="stExpander"] p,
    [data-testid="stMain"] [data-testid="stExpander"] div {
        color: var(--js-main-fg) !important;
    }

    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {
        color: #000000 !important;
        background-color: #f5f5f5 !important;
    }

    [data-testid="stSidebar"] a,
    [data-testid="stMain"] a {
        color: var(--js-accent) !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #000000 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] [data-testid="stTable"] {
        color: #000000 !important;
    }

    [data-testid="stVerticalBlock"] > div [data-testid="stMarkdown"] a {
        color: var(--js-accent) !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
        border: none;
        border-top: 1px solid rgba(210, 31, 45, 0.35);
    }

    .accent-read {
        color: var(--js-accent);
        font-weight: 600;
        font-family: "IBM Plex Sans", system-ui, sans-serif;
    }

    p.caption-hint {
        font-size: 0.85rem;
        color: #000000;
        margin: 0 0 0.4rem 0;
        line-height: 1.45;
    }

    p.caption-hint a {
        color: var(--js-accent) !important;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    import math
    if math.isnan(value) or math.isinf(value):
        return "N/A" if math.isnan(value) else ("$+∞" if value > 0 else "$-∞")
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, include_sign: bool = False) -> str:
    """Format a number as percentage."""
    if include_sign:
        return f"{value:+.1%}"
    return f"{value:.1%}"


def render_metric_card(label: str, value: str, change: Optional[str] = None, positive: bool = True):
    """Render a styled metric card."""
    change_html = ""
    if change:
        badge_class = "positive" if positive else "negative"
        change_html = f'<span class="improvement-badge {badge_class}">{change}</span>'
    
    value_class = ""
    if change:
        value_class = "positive" if positive else "negative"
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{value}</div>
            {change_html}
        </div>
    """, unsafe_allow_html=True)


def render_rule_card(rank: int, rule: RuleCandidate, *, card_class: str = ""):
    """Render a styled rule card. Optional ``card_class`` for grid tiles (e.g. ``report-tile-card``)."""
    dir_symbol = "≥" if rule.direction == "above" else "<"
    pnl_class = "positive" if rule.pnl_improvement >= 0 else "negative"
    wr_class = "positive" if rule.win_rate_improvement >= 0 else "negative"
    extra = f" {card_class.strip()}" if card_class.strip() else ""

    st.markdown(f"""
        <div class="rule-card{extra}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="rule-rank">#{rank}</span>
                    <div class="rule-expression">{rule.column} {dir_symbol} {rule.threshold:.4f}</div>
                </div>
                <div class="edge-score" title="Mean $/trade spread (better minus worse)">{rule.mean_profit_spread:.2f}</div>
            </div>
            <div class="rule-stats">
                <div class="rule-stat">
                    <span class="rule-stat-label">P&L:</span>
                    <span class="rule-stat-value {pnl_class}">{format_currency(rule.total_pnl)} ({rule.pnl_improvement_pct:+.1f}%)</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Win Rate:</span>
                    <span class="rule-stat-value {wr_class}">{rule.win_rate:.1%} ({rule.win_rate_improvement:+.1%})</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Trades:</span>
                    <span class="rule-stat-value">{rule.trades_remaining:,}</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Profit Factor:</span>
                    <span class="rule-stat-value">{rule.profit_factor:.2f}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_equity_curve(df: pd.DataFrame, pnl_column: str, title: str = "Equity Curve") -> go.Figure:
    """Create an interactive equity curve plot. Sorts by date/entry_time when present so curve is chronological."""
    sort_cols = [c for c in ["date", "entry_date", "entry_time", "EntryTime"] if c in df.columns]
    plot_df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    cumulative_pnl = plot_df[pnl_column].cumsum()
    
    fig = go.Figure()
    
    # Main equity curve
    fig.add_trace(go.Scatter(
        x=list(range(len(cumulative_pnl))),
        y=cumulative_pnl.values,
        mode='lines',
        name='Equity',
        line=dict(color=_JS_THEME["accent"], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(210, 31, 45, 0.08)',
    ))
    
    # Running max (for drawdown visualization)
    running_max = cumulative_pnl.cummax()
    fig.add_trace(go.Scatter(
        x=list(range(len(running_max))),
        y=running_max.values,
        mode='lines',
        name='High water',
        line=dict(color=_JS_THEME["neutral"], width=1, dash='dot'),
        opacity=0.85,
    ))
    
    fig.update_layout(
        **_plotly_chart_layout(
            title=dict(
                text=title,
                font=dict(size=15, color=_JS_THEME["text"], family=_JS_THEME["font_mono"]),
            ),
            xaxis_title="Trade #",
            yaxis_title="Cumulative P&L ($)",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            margin=dict(l=60, r=20, t=60, b=40),
        )
    )
    
    fig.update_xaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5)
    fig.update_yaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5, tickformat='$,.0f')
    
    return fig


def create_distribution_plot(df: pd.DataFrame, pnl_column: str) -> go.Figure:
    """Create P&L distribution histogram."""
    fig = go.Figure()
    
    pnl_values = df[pnl_column]
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=pnl_values,
        nbinsx=50,
        name='P&L distribution',
        marker_color=_JS_THEME["accent"],
        opacity=0.75
    ))
    
    # Add mean line
    mean_pnl = pnl_values.mean()
    fig.add_vline(
        x=mean_pnl,
        line_dash="dash",
        line_color=_JS_THEME["neutral"],
        annotation_text=f"Mean: {format_currency(mean_pnl)}",
        annotation_font_color=_JS_THEME["text"],
    )

    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=_JS_THEME["negative"],
        line_width=1,
        annotation_font_color=_JS_THEME["text"],
    )
    
    fig.update_layout(
        **_plotly_chart_layout(
            title=dict(
                text="P&L distribution",
                font=dict(size=15, color=_JS_THEME["text"], family=_JS_THEME["font_mono"]),
            ),
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            showlegend=False,
            margin=dict(l=60, r=20, t=60, b=40),
        )
    )
    
    fig.update_xaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5, tickformat='$,.0f')
    fig.update_yaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5)
    
    return fig


def _heatmap_cell_text(val: object) -> str:
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(fv):
        return "—"
    return f"{fv:.1f}"


def create_indicator_heatmap(rules: List[RuleCandidate], top_n: int = 15) -> go.Figure:
    """Create a heatmap showing indicator effectiveness."""
    top_rules = rules[:top_n]
    
    indicators = [f"{r.column} {'≥' if r.direction == 'above' else '<'} {r.threshold:.2f}" 
                  for r in top_rules]
    
    metrics = ['PnL Improvement %', 'Win Rate Change', 'Mean $/trade spread', 'Trades Kept %']
    
    # Build the data matrix
    data = []
    for rule in top_rules:
        total = rule.trades_remaining + rule.trades_filtered
        pct_kept = (rule.trades_remaining / total * 100) if total > 0 else 0
        row = [
            rule.pnl_improvement_pct,
            rule.win_rate_improvement * 100,
            rule.mean_profit_spread,
            pct_kept
        ]
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=metrics,
        y=indicators,
        colorscale=[
            [0, _JS_THEME["negative"]],
            [0.5, "#e8e8e8"],
            [1, _JS_THEME["positive"]],
        ],
        text=[[_heatmap_cell_text(val) for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 10, "color": _JS_THEME["text"]},
        hoverongaps=False
    ))
    
    fig.update_layout(
        **_plotly_chart_layout(
            title=dict(
                text="Rule effectiveness",
                font=dict(size=15, color=_JS_THEME["text"], family=_JS_THEME["font_mono"]),
            ),
            height=max(400, len(indicators) * 35),
            margin=dict(l=250, r=20, t=60, b=40),
        )
    )
    
    return fig


def create_threshold_analysis_plot(
    df: pd.DataFrame, 
    column: str, 
    pnl_column: str,
    direction: str = 'above'
) -> Optional[go.Figure]:
    """Create a detailed threshold analysis plot for a specific indicator."""
    values = df[column].dropna()
    pnl_values = df.loc[values.index, pnl_column].values
    col_values = values.values
    
    # Generate thresholds
    percentiles = np.linspace(5, 95, 50)
    thresholds = np.percentile(col_values, percentiles)
    
    results = []
    for thresh in thresholds:
        if direction == 'above':
            mask = col_values >= thresh
        else:
            mask = col_values < thresh
        
        filtered_pnl = pnl_values[mask]
        if len(filtered_pnl) > 10:
            results.append({
                'threshold': thresh,
                'total_pnl': filtered_pnl.sum(),
                'win_rate': (filtered_pnl > 0).mean(),
                'n_trades': len(filtered_pnl),
                'avg_pnl': filtered_pnl.mean()
            })
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total P&L vs Threshold', 'Win Rate vs Threshold',
                       'Trade Count vs Threshold', 'Avg P&L per Trade vs Threshold'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Total PnL
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['total_pnl'],
        mode='lines+markers', name='Total P&L',
        line=dict(color=_JS_THEME["accent"], width=1.5),
        marker=dict(size=4)
    ), row=1, col=1)
    
    # Win Rate
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['win_rate'],
        mode='lines+markers', name='Win Rate',
        line=dict(color=_JS_THEME["positive"], width=1.5),
        marker=dict(size=4)
    ), row=1, col=2)
    
    # Trade Count
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['n_trades'],
        mode='lines+markers', name='Trades',
        line=dict(color=_JS_THEME["neutral"], width=1.5),
        marker=dict(size=4)
    ), row=2, col=1)
    
    # Avg PnL
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['avg_pnl'],
        mode='lines+markers', name='Avg P&L',
        line=dict(color=_JS_THEME["negative"], width=1.5),
        marker=dict(size=4)
    ), row=2, col=2)
    
    fig.update_layout(
        **_plotly_chart_layout(
            title=dict(
                text=f"Threshold sweep: {column} ({direction})",
                font=dict(size=15, color=_JS_THEME["text"], family=_JS_THEME["font_mono"]),
            ),
            showlegend=False,
            height=500,
            margin=dict(l=60, r=20, t=80, b=40),
        )
    )
    
    # Update all axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5, row=i, col=j)
            fig.update_yaxes(gridcolor=_JS_THEME["grid"], gridwidth=0.5, row=i, col=j)
    
    return fig


@st.cache_data(show_spinner=False)
def _banner_data_uri(logo_path: str) -> str:
    """Build a data URI for banner image; cached to avoid re-encoding large GIFs every rerun."""
    if os.path.isfile(logo_path):
        ext = os.path.splitext(logo_path)[1].lower().lstrip(".")
        if ext == "png":
            mime = "png"
        elif ext == "gif":
            mime = "gif"
        elif ext in ("jpg", "jpeg"):
            mime = "jpeg"
        else:
            mime = "png"
        with open(logo_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        return f"data:image/{mime};base64,{b64}"
    return ""


def _fireeye_top_banner_html(logo_path: str) -> str:
    """HTML for centered top banner (logo base64-inlined for reliable fixed positioning)."""
    src = _banner_data_uri(logo_path)
    if src:
        inner = f'<img src="{src}" alt="FireEye" class="fireeye-banner-logo" />'
        extra = ""
    else:
        inner = '<span class="fireeye-wordmark">FireEye</span>'
        extra = " fireeye-floating-banner--text"
    return (
        f'<div class="fireeye-floating-banner{extra}">{inner}</div>'
        '<div class="fireeye-banner-spacer" aria-hidden="true"></div>'
    )


def _startup_diagnostic_text() -> str:
    """Single block for terminal + UI so a run is obviously alive and versions are visible."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "\n".join(
        [
            f"FireEye (Strategy Cruncher) · {ts} UTC",
            f"streamlit {st.__version__} · pandas {pd.__version__} · numpy {np.__version__}",
            f"app_dir: {_APP_DIR}",
            f"cwd: {os.getcwd()}",
            "imports: cruncher + excel_io OK",
        ]
    )


def main():
    _diag = _startup_diagnostic_text()
    if "fireeye_startup_logged" not in st.session_state:
        st.session_state["fireeye_startup_logged"] = True
        print(_diag, flush=True)

    # Header: centered partner wordmark + red rule; fixed below Streamlit chrome while scrolling
    st.markdown(_fireeye_top_banner_html(_banner_image_path()), unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Threshold discovery on backtest columns — '
        "systematic filters, same spirit as Dave Mabe's crunch workflow.</p>",
        unsafe_allow_html=True,
    )
    
    # Sidebar
    with st.sidebar:
        with st.expander("Startup test output", expanded=True):
            st.code(_diag, language="text")

        st.markdown("### Data")

        uploaded_file = st.file_uploader(
            "Upload backtest (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload your backtest results with indicator columns",
        )

        demo_backtest = st.checkbox(
            "Use built-in demo backtest (no file upload)",
            value=False,
            key="fireeye_demo_mode",
            help=(
                "Synthetic Excel in memory (~320 rows, tuned for 10 Report column cards) — same "
                "Report / Crunch / recommendations UI as a real upload. A real file upload overrides this."
            ),
        )
        if demo_backtest:
            st.caption(
                "Turn on **Analyze Column Library** to exercise the recommendations block "
                "(demo uses a small auto-generated library in your temp folder)."
            )

        st.markdown("---")
        st.markdown("### Parameters")
        
        pnl_column = st.text_input(
            "P&L Column Name",
            value="net_pnl",
            help="Name of the column containing profit/loss values"
        )
        
        crunch_mode = st.checkbox(
            "Dave Mabe Crunch Mode (iterative, one rule at a time)",
            value=True,
            help="Apply rules iteratively: find best rule → apply → re-crunch until no more good rules"
        )
        
        optimize_labels = [t[0] for t in OPTIMIZE_ON_OPTIONS]
        optimize_metric_by_label = dict(OPTIMIZE_ON_OPTIONS)
        optimize_label = st.selectbox(
            "Optimize on",
            options=optimize_labels,
            index=0,
            help=(
                "Metric to maximize when searching for each filter rule. "
                "Profit = total P&L; other options match common backtest stats."
            ),
        )
        optimize_metric = optimize_metric_by_label[optimize_label]
        
        st.markdown("##### Aggression factor")
        st.caption(
            "Smallest filtered trade set as a percent of your backtest (1–49). "
            "Default 33 → the optimizer only considers rules that leave at least that "
            "fraction of trades (e.g. 1000 trades → at least 333). "
            "Higher = larger minimum subset; lower = allow smaller subsets."
        )
        # Single slider (avoids Streamlit slider+number_input key/state desync bugs)
        aggression_pct = st.slider(
            "Aggression factor (1–49% of backtest trades)",
            min_value=1,
            max_value=49,
            value=33,
            key="aggression_pct",
            help=(
                "Minimum trades after each filter = this percent of total backtest rows "
                "(Dave Mabe aggression factor)."
            ),
        )
        
        analyze_library = st.checkbox(
            "Analyze Column Library",
            value=False,
            help="Analyze column_library.xlsx to recommend additional columns to add"
        )
        
        library_path = st.text_input(
            "Column Library Path",
            value="column_library.xlsx",
            help="Path to your column library Excel file",
            disabled=not analyze_library
        )

        st.caption(
            "Heavy analysis is **cached** for this upload and sidebar settings. "
            "Switching tabs or report sort does not re-run it."
        )
        if st.button("Clear analysis cache", help="Re-run from scratch (e.g. after replacing the file)."):
            _cached_loaded_df.clear()
            _cached_report_analysis.clear()
            _cached_crunch_analysis.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        <div class="info-box">
            <p>FireEye finds optimal indicator thresholds to filter bad trades from your backtest.</p>
            <br>
            <p><strong>Process:</strong></p>
            <p>1. Cast a wide net with initial backtest</p>
            <p>2. Add indicator columns</p>
            <p>3. Let the cruncher find optimal cutoffs</p>
            <p>4. Apply rules that make sense to you</p>
            <br>
            <p><strong>Column Library:</strong> Enable to analyze your column library Excel file and get recommendations for additional columns to add.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content — real upload takes precedence over demo mode
    _data_source: Optional[str] = None
    if uploaded_file is not None:
        _data_source = "upload"
    elif demo_backtest:
        _data_source = "demo"

    if _data_source is not None:
        progress = None
        try:
            if _data_source == "upload":
                fname = getattr(uploaded_file, "name", "") or ""
                name_l = fname.lower()
                mime = (getattr(uploaded_file, "type", "") or "").lower()
                file_bytes = uploaded_file.getvalue()
            else:
                name_l = "fireeye_demo_backtest.xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_bytes = _demo_backtest_excel_bytes()

            lib_path = library_path.strip() if analyze_library else "column_library.xlsx"
            if _data_source == "demo" and analyze_library:
                lib_path = _demo_column_library_xlsx_path()

            progress = st.progress(0, text="Starting FireEye analysis…")

            progress.progress(15, text="Loading workbook…")
            df = _cached_loaded_df(file_bytes, name_l, mime).copy()

            progress.progress(45, text="Scanning columns and building report candidates…")
            (
                _df_report,
                report_result,
                baseline_metrics,
                min_trades,
            ) = _cached_report_analysis(
                file_bytes,
                name_l,
                mime,
                pnl_column.strip(),
                aggression_pct,
                optimize_metric,
                analyze_library,
                lib_path,
            )
            # keep a local mutable copy for downstream display operations
            df = _df_report.copy()

            if crunch_mode:
                progress.progress(80, text="Running iterative crunch rules…")
                (
                    crunch_rules,
                    filtered_df,
                    final_metrics,
                ) = _cached_crunch_analysis(
                    file_bytes,
                    name_l,
                    mime,
                    pnl_column.strip(),
                    aggression_pct,
                    optimize_metric,
                )
            else:
                crunch_rules = None
                filtered_df = None
                final_metrics = None

            progress.progress(100, text="Analysis complete.")
            progress.empty()

            if _data_source == "demo":
                st.info(
                    "**Demo backtest** — synthetic workbook in memory (~320 rows; expect **10** top column cards on Report). "
                    + (
                        "Recommendations use **fireeye_demo_column_library.xlsx** in your temp directory."
                        if analyze_library
                        else "Enable **Analyze Column Library** in the sidebar to see the recommendations layout."
                    )
                )

            n_trades_bt = len(df)

            cruncher = StrategyCruncher(
                min_trades_remaining=min_trades,
                min_improvement_pct=_DEFAULT_MIN_IMPROVEMENT_PCT,
                n_threshold_bins=_DEFAULT_N_THRESHOLD_BINS,
                optimize_metric=optimize_metric,
            )

            st.caption(
                f"Aggression **{aggression_pct}%** → at least **{min_trades:,}** trades per filtered subset "
                f"(backtest has **{n_trades_bt:,}** rows). Optimizing on **{optimize_label}**."
            )

            results = report_result

            st.session_state['results'] = results
            st.session_state['report_result'] = report_result
            st.session_state['df'] = df
            st.session_state['pnl_column'] = pnl_column
            st.session_state['crunch_mode'] = crunch_mode
            st.session_state['crunch_rules'] = crunch_rules
            st.session_state['filtered_df'] = filtered_df
            st.session_state['baseline_metrics'] = baseline_metrics
            st.session_state['final_metrics'] = final_metrics
            
            tab_report, tab_iter, tab_extra = st.tabs(
                ["Report", "Iterative crunch", "More analysis"]
            )
            
            with tab_report:
                st.markdown(
                    '<p class="caption-hint"><span class="accent-read">Read</span> '
                    '<a href="https://app.davemabe.com/report/gxMbgmojHasQHUEz9M4t/" target="_blank" rel="noopener">Dave Mabe’s sample report</a> '
                    "for the same layout idea.</p>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "**Baseline**, then **Top columns** (filters, most important first). "
                    "Each row: green/red equity curves, rule, rank, **Better / Worse** table vs baseline. "
                    "Cloud Strategy Cruncher has a green **Filter** button; here use the Python snippet or **Iterative crunch**."
                )
                bl = report_result.baseline_metrics
                st.markdown("### Baseline")
                b1, b2, b3, b4, b5 = st.columns(5)
                with b1:
                    st.metric("Trades", f"{bl['n_trades']:,}")
                with b2:
                    st.metric("Total Profit", format_currency(bl["total_pnl"]))
                with b3:
                    st.metric("Profit Factor", _fmt_pf(bl["profit_factor"]))
                with b4:
                    st.metric("Win Percent", f"{bl['win_rate']:.1%}")
                with b5:
                    st.metric("Calmar", _fmt_calmar(bl["calmar_ratio"]))

                pool = report_result.get_top_rules(10)
                sort_by = st.radio(
                    "Sort by",
                    (
                        "Rank (importance)",
                        "Trades (Better)",
                        "Calmar (Better)",
                        "Mean Profit (Better)",
                        "Total Profit (Better)",
                    ),
                    horizontal=True,
                    key="report_sort_by",
                    help="Same dimensions as the cloud report: rank, size of kept set, Calmar on kept trades, expectancy, total P&L on kept trades.",
                )
                st.markdown(f"### Top columns ({len(pool)})")
                st.caption(
                    "Suggestions are ordered by importance (mean $/trade spread); **start at the top**. "
                    "Cards are shown in a **tile grid** (like Dave Mabe’s sample report). "
                    "Green curve = cumulative P&L for trades that **pass** the rule; red = for trades **removed**."
                )
                st.markdown("#### Report metrics (Dave Mabe definitions)")
                st.markdown(
                    '<p class="caption-hint"><span class="accent-read">Read</span> the '
                    '<a href="https://app.davemabe.com/docs/report-metrics" target="_blank" rel="noopener">report metrics glossary</a> '
                    "for **≥** / **&lt;**, profit improvement, curve quality, better/worse counts.</p>",
                    unsafe_allow_html=True,
                )
                _rm_pool = list(pool)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Column": r.column,
                                "Comparison": "≥" if r.direction == "above" else "<",
                                "Value": r.threshold,
                                "Rank": i + 1,
                                "Total profit improvement": format_currency(r.pnl_improvement),
                                "Curve quality improvement": round(r.curve_quality_improvement, 4),
                                "Better trades": r.trades_remaining,
                                "Worse trades": r.trades_filtered,
                            }
                            for i, r in enumerate(_rm_pool)
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                ranked = list(enumerate(pool, start=1))
                if sort_by == "Rank (importance)":
                    display_order = sorted(ranked, key=lambda x: x[0])
                elif sort_by == "Trades (Better)":
                    display_order = sorted(
                        ranked, key=lambda x: x[1].trades_remaining, reverse=True
                    )
                elif sort_by == "Calmar (Better)":
                    display_order = sorted(
                        ranked, key=lambda x: rule_calmar_from_candidate(x[1]), reverse=True
                    )
                elif sort_by == "Mean Profit (Better)":
                    display_order = sorted(ranked, key=lambda x: x[1].expectancy, reverse=True)
                elif sort_by == "Total Profit (Better)":
                    display_order = sorted(ranked, key=lambda x: x[1].total_pnl, reverse=True)
                else:
                    display_order = sorted(ranked, key=lambda x: x[0])
                
                if not pool:
                    st.warning(
                        "No column rules met the aggression and improvement thresholds. "
                        "Try lowering aggression or the minimum improvement in code defaults."
                    )
                else:
                    tiles = list(enumerate(display_order))
                    ncols = _REPORT_TOP_COLUMNS_GRID_COLS
                    for row_start in range(0, len(tiles), ncols):
                        row = tiles[row_start : row_start + ncols]
                        col_slots = st.columns(ncols)
                        for slot, (tile_i, (importance_rank, rule)) in enumerate(row):
                            with col_slots[slot]:
                                try:
                                    _tile_ctx = st.container(border=True)
                                except TypeError:
                                    _tile_ctx = st.container()
                                with _tile_ctx:
                                    render_rule_card(
                                        importance_rank, rule, card_class="report-tile-card"
                                    )
                                    st.plotly_chart(
                                        create_better_worse_equity_figure(
                                            df,
                                            pnl_column,
                                            rule,
                                            compact=True,
                                            show_title=True,
                                        ),
                                        use_container_width=True,
                                        key=f"bw_eq_{tile_i}",
                                    )
                                    exp_label = (
                                        f"Better / Worse & code — {rule.column} "
                                        f"({'≥' if rule.direction == 'above' else '<'} {rule.threshold:.4g})"
                                    )
                                    with st.expander(exp_label, expanded=False):
                                        st.caption(
                                            "**Better** = trades that remain if you apply this rule; **Worse** = trades removed. "
                                            "**Trades** = count; **Total Profit** = sum of P&L; **Profit Factor** = gross profit ÷ gross loss (>1 ⇒ net winning); "
                                            "**Win Rate** = % winners; **Calmar** = total return vs max drawdown (higher is better). "
                                            "**Avg Profit / Trade** = expectancy on each side."
                                        )
                                        passes = rule_passes_series(df, rule)
                                        better_df = df[passes]
                                        worse_df = df[~passes]
                                        mb = cruncher._calculate_metrics(better_df, pnl_column)
                                        mw = cruncher._calculate_metrics(worse_df, pnl_column)
                                        cmp_tbl = pd.DataFrame(
                                            {
                                                "Metric": [
                                                    "Trades",
                                                    "Total Profit",
                                                    "Avg Profit / Trade",
                                                    "Profit Factor",
                                                    "Win Rate",
                                                    "Calmar",
                                                ],
                                                "Better": [
                                                    f"{mb['n_trades']:,}",
                                                    format_currency(mb["total_pnl"]),
                                                    format_currency(mb["expectancy"]),
                                                    _fmt_pf(mb["profit_factor"]),
                                                    f"{mb['win_rate']:.1%}",
                                                    _fmt_calmar(mb["calmar_ratio"]),
                                                ],
                                                "Worse": [
                                                    f"{mw['n_trades']:,}",
                                                    format_currency(mw["total_pnl"]),
                                                    format_currency(mw["expectancy"]),
                                                    _fmt_pf(mw["profit_factor"]),
                                                    f"{mw['win_rate']:.1%}",
                                                    _fmt_calmar(mw["calmar_ratio"]),
                                                ],
                                            }
                                        )
                                        st.dataframe(
                                            cmp_tbl,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        op = ">=" if rule.direction == "above" else "<"
                                        py_snippet = (
                                            f"mask = df['{rule.column}'] {op} {rule.threshold}\n"
                                            f"df_better = df[mask]\n"
                                            f"df_worse = df[~mask]"
                                        )
                                        st.caption(
                                            "Apply this filter in your platform or notebook (same idea as Dave Mabe's green Filter control). "
                                            "Rows with NaN in this column count as not passing the rule."
                                        )
                                        st.code(py_snippet, language="python")
            
            with tab_iter:
                if crunch_mode:
                    st.markdown(
                        '<p class="caption-hint"><span class="accent-read">Read</span> — '
                        "Rules apply in order; use the subtabs for summary, table, equity curves, and raw rows.</p>",
                        unsafe_allow_html=True,
                    )
                    baseline = baseline_metrics
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["Iterative summary", "Rules table", "Equity curves", "Raw rows"]
                    )
                    
                    with tab1:
                        st.markdown(
                            '<div class="section-title">Dave Mabe Crunch - Before vs After</div>',
                            unsafe_allow_html=True,
                        )
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            render_metric_card("Trades", f"{baseline['n_trades']:,} → {final_metrics['n_trades']:,}")
                            render_metric_card("Profit Factor", f"{baseline['profit_factor']:.2f} → {final_metrics['profit_factor']:.2f}")
                        with col2:
                            render_metric_card(
                                "Total P&L",
                                format_currency(baseline['total_pnl']) + " → " + format_currency(final_metrics['total_pnl']),
                                positive=final_metrics['total_pnl'] >= 0,
                            )
                            render_metric_card("Sharpe", f"{baseline['sharpe_ratio']:.2f} → {final_metrics['sharpe_ratio']:.2f}")
                        with col3:
                            render_metric_card("Win Rate", f"{baseline['win_rate']:.1%} → {final_metrics['win_rate']:.1%}")
                            render_metric_card(
                                "Max DD",
                                format_currency(baseline['max_drawdown']) + " → " + format_currency(final_metrics['max_drawdown']),
                            )
                        with col4:
                            render_metric_card("Rules", f"{len(crunch_rules)}")
                        st.markdown("#### P&L Distribution (Before / After)")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(create_distribution_plot(df, pnl_column), use_container_width=True, key="crunch_dist_before")
                        with c2:
                            st.plotly_chart(create_distribution_plot(filtered_df, pnl_column) if len(filtered_df) > 0 else create_distribution_plot(df, pnl_column), use_container_width=True, key="crunch_dist_after")
                    
                    with tab2:
                        st.markdown('<div class="section-title">Rules Applied (Iterative)</div>', unsafe_allow_html=True)
                        if crunch_rules:
                            rules_table = pd.DataFrame(crunch_rules)
                            display_cols = [
                                'rule_num', 'column', 'direction', 'threshold',
                                'new_metric', 'improvement_pct', 'trades_remaining',
                            ]
                            st.dataframe(
                                rules_table[[c for c in display_cols if c in rules_table.columns]],
                                use_container_width=True,
                                hide_index=True,
                            )
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                csv_buf = io.StringIO()
                                filtered_df.to_csv(csv_buf, index=False)
                                st.download_button(
                                    "Download Filtered Trades",
                                    data=csv_buf.getvalue(),
                                    file_name="crunch_filtered_trades.csv",
                                    mime="text/csv",
                                )
                            with col_b:
                                rules_csv = io.StringIO()
                                rules_table.to_csv(rules_csv, index=False)
                                st.download_button(
                                    "Download Rules (CSV)",
                                    data=rules_csv.getvalue(),
                                    file_name="crunch_rules.csv",
                                    mime="text/csv",
                                )
                            with col_c:
                                import json

                                def _to_json(v):
                                    if isinstance(v, (np.floating, np.integer)):
                                        return float(v)
                                    return v

                                rules_json = json.dumps(
                                    [
                                        {k: _to_json(v) for k, v in r.items() if k in display_cols}
                                        for r in crunch_rules
                                    ],
                                    indent=2,
                                )
                                st.download_button(
                                    "Download Rules (JSON)",
                                    data=rules_json,
                                    file_name="crunch_rules.json",
                                    mime="application/json",
                                )
                        else:
                            st.info("No rules met the criteria.")
                    
                    with tab3:
                        st.markdown('<div class="section-title">Before vs After Equity Curve</div>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_equity_curve(df, pnl_column, "Before (Baseline)"), use_container_width=True, key="crunch_equity_before")
                        with col2:
                            st.plotly_chart(create_equity_curve(filtered_df, pnl_column, "After (Filtered)") if len(filtered_df) > 0 else create_equity_curve(df, pnl_column, "After"), use_container_width=True, key="crunch_equity_after")
                    
                    with tab4:
                        st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
                        st.dataframe(df.head(500), use_container_width=True, hide_index=True)
                else:
                    st.info("Turn on **Dave Mabe Crunch Mode** in the sidebar to run iterative optimization here.")
            
            with tab_extra:
                st.markdown("### Extra tools")
                st.markdown(
                    '<p class="caption-hint"><span class="accent-read">Read</span> — '
                    "Extra charts and heatmaps; same cached run as Report / Crunch.</p>",
                    unsafe_allow_html=True,
                )
                st.caption("Baseline equity, distribution, rule cards, heatmap, and threshold plots.")
                baseline = baseline_metrics
                st.markdown('<div class="section-title">Baseline equity and distribution</div>', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(
                        create_equity_curve(df, pnl_column, "Baseline equity"),
                        use_container_width=True,
                        key="baseline_equity_extra",
                    )
                with col2:
                    st.plotly_chart(
                        create_distribution_plot(df, pnl_column),
                        use_container_width=True,
                        key="baseline_dist_extra",
                    )
                
                st.markdown('<div class="section-title">Rule cards and heatmap</div>', unsafe_allow_html=True)
                top_rules_extra = report_result.get_top_rules(20) if report_result.rules else []
                if top_rules_extra:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        for i, rule in enumerate(top_rules_extra[:10], 1):
                            render_rule_card(i, rule)
                    with col2:
                        fig_heatmap = create_indicator_heatmap(top_rules_extra)
                        st.plotly_chart(fig_heatmap, use_container_width=True, key="rules_heatmap_extra")
                    st.markdown('<div class="section-title">Threshold deep dive</div>', unsafe_allow_html=True)
                    rule_options = [
                        f"#{i+1}: {r.column} {'≥' if r.direction == 'above' else '<'} {r.threshold:.4f}"
                        for i, r in enumerate(top_rules_extra)
                    ]
                    selected_rule_idx = st.selectbox(
                        "Select a rule",
                        range(len(rule_options)),
                        format_func=lambda x: rule_options[x],
                        key="extra_rule_select",
                    )
                    selected_rule = top_rules_extra[selected_rule_idx]
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_thresh = create_threshold_analysis_plot(
                            df, selected_rule.column, pnl_column, selected_rule.direction
                        )
                        if fig_thresh:
                            st.plotly_chart(fig_thresh, use_container_width=True, key="threshold_analysis_extra")
                    with c2:
                        if selected_rule.direction == "above":
                            fdf = df[(df[selected_rule.column] >= selected_rule.threshold).fillna(False)]
                        else:
                            fdf = df[(df[selected_rule.column] < selected_rule.threshold).fillna(False)]
                        st.plotly_chart(
                            create_equity_curve(fdf, pnl_column, "Equity after rule"),
                            use_container_width=True,
                            key="rule_filtered_equity_extra",
                        )
                else:
                    st.caption("No rules available for heatmap (check aggression thresholds).")
            
            # Column Library Recommendations (from parallel analyze)
            if report_result and report_result.column_recommendations:
                st.markdown('<div class="section-title">Column library recommendations</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>These columns from your library would be most valuable to add to your backtest.</strong></p>
                    <p>The predictive score indicates how much each column would improve your strategy if added.</p>
                </div>
                """, unsafe_allow_html=True)
                
                top_recommendations = results.get_top_column_recommendations(20)
                
                # Group by category
                categories = {}
                for rec in top_recommendations:
                    if rec.category not in categories:
                        categories[rec.category] = []
                    categories[rec.category].append(rec)
                
                for category, recs in categories.items():
                    st.markdown(f"#### {category}")
                    
                    rec_data = []
                    for rec in recs:
                        ps = rec.predictive_score
                        ps_txt = (
                            f"{ps:.3f}"
                            if isinstance(ps, (int, float)) and math.isfinite(float(ps))
                            else str(ps)
                        )
                        rec_data.append({
                            'Column Name': rec.column_name,
                            'Description': rec.description,
                            'Predictive Score': ps_txt,
                            'Status': '✓ Already Exists' if rec.calculation_method == 'Already exists'
                                     else ('✓ Can Calculate' if rec.can_calculate else '✗ Needs Data'),
                            'Method': rec.calculation_method or 'N/A'
                        })
                    
                    st.dataframe(
                        pd.DataFrame(rec_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Download recommendations
                rec_csv_buffer = io.StringIO()
                rec_df = pd.DataFrame([{
                    'Rank': i+1,
                    'Column Name': r.column_name,
                    'Category': r.category,
                    'Description': r.description,
                    'Predictive Score': r.predictive_score,
                    'Can Calculate': r.can_calculate,
                    'Calculation Method': r.calculation_method or 'N/A'
                } for i, r in enumerate(top_recommendations)])
                rec_df.to_csv(rec_csv_buffer, index=False)
                
                st.download_button(
                    label="Download column recommendations",
                    data=rec_csv_buffer.getvalue(),
                    file_name="column_recommendations.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            if progress is not None:
                progress.empty()
            st.error(f"Error processing file: {str(e)}")
            import traceback

            st.code(traceback.format_exc())
            st.stop()
    
    else:
        # No file uploaded - show instructions
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 3rem;">
            <h3 style="color: #000000; margin-bottom: 1rem; font-family: IBM Plex Sans, sans-serif; font-weight: 600; font-size: 1.1rem;">Upload a backtest file — or use the demo</h3>
            <p style="color: #000000; max-width: 640px; margin: 0 auto;">
                In the sidebar, enable <strong>Use built-in demo backtest</strong> to preview the full Report, crunch, and
                (optional) column library output without an Excel file.
                Or upload trade-level rows with a P&amp;L column plus numeric indicators — the cruncher searches thresholds per column.
            </p>
            <br>
            <p style="color: #000000;">
                <strong style="color: #000000;">Typical columns:</strong> ticker, date, net_pnl (or your P&L field),
                and indicators (e.g. gap_percent, atr, rsi).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data preview
        st.markdown('<div class="section-title">Example row format</div>', unsafe_allow_html=True)
        
        example_df = pd.DataFrame({
            'ticker': ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT'],
            'date': ['2025-01-15', '2025-01-15', '2025-01-16', '2025-01-16', '2025-01-17'],
            'net_pnl': [150.50, -75.25, 225.00, -50.00, 180.75],
            'gap_percent': [3.2, 5.8, 2.1, 7.5, 4.3],
            'arval': [2.5, 4.2, 1.8, 6.1, 3.0],
            'position_in_range': [0.75, 0.25, 0.90, 0.15, 0.65],
            'atr': [2.50, 8.75, 3.20, 4.50, 2.80]
        })
        
        st.dataframe(example_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
