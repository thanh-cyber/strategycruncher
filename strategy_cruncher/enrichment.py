"""
Indicator Enrichment Utilities

Add commonly predictive indicators to your backtest data
to give FireEye more columns to analyze.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime


def _safe_to_datetime(series, dayfirst: bool = True):
    """Parse datetime series (format='mixed' requires pandas 2.x)."""
    return pd.to_datetime(series, dayfirst=dayfirst, format="mixed")


def enrich_backtest(
    df: pd.DataFrame,
    ticker_col: str = 'ticker',
    date_col: str = 'date',
    entry_time_col: str = 'entry_time',
    entry_price_col: str = 'entry_price',
    exit_price_col: str = 'exit_price',
    pnl_col: str = 'net_pnl'
) -> pd.DataFrame:
    """
    Add commonly predictive indicator columns to backtest data.
    
    These are columns that often have predictive power for filtering trades:
    - Time-based features (hour of entry, day of week)
    - Price-based features (entry price buckets, price percentile)
    - Movement features (% move, R multiple)
    
    Args:
        df: Backtest dataframe
        ticker_col: Name of ticker column
        date_col: Name of date column
        entry_time_col: Name of entry time column
        entry_price_col: Name of entry price column
        exit_price_col: Name of exit price column
        pnl_col: Name of P&L column
        
    Returns:
        Enriched dataframe with additional indicator columns
    """
    df = df.copy()
    
    # Time-based features
    if entry_time_col in df.columns:
        df['entry_hour'] = df[entry_time_col].apply(_extract_hour)
        df['is_premarket'] = df['entry_hour'].apply(lambda h: 1 if h < 9.5 else 0)
        df['is_first_30min'] = df['entry_hour'].apply(lambda h: 1 if 9.5 <= h < 10 else 0)
        df['is_midday'] = df['entry_hour'].apply(lambda h: 1 if 11 <= h < 14 else 0)
        df['is_power_hour'] = df['entry_hour'].apply(lambda h: 1 if h >= 15 else 0)
    
    if date_col in df.columns:
        df['day_of_week'] = _safe_to_datetime(df[date_col]).dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Price-based features  
    if entry_price_col in df.columns:
        prices = df[entry_price_col]
        
        # Price buckets
        df['price_under_1'] = (prices < 1).astype(int)
        df['price_1_to_5'] = ((prices >= 1) & (prices < 5)).astype(int)
        df['price_5_to_10'] = ((prices >= 5) & (prices < 10)).astype(int)
        df['price_10_to_20'] = ((prices >= 10) & (prices < 20)).astype(int)
        df['price_over_20'] = (prices >= 20).astype(int)
        
        # Log price (useful for wide price ranges)
        df['log_price'] = np.log10(prices.clip(lower=0.01))
        
        # Price percentile within the dataset
        df['price_percentile'] = prices.rank(pct=True)
    
    # Movement features
    if entry_price_col in df.columns and exit_price_col in df.columns:
        df['pct_move'] = (df[exit_price_col] - df[entry_price_col]) / df[entry_price_col] * 100
        df['abs_pct_move'] = df['pct_move'].abs()
    
    # PnL-derived features (for analysis, not prediction)
    if pnl_col in df.columns:
        # Running metrics (useful for regime detection)
        df['pnl_5_trade_avg'] = df[pnl_col].rolling(5, min_periods=1).mean()
        df['win_streak'] = _calculate_streak(df[pnl_col] > 0)
        df['loss_streak'] = _calculate_streak(df[pnl_col] < 0)
    
    # Ticker-based features
    if ticker_col in df.columns:
        # Ticker frequency (how often this ticker appears)
        ticker_counts = df[ticker_col].value_counts()
        df['ticker_frequency'] = df[ticker_col].map(ticker_counts)
        df['is_first_trade_ticker'] = df.groupby(ticker_col).cumcount() == 0
        df['is_first_trade_ticker'] = df['is_first_trade_ticker'].astype(int)
    
    return df


def _extract_hour(time_str) -> float:
    """Extract hour as a decimal from time string."""
    try:
        if pd.isna(time_str):
            raise ValueError("entry_time is missing (NaN); cannot derive entry_hour")

        # Numeric values are interpreted as already-decimal hours.
        if isinstance(time_str, (int, float, np.integer, np.floating)):
            return float(time_str)

        # Handle pandas Timestamp / python datetime / python time.
        if hasattr(time_str, 'hour') and hasattr(time_str, 'minute'):
            return float(time_str.hour) + float(time_str.minute) / 60.0

        if isinstance(time_str, str):
            if ':' in time_str:
                parts = time_str.split(':')
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                return hour + minute / 60
        raise ValueError(f"Unrecognized time value: {time_str!r}") from None
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not parse time value: {time_str!r}") from e


def _calculate_streak(condition: pd.Series) -> pd.Series:
    """Calculate consecutive streak of a boolean condition."""
    # Group consecutive True values
    groups = (~condition).cumsum()
    # Count within each group
    streaks = condition.groupby(groups).cumsum()
    return streaks


def add_market_context(
    df: pd.DataFrame,
    spy_data: Optional[pd.DataFrame] = None,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Add market context indicators (requires SPY data).
    
    Args:
        df: Backtest dataframe
        spy_data: DataFrame with SPY daily data (date, open, high, low, close, volume)
        date_col: Name of date column
        
    Returns:
        Enriched dataframe with market context
    """
    if spy_data is None:
        return df

    if date_col not in df.columns:
        raise ValueError(
            f"add_market_context: date column {date_col!r} not in dataframe "
            f"(columns: {list(df.columns)})"
        )
    if "date" not in spy_data.columns:
        raise ValueError(
            "add_market_context: spy_data must include a 'date' column "
            f"(columns: {list(spy_data.columns)})"
        )

    df = df.copy()
    spy_data = spy_data.copy()

    # Calendar date keys only for join (never overwrite the trade sheet's date column).
    spy_data["_spy_join_date"] = pd.to_datetime(spy_data["date"]).dt.date
    df["_trade_join_date"] = pd.to_datetime(df[date_col]).dt.date

    # Calculate SPY metrics
    if "close" in spy_data.columns and "open" in spy_data.columns:
        spy_data["spy_gap"] = (
            (spy_data["open"] - spy_data["close"].shift(1))
            / spy_data["close"].shift(1)
            * 100
        )
        spy_data["spy_day_move"] = (
            (spy_data["close"] - spy_data["open"]) / spy_data["open"] * 100
        )
        spy_data["spy_range"] = (
            (spy_data["high"] - spy_data["low"]) / spy_data["open"] * 100
        )

        spy_data["spy_above_20ma"] = (
            spy_data["close"] > spy_data["close"].rolling(20).mean()
        ).astype(int)
        spy_data["spy_above_50ma"] = (
            spy_data["close"] > spy_data["close"].rolling(50).mean()
        ).astype(int)

    if "volume" in spy_data.columns:
        spy_data["spy_volume_vs_avg"] = (
            spy_data["volume"] / spy_data["volume"].rolling(20).mean()
        )

    spy_cols = [c for c in spy_data.columns if c.startswith("spy_")]
    merge_subset = spy_data[["_spy_join_date"] + spy_cols]
    df = df.merge(
        merge_subset,
        left_on="_trade_join_date",
        right_on="_spy_join_date",
        how="left",
    )
    df = df.drop(columns=["_trade_join_date", "_spy_join_date"], errors="ignore")

    return df


# Example indicator calculations that traders commonly use
def calculate_position_in_range(current: float, low: float, high: float) -> float:
    """
    Calculate position in range (0 = at low, 1 = at high).
    
    Common usage: Where is current price relative to day's range?
    """
    if high == low:
        return 0.5
    return (current - low) / (high - low)


def calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    ATR is commonly used for:
    - Position sizing
    - Stop placement
    - Filtering by volatility
    """
    high_low = highs - lows
    high_close = (highs - closes.shift()).abs()
    low_close = (lows - closes.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate relative volume (current volume vs average).
    
    RVOL > 1 means above average volume.
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enrichment.py <backtest.csv> [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_enriched.csv')
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"Original rows: {len(df)}")
    
    print("Enriching data...")
    df_enriched = enrich_backtest(df)
    
    print(f"Enriched columns: {len(df_enriched.columns)}")
    new_cols = set(df_enriched.columns) - set(df.columns)
    print(f"New columns added: {sorted(new_cols)}")
    
    print(f"Saving to {output_file}...")
    df_enriched.to_csv(output_file, index=False)
    
    print("Done!")
