"""
Column Library Analyzer

Analyzes the column library Excel file to recommend which columns would be
most valuable to add to your backtest data for optimization.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Support both package mode and direct script execution (second attempt must not omit symbols).
try:
    from .cruncher import StrategyCruncher
    from .enrichment import _extract_hour, _safe_to_datetime
except ImportError as _e_pkg:
    import os
    import sys

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from strategy_cruncher.cruncher import StrategyCruncher
        from strategy_cruncher.enrichment import _extract_hour, _safe_to_datetime
    except ImportError as _e_abs:
        raise ImportError(
            "Could not import strategy_cruncher; install the package or run from the repo root."
        ) from _e_abs


@dataclass
class ColumnRecommendation:
    """Represents a recommended column from the library."""
    column_name: str
    category: str
    description: str
    predictive_score: float
    can_calculate: bool
    calculation_method: Optional[str] = None
    sample_values: Optional[List[float]] = None
    
    def __repr__(self):
        status = "✓ Can Calculate" if self.can_calculate else "✗ Needs Data"
        return (f"{self.column_name} ({self.category}) | "
                f"Score: {self.predictive_score:.2f} | {status}")


class ColumnLibraryAnalyzer:
    """
    Analyzes column library Excel file to recommend columns for optimization.
    
    Usage:
        analyzer = ColumnLibraryAnalyzer('column_library.xlsx')
        recommendations = analyzer.analyze(backtest_df, pnl_column='net_pnl')
        for rec in recommendations[:10]:
            print(rec)
    """
    
    def __init__(self, library_path: str = 'column_library.xlsx'):
        """
        Initialize the analyzer.
        
        Args:
            library_path: Path to the column library Excel file
        """
        self.library_path = library_path
        self.library_data: Dict[str, pd.DataFrame] = {}
        self.cruncher = StrategyCruncher(min_improvement_pct=1.0, min_trades_remaining=30)
    
    def load_library(self) -> Dict[str, pd.DataFrame]:
        """Load the column library Excel file. Raises on missing file, bad sheets, or empty data."""
        from .excel_io import excel_engine_for_path

        engine = excel_engine_for_path(self.library_path)
        xl = pd.ExcelFile(self.library_path, engine=engine)
        self.library_data = {}

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(
                self.library_path, sheet_name=sheet_name, engine=engine
            )

            if 'Column Name' in df.columns:
                col_col = 'Column Name'
            elif 'column_name' in df.columns:
                col_col = 'column_name'
            elif 'Column' in df.columns:
                col_col = 'Column'
            elif len(df.columns) > 0:
                col_col = df.columns[0]
            else:
                raise ValueError(
                    f"Sheet {sheet_name!r} in {self.library_path!r} has no columns."
                )

            if 'Description' in df.columns:
                desc_col = 'Description'
            elif 'description' in df.columns:
                desc_col = 'description'
            elif 'Desc' in df.columns:
                desc_col = 'Desc'
            else:
                desc_col = df.columns[1] if len(df.columns) > 1 else None

            if desc_col:
                df_clean = df[[col_col, desc_col]].copy()
                df_clean.columns = ['column_name', 'description']
            else:
                df_clean = df[[col_col]].copy()
                df_clean.columns = ['column_name']
                df_clean['description'] = ''

            df_clean = df_clean.dropna(subset=['column_name'])
            df_clean = df_clean[df_clean['column_name'].astype(str).str.strip() != '']
            df_clean = df_clean[df_clean['column_name'].astype(str).str.lower() != 'column name']

            if len(df_clean) == 0:
                raise ValueError(
                    f"Sheet {sheet_name!r} in {self.library_path!r} has no valid column rows after cleaning."
                )

            self.library_data[sheet_name] = df_clean

        if not self.library_data:
            raise ValueError(f"No sheets found in {self.library_path!r}")
        return self.library_data
    
    def analyze(
        self,
        backtest_df: pd.DataFrame,
        pnl_column: str = 'net_pnl',
        test_calculated: bool = True
    ) -> List[ColumnRecommendation]:
        """
        Analyze the column library and recommend columns.
        
        Args:
            backtest_df: Your backtest dataframe
            pnl_column: Name of P&L column
            test_calculated: Whether to test columns we can calculate
            
        Returns:
            List of ColumnRecommendation objects, sorted by predictive score
        """
        if not self.library_data:
            self.load_library()

        if pnl_column not in backtest_df.columns:
            raise ValueError(
                f"Column library analyze: PnL column {pnl_column!r} not found "
                f"(columns: {list(backtest_df.columns)})"
            )

        recommendations = []

        # Get baseline metrics (raises on NaN / non-numeric PnL like cruncher)
        baseline = self.cruncher._calculate_metrics(backtest_df, pnl_column)
        
        # Analyze each category
        for category, df_lib in self.library_data.items():
            for _, row in df_lib.iterrows():
                col_name = str(row['column_name']).strip()
                description = str(row.get('description', '')).strip()
                
                if not col_name or col_name.lower() == 'nan':
                    continue
                
                # Check if column already exists
                if col_name in backtest_df.columns:
                    # Test existing column
                    score = self._test_column_predictive_power(
                        backtest_df, col_name, pnl_column, baseline
                    )
                    recommendations.append(ColumnRecommendation(
                        column_name=col_name,
                        category=category,
                        description=description,
                        predictive_score=score,
                        can_calculate=True,
                        calculation_method="Already exists"
                    ))
                
                elif test_calculated:
                    # Try to calculate the column
                    calculated_col, detail = self._try_calculate_column(
                        backtest_df, col_name, category
                    )

                    if calculated_col is not None:
                        # Test the calculated column
                        test_df = backtest_df.copy()
                        test_df[col_name] = calculated_col

                        score = self._test_column_predictive_power(
                            test_df, col_name, pnl_column, baseline
                        )

                        recommendations.append(ColumnRecommendation(
                            column_name=col_name,
                            category=category,
                            description=description,
                            predictive_score=score,
                            can_calculate=True,
                            calculation_method=detail,
                            sample_values=calculated_col.head(5).tolist() if len(calculated_col) > 0 else None
                        ))
                    else:
                        reason = detail or "Blocked: no derivable pattern for this library row"
                        recommendations.append(ColumnRecommendation(
                            column_name=col_name,
                            category=category,
                            description=description,
                            predictive_score=0.0,
                            can_calculate=False,
                            calculation_method=reason,
                        ))
        
        # Sort by predictive score
        recommendations.sort(key=lambda x: x.predictive_score, reverse=True)
        
        return recommendations
    
    def _test_column_predictive_power(
        self,
        df: pd.DataFrame,
        column: str,
        pnl_column: str,
        baseline: Dict
    ) -> float:
        """Test how predictive a column is."""
        if column not in df.columns:
            return 0.0
        
        # Check if column has variation
        if df[column].nunique() < 3:
            return 0.0
        
        # Find best rule for this column
        rules = self.cruncher._find_optimal_thresholds(df, column, pnl_column, baseline)
        
        if not rules:
            return 0.0
        
        # Use the best rule's edge score
        best_rule = max(rules, key=lambda r: r.edge_score)
        
        # Normalize score (edge scores can vary widely)
        # Combine edge score with improvement metrics
        es = best_rule.edge_score
        pip = best_rule.pnl_improvement_pct
        wr = best_rule.win_rate_improvement
        es_term = 0.0 if es != es else es
        pnl_raw = min(pip / 100, 1.0)
        pnl_term = 0.0 if pnl_raw != pnl_raw else pnl_raw
        wr_raw = min(wr, 0.5)
        wr_term = 0.0 if wr_raw != wr_raw else wr_raw
        score = es_term * 0.4 + pnl_term * 0.3 + wr_term * 0.3
        return 0.0 if score != score else score
    
    def _entry_price_expanding_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Percentile of entry_price using expanding rank within time-sorted order (no future rows)."""
        sort_cols = [
            c
            for c in ("date", "entry_date", "entry_time", "EntryTime")
            if c in df.columns
        ]
        work = df.loc[:, ["entry_price"]].copy()
        if sort_cols:
            for c in sort_cols:
                work[c] = df[c].values
            work = work.sort_values(sort_cols, kind="mergesort")
        ranked = work["entry_price"].expanding(min_periods=1).rank(pct=True)
        aligned = pd.Series(ranked.values, index=work.index)
        return aligned.reindex(df.index)

    def _try_calculate_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        category: str
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        Return (series, method) on success; (None, 'Blocked: …') when inputs are missing or unsafe.
        """
        col_lower = column_name.lower().replace(' ', '_').replace('-', '_')
        cat_lower = category.lower()

        # Volume-based calculations
        if 'volume' in cat_lower or 'volume' in col_lower:
            if 'arval' in col_lower or 'relative_volume' in col_lower:
                return (
                    None,
                    "Blocked: need volume (or bar volume) — cannot infer relative volume "
                    "from entry_value/shares alone",
                )

            if 'volume_surge' in col_lower:
                return None, "Blocked: need volume time series"
            if "volume" in cat_lower:
                return (
                    None,
                    "Blocked: no implemented volume derivation for this library row "
                    "(expect relative_volume/arval or volume_surge in the column name)",
                )

        # Price-based calculations
        if 'price' in cat_lower or 'position' in cat_lower:
            if 'position_in_range' in col_lower:
                if 'entry_price' not in df.columns:
                    return None, "Blocked: need entry_price"
                return (
                    None,
                    "Blocked: need session or daily high/low (or bar range) — "
                    "position_in_range is not derivable from price alone",
                )

            if 'price_percentile' in col_lower:
                if 'entry_price' not in df.columns:
                    return None, "Blocked: need entry_price"
                ser = self._entry_price_expanding_percentile(df)
                return (
                    ser,
                    "entry_price expanding percentile (sorted by date/time columns when present)",
                )

            if 'distance_from_high' in col_lower:
                return (
                    None,
                    "Blocked: need a causal reference high (e.g. prior-day or pre-entry high). "
                    "A global max(entry_price) across the backtest leaks future information.",
                )

        # Time-based calculations
        if 'time' in cat_lower or 'hour' in col_lower:
            if 'entry_hour' in col_lower or 'entry_time' in col_lower:
                if 'entry_time' not in df.columns:
                    return None, "Blocked: need entry_time"
                return df['entry_time'].apply(_extract_hour), "extracted from entry_time"

            if 'minutes_since_open' in col_lower:
                if 'entry_time' not in df.columns:
                    return None, "Blocked: need entry_time"
                hours = df['entry_time'].apply(_extract_hour)
                return (hours - 9.5) * 60, "minutes since 9:30 AM (assumed open)"

            if 'day_of_week' in col_lower:
                if 'date' not in df.columns:
                    return None, "Blocked: need date"
                return _safe_to_datetime(df['date']).dt.dayofweek, "extracted from date"

        # Volatility-based
        if 'volatility' in cat_lower or 'atr' in col_lower:
            if 'atr' in col_lower:
                return (
                    None,
                    "Blocked: need ATR column or OHLC history to compute ATR",
                )

        # Momentum-based
        if 'momentum' in cat_lower:
            if 'rsi' in col_lower:
                return None, "Blocked: need OHLC/close history to compute RSI"
            if 'macd' in col_lower:
                return None, "Blocked: need OHLC/close history to compute MACD"

        return None, None
    
    def generate_report(
        self,
        recommendations: List[ColumnRecommendation],
        top_n: int = 20,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a recommendation report.
        
        Args:
            recommendations: List of ColumnRecommendation objects
            top_n: Number of top recommendations to include
            output_path: Optional path to save report CSV
            
        Returns:
            DataFrame with recommendations
        """
        top_recs = recommendations[:top_n]
        
        report_data = []
        for i, rec in enumerate(top_recs, 1):
            report_data.append({
                'Rank': i,
                'Column Name': rec.column_name,
                'Category': rec.category,
                'Description': rec.description,
                'Predictive Score': rec.predictive_score,
                'Can Calculate': 'Yes' if rec.can_calculate else 'No',
                'Calculation Method': rec.calculation_method or 'N/A',
                'Status': 'Already Exists' if rec.calculation_method == 'Already exists' 
                         else ('Can Calculate' if rec.can_calculate else 'Needs Additional Data')
            })
        
        df_report = pd.DataFrame(report_data)
        
        if output_path:
            df_report.to_csv(output_path, index=False)
            print(f"Report saved to {output_path}")
        
        return df_report


def analyze_column_library(
    backtest_csv: str,
    library_path: str = 'column_library.xlsx',
    pnl_column: str = 'net_pnl',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Convenience function to analyze column library and generate report.
    
    Args:
        backtest_csv: Path to backtest CSV
        library_path: Path to column library Excel file
        pnl_column: Name of P&L column
        top_n: Number of top recommendations
        
    Returns:
        DataFrame with recommendations
    """
    analyzer = ColumnLibraryAnalyzer(library_path)
    df = pd.read_csv(backtest_csv)
    
    print(f"Analyzing column library...")
    print(f"Found {len(analyzer.load_library())} categories")
    
    recommendations = analyzer.analyze(df, pnl_column=pnl_column)
    
    print(f"\nFound {len(recommendations)} columns to analyze")
    print(f"Top {top_n} recommendations:")
    print()
    
    report = analyzer.generate_report(recommendations, top_n=top_n)
    print(report.to_string(index=False))
    
    return report


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python column_library_analyzer.py <backtest.csv> [library.xlsx]")
        sys.exit(1)
    
    backtest_file = sys.argv[1]
    library_file = sys.argv[2] if len(sys.argv) > 2 else 'column_library.xlsx'
    
    analyze_column_library(backtest_file, library_file)
