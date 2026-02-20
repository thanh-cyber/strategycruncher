"""
Column Library Analyzer

Analyzes the column library Excel file to recommend which columns would be
most valuable to add to your backtest data for optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Support both package mode and direct script execution.
try:
    from .cruncher import StrategyCruncher
    from .enrichment import _extract_hour
except ImportError:
    import os
    import sys

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from strategy_cruncher.cruncher import StrategyCruncher
    from strategy_cruncher.enrichment import _extract_hour


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
        
        for rec in recommendations.get_top_recommendations(10):
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
        """Load the column library Excel file."""
        try:
            xl = pd.ExcelFile(self.library_path)
            self.library_data = {}
            
            for sheet_name in xl.sheet_names:
                try:
                    df = pd.read_excel(self.library_path, sheet_name=sheet_name)
                    
                    # Handle different possible column name formats
                    if 'Column Name' in df.columns:
                        col_col = 'Column Name'
                    elif 'column_name' in df.columns:
                        col_col = 'column_name'
                    elif 'Column' in df.columns:
                        col_col = 'Column'
                    else:
                        # Try first column
                        col_col = df.columns[0] if len(df.columns) > 0 else None
                    
                    if not col_col:
                        continue
                    
                    if 'Description' in df.columns:
                        desc_col = 'Description'
                    elif 'description' in df.columns:
                        desc_col = 'description'
                    elif 'Desc' in df.columns:
                        desc_col = 'Desc'
                    else:
                        desc_col = df.columns[1] if len(df.columns) > 1 else None
                    
                    # Clean up the dataframe
                    if desc_col:
                        df_clean = df[[col_col, desc_col]].copy()
                        df_clean.columns = ['column_name', 'description']
                    else:
                        df_clean = df[[col_col]].copy()
                        df_clean.columns = ['column_name']
                        df_clean['description'] = ''
                    
                    # Remove empty rows and header row if it exists
                    df_clean = df_clean.dropna(subset=['column_name'])
                    df_clean = df_clean[df_clean['column_name'].astype(str).str.strip() != '']
                    # Remove if column_name is actually the header
                    df_clean = df_clean[df_clean['column_name'].astype(str).str.lower() != 'column name']
                    
                    if len(df_clean) > 0:
                        self.library_data[sheet_name] = df_clean
                        print(f"Loaded {len(df_clean)} columns from {sheet_name}")
                
                except Exception as e:
                    print(f"Warning: Could not load sheet '{sheet_name}': {e}")
                    continue
            
            return self.library_data
        
        except Exception as e:
            print(f"Warning: Could not load column library: {e}")
            print("Creating default library structure...")
            return self._create_default_library()
    
    def _create_default_library(self) -> Dict[str, pd.DataFrame]:
        """Create a default column library structure based on common indicators."""
        default_library = {
            'Volume-Based': pd.DataFrame({
                'column_name': [
                    'arval', 'relative_volume', 'volume_surge', 'volume_ma_ratio',
                    'volume_percent_of_float', 'volume_decay_rate', 'vap_density'
                ],
                'description': [
                    'Average relative volume (current vs average)',
                    'Volume relative to average',
                    'Sudden volume spike',
                    'Volume vs moving average',
                    'Volume as % of float shares',
                    'Volume decline from peak',
                    'Volume at Price density'
                ]
            }),
            'Price-Based': pd.DataFrame({
                'column_name': [
                    'position_in_range', 'price_percentile', 'distance_from_high',
                    'distance_from_low', 'price_momentum', 'price_acceleration'
                ],
                'description': [
                    'Position in day range (0-1)',
                    'Price percentile in dataset',
                    'Distance from day high',
                    'Distance from day low',
                    'Price momentum',
                    'Price acceleration'
                ]
            }),
            'Volatility-Based': pd.DataFrame({
                'column_name': [
                    'atr', 'atr_percent', 'volatility_ratio', 'bb_position',
                    'bb_width', 'price_volatility'
                ],
                'description': [
                    'Average True Range',
                    'ATR as % of price',
                    'Volatility vs average',
                    'Bollinger Band position',
                    'Bollinger Band width',
                    'Price volatility measure'
                ]
            }),
            'Momentum-Based': pd.DataFrame({
                'column_name': [
                    'rsi', 'macd', 'momentum', 'rate_of_change',
                    'stochastic', 'williams_r'
                ],
                'description': [
                    'Relative Strength Index',
                    'MACD indicator',
                    'Price momentum',
                    'Rate of change',
                    'Stochastic oscillator',
                    'Williams %R'
                ]
            }),
            'Time-Based': pd.DataFrame({
                'column_name': [
                    'entry_hour', 'minutes_since_open', 'time_of_day',
                    'day_of_week', 'is_premarket', 'is_power_hour'
                ],
                'description': [
                    'Hour of entry',
                    'Minutes since market open',
                    'Time of day indicator',
                    'Day of week (0-6)',
                    'Is pre-market trade',
                    'Is power hour trade'
                ]
            })
        }
        
        self.library_data = default_library
        return default_library
    
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
        
        recommendations = []
        
        # Get baseline metrics
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
                    calculated_col, method = self._try_calculate_column(
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
                            calculation_method=method,
                            sample_values=calculated_col.head(5).tolist() if len(calculated_col) > 0 else None
                        ))
                    else:
                        # Can't calculate - still add with low score
                        recommendations.append(ColumnRecommendation(
                            column_name=col_name,
                            category=category,
                            description=description,
                            predictive_score=0.0,
                            can_calculate=False,
                            calculation_method=None
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
        score = (
            best_rule.edge_score * 0.4 +
            min(best_rule.pnl_improvement_pct / 100, 1.0) * 0.3 +
            min(best_rule.win_rate_improvement, 0.5) * 0.3
        )
        
        return score
    
    def _try_calculate_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        category: str
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Try to calculate a column from available data."""
        col_lower = column_name.lower().replace(' ', '_').replace('-', '_')
        
        # Volume-based calculations
        if 'volume' in category.lower() or 'volume' in col_lower:
            if 'arval' in col_lower or 'relative_volume' in col_lower:
                if 'shares' in df.columns and 'entry_value' in df.columns:
                    # Approximate: entry_value / shares gives price, can estimate volume
                    return None, None  # Need actual volume data
            
            if 'volume_surge' in col_lower:
                return None, None  # Need volume data
        
        # Price-based calculations
        if 'price' in category.lower() or 'position' in col_lower:
            if 'position_in_range' in col_lower:
                if 'entry_price' in df.columns:
                    # Would need high/low of day - approximate with entry_price
                    return None, None
            
            if 'price_percentile' in col_lower:
                if 'entry_price' in df.columns:
                    return df['entry_price'].rank(pct=True), "entry_price.rank(pct=True)"
            
            if 'distance_from_high' in col_lower:
                if 'entry_price' in df.columns:
                    # Approximate: use max entry_price as proxy
                    max_price = df['entry_price'].max()
                    return (max_price - df['entry_price']) / max_price, "distance from max entry_price"
        
        # Time-based calculations
        if 'time' in category.lower() or 'hour' in col_lower:
            if 'entry_hour' in col_lower or 'entry_time' in col_lower:
                if 'entry_time' in df.columns:
                    return df['entry_time'].apply(_extract_hour), "extracted from entry_time"
            
            if 'minutes_since_open' in col_lower:
                if 'entry_time' in df.columns:
                    hours = df['entry_time'].apply(_extract_hour)
                    # Assume market opens at 9:30
                    return (hours - 9.5) * 60, "minutes since 9:30 AM"
            
            if 'day_of_week' in col_lower:
                if 'date' in df.columns:
                    return pd.to_datetime(df['date'], dayfirst=True, format='mixed').dt.dayofweek, "extracted from date"
        
        # Volatility-based
        if 'volatility' in category.lower() or 'atr' in col_lower:
            if 'atr' in col_lower:
                # Would need high/low/close data
                return None, None
        
        # Momentum-based
        if 'momentum' in category.lower():
            if 'rsi' in col_lower:
                return None, None  # Need price history
            if 'macd' in col_lower:
                return None, None  # Need price history
        
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
        for rec in top_recs:
            report_data.append({
                'Rank': top_recs.index(rec) + 1,
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
