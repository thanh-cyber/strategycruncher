# Strategy Cruncher - Backtest Optimization Tool
# Analyzes backtest data to find optimal indicator thresholds

from .cruncher import StrategyCruncher, OptimizationResult, RuleCandidate
from .enrichment import enrich_backtest, add_market_context
from .column_library_analyzer import ColumnLibraryAnalyzer, ColumnRecommendation, analyze_column_library

__all__ = [
    'StrategyCruncher', 'OptimizationResult', 'RuleCandidate',
    'enrich_backtest', 'add_market_context',
    'ColumnLibraryAnalyzer', 'ColumnRecommendation', 'analyze_column_library'
]
__version__ = '1.0.0'
