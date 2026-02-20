# Strategy Cruncher - Backtest Optimization Tool
# Dave Mabe style iterative Filter Phase - one rule at a time

from .cruncher import StrategyCruncher, OptimizationResult, RuleCandidate, CrunchResult
from .enrichment import enrich_backtest, add_market_context
from .column_library_analyzer import ColumnLibraryAnalyzer, ColumnRecommendation, analyze_column_library

__all__ = [
    'StrategyCruncher', 'OptimizationResult', 'RuleCandidate', 'CrunchResult',
    'enrich_backtest', 'add_market_context',
    'ColumnLibraryAnalyzer', 'ColumnRecommendation', 'analyze_column_library'
]
__version__ = '1.0.0'
