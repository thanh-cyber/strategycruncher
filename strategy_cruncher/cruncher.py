# =====================================================
# Dave Mabe Style Iterative Strategy Cruncher
# One wide-net backtest -> Smart Filter Phase
# =====================================================
#
# Run one wide-net backtest -> add column library -> cruncher finds best single rule
# -> apply it -> re-crunch remaining columns -> repeat until no more good rules.
#
# Inspired by Dave Mabe's Filter Phase: iterative, one-rule-at-a-time, super explainable.

"""
Strategy Cruncher - Core Optimization Engine

Analyzes backtest trade data to find optimal indicator thresholds
that maximize profit, win rate, or other metrics while filtering bad trades.

Inspired by Dave Mabe's approach: cast a wide net with the initial backtest,
then use optimization to find the subset of trades worth taking.

Predictive columns: The rules returned by crunch() use the 'column' field (e.g. Entry_Col_ATR14).
Those column names are exactly the predictive (entry) columns for your strategy—use them to
filter or weight entries when you crunch your Excel/CSV from the backtest.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal
import warnings


@dataclass
class RuleCandidate:
    """Represents a potential filtering rule for a strategy."""
    column: str
    direction: Literal['above', 'below']  # Keep trades above or below threshold
    threshold: float
    
    # Metrics after applying this rule
    trades_remaining: int
    trades_filtered: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    expectancy: float
    
    # Improvement metrics (compared to baseline)
    pnl_improvement: float
    pnl_improvement_pct: float
    win_rate_improvement: float
    expectancy_improvement: float
    
    # Statistical significance
    edge_score: float  # Composite score ranking this rule's power
    
    def __repr__(self):
        dir_symbol = '>' if self.direction == 'above' else '<'
        return (f"Rule: {self.column} {dir_symbol} {self.threshold:.4f} | "
                f"PnL: ${self.total_pnl:,.2f} ({self.pnl_improvement_pct:+.1f}%) | "
                f"Win Rate: {self.win_rate:.1%} | "
                f"Trades: {self.trades_remaining} | "
                f"Edge Score: {self.edge_score:.2f}")


@dataclass
class OptimizationResult:
    """Complete optimization results for a backtest."""
    baseline_metrics: Dict
    rules: List[RuleCandidate]
    applied_rules: List[RuleCandidate] = field(default_factory=list)
    filtered_df: Optional[pd.DataFrame] = None
    column_recommendations: Optional[List] = None  # ColumnRecommendation objects
    
    def get_top_rules(self, n: int = 10, metric: str = 'edge_score') -> List[RuleCandidate]:
        """Get top N rules ranked by specified metric."""
        return sorted(self.rules, key=lambda r: getattr(r, metric), reverse=True)[:n]
    
    def get_top_column_recommendations(self, n: int = 20) -> List:
        """Get top N column recommendations."""
        if not self.column_recommendations:
            return []
        return self.column_recommendations[:n]
    
    def apply_rule(self, rule: RuleCandidate, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a rule and return filtered dataframe."""
        if rule.direction == 'above':
            mask = df[rule.column] > rule.threshold
        else:
            mask = df[rule.column] < rule.threshold
        
        self.applied_rules.append(rule)
        self.filtered_df = df[mask].copy()
        return self.filtered_df


@dataclass
class CrunchResult:
    """Result of Dave Mabe style iterative crunch (one rule at a time)."""
    rules: List[Dict]  # List of applied rule dicts
    filtered_df: pd.DataFrame
    baseline_df: pd.DataFrame
    baseline_metric: float
    final_metric: float
    target_metric: str
    pnl_column: str


class StrategyCruncher:
    """
    Analyzes backtest CSV data to find optimal indicator thresholds.
    
    Usage:
        cruncher = StrategyCruncher()
        results = cruncher.analyze('backtest.csv', pnl_column='net_pnl')
        
        # View top rules
        for rule in results.get_top_rules(10):
            print(rule)
    """
    
    def __init__(
        self,
        min_trades_remaining: int = 50,
        min_improvement_pct: float = 5.0,
        n_threshold_bins: int = 100,
        optimize_metric: Literal['pnl', 'sharpe', 'profit_factor', 'expectancy'] = 'pnl'
    ):
        """
        Initialize the Strategy Cruncher.
        
        Args:
            min_trades_remaining: Minimum trades that must remain after applying a rule
            min_improvement_pct: Minimum PnL improvement % to consider a rule
            n_threshold_bins: Number of threshold values to test per column
            optimize_metric: Primary metric to optimize for
        """
        self.min_trades_remaining = min_trades_remaining
        self.min_improvement_pct = min_improvement_pct
        self.n_threshold_bins = n_threshold_bins
        self.optimize_metric = optimize_metric
    
    def _calculate_metric(self, df: pd.DataFrame, metric: str, pnl_column: str) -> float:
        """Calculate a single scalar metric for a dataframe (used by crunch)."""
        if len(df) == 0:
            return 0.0
        pnl = df[pnl_column]
        if metric == "profit_factor":
            wins = pnl[pnl > 0].sum()
            losses = abs(pnl[pnl < 0].sum())
            return wins / losses if losses > 0 else (float('inf') if wins > 0 else 0.0)
        elif metric == "expectancy":
            return float(pnl.mean())
        elif metric == "win_rate":
            return float((pnl > 0).mean() * 100)
        elif metric == "total_profit":
            return float(pnl.sum())
        return float(pnl.sum())

    def _is_usable_numeric_series(self, series: pd.Series) -> bool:
        """Return True when a column is numeric and has enough variation."""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        if series.isna().all():
            return False
        if series.nunique() < 3:
            return False
        return True

    def _get_entry_exclusion_terms(self) -> List[str]:
        """Column name terms that imply forward-looking/exit-derived values."""
        return [
            "MFE", "MAE", "UnrealizedPL", "DistToInitialStop", "BarsTo",
            "BarsToMFE", "BarsToMAE", "MaxDrawdownFromMFE",
            "MaxFavorableExcursion", "MaxAdverseExcursion",
            "Exit", "Hold", "holding_minutes", "entry_time", "exit_time",
            "holding", "Unrealized", "Realized",
        ]

    # Column naming convention (aligns with backtestlibrary + librarycolumn):
    # - Entry_Col_*   = snapshot at entry time (use for ENTRY crunch only)
    # - Col_*_Exit   = snapshot at exit time (exit crunch only; never for entry)
    # - Continuous_Col_* / Cont_* = continuous tracking (_Entry, _Exit, _Max, _Min, _At30min...); exit-side, NOT entry

    def _detect_entry_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect entry-only columns for entry crunch.
        Uses ONLY: Entry_Col_* (entry snapshot) and Col_* that are not exit/forward-looking.
        Excludes: Col_*_Exit, Continuous_* / Cont_* (continuous), and any name with MFE/MAE/Unrealized/etc.
        """
        exit_substrings = self._get_entry_exclusion_terms()
        entry_cols = []
        for c in df.columns:
            # Must be entry-prefixed or bare Col_* (no continuous or exit columns)
            if c.startswith("Cont_") or c.startswith("Continuous_"):
                continue
            is_entry_prefixed = c.startswith("Entry_Col_")
            is_bare_col = c.startswith("Col_") and not c.endswith("_Exit") and "_Exit" not in c
            if not (is_entry_prefixed or is_bare_col):
                continue
            if any(x.lower() in c.lower() for x in exit_substrings):
                continue
            if not self._is_usable_numeric_series(df[c]):
                continue
            entry_cols.append(c)
        return entry_cols
    
    def crunch(
        self,
        df: pd.DataFrame,
        pnl_column: str = "net_pnl",
        target_metric: str = "profit_factor",
        min_trades: int = 300,
        min_improvement_pct: float = 8.0,
        max_rules: int = 8,
        verbose: bool = True
    ) -> Tuple[List[Dict], pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Dave Mabe style iterative Filter Phase: one rule at a time.
        
        Finds best single rule -> applies it -> re-crunches remaining columns -> repeat.
        
        Args:
            df: Backtest DataFrame
            pnl_column: P&L column name (net_pnl, profit, etc.)
            target_metric: 'profit_factor', 'expectancy', 'win_rate', or 'total_profit'
            min_trades: Minimum trades after each rule
            min_improvement_pct: Minimum % improvement to accept a rule
            max_rules: Maximum rules to apply
            verbose: Print progress to stdout
            
        Returns:
            (rules, filtered_df, before_curve, after_curve)
            - rules: list of applied rule dicts; each rule['column'] is a predictive (entry) column name
            - filtered_df: trades remaining after all rules
            - before_curve: cumulative P&L of original trades (chronological when time cols present)
            - after_curve: cumulative P&L of filtered trades (chronological when time cols present)
        """
        if pnl_column not in df.columns:
            raise ValueError(f"PnL column '{pnl_column}' not found. Available: {list(df.columns)}")
        
        # Auto-detect entry columns (no exit columns)
        self.entry_columns = self._detect_entry_columns(df)
        if not self.entry_columns:
            # Fallback: use indicator columns (works when CSV has no Col_* columns)
            self.entry_columns = self._detect_indicator_columns(df, pnl_column, None)
        
        current_df = df.copy()
        baseline = self._calculate_metric(current_df, target_metric, pnl_column)
        results: List[Dict] = []
        # Chronological equity curve when time columns exist (e.g. backtest export)
        sort_cols = [c for c in ["date", "entry_date", "entry_time", "EntryTime"] if c in df.columns]
        _df_sorted = df.sort_values(sort_cols) if sort_cols else df
        before_curve = (
            np.cumsum(_df_sorted[pnl_column].values) if pnl_column in _df_sorted.columns else np.array([])
        )
        
        if verbose:
            n0 = len(current_df)
            metric_label = target_metric.replace("_", " ").title()
            print(f"\n=== Dave Mabe Iterative Filter Phase ===")
            print(f"Baseline {metric_label}: {baseline:.3f} | Trades: {n0:,}\n")
        
        for rule_num in range(1, max_rules + 1):
            best_rule = None
            best_score = -np.inf
            
            for col in self.entry_columns:
                if col not in current_df.columns:
                    continue
                vals = current_df[col].dropna()
                if len(vals) < 3:
                    continue
                thresholds = np.percentile(vals, np.arange(5, 96, 5))
                for thresh in np.unique(thresholds):
                    for direction in [">", "<"]:
                        if direction == ">":
                            mask = current_df[col] > thresh
                        else:
                            mask = current_df[col] < thresh
                        filtered = current_df[mask]
                        if len(filtered) < min_trades:
                            continue
                        score = self._calculate_metric(filtered, target_metric, pnl_column)
                        # Handle inf for profit_factor
                        if score == float('inf'):
                            score = 999.0
                        denom = abs(baseline) if baseline != 0 else 1e-9
                        improvement = (score - baseline) / denom * 100
                        if improvement >= min_improvement_pct and score > best_score:
                            best_score = score
                            best_rule = {
                                "rule_num": rule_num,
                                "column": col,
                                "direction": direction,
                                "threshold": round(float(thresh), 4),
                                "new_metric": score,
                                "improvement_pct": round(improvement, 1),
                                "trades_remaining": len(filtered)
                            }
            
            if best_rule is None:
                if verbose:
                    print(f"No more good rules found after {rule_num - 1} rules.")
                break

            trades_before = len(current_df)
            prev_baseline = self._calculate_metrics(current_df, pnl_column)
            col = best_rule["column"]
            thresh = best_rule["threshold"]
            if best_rule["direction"] == ">":
                current_df = current_df[current_df[col] > thresh]
            else:
                current_df = current_df[current_df[col] < thresh]

            # Enrich rule with full metrics (for analyze() conversion to RuleCandidate)
            metrics = self._calculate_metrics(current_df, pnl_column)
            best_rule["trades_filtered"] = trades_before - len(current_df)
            best_rule["total_pnl"] = metrics["total_pnl"]
            best_rule["win_rate"] = metrics["win_rate"]
            best_rule["avg_win"] = metrics["avg_win"]
            best_rule["avg_loss"] = metrics["avg_loss"]
            best_rule["profit_factor"] = metrics["profit_factor"]
            best_rule["sharpe_ratio"] = metrics["sharpe_ratio"]
            best_rule["max_drawdown"] = metrics["max_drawdown"]
            best_rule["expectancy"] = metrics["expectancy"]
            best_rule["pnl_improvement"] = metrics["total_pnl"] - prev_baseline["total_pnl"]
            denom = abs(prev_baseline["total_pnl"]) or 1e-9
            best_rule["pnl_improvement_pct"] = best_rule["pnl_improvement"] / denom * 100
            best_rule["win_rate_improvement"] = metrics["win_rate"] - prev_baseline["win_rate"]
            best_rule["expectancy_improvement"] = metrics["expectancy"] - prev_baseline["expectancy"]
            best_rule["edge_score"] = best_rule["improvement_pct"] / 100.0

            results.append(best_rule)
            baseline = best_rule["new_metric"]
            
            if verbose:
                dir_sym = best_rule['direction']
                pf = best_rule['new_metric']
                imp = best_rule['improvement_pct']
                n_tr = best_rule['trades_remaining']
                print(f"Rule {rule_num}: {col} {dir_sym} {thresh}   -> PF {pf:.2f} (+{imp:.0f}%) | Trades: {n_tr:,}")
        
        _after_sorted = (
            current_df.sort_values(sort_cols) if sort_cols and len(current_df) > 0 else current_df
        )
        after_curve = (
            np.cumsum(_after_sorted[pnl_column].values)
            if len(_after_sorted) > 0 and pnl_column in _after_sorted.columns
            else np.array([])
        )
        
        if verbose and results:
            initial_metric = self._calculate_metric(df, target_metric, pnl_column)
            final_metric = self._calculate_metric(current_df, target_metric, pnl_column)
            pct = (final_metric - initial_metric) / (abs(initial_metric) or 1e-9) * 100
            print(f"\nFinal Strategy: {len(results)} rules | PF {final_metric:.2f} (+{pct:.0f}%) | Trades: {len(current_df):,}")
            print("=" * 60)
        
        return results, current_df, before_curve, after_curve

    @staticmethod
    def load_trade_file(path: str) -> pd.DataFrame:
        """Load trade data from CSV or Excel (.xlsx, .xls). Use for backtest exports."""
        path_lower = path.lower()
        if path_lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, engine="openpyxl" if path_lower.endswith(".xlsx") else None)
        return pd.read_csv(path)

    def _crunch_rules_to_rule_candidates(
        self, crunch_rules: List[Dict], baseline_metrics: Dict
    ) -> List[RuleCandidate]:
        """Convert crunch rule dicts to RuleCandidate objects."""
        out = []
        for r in crunch_rules:
            direction = "above" if r["direction"] == ">" else "below"
            out.append(
                RuleCandidate(
                    column=r["column"],
                    direction=direction,
                    threshold=r["threshold"],
                    trades_remaining=r["trades_remaining"],
                    trades_filtered=r.get("trades_filtered", 0),
                    total_pnl=r.get("total_pnl", 0),
                    win_rate=r.get("win_rate", 0),
                    avg_win=r.get("avg_win", 0),
                    avg_loss=r.get("avg_loss", 0),
                    profit_factor=r.get("profit_factor", 0),
                    sharpe_ratio=r.get("sharpe_ratio", 0),
                    max_drawdown=r.get("max_drawdown", 0),
                    expectancy=r.get("expectancy", 0),
                    pnl_improvement=r.get("pnl_improvement", 0),
                    pnl_improvement_pct=r.get("pnl_improvement_pct", 0),
                    win_rate_improvement=r.get("win_rate_improvement", 0),
                    expectancy_improvement=r.get("expectancy_improvement", 0),
                    edge_score=r.get("edge_score", r.get("improvement_pct", 0) / 100.0),
                )
            )
        return out

    def analyze(
        self,
        data: pd.DataFrame | str,
        pnl_column: str = 'net_pnl',
        indicator_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        analyze_column_library: bool = False,
        library_path: str = 'column_library.xlsx',
        iterative: bool = True,
        max_rules: int = 8,
    ) -> OptimizationResult:
        """
        Analyze backtest data to find optimal indicator thresholds.
        
        By default uses Dave Mabe iterative crunch (one rule at a time).
        Set iterative=False for legacy single-pass mode (deprecated).
        
        Args:
            data: DataFrame or path to CSV file
            pnl_column: Name of the P&L column
            indicator_columns: List of columns to analyze (if None, auto-detect)
            exclude_columns: List of columns to exclude from analysis
            analyze_column_library: Whether to analyze column library for recommendations
            library_path: Path to column library Excel file
            iterative: If True (default), use crunch() internally. If False, legacy single-pass.
            max_rules: Maximum rules to apply in iterative crunch (only when iterative=True).
            
        Returns:
            OptimizationResult with ranked rules and optional column recommendations
        """
        # Load data (CSV or Excel from backtest export)
        if isinstance(data, str):
            df = self.load_trade_file(data)
        else:
            df = data.copy()

        # Validate PnL column exists
        if pnl_column not in df.columns:
            raise ValueError(
                f"PnL column '{pnl_column}' not found. Available: {list(df.columns)}"
            )

        if iterative:
            # Dave Mabe style: delegate to crunch(), convert to OptimizationResult
            crunch_rules, filtered_df, _, _ = self.crunch(
                df,
                pnl_column=pnl_column,
                target_metric=self.optimize_metric,
                min_trades=self.min_trades_remaining,
                min_improvement_pct=self.min_improvement_pct,
                max_rules=max_rules,
                verbose=False,
            )
            baseline = self._calculate_metrics(df, pnl_column)
            rule_candidates = self._crunch_rules_to_rule_candidates(
                crunch_rules, baseline
            )
            column_recommendations = None
            if analyze_column_library:
                try:
                    from .column_library_analyzer import ColumnLibraryAnalyzer
                    analyzer = ColumnLibraryAnalyzer(library_path)
                    column_recommendations = analyzer.analyze(
                        df, pnl_column=pnl_column
                    )
                except Exception as e:
                    warnings.warn(f"Could not analyze column library: {e}")
            result = OptimizationResult(
                baseline_metrics=baseline,
                rules=rule_candidates,
                filtered_df=filtered_df,
                column_recommendations=column_recommendations,
            )
            return result

        # Legacy single-pass mode (deprecated)
        warnings.warn(
            "analyze(iterative=False) is deprecated. Prefer crunch() or analyze(iterative=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        if indicator_columns is None:
            indicator_columns = self._detect_indicator_columns(
                df, pnl_column, exclude_columns
            )
        
        # Calculate baseline metrics
        baseline = self._calculate_metrics(df, pnl_column)
        
        # Find optimal rules for each indicator
        all_rules = []
        for col in indicator_columns:
            rules = self._find_optimal_thresholds(df, col, pnl_column, baseline)
            all_rules.extend(rules)
        
        # Sort by edge score
        all_rules.sort(key=lambda r: r.edge_score, reverse=True)
        
        # Analyze column library if requested
        column_recommendations = None
        if analyze_column_library:
            try:
                from .column_library_analyzer import ColumnLibraryAnalyzer
                analyzer = ColumnLibraryAnalyzer(library_path)
                column_recommendations = analyzer.analyze(df, pnl_column=pnl_column)
            except Exception as e:
                warnings.warn(f"Could not analyze column library: {e}")
        
        result = OptimizationResult(
            baseline_metrics=baseline,
            rules=all_rules,
            filtered_df=df.copy()
        )
        
        # Add column recommendations to result
        if column_recommendations:
            result.column_recommendations = column_recommendations
        
        return result
    
    def _detect_indicator_columns(
        self,
        df: pd.DataFrame,
        pnl_column: str,
        exclude_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Fallback for entry-column detection when no Entry_Col_* / Col_* found.
        Returns only columns safe for ENTRY crunch: excludes exit snapshots,
        continuous columns, and forward-looking terms (MFE, MAE, Unrealized, etc.).
        """
        default_exclude = {
            'shares', 'entry_value', 'exit_value', 'gross_pnl', 'commission',
            'sec_taf_fee', 'slippage', 'gst', 'borrow_cost', 'net_pnl',
            'account_balance_before', 'account_balance_after', 'entry_price',
            'exit_price', 'entry_time', 'exit_time', 'holding_minutes',
            'bars_held', 'trade_duration', 'benchmark_price', 'Unnamed: 0', 'index',
            'pct_move', 'abs_pct_move', 'win_streak', 'loss_streak',
            'pnl_5_trade_avg', 'pnl_10_trade_avg', 'pnl_20_trade_avg',
            'cumulative_pnl', 'running_pnl',
        }
        if exclude_columns:
            default_exclude.update(exclude_columns)
        default_exclude.add(pnl_column)

        exit_substrings = self._get_entry_exclusion_terms()

        indicator_cols = []
        for col in df.columns:
            if col in default_exclude:
                continue
            # Do not use exit or continuous columns for entry crunch
            if col.startswith("Cont_") or col.startswith("Continuous_"):
                continue
            if "_Exit" in col or col.endswith("_Exit"):
                continue
            if any(x.lower() in col.lower() for x in exit_substrings):
                continue
            if not self._is_usable_numeric_series(df[col]):
                continue
            indicator_cols.append(col)
        return indicator_cols
    
    def _calculate_metrics(self, df: pd.DataFrame, pnl_column: str) -> Dict:
        """Calculate comprehensive trading metrics."""
        pnl = df[pnl_column].values
        n_trades = len(pnl)
        
        if n_trades == 0:
            return {
                'n_trades': 0, 'total_pnl': 0, 'win_rate': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'expectancy': 0
            }
        
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        n_wins = len(wins)
        n_losses = len(losses)
        
        total_pnl = pnl.sum()
        win_rate = n_wins / n_trades if n_trades > 0 else 0
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        
        # Sharpe ratio (annualized, assuming daily trades)
        if pnl.std() > 0:
            sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        # Expectancy (average profit per trade)
        expectancy = total_pnl / n_trades if n_trades > 0 else 0
        
        return {
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy
        }
    
    def _find_optimal_thresholds(
        self,
        df: pd.DataFrame,
        column: str,
        pnl_column: str,
        baseline: Dict
    ) -> List[RuleCandidate]:
        """Find optimal threshold values for a single indicator column."""
        
        rules = []
        values = df[column].dropna()
        
        if len(values) < self.min_trades_remaining:
            return rules
        
        # Generate threshold candidates
        thresholds = self._generate_thresholds(values)
        
        pnl_values = df[pnl_column].values
        col_values = df[column].values
        
        # Test both directions: keep above threshold, keep below threshold
        for direction in ['above', 'below']:
            best_rule = None
            best_score = -np.inf
            
            for thresh in thresholds:
                if direction == 'above':
                    mask = col_values > thresh
                else:
                    mask = col_values < thresh
                
                # Handle NaN values - exclude them
                nan_mask = ~np.isnan(col_values)
                final_mask = mask & nan_mask
                
                trades_remaining = final_mask.sum()
                
                if trades_remaining < self.min_trades_remaining:
                    continue
                
                # Calculate metrics for filtered trades
                filtered_pnl = pnl_values[final_mask]
                metrics = self._calculate_metrics_from_pnl(filtered_pnl)
                
                # Calculate improvements
                pnl_improvement = metrics['total_pnl'] - baseline['total_pnl']
                if baseline['total_pnl'] != 0:
                    pnl_improvement_pct = (pnl_improvement / abs(baseline['total_pnl']) * 100)
                elif pnl_improvement > 0:
                    pnl_improvement_pct = float('inf')
                else:
                    pnl_improvement_pct = 0.0
                
                if pnl_improvement_pct < self.min_improvement_pct:
                    continue
                
                win_rate_improvement = metrics['win_rate'] - baseline['win_rate']
                expectancy_improvement = metrics['expectancy'] - baseline['expectancy']
                
                # Calculate edge score (composite ranking metric)
                edge_score = self._calculate_edge_score(
                    metrics, baseline, trades_remaining, len(df)
                )
                
                if edge_score > best_score:
                    best_score = edge_score
                    best_rule = RuleCandidate(
                        column=column,
                        direction=direction,
                        threshold=thresh,
                        trades_remaining=trades_remaining,
                        trades_filtered=len(df) - trades_remaining,
                        total_pnl=metrics['total_pnl'],
                        win_rate=metrics['win_rate'],
                        avg_win=metrics['avg_win'],
                        avg_loss=metrics['avg_loss'],
                        profit_factor=metrics['profit_factor'],
                        sharpe_ratio=metrics['sharpe_ratio'],
                        max_drawdown=metrics['max_drawdown'],
                        expectancy=metrics['expectancy'],
                        pnl_improvement=pnl_improvement,
                        pnl_improvement_pct=pnl_improvement_pct,
                        win_rate_improvement=win_rate_improvement,
                        expectancy_improvement=expectancy_improvement,
                        edge_score=edge_score
                    )
            
            if best_rule is not None:
                rules.append(best_rule)
        
        return rules
    
    def _generate_thresholds(self, values: pd.Series) -> np.ndarray:
        """Generate threshold values to test."""
        # Use percentiles for more robust threshold selection
        percentiles = np.linspace(5, 95, self.n_threshold_bins)
        thresholds = np.percentile(values.dropna(), percentiles)
        
        # Remove duplicates and sort
        thresholds = np.unique(thresholds)
        
        return thresholds
    
    def _calculate_metrics_from_pnl(self, pnl: np.ndarray) -> Dict:
        """Calculate metrics from a PnL array."""
        n_trades = len(pnl)
        
        if n_trades == 0:
            return {
                'n_trades': 0, 'total_pnl': 0, 'win_rate': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'expectancy': 0
            }
        
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        total_pnl = pnl.sum()
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        
        if pnl.std() > 0:
            sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        expectancy = total_pnl / n_trades if n_trades > 0 else 0
        
        return {
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy
        }
    
    def _calculate_edge_score(
        self,
        metrics: Dict,
        baseline: Dict,
        trades_remaining: int,
        total_trades: int
    ) -> float:
        """
        Calculate a composite edge score for ranking rules.
        
        Higher score = more valuable rule.
        Balances improvement in metrics with trade retention.
        """
        # Normalize improvements
        pnl_factor = 0
        if baseline['total_pnl'] != 0:
            pnl_factor = (metrics['total_pnl'] - baseline['total_pnl']) / abs(baseline['total_pnl'])
        elif metrics['total_pnl'] > 0:
            pnl_factor = 1.0  # Going from 0 to positive is good
        
        # Win rate improvement (weighted)
        wr_factor = (metrics['win_rate'] - baseline['win_rate']) * 2
        
        # Profit factor improvement
        pf_baseline = baseline['profit_factor'] if baseline['profit_factor'] != float('inf') else 2
        pf_current = metrics['profit_factor'] if metrics['profit_factor'] != float('inf') else 2
        pf_factor = (pf_current - pf_baseline) / max(pf_baseline, 0.5) * 0.5
        
        # Trade retention penalty (prefer keeping more trades)
        retention = trades_remaining / total_trades
        retention_factor = retention ** 0.5  # Square root to soften penalty
        
        # Sharpe improvement
        sharpe_factor = 0
        if baseline['sharpe_ratio'] != 0:
            sharpe_factor = (metrics['sharpe_ratio'] - baseline['sharpe_ratio']) / abs(baseline['sharpe_ratio']) * 0.3
        
        # Composite score
        score = (pnl_factor + wr_factor + pf_factor + sharpe_factor) * retention_factor
        
        # Bonus for significant improvements with good retention
        if pnl_factor > 0.2 and retention > 0.3:
            score *= 1.2
        
        return score
    
    def analyze_rule_combinations(
        self,
        df: pd.DataFrame,
        rules: List[RuleCandidate],
        pnl_column: str = 'net_pnl',
        max_rules: int = 3
    ) -> List[Tuple[List[RuleCandidate], Dict]]:
        """
        Analyze combinations of rules to find synergistic effects.
        
        Args:
            df: Original dataframe
            rules: List of individual rules to combine
            pnl_column: PnL column name
            max_rules: Maximum rules to combine
            
        Returns:
            List of (rule_combination, metrics) tuples
        """
        from itertools import combinations
        
        results = []
        baseline = self._calculate_metrics(df, pnl_column)
        
        for n in range(2, max_rules + 1):
            for rule_combo in combinations(rules[:10], n):  # Top 10 rules only
                filtered_df = df.copy()
                
                for rule in rule_combo:
                    if rule.direction == 'above':
                        filtered_df = filtered_df[filtered_df[rule.column] > rule.threshold]
                    else:
                        filtered_df = filtered_df[filtered_df[rule.column] < rule.threshold]
                
                if len(filtered_df) >= self.min_trades_remaining:
                    metrics = self._calculate_metrics(filtered_df, pnl_column)
                    metrics['n_rules'] = len(rule_combo)
                    metrics['rule_names'] = [f"{r.column} {r.direction} {r.threshold:.4f}" for r in rule_combo]
                    metrics['improvement'] = metrics['total_pnl'] - baseline['total_pnl']
                    results.append((list(rule_combo), metrics))
        
        # Sort by total PnL improvement
        results.sort(key=lambda x: x[1]['total_pnl'], reverse=True)
        
        return results[:20]  # Top 20 combinations


def quick_analyze(
    csv_path: str,
    pnl_column: str = 'net_pnl',
    top_n: int = 10
) -> None:
    """
    Quick analysis function for command-line use.
    
    Args:
        csv_path: Path to backtest CSV
        pnl_column: Name of PnL column
        top_n: Number of top rules to display
    """
    print(f"\n{'='*70}")
    print("STRATEGY CRUNCHER - Backtest Optimization Analysis")
    print(f"{'='*70}\n")
    
    cruncher = StrategyCruncher()
    results = cruncher.analyze(csv_path, pnl_column=pnl_column)
    
    baseline = results.baseline_metrics
    
    print("BASELINE METRICS (Before Optimization)")
    print("-" * 40)
    print(f"  Total Trades:    {baseline['n_trades']:,}")
    print(f"  Total P&L:       ${baseline['total_pnl']:,.2f}")
    print(f"  Win Rate:        {baseline['win_rate']:.1%}")
    print(f"  Avg Win:         ${baseline['avg_win']:,.2f}")
    print(f"  Avg Loss:        ${baseline['avg_loss']:,.2f}")
    print(f"  Profit Factor:   {baseline['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:    {baseline['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    ${baseline['max_drawdown']:,.2f}")
    print(f"  Expectancy:      ${baseline['expectancy']:.2f}/trade")
    
    print(f"\n\nTOP {top_n} OPTIMIZATION RULES (Ranked by Edge Score)")
    print("=" * 70)
    
    for i, rule in enumerate(results.get_top_rules(top_n), 1):
        dir_symbol = '>' if rule.direction == 'above' else '<'
        print(f"\n#{i}. {rule.column} {dir_symbol} {rule.threshold:.4f}")
        print(f"    Edge Score:      {rule.edge_score:.3f}")
        print(f"    Trades Kept:     {rule.trades_remaining:,} ({rule.trades_remaining/baseline['n_trades']:.1%} of original)")
        print(f"    Total P&L:       ${rule.total_pnl:,.2f} ({rule.pnl_improvement_pct:+.1f}%)")
        print(f"    Win Rate:        {rule.win_rate:.1%} ({rule.win_rate_improvement:+.1%})")
        print(f"    Profit Factor:   {rule.profit_factor:.2f}")
        print(f"    Expectancy:      ${rule.expectancy:.2f}/trade ({rule.expectancy_improvement:+.2f})")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cruncher.py <backtest.csv> [pnl_column]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    pnl_col = sys.argv[2] if len(sys.argv) > 2 else 'net_pnl'
    
    quick_analyze(csv_path, pnl_col)
