# =====================================================
# Dave Mabe Style Iterative Filter Phase (FireEye)
# One wide-net backtest -> Smart Filter Phase
# =====================================================
#
# Run one wide-net backtest -> add column library -> cruncher finds best single rule
# -> apply it -> re-crunch remaining columns -> repeat until no more good rules.
#
# Inspired by Dave Mabe's Filter Phase: iterative, one-rule-at-a-time, super explainable.

"""
FireEye — Core Optimization Engine

Analyzes backtest trade data to find optimal indicator thresholds
that maximize profit, win rate, or other metrics while filtering bad trades.

Inspired by Dave Mabe's approach: cast a wide net with the initial backtest,
then use optimization to find the subset of trades worth taking.

Predictive columns: The rules returned by crunch() use the 'column' field (e.g. Entry_Col_ATR14).
Naming matches backtestlibrary (see columns.py): Entry_Col_* = entry bar snapshot, Exit_Col_* = exit
bar snapshot, Continuous_Col_*_* = intra-trade tracking (suffixes _Entry, _Exit, _Max, _Min, _At30min, _At60min).

Calmar-style ratio (aligned with common trading reports): total P&L divided by maximum drawdown
on the cumulative equity curve (higher is better when drawdown is positive).
Those column names are exactly the predictive (entry) columns for your strategy—use them to
filter or weight entries when you crunch your Excel/CSV from the backtest.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal


def calmar_ratio(total_pnl: float, max_drawdown: float) -> float:
    """Total P&L divided by max equity drawdown (Dave Mabe-style Calmar column)."""
    if max_drawdown is None or max_drawdown <= 1e-12:
        return float("inf") if total_pnl > 0 else 0.0
    return float(total_pnl / max_drawdown)


def _report_sort_columns(df: pd.DataFrame) -> List[str]:
    """Chronological order for equity-style metrics (matches report dual curves)."""
    return [c for c in ("date", "entry_date", "entry_time", "EntryTime") if c in df.columns]


def equity_curve_quality(cumulative: np.ndarray) -> float:
    """
    Correlation of cumulative P&L with trade index (ideal smooth uptrend ≈ 1).
    Per Dave Mabe report docs: relates equity path to a 45° rising line in (index, equity) space;
    can range from -1 to 1; improvement compares better vs worse trade-set curves.
    """
    n = len(cumulative)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    sy = float(np.std(cumulative))
    if sy < 1e-15:
        return 0.0
    r = np.corrcoef(x, cumulative.astype(float))[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def curve_quality_metrics_for_rule(
    df: pd.DataFrame,
    pnl_column: str,
    column: str,
    direction: Literal["above", "below"],
    threshold: float,
) -> Tuple[float, float, float]:
    """
    (quality_better, quality_worse, quality_improvement) using green/red cumulative paths
    on time-sorted trades — see https://app.davemabe.com/docs/report-metrics
    """
    sort_cols = _report_sort_columns(df)
    plot_df = (
        df.sort_values(sort_cols).reset_index(drop=True)
        if sort_cols
        else df.reset_index(drop=True)
    )
    pnl = plot_df[pnl_column].astype(float).values
    col = plot_df[column]
    if direction == "above":
        passes = (col >= threshold).fillna(False).values
    else:
        passes = (col < threshold).fillna(False).values
    cum_better = np.cumsum(np.where(passes, pnl, 0.0))
    cum_worse = np.cumsum(np.where(~passes, pnl, 0.0))
    qb = equity_curve_quality(cum_better)
    qw = equity_curve_quality(cum_worse)
    return qb, qw, qb - qw


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
    
    # Dave Mabe report metrics (https://app.davemabe.com/docs/report-metrics)
    mean_profit_spread: float  # mean P&L | better − mean P&L | worse (optimization target for Value)
    curve_quality_better: float  # correlation of green cumulative curve with rising trend
    curve_quality_worse: float  # same for red curve
    curve_quality_improvement: float  # curve_quality_better − curve_quality_worse (max 2.0)
    
    # Ranking (legacy name: defaults to scale comparable to mean spread in single-pass mode)
    edge_score: float
    
    def __repr__(self):
        dir_symbol = '>=' if self.direction == 'above' else '<'
        return (f"Rule: {self.column} {dir_symbol} {self.threshold:.4f} | "
                f"PnL: ${self.total_pnl:,.2f} ({self.pnl_improvement_pct:+.1f}%) | "
                f"Win Rate: {self.win_rate:.1%} | "
                f"Trades: {self.trades_remaining} | "
                f"Mean spread: {self.mean_profit_spread:.4f}")


@dataclass
class OptimizationResult:
    """Complete optimization results for a backtest."""
    baseline_metrics: Dict
    rules: List[RuleCandidate]
    applied_rules: List[RuleCandidate] = field(default_factory=list)
    filtered_df: Optional[pd.DataFrame] = None
    column_recommendations: Optional[List] = None  # ColumnRecommendation objects
    
    def get_top_rules(self, n: int = 10, metric: str = 'mean_profit_spread') -> List[RuleCandidate]:
        """Get top N rules ranked by specified metric (default: mean $/trade spread, Dave Mabe report style)."""
        return sorted(self.rules, key=lambda r: getattr(r, metric), reverse=True)[:n]
    
    def get_top_column_recommendations(self, n: int = 20) -> List:
        """Get top N column recommendations."""
        if not self.column_recommendations:
            return []
        return self.column_recommendations[:n]
    
    def apply_rule(self, rule: RuleCandidate, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a rule and return filtered dataframe."""
        col = df[rule.column]
        if rule.direction == 'above':
            mask = col.ge(rule.threshold).fillna(False)
        else:
            mask = col.lt(rule.threshold).fillna(False)
        
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
        optimize_metric: Literal[
            'pnl', 'total_profit', 'win_rate', 'profit_factor', 'expectancy', 'sharpe', 'sharpe_ratio'
        ] = 'total_profit'
    ):
        """
        Initialize the FireEye optimizer (StrategyCruncher).
        
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
        pnl = pd.to_numeric(df[pnl_column], errors="coerce")
        if pnl.isna().any():
            raise ValueError(
                f"PnL column {pnl_column!r} contains NaN or non-numeric values; remove or fix those rows."
            )
        if metric == "profit_factor":
            wins = pnl[pnl > 0].sum()
            losses = abs(pnl[pnl < 0].sum())
            return wins / losses if losses > 0 else (float('inf') if wins > 0 else 0.0)
        elif metric == "expectancy":
            return float(pnl.mean())
        elif metric == "win_rate":
            return float((pnl > 0).mean() * 100)
        elif metric == "total_profit" or metric == "pnl":
            return float(pnl.sum())
        elif metric in ("sharpe", "sharpe_ratio"):
            if len(pnl) < 2 or pnl.std() == 0:
                return 0.0
            return float((pnl.mean() / pnl.std()) * np.sqrt(252))
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

    # Column naming — backtestlibrary.columns (ENTRY_COLUMN_PREFIX / EXIT_COLUMN_PREFIX /
    # CONTINUOUS_COLUMN_PREFIX): apply_entry_columns writes Entry_Col_{Col_*}, apply_exit_columns
    # writes Exit_Col_{Col_*}, continuous tracking writes Continuous_Col_{base}_{Entry|Exit|Max|Min|At30min|At60min}.

    def _detect_entry_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect entry-only columns for entry crunch.
        Accepts: Entry_Col_* (backtestlibrary entry snapshots) and bare Col_* when present
        (e.g. wide-format rows without the Entry_ prefix).
        Rejects: Exit_Col_* (exit snapshots), Continuous_* / Cont_* (intra-trade paths), and
        names matching forward-looking / exit-derived heuristics (MFE, MAE, Unrealized, etc.).
        """
        exit_substrings = self._get_entry_exclusion_terms()
        entry_cols = []
        for c in df.columns:
            if c.startswith("Cont_") or c.startswith("Continuous_"):
                continue
            if c.startswith("Exit_"):
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

    def _require_entry_columns(
        self, df: pd.DataFrame, where: str, entry_cols: List[str]
    ) -> None:
        """Raise if no entry-style columns; lists file headers for debugging."""
        if entry_cols:
            return
        raise ValueError(
            f"{where}: No usable entry-style columns found. "
            "Expected columns like backtestlibrary exports: 'Entry_Col_*' (entry snapshot) "
            "or bare 'Col_*' in wide format — not 'Exit_Col_*', not 'Continuous_Col_*', "
            "not forward-looking exit metrics. "
            f"Columns in file: {list(df.columns)!r}"
        )

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
            target_metric: profit_factor, expectancy, win_rate, total_profit, sharpe_ratio, etc.
            min_trades: Minimum trades after each rule (use aggression factor in the UI)
            min_improvement_pct: Minimum % improvement to accept a rule
            max_rules: Maximum rules to apply
            verbose: Print progress to stdout
            
        Returns:
            (rules, filtered_df, before_curve, after_curve)
            - rules: list of applied rule dicts; each rule['column'] is a predictive (entry) column name
            - filtered_df: trades remaining after all rules
            - before_curve: cumulative P&L of original trades (chronological when time cols present)
            - after_curve: cumulative P&L of filtered trades (chronological when time cols present)

        Raises:
            ValueError: If the frame has no usable Entry_Col_* / Col_* columns (lists all headers;
                naming matches backtestlibrary ``columns.py``).
        """
        if pnl_column not in df.columns:
            raise ValueError(f"PnL column '{pnl_column}' not found. Available: {list(df.columns)}")
        
        # Auto-detect entry columns (no exit columns); no silent fallback
        self.entry_columns = self._detect_entry_columns(df)
        self._require_entry_columns(df, "crunch()", self.entry_columns)

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
                            mask = (current_df[col] >= thresh).fillna(False)
                        else:
                            mask = (current_df[col] < thresh).fillna(False)
                        filtered = current_df[mask]
                        if len(filtered) < min_trades:
                            continue
                        score = self._calculate_metric(filtered, target_metric, pnl_column)
                        # Handle inf for profit_factor
                        if score == float('inf'):
                            score = 999.0
                        denom = abs(baseline) if baseline != 0 else 1e-9
                        if not np.isfinite(score) or not np.isfinite(baseline):
                            continue
                        improvement = (score - baseline) / denom * 100
                        if not np.isfinite(improvement):
                            continue
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
                current_df = current_df[(current_df[col] >= thresh).fillna(False)]
            else:
                current_df = current_df[(current_df[col] < thresh).fillna(False)]

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
                mval = best_rule['new_metric']
                imp = best_rule['improvement_pct']
                n_tr = best_rule['trades_remaining']
                ml = target_metric.replace("_", " ").title()
                print(f"Rule {rule_num}: {col} {dir_sym} {thresh}   -> {ml} {mval:.4g} (+{imp:.0f}%) | Trades: {n_tr:,}")
        
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
            ml = target_metric.replace("_", " ").title()
            print(f"\nFinal Strategy: {len(results)} rules | {ml} {final_metric:.4g} (+{pct:.0f}%) | Trades: {len(current_df):,}")
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
        self,
        crunch_rules: List[Dict],
        baseline_metrics: Dict,
        df: pd.DataFrame,
        pnl_column: str,
    ) -> List[RuleCandidate]:
        """Convert crunch rule dicts to RuleCandidate objects (report metrics on full backtest)."""
        out = []
        for r in crunch_rules:
            direction = "above" if r["direction"] == ">" else "below"
            qb, qw, qimprove = curve_quality_metrics_for_rule(
                df, pnl_column, r["column"], direction, float(r["threshold"])
            )
            col = df[r["column"]]
            if direction == "above":
                passes = (col >= float(r["threshold"])).fillna(False)
            else:
                passes = (col < float(r["threshold"])).fillna(False)
            pnl = df[pnl_column].astype(float)
            better_mean = float(pnl[passes].mean()) if passes.any() else 0.0
            worse_mean = float(pnl[~passes].mean()) if (~passes).any() else 0.0
            spread = better_mean - worse_mean
            out.append(
                RuleCandidate(
                    column=r["column"],
                    direction=direction,
                    threshold=r["threshold"],
                    trades_remaining=r["trades_remaining"],
                    trades_filtered=r["trades_filtered"],
                    total_pnl=r["total_pnl"],
                    win_rate=r["win_rate"],
                    avg_win=r["avg_win"],
                    avg_loss=r["avg_loss"],
                    profit_factor=r["profit_factor"],
                    sharpe_ratio=r["sharpe_ratio"],
                    max_drawdown=r["max_drawdown"],
                    expectancy=r["expectancy"],
                    pnl_improvement=r["pnl_improvement"],
                    pnl_improvement_pct=r["pnl_improvement_pct"],
                    win_rate_improvement=r["win_rate_improvement"],
                    expectancy_improvement=r["expectancy_improvement"],
                    mean_profit_spread=spread,
                    curve_quality_better=qb,
                    curve_quality_worse=qw,
                    curve_quality_improvement=qimprove,
                    edge_score=spread,
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
            indicator_columns: Columns to analyze; if None, uses entry-style columns only
                (Entry_Col_* / Col_*). Raises ValueError if none are found—fix the file or pass an explicit list.
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
                crunch_rules, baseline, df, pnl_column
            )
            column_recommendations = None
            if analyze_column_library:
                from .column_library_analyzer import ColumnLibraryAnalyzer
                analyzer = ColumnLibraryAnalyzer(library_path)
                column_recommendations = analyzer.analyze(
                    df, pnl_column=pnl_column
                )
            result = OptimizationResult(
                baseline_metrics=baseline,
                rules=rule_candidates,
                filtered_df=filtered_df,
                column_recommendations=column_recommendations,
            )
            return result

        # Legacy single-pass mode (parallel top-column rules for reports)
        if indicator_columns is None:
            indicator_columns = self._detect_entry_columns(df)
            self._require_entry_columns(
                df, "analyze(iterative=False)", indicator_columns
            )
            if exclude_columns:
                _ex = set(exclude_columns)
                _before = list(indicator_columns)
                indicator_columns = [c for c in indicator_columns if c not in _ex]
                if not indicator_columns:
                    raise ValueError(
                        "analyze(iterative=False): exclude_columns removed every "
                        "entry-style column. "
                        f"exclude_columns={list(exclude_columns)!r}, "
                        f"detected entry columns before filter: {_before!r}"
                    )
        
        # Calculate baseline metrics
        baseline = self._calculate_metrics(df, pnl_column)
        
        # Find optimal rules for each indicator
        all_rules = []
        for col in indicator_columns:
            rules = self._find_optimal_thresholds(df, col, pnl_column, baseline)
            all_rules.extend(rules)
        
        # Importance ≈ mean-profit spread (Dave Mabe report ranking)
        all_rules.sort(key=lambda r: r.mean_profit_spread, reverse=True)
        
        # Analyze column library if requested (errors propagate — no silent fallback)
        column_recommendations = None
        if analyze_column_library:
            from .column_library_analyzer import ColumnLibraryAnalyzer
            analyzer = ColumnLibraryAnalyzer(library_path)
            column_recommendations = analyzer.analyze(df, pnl_column=pnl_column)
        
        result = OptimizationResult(
            baseline_metrics=baseline,
            rules=all_rules,
            filtered_df=df.copy()
        )
        
        # Add column recommendations to result
        if column_recommendations:
            result.column_recommendations = column_recommendations
        
        return result

    def _calculate_metrics(self, df: pd.DataFrame, pnl_column: str) -> Dict:
        """Calculate comprehensive trading metrics."""
        pnl = pd.to_numeric(df[pnl_column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(pnl).any():
            raise ValueError(
                f"PnL column {pnl_column!r} contains NaN or non-numeric values; remove or fix those rows."
            )
        n_trades = len(pnl)
        
        if n_trades == 0:
            return {
                'n_trades': 0, 'total_pnl': 0, 'win_rate': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'expectancy': 0,
                'calmar_ratio': 0.0,
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
        
        cr = calmar_ratio(total_pnl, max_drawdown)
        
        return {
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'calmar_ratio': cr,
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
        
        col_series = df[column]
        pnl_values = pd.to_numeric(df[pnl_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(pnl_values).all():
            raise ValueError(
                f"PnL column {pnl_column!r} contains NaN or non-numeric values; remove or fix those rows."
            )
        
        # Test both directions: keep above threshold, keep below threshold
        for direction in ['above', 'below']:
            best_rule = None
            best_score = -np.inf
            
            for thresh in thresholds:
                tf = float(thresh)
                if direction == 'above':
                    passes = col_series.ge(tf).fillna(False).to_numpy()
                else:
                    passes = col_series.lt(tf).fillna(False).to_numpy()
                final_mask = passes
                worse_mask = ~passes
                
                trades_remaining = int(final_mask.sum())
                n_worse = int(worse_mask.sum())
                
                if trades_remaining < self.min_trades_remaining:
                    continue
                if n_worse < max(10, min(self.min_trades_remaining // 5, 100)):
                    continue
                
                # Calculate metrics for filtered trades (better set)
                filtered_pnl = pnl_values[final_mask]
                metrics = self._calculate_metrics_from_pnl(filtered_pnl)
                
                # Calculate improvements vs full backtest baseline
                pnl_improvement = metrics['total_pnl'] - baseline['total_pnl']
                if baseline['total_pnl'] != 0:
                    pnl_improvement_pct = (pnl_improvement / abs(baseline['total_pnl']) * 100)
                elif pnl_improvement > 0:
                    pnl_improvement_pct = float('inf')
                else:
                    pnl_improvement_pct = 0.0
                
                if pnl_improvement_pct < self.min_improvement_pct:
                    continue
                
                mean_better = float(np.mean(pnl_values[final_mask]))
                mean_worse = float(np.mean(pnl_values[worse_mask])) if n_worse > 0 else 0.0
                mean_spread = mean_better - mean_worse
                # Dave Mabe: optimal split maximizes avg-profit gap between better and worse sets
                if mean_spread <= 0:
                    continue
                
                if mean_spread <= best_score:
                    continue
                
                win_rate_improvement = metrics['win_rate'] - baseline['win_rate']
                expectancy_improvement = metrics['expectancy'] - baseline['expectancy']
                
                qb, qw, qimp = curve_quality_metrics_for_rule(
                    df, pnl_column, column, direction, tf
                )
                
                best_score = mean_spread
                n_not_better = int((~passes).sum())
                best_rule = RuleCandidate(
                    column=column,
                    direction=direction,
                    threshold=thresh,
                    trades_remaining=trades_remaining,
                    trades_filtered=n_not_better,
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
                    mean_profit_spread=mean_spread,
                    curve_quality_better=qb,
                    curve_quality_worse=qw,
                    curve_quality_improvement=qimp,
                    edge_score=mean_spread,
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
        pnl = np.asarray(pnl, dtype=float)
        if np.isnan(pnl).any():
            raise ValueError("Internal error: PnL slice contained NaN in _calculate_metrics_from_pnl.")
        n_trades = len(pnl)
        
        if n_trades == 0:
            return {
                'n_trades': 0, 'total_pnl': 0, 'win_rate': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0, 'expectancy': 0,
                'calmar_ratio': 0.0,
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
        
        cr = calmar_ratio(total_pnl, max_drawdown)
        
        return {
            'n_trades': n_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'calmar_ratio': cr,
        }
    
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
                    col = filtered_df[rule.column]
                    if rule.direction == 'above':
                        m = col.ge(rule.threshold).fillna(False)
                    else:
                        m = col.lt(rule.threshold).fillna(False)
                    filtered_df = filtered_df[m]
                
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
    print("FIREEYE - Backtest Optimization Analysis")
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
    
    print(f"\n\nTOP {top_n} OPTIMIZATION RULES (Ranked by mean $/trade spread)")
    print("=" * 70)
    
    for i, rule in enumerate(results.get_top_rules(top_n), 1):
        dir_symbol = '>=' if rule.direction == 'above' else '<'
        print(f"\n#{i}. {rule.column} {dir_symbol} {rule.threshold:.4f}")
        print(f"    Mean $ spread:   {rule.mean_profit_spread:.4f}")
        print(f"    Curve Q Δ:       {rule.curve_quality_improvement:.4f}")
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
