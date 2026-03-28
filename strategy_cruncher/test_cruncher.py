"""
Test script for FireEye (strategy_cruncher package)
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_test_df():
    """Load backtest CSV if present; otherwise return synthetic DataFrame so tests can run."""
    import pandas as pd
    import numpy as np

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real5backtest_trades_10000.csv')
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path), csv_path

    # Synthetic data: enough rows and columns for crunch/analyze to exercise logic
    n = 400
    np.random.seed(42)
    df = pd.DataFrame({
        'ticker': np.random.choice(['AAPL', 'TSLA', 'NVDA'], n),
        'date': pd.date_range('2024-01-01', periods=n, freq='h').strftime('%Y-%m-%d').tolist(),
        'entry_time': pd.date_range('2024-01-01', periods=n, freq='h').strftime('%H:%M:%S').tolist(),
        'entry_price': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'exit_price': 100 + np.cumsum(np.random.randn(n) * 0.5) + np.random.randn(n) * 2,
        'net_pnl': np.random.randn(n) * 50,
        'Entry_Col_ATR14': np.abs(np.random.randn(n)) * 2 + 1,
        'Entry_Col_RSI14': np.random.uniform(30, 70, n),
    })
    return df, None


def run_tests():
    print("=" * 60)
    print("FIREEYE - Test Suite")
    print("=" * 60)

    # Test 1: Import all modules
    print("\nTest 1: Importing modules...")
    try:
        from strategy_cruncher import StrategyCruncher, OptimizationResult, RuleCandidate
        from strategy_cruncher import enrich_backtest, add_market_context
        print("  [OK] PASS: All imports successful")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        return False

    # Load test data once (file or synthetic)
    df, csv_path = _get_test_df()
    if csv_path:
        print(f"\n  Using file: {csv_path}")
    else:
        print(f"\n  Using synthetic data ({len(df)} rows)")

    # Test 2: Load and analyze data
    print("\nTest 2: Analyzing backtest data...")
    try:
        cruncher = StrategyCruncher(min_improvement_pct=5.0, min_trades_remaining=50)
        if csv_path:
            results = cruncher.analyze(csv_path, pnl_column='net_pnl')
        else:
            results = cruncher.analyze(df, pnl_column='net_pnl')
        print(f"  [OK] PASS: Found {len(results.rules)} rules")
        n_trades = results.baseline_metrics['n_trades']
        total_pnl = results.baseline_metrics['total_pnl']
        print(f"  Baseline: {n_trades:,} trades, PnL=${total_pnl:,.0f}")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Test enrichment
    print("\nTest 3: Testing enrichment...")
    try:
        df_enriched = enrich_backtest(df.copy())
        new_cols = len(df_enriched.columns) - len(df.columns)
        print(f"  [OK] PASS: Added {new_cols} new columns")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        return False

    # Test 4: Test rule access
    print("\nTest 4: Testing rule access...")
    try:
        if results.rules:
            top_rule = results.get_top_rules(1)[0]
            dir_symbol = '>=' if top_rule.direction == 'above' else '<'
            print(f"  Top rule: {top_rule.column} {dir_symbol} {top_rule.threshold:.2f}")
            print(f"  Mean $/trade spread: {top_rule.mean_profit_spread:.2f}")
            print(f"  PnL improvement: {top_rule.pnl_improvement_pct:+.1f}%")
            print("  [OK] PASS: Rule access works")
        else:
            print("  [OK] PASS: No rules found (expected with high threshold)")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        return False

    # Test 5: Test metrics calculation
    print("\nTest 5: Testing metrics calculation...")
    try:
        baseline = results.baseline_metrics
        assert 'n_trades' in baseline
        assert 'total_pnl' in baseline
        assert 'win_rate' in baseline
        assert 'profit_factor' in baseline
        assert 'sharpe_ratio' in baseline
        assert 'max_drawdown' in baseline
        assert 'expectancy' in baseline
        assert 'calmar_ratio' in baseline
        if results.rules:
            tr0 = results.rules[0]
            assert hasattr(tr0, 'mean_profit_spread')
            assert hasattr(tr0, 'curve_quality_improvement')
        print("  [OK] PASS: All metrics present")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        return False

    # Test 6: Test rule combination analysis
    print("\nTest 6: Testing rule combinations...")
    try:
        if len(results.rules) >= 2:
            combos = cruncher.analyze_rule_combinations(
                df, results.rules[:5], pnl_column='net_pnl', max_rules=2
            )
            print(f"  [OK] PASS: Found {len(combos)} combinations")
        else:
            print("  [OK] PASS: Not enough rules for combinations (expected)")
    except Exception as e:
        print(f"  [X] FAIL: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
