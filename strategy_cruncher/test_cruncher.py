"""
Test script for Strategy Cruncher
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    print("=" * 60)
    print("STRATEGY CRUNCHER - Test Suite")
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
    
    # Test 2: Load and analyze data
    print("\nTest 2: Analyzing backtest data...")
    try:
        cruncher = StrategyCruncher(min_improvement_pct=5.0, min_trades_remaining=50)
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'real5backtest_trades_10000.csv')
        results = cruncher.analyze(csv_path, pnl_column='net_pnl')
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
        import pandas as pd
        df = pd.read_csv(csv_path)
        df_enriched = enrich_backtest(df)
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
            dir_symbol = '>' if top_rule.direction == 'above' else '<'
            print(f"  Top rule: {top_rule.column} {dir_symbol} {top_rule.threshold:.2f}")
            print(f"  Edge score: {top_rule.edge_score:.2f}")
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
