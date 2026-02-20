"""
Strategy Cruncher - Quick Start Script

Run the Strategy Cruncher from the command line:
    python -m strategy_cruncher.run backtest.csv [pnl_column]

Or launch the web app:
    python -m strategy_cruncher.run --app
    
Or:
    streamlit run strategy_cruncher/app.py
"""

import sys
import os

# Add parent directory to path for imports when running as script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    if sys.argv[1] == '--app' or sys.argv[1] == '-a':
        # Launch Streamlit app
        import subprocess
        app_path = os.path.join(os.path.dirname(__file__), 'app.py')
        subprocess.run(['streamlit', 'run', app_path])
    
    elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print_help()
    
    else:
        # Run command-line analysis
        from strategy_cruncher import StrategyCruncher

        # Parse positional args while skipping values for known flags.
        raw_args = sys.argv[1:]
        flags_with_value = {'--library', '-l', '--max-rules', '--min-improvement'}
        positionals = []
        i = 0
        while i < len(raw_args):
            arg = raw_args[i]
            if arg in flags_with_value:
                i += 2
                continue
            if arg.startswith('-'):
                i += 1
                continue
            positionals.append(arg)
            i += 1

        if not positionals:
            print("Error: No CSV file specified.")
            print_help()
            sys.exit(1)
        csv_path = positionals[0]
        pnl_column = positionals[1] if len(positionals) > 1 else 'net_pnl'
        
        if not os.path.exists(csv_path):
            print(f"Error: File '{csv_path}' not found.")
            sys.exit(1)
        
        # Dave Mabe crunch is default; --legacy uses old single-pass analyze
        use_legacy = '--legacy' in sys.argv or '-L' in sys.argv
        use_crunch = not use_legacy
        
        print(f"\n{'='*70}")
        mode = " [Dave Mabe Crunch]" if use_crunch else " [Legacy]"
        print("STRATEGY CRUNCHER - Backtest Optimization Analysis" + mode)
        print(f"{'='*70}")
        print(f"\nAnalyzing: {csv_path}")
        print(f"P&L Column: {pnl_column}")
        
        cruncher = StrategyCruncher()
        
        # Check if user wants to analyze column library (skip for crunch mode)
        analyze_library = not use_crunch and ('--library' in sys.argv or '-l' in sys.argv)
        library_path = 'column_library.xlsx'
        if analyze_library:
            # Check for library path argument
            lib_idx = None
            if '--library' in sys.argv:
                lib_idx = sys.argv.index('--library')
            elif '-l' in sys.argv:
                lib_idx = sys.argv.index('-l')
            
            if lib_idx is not None and lib_idx + 1 < len(sys.argv):
                library_path = sys.argv[lib_idx + 1]
        
        # Parse optional crunch args
        max_rules = 8
        min_improvement = 8.0
        if '--max-rules' in sys.argv:
            idx = sys.argv.index('--max-rules')
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
                max_rules = int(sys.argv[idx + 1])
        if '--min-improvement' in sys.argv:
            idx = sys.argv.index('--min-improvement')
            if idx + 1 < len(sys.argv):
                try:
                    min_improvement = float(sys.argv[idx + 1])
                except ValueError:
                    pass

        try:
            if use_crunch:
                import pandas as pd
                df = pd.read_csv(csv_path)
                crunch_rules, filtered_df, _, _ = cruncher.crunch(
                    df, pnl_column=pnl_column,
                    target_metric="profit_factor",
                    min_trades=300,
                    min_improvement_pct=min_improvement,
                    max_rules=max_rules,
                    verbose=True
                )
                baseline_m = cruncher._calculate_metric(df, "profit_factor", pnl_column)
                final_m = cruncher._calculate_metric(filtered_df, "profit_factor", pnl_column) if len(filtered_df) > 0 else 0
                print(f"\n{'='*70}")
                print("RULES APPLIED (Dave Mabe Iterative)")
                print("=" * 70)
                for r in crunch_rules:
                    print(f"  {r['rule_num']}. {r['column']} {r['direction']} {r['threshold']} -> "
                          f"PF {r['new_metric']:.3f} (+{r['improvement_pct']}%) | Trades: {r['trades_remaining']}")
                print(f"\n{'='*70}")
                print(f"FINAL SUMMARY")
                print("-" * 40)
                print(f"  Rules applied:     {len(crunch_rules)}")
                print(f"  Trades:            {len(df):,} -> {len(filtered_df):,}")
                print(f"  Profit Factor:     {baseline_m:.3f} -> {final_m:.3f}")
                if baseline_m and baseline_m != float('inf'):
                    pct = (final_m - baseline_m) / abs(baseline_m) * 100
                    print(f"  Edge improvement:  {pct:+.1f}%")
                print(f"{'='*70}\n")
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    results = cruncher.analyze(
                        csv_path,
                        pnl_column=pnl_column,
                        analyze_column_library=analyze_library,
                        library_path=library_path,
                        iterative=False,  # legacy single-pass
                    )
                baseline = results.baseline_metrics
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        if not use_crunch:
            baseline = results.baseline_metrics
            print(f"\n{'='*70}")
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
            
            print(f"\n\n{'='*70}")
            print("TOP 15 OPTIMIZATION RULES (Ranked by Edge Score)")
            print("=" * 70)
            
            if not results.rules:
                print("\nWARNING: No optimization rules found that meet the criteria.")
                print("    Try lowering the minimum improvement threshold.\n")
            else:
                for i, rule in enumerate(results.get_top_rules(15), 1):
                    dir_symbol = '>' if rule.direction == 'above' else '<'
                    print(f"\n#{i}. {rule.column} {dir_symbol} {rule.threshold:.4f}")
                    print(f"    Edge Score:      {rule.edge_score:.3f}")
                    print(f"    Trades Kept:     {rule.trades_remaining:,} ({rule.trades_remaining/baseline['n_trades']:.1%} of original)")
                    print(f"    Total P&L:       ${rule.total_pnl:,.2f} ({rule.pnl_improvement_pct:+.1f}%)")
                    print(f"    Win Rate:        {rule.win_rate:.1%} ({rule.win_rate_improvement:+.1%})")
                    print(f"    Profit Factor:   {rule.profit_factor:.2f}")
                    print(f"    Expectancy:      ${rule.expectancy:.2f}/trade ({rule.expectancy_improvement:+.2f})")
            
            # Column Library Recommendations
            if results.column_recommendations:
                print(f"\n\n{'='*70}")
                print("COLUMN LIBRARY RECOMMENDATIONS")
                print("=" * 70)
                print(f"\nTop 15 columns from your library that would improve your strategy:\n")
                
                top_recs = results.get_top_column_recommendations(15)
                for i, rec in enumerate(top_recs, 1):
                    status = "EXISTS" if rec.calculation_method == 'Already exists' else ("CAN CALC" if rec.can_calculate else "NEEDS DATA")
                    print(f"{i:2}. {rec.column_name:30} ({rec.category:20})")
                    print(f"    Score: {rec.predictive_score:.3f} | {status}")
                    if rec.description:
                        print(f"    {rec.description[:70]}")
                    print()
            
            print(f"\n{'='*70}")
            print("Analysis complete!")
            print(f"\nTip: --legacy for single-pass | --max-rules N | --min-improvement X | --app for web UI")
            print(f"{'='*70}\n")


def print_help():
    help_text = """
======================================================================
                       STRATEGY CRUNCHER                              
            Backtest Optimization & Rule Discovery Tool               
======================================================================

USAGE:
    python -m strategy_cruncher.run <backtest.csv> [pnl_column]
    python -m strategy_cruncher.run <backtest.csv> [pnl_column]   # Dave Mabe crunch (default)
    python -m strategy_cruncher.run <backtest.csv> [pnl_column] --library [library.xlsx]
    python -m strategy_cruncher.run --app

ARGUMENTS:
    backtest.csv    Path to your backtest CSV file
    pnl_column      Name of the P&L column (default: 'net_pnl')

OPTIONS:
    --app, -a           Launch the interactive web application
    --crunch            Dave Mabe iterative crunch (default)
    --legacy, -L        Use legacy single-pass analyze
    --max-rules N       Max rules to apply in crunch mode (default: 8)
    --min-improvement X Min %% improvement to accept a rule (default: 8.0)
    --library, -l       Analyze column library (legacy mode only)
    --help, -h          Show this help message

EXAMPLES:
    # Analyze a backtest file (Dave Mabe crunch - default)
    python -m strategy_cruncher.run real5backtest_trades_10000.csv net_pnl

    # Crunch with custom rules
    python -m strategy_cruncher.run backtest.csv net_pnl --max-rules 5 --min-improvement 10

    # Launch the web interface
    python -m strategy_cruncher.run --app

    # Or use streamlit directly
    streamlit run strategy_cruncher/app.py

WHAT IT DOES:
    The Strategy Cruncher analyzes your backtest data to find optimal
    indicator thresholds that filter out bad trades. It:
    
    1. Scans all numeric columns as potential indicators
    2. Tests thousands of threshold values
    3. Finds the optimal cutoff for each indicator
    4. Ranks rules by their "edge score" (improvement vs. trade retention)
    5. Shows you exactly which rules will improve your strategy

EXPECTED CSV FORMAT:
    Your CSV should have:
    - A P&L column (net_pnl, gross_pnl, profit, etc.)
    - Indicator columns (gap_percent, arval, atr, position_in_range, etc.)
    - Trade metadata (ticker, date, entry_time, etc.)

For more info, see: https://github.com/thanh-cyber/strategycruncher
"""
    print(help_text)


if __name__ == '__main__':
    main()
