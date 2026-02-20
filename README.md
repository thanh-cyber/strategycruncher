# strategycruncher

Backtest optimization and rule discovery library for systematic trading workflows.

It scans numeric backtest columns, tests threshold filters, and ranks rules by edge score.

## Install

### From GitHub

```bash
pip install "git+https://github.com/thanh-cyber/strategycruncher.git"
```

### Editable (local development)

```bash
git clone https://github.com/thanh-cyber/strategycruncher.git
cd strategycruncher
pip install -e .
```

## Quick Usage

```python
from strategy_cruncher import StrategyCruncher

cruncher = StrategyCruncher(
    min_trades_remaining=50,
    min_improvement_pct=5.0,
    n_threshold_bins=100,
)

results = cruncher.analyze("backtest.csv", pnl_column="net_pnl")

for rule in results.get_top_rules(10):
    print(rule)
```

## Launch Web App

```bash
strategy-cruncher --app
```

or:

```bash
streamlit run strategy_cruncher/app.py
```

## Public API

- `StrategyCruncher`
- `OptimizationResult`
- `RuleCandidate`
- `enrich_backtest`
- `add_market_context`
- `ColumnLibraryAnalyzer`
- `ColumnRecommendation`
- `analyze_column_library`
