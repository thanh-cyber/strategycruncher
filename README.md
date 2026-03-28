# FireEye (`strategycruncher`)

**FireEye** is the product name for this repo’s backtest optimization and rule-discovery tool. The Python package remains `strategy_cruncher` for imports.

It scans numeric backtest columns, tests threshold filters, and ranks rules by mean $/trade spread (Dave Mabe–style report metrics).

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
fireeye --app
```

or:

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
