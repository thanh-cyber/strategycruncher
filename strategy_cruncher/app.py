"""
Strategy Cruncher - Interactive Web Application

A powerful tool for analyzing backtest data and finding optimal indicator thresholds.
Inspired by Dave Mabe's systematic trading approach.

Run with: streamlit run strategy_cruncher/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import io
import sys
import os

# Handle both direct execution and package import
try:
    from .cruncher import StrategyCruncher, OptimizationResult, RuleCandidate
except ImportError:
    # When run directly with streamlit
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from strategy_cruncher.cruncher import StrategyCruncher, OptimizationResult, RuleCandidate

# Page configuration
st.set_page_config(
    page_title="Strategy Cruncher",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a distinctive, professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-amber: #f59e0b;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --border-color: #374151;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%);
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a2332 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    
    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }
    
    .rule-card {
        background: linear-gradient(145deg, #1e293b 0%, #1a2332 100%);
        border: 1px solid #374151;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }
    
    .rule-card:hover {
        border-left-color: #10b981;
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .rule-rank {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        color: #8b5cf6;
        font-weight: 600;
    }
    
    .rule-expression {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #f3f4f6;
        margin: 0.5rem 0;
    }
    
    .rule-stats {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    
    .rule-stat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    
    .rule-stat-label { color: #6b7280; }
    .rule-stat-value { color: #d1d5db; font-weight: 500; }
    
    .edge-score {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f59e0b;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #f3f4f6;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .improvement-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .improvement-badge.positive {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .improvement-badge.negative {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0a0e17 100%);
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    .info-box p {
        color: #d1d5db;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    import math
    if math.isnan(value) or math.isinf(value):
        return "N/A" if math.isnan(value) else ("$+∞" if value > 0 else "$-∞")
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, include_sign: bool = False) -> str:
    """Format a number as percentage."""
    if include_sign:
        return f"{value:+.1%}"
    return f"{value:.1%}"


def render_metric_card(label: str, value: str, change: Optional[str] = None, positive: bool = True):
    """Render a styled metric card."""
    change_html = ""
    if change:
        badge_class = "positive" if positive else "negative"
        change_html = f'<span class="improvement-badge {badge_class}">{change}</span>'
    
    value_class = ""
    if change:
        value_class = "positive" if positive else "negative"
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{value}</div>
            {change_html}
        </div>
    """, unsafe_allow_html=True)


def render_rule_card(rank: int, rule: RuleCandidate):
    """Render a styled rule card."""
    dir_symbol = ">" if rule.direction == "above" else "<"
    pnl_class = "positive" if rule.pnl_improvement >= 0 else "negative"
    wr_class = "positive" if rule.win_rate_improvement >= 0 else "negative"
    
    st.markdown(f"""
        <div class="rule-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="rule-rank">#{rank}</span>
                    <div class="rule-expression">{rule.column} {dir_symbol} {rule.threshold:.4f}</div>
                </div>
                <div class="edge-score">{rule.edge_score:.2f}</div>
            </div>
            <div class="rule-stats">
                <div class="rule-stat">
                    <span class="rule-stat-label">P&L:</span>
                    <span class="rule-stat-value {pnl_class}">{format_currency(rule.total_pnl)} ({rule.pnl_improvement_pct:+.1f}%)</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Win Rate:</span>
                    <span class="rule-stat-value {wr_class}">{rule.win_rate:.1%} ({rule.win_rate_improvement:+.1%})</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Trades:</span>
                    <span class="rule-stat-value">{rule.trades_remaining:,}</span>
                </div>
                <div class="rule-stat">
                    <span class="rule-stat-label">Profit Factor:</span>
                    <span class="rule-stat-value">{rule.profit_factor:.2f}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_equity_curve(df: pd.DataFrame, pnl_column: str, title: str = "Equity Curve") -> go.Figure:
    """Create an interactive equity curve plot."""
    cumulative_pnl = df[pnl_column].cumsum()
    
    fig = go.Figure()
    
    # Main equity curve
    fig.add_trace(go.Scatter(
        x=list(range(len(cumulative_pnl))),
        y=cumulative_pnl,
        mode='lines',
        name='Equity',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    # Running max (for drawdown visualization)
    running_max = cumulative_pnl.cummax()
    fig.add_trace(go.Scatter(
        x=list(range(len(running_max))),
        y=running_max,
        mode='lines',
        name='High Water Mark',
        line=dict(color='#3b82f6', width=1, dash='dot'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#f3f4f6')),
        xaxis_title="Trade #",
        yaxis_title="Cumulative P&L ($)",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        font=dict(family="JetBrains Mono", color='#9ca3af'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    fig.update_xaxes(gridcolor='#374151', gridwidth=0.5)
    fig.update_yaxes(gridcolor='#374151', gridwidth=0.5, tickformat='$,.0f')
    
    return fig


def create_distribution_plot(df: pd.DataFrame, pnl_column: str) -> go.Figure:
    """Create P&L distribution histogram."""
    fig = go.Figure()
    
    pnl_values = df[pnl_column]
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=pnl_values,
        nbinsx=50,
        name='P&L Distribution',
        marker_color='#3b82f6',
        opacity=0.8
    ))
    
    # Add mean line
    mean_pnl = pnl_values.mean()
    fig.add_vline(x=mean_pnl, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Mean: {format_currency(mean_pnl)}")
    
    # Add zero line
    fig.add_vline(x=0, line_dash="solid", line_color="#ef4444", line_width=2)
    
    fig.update_layout(
        title=dict(text="P&L Distribution", font=dict(size=18, color='#f3f4f6')),
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        font=dict(family="JetBrains Mono", color='#9ca3af'),
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    fig.update_xaxes(gridcolor='#374151', gridwidth=0.5, tickformat='$,.0f')
    fig.update_yaxes(gridcolor='#374151', gridwidth=0.5)
    
    return fig


def create_indicator_heatmap(rules: List[RuleCandidate], top_n: int = 15) -> go.Figure:
    """Create a heatmap showing indicator effectiveness."""
    top_rules = rules[:top_n]
    
    indicators = [f"{r.column} {'>' if r.direction == 'above' else '<'} {r.threshold:.2f}" 
                  for r in top_rules]
    
    metrics = ['PnL Improvement %', 'Win Rate Change', 'Edge Score', 'Trades Kept %']
    
    # Build the data matrix
    data = []
    for rule in top_rules:
        total = rule.trades_remaining + rule.trades_filtered
        pct_kept = (rule.trades_remaining / total * 100) if total > 0 else 0
        row = [
            rule.pnl_improvement_pct,
            rule.win_rate_improvement * 100,
            rule.edge_score * 100,  # Scale for visibility
            pct_kept
        ]
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=metrics,
        y=indicators,
        colorscale=[
            [0, '#ef4444'],
            [0.5, '#1f2937'],
            [1, '#10b981']
        ],
        text=[[f"{val:.1f}" for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text="Rule Effectiveness Matrix", font=dict(size=18, color='#f3f4f6')),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        font=dict(family="JetBrains Mono", color='#9ca3af'),
        height=max(400, len(indicators) * 35),
        margin=dict(l=250, r=20, t=60, b=40)
    )
    
    return fig


def create_threshold_analysis_plot(
    df: pd.DataFrame, 
    column: str, 
    pnl_column: str,
    direction: str = 'above'
) -> go.Figure:
    """Create a detailed threshold analysis plot for a specific indicator."""
    values = df[column].dropna()
    pnl_values = df.loc[values.index, pnl_column].values
    col_values = values.values
    
    # Generate thresholds
    percentiles = np.linspace(5, 95, 50)
    thresholds = np.percentile(col_values, percentiles)
    
    results = []
    for thresh in thresholds:
        if direction == 'above':
            mask = col_values > thresh
        else:
            mask = col_values < thresh
        
        filtered_pnl = pnl_values[mask]
        if len(filtered_pnl) > 10:
            results.append({
                'threshold': thresh,
                'total_pnl': filtered_pnl.sum(),
                'win_rate': (filtered_pnl > 0).mean(),
                'n_trades': len(filtered_pnl),
                'avg_pnl': filtered_pnl.mean()
            })
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total P&L vs Threshold', 'Win Rate vs Threshold',
                       'Trade Count vs Threshold', 'Avg P&L per Trade vs Threshold'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Total PnL
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['total_pnl'],
        mode='lines+markers', name='Total P&L',
        line=dict(color='#10b981', width=2),
        marker=dict(size=4)
    ), row=1, col=1)
    
    # Win Rate
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['win_rate'],
        mode='lines+markers', name='Win Rate',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4)
    ), row=1, col=2)
    
    # Trade Count
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['n_trades'],
        mode='lines+markers', name='Trades',
        line=dict(color='#f59e0b', width=2),
        marker=dict(size=4)
    ), row=2, col=1)
    
    # Avg PnL
    fig.add_trace(go.Scatter(
        x=results_df['threshold'], y=results_df['avg_pnl'],
        mode='lines+markers', name='Avg P&L',
        line=dict(color='#8b5cf6', width=2),
        marker=dict(size=4)
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"Threshold Analysis: {column} ({direction})",
            font=dict(size=18, color='#f3f4f6')
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17, 24, 39, 0.8)',
        font=dict(family="JetBrains Mono", color='#9ca3af'),
        showlegend=False,
        height=500,
        margin=dict(l=60, r=20, t=80, b=40)
    )
    
    # Update all axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='#374151', gridwidth=0.5, row=i, col=j)
            fig.update_yaxes(gridcolor='#374151', gridwidth=0.5, row=i, col=j)
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Strategy Cruncher</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize your backtest • Find the edge • Skip bad trades</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Input")
        
        uploaded_file = st.file_uploader(
            "Upload Backtest CSV",
            type=['csv'],
            help="Upload your backtest results with indicator columns"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        pnl_column = st.text_input(
            "P&L Column Name",
            value="net_pnl",
            help="Name of the column containing profit/loss values"
        )
        
        crunch_mode = st.checkbox(
            "Dave Mabe Crunch Mode (iterative, one rule at a time)",
            value=True,
            help="Apply rules iteratively: find best rule → apply → re-crunch until no more good rules"
        )
        
        min_trades = st.slider(
            "Min Trades After Filter",
            min_value=10,
            max_value=500,
            value=300 if crunch_mode else 50,
            help="Minimum trades required after applying a rule"
        )
        
        min_improvement = st.slider(
            "Min P&L Improvement %",
            min_value=0.0,
            max_value=50.0,
            value=8.0 if crunch_mode else 5.0,
            step=1.0,
            help="Minimum P&L improvement to consider a rule"
        )
        
        max_rules = st.slider(
            "Max Rules (Crunch)",
            min_value=1,
            max_value=15,
            value=8,
            help="Maximum rules to apply in Dave Mabe crunch mode"
        ) if crunch_mode else 8
        
        n_thresholds = st.slider(
            "Threshold Granularity",
            min_value=20,
            max_value=200,
            value=100,
            help="Number of threshold values to test per indicator"
        )
        
        analyze_library = st.checkbox(
            "Analyze Column Library",
            value=False,
            help="Analyze column_library.xlsx to recommend additional columns to add"
        )
        
        library_path = st.text_input(
            "Column Library Path",
            value="column_library.xlsx",
            help="Path to your column library Excel file",
            disabled=not analyze_library
        )
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        <div class="info-box">
            <p>Strategy Cruncher finds optimal indicator thresholds to filter bad trades from your backtest.</p>
            <br>
            <p><strong>Process:</strong></p>
            <p>1. Cast a wide net with initial backtest</p>
            <p>2. Add indicator columns</p>
            <p>3. Let the cruncher find optimal cutoffs</p>
            <p>4. Apply rules that make sense to you</p>
            <br>
            <p><strong>Column Library:</strong> Enable to analyze your column library Excel file and get recommendations for additional columns to add.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            
            if pnl_column not in df.columns:
                st.error(f"❌ Column '{pnl_column}' not found in data. Available columns: {list(df.columns)}")
                return
            
            # Initialize cruncher
            cruncher = StrategyCruncher(
                min_trades_remaining=min_trades,
                min_improvement_pct=min_improvement,
                n_threshold_bins=n_thresholds
            )
            
            
            # Run analysis
            with st.spinner("Crunching..." if crunch_mode else "Analyzing backtest data..."):
                if crunch_mode:
                    crunch_rules, filtered_df, before_curve, after_curve = cruncher.crunch(
                        df, pnl_column=pnl_column,
                        target_metric="profit_factor",
                        min_trades=min_trades,
                        min_improvement_pct=min_improvement,
                        max_rules=max_rules,
                        verbose=False
                    )
                    baseline_metrics = cruncher._calculate_metrics(df, pnl_column)
                    final_metrics = cruncher._calculate_metrics(filtered_df, pnl_column) if len(filtered_df) > 0 else baseline_metrics
                    results = None
                else:
                    results = cruncher.analyze(
                        df, 
                        pnl_column=pnl_column,
                        analyze_column_library=analyze_library,
                        library_path=library_path if analyze_library else 'column_library.xlsx'
                    )
                    crunch_rules = None
                    filtered_df = None
                    baseline_metrics = None
                    final_metrics = None
            
            # Store in session state
            st.session_state['results'] = results
            st.session_state['df'] = df
            st.session_state['pnl_column'] = pnl_column
            st.session_state['crunch_mode'] = crunch_mode
            st.session_state['crunch_rules'] = crunch_rules
            st.session_state['filtered_df'] = filtered_df
            st.session_state['baseline_metrics'] = baseline_metrics
            st.session_state['final_metrics'] = final_metrics
            
            if crunch_mode:
                # Dave Mabe Crunch: Tabbed layout
                baseline = baseline_metrics
                tab1, tab2, tab3, tab4 = st.tabs(["Iterative Crunch", "Rules Table", "Equity Curves", "Raw Results"])
                
                with tab1:
                    st.markdown('<div class="section-title">Dave Mabe Crunch - Before vs After</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_metric_card("Trades", f"{baseline['n_trades']:,} → {final_metrics['n_trades']:,}")
                        render_metric_card("Profit Factor", f"{baseline['profit_factor']:.2f} → {final_metrics['profit_factor']:.2f}")
                    with col2:
                        render_metric_card("Total P&L", format_currency(baseline['total_pnl']) + " → " + format_currency(final_metrics['total_pnl']), positive=final_metrics['total_pnl'] >= 0)
                        render_metric_card("Sharpe", f"{baseline['sharpe_ratio']:.2f} → {final_metrics['sharpe_ratio']:.2f}")
                    with col3:
                        render_metric_card("Win Rate", f"{baseline['win_rate']:.1%} → {final_metrics['win_rate']:.1%}")
                        render_metric_card("Max DD", format_currency(baseline['max_drawdown']) + " → " + format_currency(final_metrics['max_drawdown']))
                    with col4:
                        render_metric_card("Rules", f"{len(crunch_rules)}")
                    st.markdown("#### P&L Distribution (Before / After)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(create_distribution_plot(df, pnl_column), use_container_width=True)
                    with c2:
                        st.plotly_chart(create_distribution_plot(filtered_df, pnl_column) if len(filtered_df) > 0 else create_distribution_plot(df, pnl_column), use_container_width=True)
                
                with tab2:
                    st.markdown('<div class="section-title">Rules Applied (Iterative)</div>', unsafe_allow_html=True)
                    if crunch_rules:
                        rules_table = pd.DataFrame(crunch_rules)
                        display_cols = ['rule_num', 'column', 'direction', 'threshold', 'new_metric', 'improvement_pct', 'trades_remaining']
                        st.dataframe(rules_table[[c for c in display_cols if c in rules_table.columns]], use_container_width=True, hide_index=True)
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            csv_buf = io.StringIO()
                            filtered_df.to_csv(csv_buf, index=False)
                            st.download_button("Download Filtered Trades", data=csv_buf.getvalue(), file_name="crunch_filtered_trades.csv", mime="text/csv")
                        with col_b:
                            rules_csv = io.StringIO()
                            rules_table.to_csv(rules_csv, index=False)
                            st.download_button("Download Rules (CSV)", data=rules_csv.getvalue(), file_name="crunch_rules.csv", mime="text/csv")
                        with col_c:
                            import json
                            def _to_json(v):
                                if isinstance(v, (np.floating, np.integer)):
                                    return float(v)
                                return v
                            rules_json = json.dumps([{k: _to_json(v) for k, v in r.items() if k in display_cols} for r in crunch_rules], indent=2)
                            st.download_button("Download Rules (JSON)", data=rules_json, file_name="crunch_rules.json", mime="application/json")
                    else:
                        st.info("No rules met the criteria.")
                
                with tab3:
                    st.markdown('<div class="section-title">Before vs After Equity Curve</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_equity_curve(df, pnl_column, "Before (Baseline)"), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_equity_curve(filtered_df, pnl_column, "After (Filtered)") if len(filtered_df) > 0 else create_equity_curve(df, pnl_column, "After"), use_container_width=True)
                
                with tab4:
                    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
                    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
            else:
                # Standard analyze mode
                baseline = results.baseline_metrics
                st.markdown('<div class="section-title">📊 Baseline Metrics</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("Total Trades", f"{baseline['n_trades']:,}")
                    render_metric_card("Profit Factor", f"{baseline['profit_factor']:.2f}")
                with col2:
                    pnl_positive = baseline['total_pnl'] >= 0
                    render_metric_card("Total P&L", format_currency(baseline['total_pnl']), positive=pnl_positive)
                    render_metric_card("Sharpe Ratio", f"{baseline['sharpe_ratio']:.2f}")
                with col3:
                    render_metric_card("Win Rate", f"{baseline['win_rate']:.1%}")
                    render_metric_card("Max Drawdown", format_currency(baseline['max_drawdown']))
                with col4:
                    render_metric_card("Avg Win", format_currency(baseline['avg_win']))
                    render_metric_card("Avg Loss", format_currency(baseline['avg_loss']))
                
                st.markdown('<div class="section-title">📈 Equity Curve</div>', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_equity = create_equity_curve(df, pnl_column, "Baseline Equity Curve")
                    st.plotly_chart(fig_equity, use_container_width=True)
                with col2:
                    fig_dist = create_distribution_plot(df, pnl_column)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown('<div class="section-title">🎯 Top Optimization Rules</div>', unsafe_allow_html=True)
            
            if not crunch_mode:
                if not results.rules:
                    st.warning("⚠️ No optimization rules found that meet the criteria. Try lowering the minimum improvement threshold.")
                else:
                    # Rules table
                    top_rules = results.get_top_rules(20)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        for i, rule in enumerate(top_rules[:10], 1):
                            render_rule_card(i, rule)
                    
                    with col2:
                        # Rule effectiveness heatmap
                        fig_heatmap = create_indicator_heatmap(top_rules)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Detailed Rule Analysis
                    st.markdown('<div class="section-title">🔬 Deep Dive Analysis</div>', unsafe_allow_html=True)
                    
                    # Rule selector
                    rule_options = [f"#{i+1}: {r.column} {'>' if r.direction == 'above' else '<'} {r.threshold:.4f}" 
                                  for i, r in enumerate(top_rules)]
                    
                    selected_rule_idx = st.selectbox(
                        "Select a rule to analyze in detail",
                        range(len(rule_options)),
                        format_func=lambda x: rule_options[x]
                    )
                    
                    selected_rule = top_rules[selected_rule_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Threshold analysis plot
                        fig_thresh = create_threshold_analysis_plot(
                            df, selected_rule.column, pnl_column, selected_rule.direction
                        )
                        if fig_thresh:
                            st.plotly_chart(fig_thresh, use_container_width=True)
                    
                    with col2:
                        # Apply rule and show new equity curve
                        if selected_rule.direction == 'above':
                            filtered_df = df[df[selected_rule.column] > selected_rule.threshold]
                        else:
                            filtered_df = df[df[selected_rule.column] < selected_rule.threshold]
                        
                        fig_filtered = create_equity_curve(
                            filtered_df, pnl_column, 
                            f"Equity Curve After Applying Rule"
                        )
                        st.plotly_chart(fig_filtered, use_container_width=True)
                    
                    # Comparison metrics
                    st.markdown("#### 📊 Before vs After Comparison")
                    
                    new_metrics = cruncher._calculate_metrics(filtered_df, pnl_column)
                    
                    comparison_data = {
                        'Metric': ['Total Trades', 'Total P&L', 'Win Rate', 'Avg Win', 'Avg Loss', 
                                  'Profit Factor', 'Sharpe Ratio', 'Expectancy'],
                        'Before': [
                            f"{baseline['n_trades']:,}",
                            format_currency(baseline['total_pnl']),
                            f"{baseline['win_rate']:.1%}",
                            format_currency(baseline['avg_win']),
                            format_currency(baseline['avg_loss']),
                            f"{baseline['profit_factor']:.2f}",
                            f"{baseline['sharpe_ratio']:.2f}",
                            format_currency(baseline['expectancy'])
                        ],
                        'After': [
                            f"{new_metrics['n_trades']:,}",
                            format_currency(new_metrics['total_pnl']),
                            f"{new_metrics['win_rate']:.1%}",
                            format_currency(new_metrics['avg_win']),
                            format_currency(new_metrics['avg_loss']),
                            f"{new_metrics['profit_factor']:.2f}",
                            f"{new_metrics['sharpe_ratio']:.2f}",
                            format_currency(new_metrics['expectancy'])
                        ],
                        'Change': [
                            f"{new_metrics['n_trades'] - baseline['n_trades']:+,}",
                            f"{((new_metrics['total_pnl'] - baseline['total_pnl']) / abs(baseline['total_pnl']) * 100) if baseline['total_pnl'] != 0 else 0:+.1f}%",
                            f"{(new_metrics['win_rate'] - baseline['win_rate']) * 100:+.1f}pp",
                            f"{((new_metrics['avg_win'] - baseline['avg_win']) / baseline['avg_win'] * 100) if baseline['avg_win'] != 0 else 0:+.1f}%",
                            f"{((new_metrics['avg_loss'] - baseline['avg_loss']) / baseline['avg_loss'] * 100) if baseline['avg_loss'] != 0 else 0:+.1f}%",
                            f"{new_metrics['profit_factor'] - baseline['profit_factor']:+.2f}",
                            f"{new_metrics['sharpe_ratio'] - baseline['sharpe_ratio']:+.2f}",
                            f"{((new_metrics['expectancy'] - baseline['expectancy']) / abs(baseline['expectancy']) * 100) if baseline['expectancy'] != 0 else 0:+.1f}%"
                        ]
                    }
                    
                    st.dataframe(
                        pd.DataFrame(comparison_data),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download filtered data
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        # Download filtered trades
                        csv_buffer = io.StringIO()
                        filtered_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Filtered Trades",
                            data=csv_buffer.getvalue(),
                            file_name="filtered_trades.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Download all rules
                        rules_data = []
                        for r in top_rules:
                            rules_data.append({
                                'Rank': top_rules.index(r) + 1,
                                'Column': r.column,
                                'Direction': r.direction,
                                'Threshold': r.threshold,
                                'Edge Score': r.edge_score,
                                'Total PnL': r.total_pnl,
                                'PnL Improvement %': r.pnl_improvement_pct,
                                'Win Rate': r.win_rate,
                                'Win Rate Change': r.win_rate_improvement,
                                'Trades Remaining': r.trades_remaining,
                                'Trades Filtered': r.trades_filtered,
                                'Profit Factor': r.profit_factor,
                                'Expectancy': r.expectancy
                            })
                        
                        rules_df = pd.DataFrame(rules_data)
                        rules_csv = io.StringIO()
                        rules_df.to_csv(rules_csv, index=False)
                        
                        st.download_button(
                            label="📥 Download All Rules",
                            data=rules_csv.getvalue(),
                            file_name="optimization_rules.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # Generate rule code
                        rule_code = f"""# Apply this rule to your backtest
# Rule: {selected_rule.column} {'>' if selected_rule.direction == 'above' else '<'} {selected_rule.threshold}

df_filtered = df[df['{selected_rule.column}'] {'>' if selected_rule.direction == 'above' else '<'} {selected_rule.threshold}]
"""
                        st.code(rule_code, language='python')
            
            # Column Library Recommendations Section (standard mode only)
            if not crunch_mode and results and results.column_recommendations:
                st.markdown('<div class="section-title">📚 Column Library Recommendations</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>These columns from your library would be most valuable to add to your backtest.</strong></p>
                    <p>The predictive score indicates how much each column would improve your strategy if added.</p>
                </div>
                """, unsafe_allow_html=True)
                
                top_recommendations = results.get_top_column_recommendations(20)
                
                # Group by category
                categories = {}
                for rec in top_recommendations:
                    if rec.category not in categories:
                        categories[rec.category] = []
                    categories[rec.category].append(rec)
                
                for category, recs in categories.items():
                    st.markdown(f"#### {category}")
                    
                    rec_data = []
                    for rec in recs:
                        rec_data.append({
                            'Column Name': rec.column_name,
                            'Description': rec.description,
                            'Predictive Score': f"{rec.predictive_score:.3f}",
                            'Status': '✓ Already Exists' if rec.calculation_method == 'Already exists' 
                                     else ('✓ Can Calculate' if rec.can_calculate else '✗ Needs Data'),
                            'Method': rec.calculation_method or 'N/A'
                        })
                    
                    st.dataframe(
                        pd.DataFrame(rec_data),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Download recommendations
                rec_csv_buffer = io.StringIO()
                rec_df = pd.DataFrame([{
                    'Rank': i+1,
                    'Column Name': r.column_name,
                    'Category': r.category,
                    'Description': r.description,
                    'Predictive Score': r.predictive_score,
                    'Can Calculate': r.can_calculate,
                    'Calculation Method': r.calculation_method or 'N/A'
                } for i, r in enumerate(top_recommendations)])
                rec_df.to_csv(rec_csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Column Recommendations",
                    data=rec_csv_buffer.getvalue(),
                    file_name="column_recommendations.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # No file uploaded - show instructions
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f3f4f6; margin-bottom: 1rem;">👈 Upload a backtest CSV to get started</h3>
            <p style="color: #9ca3af; max-width: 600px; margin: 0 auto;">
                Your CSV should contain trade data with a P&L column and indicator columns.
                The Strategy Cruncher will analyze each indicator to find optimal thresholds
                that maximize your trading performance.
            </p>
            <br>
            <p style="color: #6b7280;">
                <strong>Expected columns:</strong> ticker, date, net_pnl (or your P&L column),
                and any numeric indicator columns (gap_percent, arval, position_in_range, atr, etc.)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data preview
        st.markdown('<div class="section-title">📋 Example Data Format</div>', unsafe_allow_html=True)
        
        example_df = pd.DataFrame({
            'ticker': ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT'],
            'date': ['2025-01-15', '2025-01-15', '2025-01-16', '2025-01-16', '2025-01-17'],
            'net_pnl': [150.50, -75.25, 225.00, -50.00, 180.75],
            'gap_percent': [3.2, 5.8, 2.1, 7.5, 4.3],
            'arval': [2.5, 4.2, 1.8, 6.1, 3.0],
            'position_in_range': [0.75, 0.25, 0.90, 0.15, 0.65],
            'atr': [2.50, 8.75, 3.20, 4.50, 2.80]
        })
        
        st.dataframe(example_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
