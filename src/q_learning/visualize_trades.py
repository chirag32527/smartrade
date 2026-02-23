"""
Visualization Script for Trade Analysis

This script analyzes and visualizes trade logs from the NIFTY options trading agent.

Usage:
    python visualize_trades.py trade_logs/trades_50000.csv
    python visualize_trades.py trade_logs/  # Process all CSV files in directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path


def load_trade_data(filepath):
    """Load trade data from CSV."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Filter only completed trades (those with P&L)
    df = df[df['pnl'].notna()].copy()

    if len(df) == 0:
        print(f"Warning: No completed trades found in {filepath}")
        return None

    return df


def print_summary_statistics(df):
    """Print detailed summary statistics."""
    print("\n" + "="*70)
    print("TRADE ANALYSIS SUMMARY")
    print("="*70)

    # Overall statistics
    print(f"\nTotal Completed Trades: {len(df)}")
    print(f"Episodes Covered: {df['episode'].min()} to {df['episode'].max()}")

    # P&L statistics
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]

    print(f"\n📊 P&L Statistics:")
    print(f"  Total P&L: ₹{total_pnl:,.2f}")
    print(f"  Avg P&L per Trade: ₹{avg_pnl:,.2f}")
    print(f"  Winning Trades: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    print(f"  Losing Trades: {len(losses)} ({len(losses)/len(df)*100:.1f}%)")

    if len(wins) > 0:
        print(f"  Avg Win: ₹{wins['pnl'].mean():,.2f}")
        print(f"  Max Win: ₹{wins['pnl'].max():,.2f}")

    if len(losses) > 0:
        print(f"  Avg Loss: ₹{losses['pnl'].mean():,.2f}")
        print(f"  Max Loss: ₹{losses['pnl'].min():,.2f}")

    if len(losses) > 0 and losses['pnl'].sum() != 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"  Profit Factor: {profit_factor:.2f}")

    # Strategy breakdown
    print(f"\n🎯 Strategy Breakdown:")

    # By direction
    long_trades = df[df['direction'] == 'long']
    short_trades = df[df['direction'] == 'short']

    print(f"\n  LONG Positions:")
    print(f"    Count: {len(long_trades)} ({len(long_trades)/len(df)*100:.1f}%)")
    if len(long_trades) > 0:
        long_wins = long_trades[long_trades['pnl'] > 0]
        print(f"    Win Rate: {len(long_wins)/len(long_trades)*100:.1f}%")
        print(f"    Total P&L: ₹{long_trades['pnl'].sum():,.2f}")
        print(f"    Avg P&L: ₹{long_trades['pnl'].mean():,.2f}")

    print(f"\n  SHORT Positions:")
    print(f"    Count: {len(short_trades)} ({len(short_trades)/len(df)*100:.1f}%)")
    if len(short_trades) > 0:
        short_wins = short_trades[short_trades['pnl'] > 0]
        print(f"    Win Rate: {len(short_wins)/len(short_trades)*100:.1f}%")
        print(f"    Total P&L: ₹{short_trades['pnl'].sum():,.2f}")
        print(f"    Avg P&L: ₹{short_trades['pnl'].mean():,.2f}")

    # By option type
    print(f"\n  By Option Type:")
    for opt_type in ['CE', 'PE']:
        opt_trades = df[df['option_type'] == opt_type]
        if len(opt_trades) > 0:
            opt_wins = opt_trades[opt_trades['pnl'] > 0]
            print(f"    {opt_type}: {len(opt_trades)} trades, "
                  f"{len(opt_wins)/len(opt_trades)*100:.1f}% win rate, "
                  f"₹{opt_trades['pnl'].sum():,.2f} total")

    # By strike offset
    print(f"\n  By Strike Offset:")
    for offset in sorted(df['strike_offset'].unique()):
        offset_trades = df[df['strike_offset'] == offset]
        offset_wins = offset_trades[offset_trades['pnl'] > 0]
        strike_label = f"ATM{offset:+d}" if offset != 0 else "ATM"
        print(f"    {strike_label}: {len(offset_trades)} trades, "
              f"{len(offset_wins)/len(offset_trades)*100:.1f}% win rate")

    # Holding period
    print(f"\n⏱️  Holding Period:")
    print(f"  Average: {df['holding_period'].mean():.1f} days")
    print(f"  Median: {df['holding_period'].median():.1f} days")
    print(f"  Min: {df['holding_period'].min():.0f} days")
    print(f"  Max: {df['holding_period'].max():.0f} days")

    # VIX conditions
    print(f"\n📈 VIX Conditions:")
    print(f"  Avg VIX at Entry: {df['vix'].mean():.2f}")
    print(f"  VIX Range: {df['vix'].min():.2f} - {df['vix'].max():.2f}")

    high_vix = df[df['vix'] > df['vix'].median()]
    low_vix = df[df['vix'] <= df['vix'].median()]

    if len(high_vix) > 0 and len(low_vix) > 0:
        print(f"  High VIX trades: {len(high_vix)}, Avg P&L: ₹{high_vix['pnl'].mean():,.2f}")
        print(f"  Low VIX trades: {len(low_vix)}, Avg P&L: ₹{low_vix['pnl'].mean():,.2f}")

    print("="*70 + "\n")


def create_visualizations(df, save_path=None):
    """Create comprehensive visualizations."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('NIFTY Options Trading Agent - Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Cumulative P&L over time
    ax = axes[0, 0]
    cumulative_pnl = df.sort_values('episode')['pnl'].cumsum()
    ax.plot(cumulative_pnl.values, linewidth=2, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Cumulative P&L Over Time', fontweight='bold')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative P&L (₹)')
    ax.grid(True, alpha=0.3)

    # 2. P&L distribution
    ax = axes[0, 1]
    ax.hist(df['pnl'], bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=df['pnl'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: ₹{df["pnl"].mean():.0f}')
    ax.set_title('P&L Distribution', fontweight='bold')
    ax.set_xlabel('P&L (₹)')
    ax.set_ylabel('Frequency')
    ax.legend()

    # 3. Win rate by strategy
    ax = axes[0, 2]
    strategies = []
    win_rates = []

    for direction in ['long', 'short']:
        for opt_type in ['CE', 'PE']:
            subset = df[(df['direction'] == direction) & (df['option_type'] == opt_type)]
            if len(subset) > 0:
                wins = len(subset[subset['pnl'] > 0])
                win_rate = wins / len(subset) * 100
                strategies.append(f"{direction.upper()} {opt_type}")
                win_rates.append(win_rate)

    colors = ['#06A77D' if wr >= 50 else '#D81159' for wr in win_rates]
    ax.barh(strategies, win_rates, color=colors, edgecolor='black')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% Break-even')
    ax.set_title('Win Rate by Strategy', fontweight='bold')
    ax.set_xlabel('Win Rate (%)')
    ax.legend()

    # 4. P&L by strike offset
    ax = axes[1, 0]
    strike_pnl = df.groupby('strike_offset')['pnl'].sum().sort_index()
    colors_strike = ['#06A77D' if v >= 0 else '#D81159' for v in strike_pnl.values]
    strike_labels = [f"ATM{o:+d}" if o != 0 else "ATM" for o in strike_pnl.index]
    ax.bar(strike_labels, strike_pnl.values, color=colors_strike, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title('Total P&L by Strike Offset', fontweight='bold')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Total P&L (₹)')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Holding period vs P&L
    ax = axes[1, 1]
    scatter = ax.scatter(df['holding_period'], df['pnl'], c=df['pnl'],
                        cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Holding Period vs P&L', fontweight='bold')
    ax.set_xlabel('Holding Period (days)')
    ax.set_ylabel('P&L (₹)')
    plt.colorbar(scatter, ax=ax, label='P&L (₹)')

    # 6. Trade count by direction and type
    ax = axes[1, 2]
    trade_counts = df.groupby(['direction', 'option_type']).size().unstack(fill_value=0)
    trade_counts.plot(kind='bar', ax=ax, color=['#FF6B35', '#004E89'], edgecolor='black', width=0.7)
    ax.set_title('Trade Count by Direction & Type', fontweight='bold')
    ax.set_xlabel('Direction')
    ax.set_ylabel('Count')
    ax.legend(title='Option Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # 7. VIX at entry vs P&L
    ax = axes[2, 0]
    ax.scatter(df['vix'], df['pnl'], c=df['pnl'], cmap='RdYlGn',
              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('VIX at Entry vs P&L', fontweight='bold')
    ax.set_xlabel('VIX at Entry')
    ax.set_ylabel('P&L (₹)')

    # 8. Long vs Short performance
    ax = axes[2, 1]
    direction_pnl = df.groupby('direction').agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()

    x = np.arange(len(direction_pnl))
    width = 0.25

    total_pnl = [direction_pnl.iloc[i][('pnl', 'sum')] for i in range(len(direction_pnl))]
    avg_pnl = [direction_pnl.iloc[i][('pnl', 'mean')] for i in range(len(direction_pnl))]

    colors_dir = ['#06A77D' if v >= 0 else '#D81159' for v in total_pnl]

    ax.bar(x, total_pnl, width, label='Total P&L', color=colors_dir, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title('Long vs Short Performance', fontweight='bold')
    ax.set_ylabel('Total P&L (₹)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in direction_pnl['direction']])
    ax.legend()

    # 9. Episode progression - avg P&L per episode
    ax = axes[2, 2]
    episode_pnl = df.groupby('episode')['pnl'].mean()
    ax.plot(episode_pnl.index, episode_pnl.values, marker='o', linewidth=1, markersize=3, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Add trend line
    z = np.polyfit(episode_pnl.index, episode_pnl.values, 1)
    p = np.poly1d(z)
    ax.plot(episode_pnl.index, p(episode_pnl.index), "r--", alpha=0.5, linewidth=2, label='Trend')

    ax.set_title('Learning Progress (Avg P&L per Episode)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg P&L (₹)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    else:
        plt.show()


def analyze_trade_file(filepath, save_plots=True):
    """Analyze a single trade file."""
    print(f"\nAnalyzing: {filepath}")

    df = load_trade_data(filepath)
    if df is None or len(df) == 0:
        return

    # Print statistics
    print_summary_statistics(df)

    # Create visualizations
    if save_plots:
        plot_path = filepath.replace('.csv', '_analysis.png')
        create_visualizations(df, save_path=plot_path)
    else:
        create_visualizations(df)


def main():
    parser = argparse.ArgumentParser(description="Analyze NIFTY Options Trading Logs")
    parser.add_argument('path', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--no-plots', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        # Single file
        analyze_trade_file(str(path), save_plots=not args.no_plots and not args.show)
    elif path.is_dir():
        # Directory - process all CSV files
        csv_files = sorted(path.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {path}")
            return

        print(f"Found {len(csv_files)} CSV files")

        # Process the latest file
        latest_file = csv_files[-1]
        analyze_trade_file(str(latest_file), save_plots=not args.no_plots and not args.show)

        # Optionally combine all files
        if len(csv_files) > 1:
            print(f"\n{'='*70}")
            print("COMBINED ANALYSIS (All Files)")
            print(f"{'='*70}")

            all_dfs = []
            for csv_file in csv_files:
                df = load_trade_data(str(csv_file))
                if df is not None:
                    all_dfs.append(df)

            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                print_summary_statistics(combined_df)

                if not args.no_plots:
                    combined_plot_path = path / 'combined_analysis.png'
                    create_visualizations(combined_df, save_path=str(combined_plot_path))
    else:
        print(f"Error: {path} is not a valid file or directory")


if __name__ == '__main__':
    main()
