"""
Hong Kong Merger Arbitrage Analysis
====================================
Analyzes M&A transactions in Hong Kong to evaluate merger arbitrage profitability.

Usage: python merger_arbitrage_analysis.py

Author: Siqi Wang
Date: November 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

print("="*80)
print("HONG KONG MERGER ARBITRAGE ANALYSIS")
print("Event-Driven Strategy: Profiting from M&A Deal Spreads")
print("="*80)

# ============================================
# PART 1: DEFINE M&A DEALS
# ============================================

# Completed deals 
completed_deals = {
    '0973.HK': {
        'announce_date': '2024-04-29',
        'offer_price': 34.00,
        'completion_date': '2024-08-06',
        'name': "L'Occitane International",
        'status': 'Completed'
    },
    '0273.HK': {
        'announce_date': '2023-06-11',
        'offer_price': 0.0338,
        'completion_date': '2023-11-13',
        'name': 'Mason Group Holdings',
        'status': 'Completed'
    },
    '1839.HK': {
        'announce_date': '2024-03-11',
        'offer_price': 7.50,
        'completion_date': '2024-06-03',
        'name': 'CIMC Vehicles',
        'status': 'Completed'
    },
    '0638.HK': {
        'announce_date': '2024-06-28',
        'offer_price': 0.72,
        'completion_date': '2024-08-23',
        'name': 'Kin Yat Holdings',
        'status': 'Completed'
    },
    '3948.HK': {
        'announce_date': '2023-06-13',
        'offer_price': 17.50,
        'completion_date': '2023-08-11',
        'name': 'Inner Mongolia Yitai Coal',
        'status': 'Completed'
    },
    '3331.HK': {
        'announce_date': '2024-03-08',
        'offer_price': 23.50,
        'completion_date': '2024-08-16',
        'name': 'Vinda International',
        'status': 'Completed'
    },
    '3799.HK': {
        'announce_date': '2023-06-27',
        'offer_price': 3.75,
        'completion_date': '2023-08-30',
        'name': 'Dali Foods Group',
        'status': 'Completed'
    },
    '1230.HK': {
        'announce_date': '2022-05-06',
        'offer_price': 1.20,
        'completion_date': '2023-07-05',
        'name': 'Yashili International',
        'status': 'Completed'
    },
}

# Withdrawn deals - announcement effect
withdrawn_deals = {
    '1970.HK': {
        'announce_date': '2023-07-12',
        'offer_price': 10.00,
        'completion_date': '2023-10-10',  # Withdrawal date
        'name': 'IMAX China',
        'status': 'Withdrawn'
    },
    '0570.HK': {
        'announce_date': '2024-02-21',
        'offer_price': 4.60,
        'completion_date': '2024-06-30',  # Estimated withdrawal date
        'name': 'China TCM Holdings',
        'status': 'Withdrawn'
    },
}

# Combine all deals
all_deals = {**completed_deals, **withdrawn_deals}

print(f"Total deals: {len(all_deals)}")
print(f"  - Completed: {len(completed_deals)}")
print(f"  - Withdrawn: {len(withdrawn_deals)}")

# ============================================
# PART 2: ANALYSIS FUNCTIONS
# ============================================

def analyze_merger_arbitrage(ticker, deal_info):
    """
    Analyze a single M&A deal
    Returns metrics and data for visualization
    """
    try:
        # Date setup
        announce_date = pd.to_datetime(deal_info['announce_date'])
        completion_date = pd.to_datetime(deal_info['completion_date'])
        start_date = announce_date - timedelta(days=60)
        end_date = completion_date + timedelta(days=30)
        
        # Download stock data
        print(f"  Downloading {deal_info['name']} ({ticker})...")
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            print(f"  ⚠️  No data available for {ticker}")
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # Download HSI for benchmark
        hsi_data = yf.download('^HSI', start=start_date, end=end_date, progress=False)
        if isinstance(hsi_data.columns, pd.MultiIndex):
            hsi_data.columns = hsi_data.columns.get_level_values(0)
        
        # Calculate returns
        stock_data['Returns'] = stock_data['Close'].pct_change()
        hsi_data['HSI_Returns'] = hsi_data['Close'].pct_change()
        
        # Merge
        data = stock_data.copy()
        data['HSI_Returns'] = hsi_data['HSI_Returns']
        
        # Abnormal returns
        data['Abnormal_Return'] = data['Returns'] - data['HSI_Returns']
        data['CAR'] = data['Abnormal_Return'].cumsum()
        
        # Deal spread
        offer_price = deal_info['offer_price']
        data['Deal_Spread'] = (offer_price - data['Close']) / data['Close'] * 100
        
        # Event window: [-5, +5] days around announcement
        event_start = announce_date - timedelta(days=7)
        event_end = announce_date + timedelta(days=7)
        event_data = data[event_start:event_end]
        
        # 5-day CAR
        car_5_days = event_data['Abnormal_Return'].sum() * 100 if not event_data.empty else 0
        
        # Arbitrage profit simulation
        days_after_announce = data.index[data.index > announce_date]
        
        if len(days_after_announce) > 0:
            buy_date = days_after_announce[0]
            buy_price = float(data.loc[buy_date, 'Close'])
            
            if deal_info['status'] == 'Completed':
                # Sell at offer price (deal completes)
                arb_profit = (offer_price - buy_price) / buy_price * 100
            else:
                # Withdrawn - use price at withdrawal date
                withdrawal_prices = data.loc[data.index <= completion_date, 'Close']
                if not withdrawal_prices.empty:
                    sell_price = float(withdrawal_prices.iloc[-1])
                    arb_profit = (sell_price - buy_price) / buy_price * 100
                else:
                    arb_profit = None
            
            holding_days = (completion_date - buy_date).days
        else:
            arb_profit = None
            holding_days = None
            buy_price = None
        
        # Get price at announcement
        if announce_date in data.index:
            price_at_announce = float(data.loc[announce_date, 'Close'])
        else:
            nearest_dates = data.index[data.index <= announce_date]
            if len(nearest_dates) > 0:
                nearest_date = nearest_dates[-1]
                price_at_announce = float(data.loc[nearest_date, 'Close'])
            else:
                price_at_announce = None
        
        return {
            'name': deal_info['name'],
            'ticker': ticker,
            'status': deal_info['status'],
            'car_5_days': car_5_days,
            'deal_spread_mean': float(data['Deal_Spread'].mean()),
            'deal_spread_at_announce': float(data['Deal_Spread'].iloc[0]) if not data.empty else None,
            'arb_profit': arb_profit,
            'holding_days': holding_days,
            'buy_price': buy_price,
            'offer_price': offer_price,
            'price_at_announce': price_at_announce,
            'data': data,
            'announce_date': announce_date,
            'completion_date': completion_date
        }
    
    except Exception as e:
        print(f"  ❌ Error analyzing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# PART 3: RUN ANALYSIS ON ALL DEALS
# ============================================

print("\n" + "="*80)
print("RUNNING MERGER ARBITRAGE ANALYSIS")
print("="*80 + "\n")

results = {}
for ticker, deal_info in all_deals.items():
    result = analyze_merger_arbitrage(ticker, deal_info)
    if result is not None:
        results[ticker] = result
        status_emoji = "✓" if result['status'] == 'Completed' else "✗"
        arb_str = f"{result['arb_profit']:.2f}%" if result['arb_profit'] is not None else "N/A"
        print(f"  {status_emoji} {result['name']}: CAR = {result['car_5_days']:.2f}%, Arb Profit = {arb_str} ({result['status']})")

if len(results) == 0:
    print("❌ No deals successfully analyzed. Check your deal dates and ticker symbols.")
    exit()

print(f"\n✓ Successfully analyzed {len(results)} out of {len(all_deals)} deals\n")

# ============================================
# PART 4: PORTFOLIO-LEVEL STATISTICS
# ============================================

print("="*80)
print("PORTFOLIO SUMMARY STATISTICS")
print("="*80 + "\n")

# Separate completed and withdrawn
completed_results = {k: v for k, v in results.items() if v['status'] == 'Completed'}
withdrawn_results = {k: v for k, v in results.items() if v['status'] == 'Withdrawn'}

# Extract metrics - ALL deals
all_car = [r['car_5_days'] for r in results.values() if r['car_5_days'] is not None]

# Extract metrics - COMPLETED only
completed_arb_profit = [r['arb_profit'] for r in completed_results.values() if r['arb_profit'] is not None]
completed_spread = [r['deal_spread_at_announce'] for r in completed_results.values() if r['deal_spread_at_announce'] is not None]
completed_holding = [r['holding_days'] for r in completed_results.values() if r['holding_days'] is not None]

# Withdrawn metrics
withdrawn_arb_profit = [r['arb_profit'] for r in withdrawn_results.values() if r['arb_profit'] is not None]

print(f"="*50)
print(f"ALL DEALS ({len(results)} transactions)")
print(f"="*50)
print(f"\nCumulative Abnormal Returns (5-day window around announcement):")
print(f"  Average CAR: {np.mean(all_car):.2f}%")
print(f"  Median CAR: {np.median(all_car):.2f}%")
print(f"  Std Dev: {np.std(all_car):.2f}%")
print(f"  Min: {np.min(all_car):.2f}%")
print(f"  Max: {np.max(all_car):.2f}%")

print(f"\n" + "="*50)
print(f"COMPLETED DEALS ({len(completed_results)} transactions)")
print(f"="*50)

if len(completed_arb_profit) > 0:
    print(f"\nArbitrage Profits (Buy at T+1, Sell at Offer):")
    print(f"  Average Profit: {np.mean(completed_arb_profit):.2f}%")
    print(f"  Median Profit: {np.median(completed_arb_profit):.2f}%")
    print(f"  Min Profit: {np.min(completed_arb_profit):.2f}%")
    print(f"  Max Profit: {np.max(completed_arb_profit):.2f}%")
    print(f"  Success Rate (>0%): {len([x for x in completed_arb_profit if x > 0]) / len(completed_arb_profit) * 100:.1f}%")

if len(completed_holding) > 0:
    print(f"\nHolding Period:")
    print(f"  Average: {np.mean(completed_holding):.0f} days")
    print(f"  Min: {np.min(completed_holding):.0f} days")
    print(f"  Max: {np.max(completed_holding):.0f} days")

if len(completed_spread) > 0:
    print(f"\nDeal Spreads at Announcement:")
    print(f"  Average Spread: {np.mean(completed_spread):.2f}%")
    print(f"  Median Spread: {np.median(completed_spread):.2f}%")

# Annualized return
if len(completed_holding) > 0 and len(completed_arb_profit) > 0:
    avg_holding = np.mean(completed_holding)
    avg_profit = np.mean(completed_arb_profit)
    if avg_holding > 0:
        annualized_return = (1 + avg_profit/100) ** (365/avg_holding) - 1
        print(f"\nAnnualized Return (approximation): {annualized_return*100:.1f}%")

print(f"\n" + "="*50)
print(f"WITHDRAWN DEALS ({len(withdrawn_results)} transactions)")
print(f"="*50)

if len(withdrawn_arb_profit) > 0:
    print(f"\nArbitrage Results (if held until withdrawal):")
    print(f"  Average Loss: {np.mean(withdrawn_arb_profit):.2f}%")
    for ticker, r in withdrawn_results.items():
        if r['arb_profit'] is not None:
            print(f"    - {r['name']}: {r['arb_profit']:.2f}%")

# Statistical significance of CAR
print(f"\n" + "="*50)
print(f"STATISTICAL TESTS")
print(f"="*50)

from scipy import stats
t_stat, p_value = stats.ttest_1samp(all_car, 0)
print(f"\nStatistical Significance of CAR (H0: CAR = 0):")
print(f"  t-statistic: {t_stat:.2f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✓ Statistically significant at 5% level")
elif p_value < 0.10:
    print(f"  ~ Marginally significant at 10% level")
else:
    print(f"  ✗ Not statistically significant")

# ============================================
# PART 5: SAVE RESULTS TO CSV
# ============================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create results dataframe
results_list = []
for ticker, r in results.items():
    results_list.append({
        'Ticker': ticker,
        'Company': r['name'],
        'Status': r['status'],
        'Announcement_Date': r['announce_date'].strftime('%Y-%m-%d'),
        'Completion_Date': r['completion_date'].strftime('%Y-%m-%d'),
        'Offer_Price': r['offer_price'],
        'Price_at_Announcement': round(r['price_at_announce'], 4) if r['price_at_announce'] else None,
        'CAR_5day_%': round(r['car_5_days'], 2),
        'Deal_Spread_at_Announce_%': round(r['deal_spread_at_announce'], 2) if r['deal_spread_at_announce'] else None,
        'Buy_Price_T+1': round(r['buy_price'], 4) if r['buy_price'] else None,
        'Arbitrage_Profit_%': round(r['arb_profit'], 2) if r['arb_profit'] is not None else None,
        'Holding_Days': r['holding_days']
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('../results/deal_summary.csv', index=False)
print(f"\n✓ Results saved to results/deal_summary.csv")

# Print the table
print("\n" + "="*80)
print("DEAL-BY-DEAL SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# ============================================
# PART 6: VISUALIZATIONS
# ============================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hong Kong Merger Arbitrage Analysis\n(2022-2024 M&A Transactions)', 
             fontsize=16, fontweight='bold')

# Plot 1: Deal Spreads Over Time
ax1 = axes[0, 0]
for ticker, result in results.items():
    if result['data'] is not None and not result['data'].empty:
        days_from_announce = (result['data'].index - result['announce_date']).days
        linestyle = '-' if result['status'] == 'Completed' else '--'
        ax1.plot(days_from_announce, result['data']['Deal_Spread'], 
                label=f"{result['name']} ({result['status'][0]})", 
                alpha=0.7, linewidth=2, linestyle=linestyle)

ax1.axvline(x=0, color='red', linestyle='--', label='Announcement', linewidth=2)
ax1.set_title('Deal Spreads Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Days from Announcement')
ax1.set_ylabel('Deal Spread (%)')
ax1.legend(fontsize=7, loc='best')
ax1.grid(alpha=0.3)
ax1.set_xlim(-60, 150)

# Plot 2: CAR Distribution
ax2 = axes[0, 1]
completed_car = [r['car_5_days'] for r in completed_results.values()]
withdrawn_car = [r['car_5_days'] for r in withdrawn_results.values()]

ax2.hist(completed_car, bins=8, edgecolor='black', alpha=0.7, color='green', label='Completed')
if withdrawn_car:
    ax2.hist(withdrawn_car, bins=4, edgecolor='black', alpha=0.7, color='red', label='Withdrawn')
ax2.axvline(x=np.mean(all_car), color='blue', linestyle='--', 
            label=f'Mean: {np.mean(all_car):.2f}%', linewidth=2)
ax2.set_title('Distribution of 5-Day CAR', fontsize=12, fontweight='bold')
ax2.set_xlabel('CAR (%)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Arbitrage Profit by Deal
ax3 = axes[1, 0]
deal_names = []
arb_profits = []
colors = []
for r in results.values():
    if r['arb_profit'] is not None:
        deal_names.append(f"{r['name']}\n({r['status'][0]})")
        arb_profits.append(r['arb_profit'])
        if r['status'] == 'Completed':
            colors.append('green' if r['arb_profit'] > 0 else 'orange')
        else:
            colors.append('red')

ax3.barh(range(len(arb_profits)), arb_profits, color=colors, alpha=0.7)
ax3.set_yticks(range(len(deal_names)))
ax3.set_yticklabels(deal_names, fontsize=8)
ax3.set_title('Arbitrage Profit by Deal', fontsize=12, fontweight='bold')
ax3.set_xlabel('Profit (%)')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Risk-Return Scatter (Completed deals only)
ax4 = axes[1, 1]
for r in results.values():
    if r['deal_spread_mean'] is not None and r['arb_profit'] is not None:
        color = 'green' if r['status'] == 'Completed' else 'red'
        marker = 'o' if r['status'] == 'Completed' else 'x'
        ax4.scatter(r['deal_spread_mean'], r['arb_profit'], 
                   s=150, alpha=0.7, c=color, marker=marker)
        ax4.annotate(r['name'], (r['deal_spread_mean'], r['arb_profit']), 
                    fontsize=7, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax4.set_title('Deal Spread vs Arbitrage Profit', fontsize=12, fontweight='bold')
ax4.set_xlabel('Average Deal Spread (%)')
ax4.set_ylabel('Arbitrage Profit (%)')
ax4.grid(alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='Completed'),
                   Line2D([0], [0], marker='x', color='red', markersize=10, 
                          label='Withdrawn')]
ax4.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('../results/merger_arbitrage_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Main visualization saved to results/merger_arbitrage_analysis.png")

# Additional plot: Cumulative Abnormal Returns
fig2, ax = plt.subplots(figsize=(14, 8))
for ticker, result in results.items():
    if result['data'] is not None and not result['data'].empty:
        days_from_announce = (result['data'].index - result['announce_date']).days
        linestyle = '-' if result['status'] == 'Completed' else '--'
        ax.plot(days_from_announce, result['data']['CAR'] * 100,
               label=f"{result['name']} ({result['status'][0]})", 
               alpha=0.7, linewidth=2, linestyle=linestyle)

ax.axvline(x=0, color='red', linestyle='--', label='Announcement', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_title('Cumulative Abnormal Returns Around M&A Announcements\n(Hong Kong 2022-2024)', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Days from Announcement', fontsize=12)
ax.set_ylabel('CAR (%)', fontsize=12)
ax.legend(fontsize=8, loc='best', ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim(-60, 150)
plt.tight_layout()
plt.savefig('../results/car_evolution.png', dpi=300, bbox_inches='tight')
print("✓ CAR evolution plot saved to results/car_evolution.png")

# Summary statistics chart
fig3, ax = plt.subplots(figsize=(10, 6))

# Create summary bar chart
metrics = ['Avg CAR\n(All)', 'Avg Arb Profit\n(Completed)', 'Success Rate\n(%)']
if len(completed_arb_profit) > 0:
    values = [
        np.mean(all_car),
        np.mean(completed_arb_profit),
        len([x for x in completed_arb_profit if x > 0]) / len(completed_arb_profit) * 100
    ]
else:
    values = [np.mean(all_car), 0, 0]

colors = ['skyblue', 'lightgreen', 'gold']
bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_title('Summary Statistics: HK Merger Arbitrage\n(2022-2024)', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/summary_stats.png', dpi=300, bbox_inches='tight')
print("✓ Summary statistics saved to results/summary_stats.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📊 results/deal_summary.csv")
print("  📈 results/merger_arbitrage_analysis.png")
print("  📈 results/car_evolution.png")
print("  📈 results/summary_stats.png")
print("\n" + "="*80)
