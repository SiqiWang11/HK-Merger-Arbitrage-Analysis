"""
Hong Kong Merger Arbitrage Analysis - DEMO VERSION
===================================================
This version uses simulated data to demonstrate the analysis output.
For actual analysis, run merger_arbitrage_analysis.py on your local machine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Create results directory
os.makedirs('../results', exist_ok=True)

print("="*80)
print("HONG KONG MERGER ARBITRAGE ANALYSIS (DEMO)")
print("Event-Driven Strategy: Profiting from M&A Deal Spreads")
print("="*80)

# ============================================
# SIMULATED RESULTS BASED ON REALISTIC HK M&A DATA
# ============================================

# These are simulated results based on typical HK M&A characteristics
# Actual results will vary when running with real Yahoo Finance data

results_data = {
    '0973.HK': {
        'name': "L'Occitane International",
        'status': 'Completed',
        'announce_date': '2024-04-29',
        'completion_date': '2024-08-06',
        'offer_price': 34.00,
        'price_at_announce': 28.50,
        'buy_price_t1': 31.20,
        'car_5_days': 15.8,
        'deal_spread_at_announce': 19.3,
        'arb_profit': 8.97,
        'holding_days': 99
    },
    '0273.HK': {
        'name': 'Mason Group Holdings',
        'status': 'Completed',
        'announce_date': '2023-06-11',
        'completion_date': '2023-11-13',
        'offer_price': 0.0338,
        'price_at_announce': 0.028,
        'buy_price_t1': 0.031,
        'car_5_days': 18.2,
        'deal_spread_at_announce': 20.7,
        'arb_profit': 9.03,
        'holding_days': 155
    },
    '1839.HK': {
        'name': 'CIMC Vehicles',
        'status': 'Completed',
        'announce_date': '2024-03-11',
        'completion_date': '2024-06-03',
        'offer_price': 7.50,
        'price_at_announce': 6.35,
        'buy_price_t1': 6.95,
        'car_5_days': 12.4,
        'deal_spread_at_announce': 18.1,
        'arb_profit': 7.91,
        'holding_days': 84
    },
    '0638.HK': {
        'name': 'Kin Yat Holdings',
        'status': 'Completed',
        'announce_date': '2024-06-28',
        'completion_date': '2024-08-23',
        'offer_price': 0.72,
        'price_at_announce': 0.58,
        'buy_price_t1': 0.66,
        'car_5_days': 20.3,
        'deal_spread_at_announce': 24.1,
        'arb_profit': 9.09,
        'holding_days': 56
    },
    '3948.HK': {
        'name': 'Inner Mongolia Yitai Coal',
        'status': 'Completed',
        'announce_date': '2023-06-13',
        'completion_date': '2023-08-11',
        'offer_price': 17.50,
        'price_at_announce': 14.80,
        'buy_price_t1': 16.20,
        'car_5_days': 14.5,
        'deal_spread_at_announce': 18.2,
        'arb_profit': 8.02,
        'holding_days': 59
    },
    '3331.HK': {
        'name': 'Vinda International',
        'status': 'Completed',
        'announce_date': '2024-03-08',
        'completion_date': '2024-08-16',
        'offer_price': 23.50,
        'price_at_announce': 19.80,
        'buy_price_t1': 21.50,
        'car_5_days': 13.2,
        'deal_spread_at_announce': 18.7,
        'arb_profit': 9.30,
        'holding_days': 161
    },
    '3799.HK': {
        'name': 'Dali Foods Group',
        'status': 'Completed',
        'announce_date': '2023-06-27',
        'completion_date': '2023-08-30',
        'offer_price': 3.75,
        'price_at_announce': 3.15,
        'buy_price_t1': 3.45,
        'car_5_days': 16.8,
        'deal_spread_at_announce': 19.0,
        'arb_profit': 8.70,
        'holding_days': 64
    },
    '1230.HK': {
        'name': 'Yashili International',
        'status': 'Completed',
        'announce_date': '2022-05-06',
        'completion_date': '2023-07-05',
        'offer_price': 1.20,
        'price_at_announce': 0.95,
        'buy_price_t1': 1.08,
        'car_5_days': 22.5,
        'deal_spread_at_announce': 26.3,
        'arb_profit': 11.11,
        'holding_days': 425
    },
    '1970.HK': {
        'name': 'IMAX China',
        'status': 'Withdrawn',
        'announce_date': '2023-07-12',
        'completion_date': '2023-10-10',
        'offer_price': 10.00,
        'price_at_announce': 7.20,
        'buy_price_t1': 8.80,
        'car_5_days': 19.5,
        'deal_spread_at_announce': 38.9,
        'arb_profit': -22.73,  # Loss due to withdrawal
        'holding_days': 90
    },
    '0570.HK': {
        'name': 'China TCM Holdings',
        'status': 'Withdrawn',
        'announce_date': '2024-02-21',
        'completion_date': '2024-06-30',
        'offer_price': 4.60,
        'price_at_announce': 3.45,
        'buy_price_t1': 4.10,
        'car_5_days': 17.2,
        'deal_spread_at_announce': 33.3,
        'arb_profit': -18.29,  # Loss due to withdrawal
        'holding_days': 130
    },
}

# ============================================
# ANALYSIS OUTPUT
# ============================================

print(f"\nTotal deals analyzed: {len(results_data)}")
completed = {k: v for k, v in results_data.items() if v['status'] == 'Completed'}
withdrawn = {k: v for k, v in results_data.items() if v['status'] == 'Withdrawn'}
print(f"  - Completed: {len(completed)}")
print(f"  - Withdrawn: {len(withdrawn)}")

print("\n" + "="*80)
print("RUNNING MERGER ARBITRAGE ANALYSIS")
print("="*80 + "\n")

for ticker, r in results_data.items():
    status_emoji = "✓" if r['status'] == 'Completed' else "✗"
    print(f"  {status_emoji} {r['name']}: CAR = {r['car_5_days']:.2f}%, Arb Profit = {r['arb_profit']:.2f}% ({r['status']})")

# ============================================
# PORTFOLIO STATISTICS
# ============================================

print("\n" + "="*80)
print("PORTFOLIO SUMMARY STATISTICS")
print("="*80)

# All deals CAR
all_car = [r['car_5_days'] for r in results_data.values()]
print(f"\n{'='*50}")
print(f"ALL DEALS ({len(results_data)} transactions)")
print(f"{'='*50}")
print(f"\nCumulative Abnormal Returns (5-day window around announcement):")
print(f"  Average CAR: {np.mean(all_car):.2f}%")
print(f"  Median CAR: {np.median(all_car):.2f}%")
print(f"  Std Dev: {np.std(all_car):.2f}%")
print(f"  Min: {np.min(all_car):.2f}%")
print(f"  Max: {np.max(all_car):.2f}%")

# Completed deals
completed_arb = [r['arb_profit'] for r in completed.values()]
completed_holding = [r['holding_days'] for r in completed.values()]
completed_spread = [r['deal_spread_at_announce'] for r in completed.values()]

print(f"\n{'='*50}")
print(f"COMPLETED DEALS ({len(completed)} transactions)")
print(f"{'='*50}")
print(f"\nArbitrage Profits (Buy at T+1, Sell at Offer):")
print(f"  Average Profit: {np.mean(completed_arb):.2f}%")
print(f"  Median Profit: {np.median(completed_arb):.2f}%")
print(f"  Min Profit: {np.min(completed_arb):.2f}%")
print(f"  Max Profit: {np.max(completed_arb):.2f}%")
print(f"  Success Rate (>0%): {len([x for x in completed_arb if x > 0]) / len(completed_arb) * 100:.1f}%")

print(f"\nHolding Period:")
print(f"  Average: {np.mean(completed_holding):.0f} days")
print(f"  Min: {np.min(completed_holding):.0f} days")
print(f"  Max: {np.max(completed_holding):.0f} days")

print(f"\nDeal Spreads at Announcement:")
print(f"  Average Spread: {np.mean(completed_spread):.2f}%")
print(f"  Median Spread: {np.median(completed_spread):.2f}%")

# Annualized return
avg_holding = np.mean(completed_holding)
avg_profit = np.mean(completed_arb)
annualized_return = (1 + avg_profit/100) ** (365/avg_holding) - 1
print(f"\nAnnualized Return (approximation): {annualized_return*100:.1f}%")

# Withdrawn deals
print(f"\n{'='*50}")
print(f"WITHDRAWN DEALS ({len(withdrawn)} transactions)")
print(f"{'='*50}")
withdrawn_arb = [r['arb_profit'] for r in withdrawn.values()]
print(f"\nArbitrage Results (if held until withdrawal):")
print(f"  Average Loss: {np.mean(withdrawn_arb):.2f}%")
for ticker, r in withdrawn.items():
    print(f"    - {r['name']}: {r['arb_profit']:.2f}%")

# Statistical test
print(f"\n{'='*50}")
print(f"STATISTICAL TESTS")
print(f"{'='*50}")
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

# Risk-adjusted metrics
print(f"\n{'='*50}")
print(f"RISK-ADJUSTED METRICS")
print(f"{'='*50}")

# Expected return considering deal failure
success_rate = len(completed) / len(results_data)
fail_rate = len(withdrawn) / len(results_data)
expected_return = success_rate * np.mean(completed_arb) + fail_rate * np.mean(withdrawn_arb)
print(f"\nExpected Return (considering failure risk):")
print(f"  Success Rate: {success_rate*100:.1f}%")
print(f"  Failure Rate: {fail_rate*100:.1f}%")
print(f"  Expected Return: {expected_return:.2f}%")
print(f"  (= {success_rate*100:.0f}% × {np.mean(completed_arb):.1f}% + {fail_rate*100:.0f}% × {np.mean(withdrawn_arb):.1f}%)")

# ============================================
# SAVE TO CSV
# ============================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_list = []
for ticker, r in results_data.items():
    results_list.append({
        'Ticker': ticker,
        'Company': r['name'],
        'Status': r['status'],
        'Announcement_Date': r['announce_date'],
        'Completion_Date': r['completion_date'],
        'Offer_Price': r['offer_price'],
        'Price_at_Announcement': r['price_at_announce'],
        'CAR_5day_%': r['car_5_days'],
        'Deal_Spread_at_Announce_%': r['deal_spread_at_announce'],
        'Buy_Price_T+1': r['buy_price_t1'],
        'Arbitrage_Profit_%': r['arb_profit'],
        'Holding_Days': r['holding_days']
    })

results_df = pd.DataFrame(results_list)
results_df.to_csv('../results/deal_summary.csv', index=False)
print(f"\n✓ Results saved to results/deal_summary.csv")

print("\n" + "="*80)
print("DEAL-BY-DEAL SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hong Kong Merger Arbitrage Analysis\n(2022-2024 M&A Transactions)', 
             fontsize=16, fontweight='bold')

# Plot 1: Deal Spreads comparison
ax1 = axes[0, 0]
companies = [r['name'][:15] for r in results_data.values()]
spreads = [r['deal_spread_at_announce'] for r in results_data.values()]
colors = ['green' if r['status'] == 'Completed' else 'red' for r in results_data.values()]
ax1.barh(range(len(spreads)), spreads, color=colors, alpha=0.7)
ax1.set_yticks(range(len(companies)))
ax1.set_yticklabels(companies, fontsize=8)
ax1.set_title('Deal Spread at Announcement', fontsize=12, fontweight='bold')
ax1.set_xlabel('Deal Spread (%)')
ax1.axvline(x=np.mean(spreads), color='blue', linestyle='--', 
            label=f'Mean: {np.mean(spreads):.1f}%', linewidth=2)
ax1.legend()
ax1.grid(alpha=0.3, axis='x')

# Plot 2: CAR Distribution
ax2 = axes[0, 1]
completed_car = [r['car_5_days'] for r in completed.values()]
withdrawn_car = [r['car_5_days'] for r in withdrawn.values()]
ax2.hist(completed_car, bins=6, edgecolor='black', alpha=0.7, color='green', label='Completed')
ax2.hist(withdrawn_car, bins=3, edgecolor='black', alpha=0.7, color='red', label='Withdrawn')
ax2.axvline(x=np.mean(all_car), color='blue', linestyle='--', 
            label=f'Mean: {np.mean(all_car):.1f}%', linewidth=2)
ax2.set_title('Distribution of 5-Day CAR', fontsize=12, fontweight='bold')
ax2.set_xlabel('CAR (%)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Arbitrage Profit by Deal
ax3 = axes[1, 0]
deal_names = [f"{r['name'][:12]}\n({r['status'][0]})" for r in results_data.values()]
arb_profits = [r['arb_profit'] for r in results_data.values()]
colors = []
for r in results_data.values():
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
ax3.axvline(x=np.mean(completed_arb), color='green', linestyle='--', 
            label=f'Avg (Completed): {np.mean(completed_arb):.1f}%', linewidth=2)
ax3.legend()
ax3.grid(alpha=0.3, axis='x')

# Plot 4: Risk-Return Scatter
ax4 = axes[1, 1]
for r in results_data.values():
    color = 'green' if r['status'] == 'Completed' else 'red'
    marker = 'o' if r['status'] == 'Completed' else 'x'
    ax4.scatter(r['deal_spread_at_announce'], r['arb_profit'], 
               s=150, alpha=0.7, c=color, marker=marker)
    ax4.annotate(r['name'][:10], (r['deal_spread_at_announce'], r['arb_profit']), 
                fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax4.set_title('Deal Spread vs Arbitrage Profit', fontsize=12, fontweight='bold')
ax4.set_xlabel('Deal Spread at Announcement (%)')
ax4.set_ylabel('Arbitrage Profit (%)')
ax4.grid(alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                          markersize=10, label='Completed'),
                   Line2D([0], [0], marker='x', color='red', markersize=10, 
                          label='Withdrawn')]
ax4.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('../results/merger_arbitrage_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Main visualization saved to results/merger_arbitrage_analysis.png")

# Summary statistics chart
fig2, ax = plt.subplots(figsize=(12, 6))

# Create grouped bar chart
x = np.arange(4)
width = 0.35

metrics_labels = ['Avg CAR\n(%)', 'Avg Arb Profit\n(%)', 'Success Rate\n(%)', 'Annualized\nReturn (%)']
all_values = [np.mean(all_car), np.mean([r['arb_profit'] for r in results_data.values()]), 
              success_rate * 100, annualized_return * 100]
completed_values = [np.mean(completed_car), np.mean(completed_arb), 100, 
                    (1 + np.mean(completed_arb)/100) ** (365/np.mean(completed_holding)) * 100 - 100]

bars1 = ax.bar(x - width/2, all_values, width, label='All Deals', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, completed_values, width, label='Completed Only', color='green', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
                
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Summary Statistics: HK Merger Arbitrage Strategy\n(2022-2024)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.legend()
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../results/summary_stats.png', dpi=300, bbox_inches='tight')
print("✓ Summary statistics saved to results/summary_stats.png")

# Holding period vs Return
fig3, ax = plt.subplots(figsize=(10, 6))
for r in results_data.values():
    color = 'green' if r['status'] == 'Completed' else 'red'
    marker = 'o' if r['status'] == 'Completed' else 'x'
    ax.scatter(r['holding_days'], r['arb_profit'], s=150, alpha=0.7, c=color, marker=marker)
    ax.annotate(r['name'][:10], (r['holding_days'], r['arb_profit']), 
                fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_title('Holding Period vs Arbitrage Profit', fontsize=14, fontweight='bold')
ax.set_xlabel('Holding Period (Days)', fontsize=12)
ax.set_ylabel('Arbitrage Profit (%)', fontsize=12)
ax.grid(alpha=0.3)
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('../results/holding_vs_return.png', dpi=300, bbox_inches='tight')
print("✓ Holding period analysis saved to results/holding_vs_return.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📊 results/deal_summary.csv")
print("  📈 results/merger_arbitrage_analysis.png")
print("  📈 results/summary_stats.png")
print("  📈 results/holding_vs_return.png")
print("\n" + "="*80)
print("\nKEY FINDINGS:")
print("="*80)
print(f"""
1. ANNOUNCEMENT EFFECT:
   - Average 5-day CAR: {np.mean(all_car):.2f}% (statistically significant)
   - Market reacts positively to M&A announcements

2. MERGER ARBITRAGE PROFITABILITY:
   - Average profit (completed deals): {np.mean(completed_arb):.2f}%
   - Annualized return: {annualized_return*100:.1f}%
   - Success rate: {success_rate*100:.0f}%

3. RISK ANALYSIS:
   - Deal failure rate: {fail_rate*100:.0f}%
   - Average loss on failed deals: {np.mean(withdrawn_arb):.2f}%
   - Expected return (risk-adjusted): {expected_return:.2f}%

4. PRACTICAL IMPLICATIONS:
   - Hong Kong M&A spreads are attractive for arbitrage
   - Deal selection is critical - avoid deals with regulatory risk
   - Transaction costs (~0.5% round-trip) must be considered
""")
