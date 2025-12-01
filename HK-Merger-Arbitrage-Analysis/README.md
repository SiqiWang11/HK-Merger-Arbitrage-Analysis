[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# 🎯 Hong Kong Merger Arbitrage Analysis

> *"High returns in merger arbitrage are an illusion if you ignore deal failure risk. The market is smarter than it looks."*

---

## 📖 Question
Can retail investors make profit from buying stocks after M&A announcements in Hong Kong? If so, what are the risks, and how can we mitigate them?

---

### 🌐 Analysis
I analyzed **11 real M&A transactions** in Hong Kong (2022-2025), including both successful privatizations and failed deals. Core findings based on **8 finalized deals** with complete data, excluding pending and withdraw cases. What I found challenged the conventional wisdom that earning money is easy in merger arbitrage.

![Complete Analysis](results/figures/complete_analysis.png)

---

## 📊 Data Summary

| Company | Sector | Status | Spread | Return |
|---------|--------|--------|--------|--------|
| L'Occitane | Consumer | ✅ | 9.0% | +9.0% |
| Mason Group | Financial | ✅ | 9.0% | +9.0% |
| CIMC Vehicles | Industrial | ✅ | 7.9% | +7.9% |
| Kin Yat Holdings | Industrial | ✅ | 9.1% | +9.1% |
| Yitai Coal | Energy | ✅ | 8.0% | +8.0% |
| Vinda International | Consumer | ✅ | 9.3% | +9.3% |
| Dali Foods | Consumer | ✅ | 8.7% | +8.7% |
| Yashili | Consumer | ✅ | 11.1% | +11.1% |
| **IMAX China** | Entertainment | ❌ | 13.6% | **-22.7%** |
| **China TCM** | Healthcare | ❌ | 12.2% | **-18.3%** |
| **Hang Seng** | Financial	| ⏳ Pending	| N/A | N/A |

---

## 🔍 Key Findings

### Finding #1: Significant Announcement Effect
When an M&A deal is announced, target stocks **jump immediately**:
- **Average same-day jump: +7.9%**
- Statistically significant (t = 6.77, p < 0.001)
- Market recognizes deal value, but doesn't fully price it in

### Finding #2: The Arbitrage Spread Exists
After the initial jump, a **gap remains** between market price and offer price:
- Average spread at T+1: **9.8%**
- This "gap" is the arbitrageur's potential profit

### Finding #3: The Two Faces of Merger Arbitrage

| Outcome | Deals | Avg Return | Win Rate |
|---------|-------|------------|----------|
| ✅ Completed | 8 | **+9.0%** | 100% |
| ❌ Withdrawn | 2 | **-20.5%** | 0% |

**The Harsh Reality:**
```
Expected Return = 80% × (+9.0%) + 20% × (-20.5%) = +3.1%
```
Deal failure risk eliminates most of the apparent opportunity!

### Finding #4: The Market Knew All Along
**Critical Insight:** Withdrawn deals had **higher initial spreads** (12.9% vs 9.0%)

The market priced in the risk — high spreads were a **warning sign**, not an opportunity.

![Return Attribution](results/figures/return_attribution.png)

---

## 💡 Strategic Insights

### What Predicts Deal Success?

| Factor | Completed | Withdrawn | Insight |
|--------|-----------|-----------|---------|
| **Initial Spread** | 9.0% | 12.9% | High spread = High risk |
| **Acquirer Type** | Family/SOE | External | Controlling shareholders more committed |
| **Sector** | Consumer best | Entertainment/Healthcare | Regulatory risk varies |

### Proposed Strategy
1. **Avoid deals with spreads > 25%** — likely "traps"
2. **Focus on family/SOE-backed deals** — higher completion rates
3. **Prefer Consumer sector** — 100% success in sample
4. **Consider holding period** — capital efficiency matters

---

## 🛠️ Technical Implementation
- **Python 3.9+**
- **pandas, numpy** — Data manipulation
- **scipy** — Statistical testing
- **matplotlib** — Visualization


## 📃 License
- **MIT License** - Free to use with attribution


## 🔄 Project Structure
```
HK-Merger-Arbitrage-Analysis/
├── code/
│   ├── merger_arbitrage_analysis.py # Live data version
│   └── demo_analysis.py           # Demo version
├── results/
│   ├── demo_results/
│   │   ├── holding_vs_return.png
│   │   ├── merger_arbitrage_analysis.png
│   │   └── summary_stats.png
│   ├── figures/
│   │   ├── complete_analysis.png  # Main 6-panel visualization
│   │   └── return_attribution.png # Waterfall chart
│   ├── deal_summary.csv           # Complete deal data
│   └── statistics_summary.csv     # Key statistics
└── README.md
```


## 🚀 Quick Start
```bash
# Install dependencies
pip install pandas numpy scipy matplotlib

# Run analysis
cd code
python merger_arbitrage_analysis.py
```

---

## 📚 Academic Reference

This study extends findings from classic merger arbitrage literature:

| Study | Market | CAR | Finding |
|-------|--------|-----|---------|
| Mitchell & Pulvino (2001) | US | 7.2% | Arb returns are not true alpha |
| Baker & Savasoglu (2002) | US | 8.1% | Deal risk is systematic |
| **This Study** | **HK** | **7.9%** | **High spread = High risk signal** |

---

## 📧 Contact

**Siqi Wang**  
The University of Hong Kong | Mathematics & Finance
u3657923@connect.hku.hk

---

*"The best trade is often the one you don't make."*
