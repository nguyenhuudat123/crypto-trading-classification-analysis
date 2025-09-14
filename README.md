# Cryptocurrency Price Movement Prediction: 3-Class vs 4-Class Classification

A comprehensive research project comparing the effectiveness of 3-class and 4-class classification strategies for cryptocurrency price movement prediction using machine learning models.

## Project Overview

This project evaluates whether adding an additional classification class improves cryptocurrency trading strategy performance across multiple timeframes and assets.

### Classification Strategies
- **3-Class**: Decline (0), Sideways (1), Rise (2)
- **4-Class**: Enhanced granularity with additional price movement categories

### Assets & Timeframes
- **Cryptocurrencies**: Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB)
- **Timeframes**: Daily (1D), 4-Hour (4H), 30-Minute (30M)
- **Data Period**: 2020-2025
- **Total Samples**: 99K+ (30M) to 2K+ (1D)

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis

The project is designed to run entirely through Jupyter notebooks. Choose your path:

#### Option 1: Run Complete Analysis
```bash
# Navigate to analysis folder
cd notebook/for_analysis/

# Open and run the main analysis notebook
jupyter notebook ptbtc.ipynb
```

#### Option 2: Generate New Results

**For 3-Class Classification:**
```bash
cd notebook/for_3_classes/
jupyter notebook btc_3class.ipynb    # For Bitcoin
jupyter notebook eth_3class.ipynb    # For Ethereum  
jupyter notebook bnb_3class.ipynb    # For Binance Coin
```

**For 4-Class Classification:**
```bash
cd notebook/for_4_classes/
jupyter notebook btc_4_class.ipynb   # For Bitcoin
jupyter notebook eth_4class.ipynb    # For Ethereum
jupyter notebook bnb_4class.ipynb    # For Binance Coin
```

#### Option 3: Download Fresh Data
```bash
cd notebook/download/
jupyter notebook download.ipynb
```

## Project Structure

```
├── data/                           # Raw cryptocurrency data
│   ├── btc_1d.csv, btc_4h.csv, btc_30m.csv
│   ├── eth_1d.csv, eth_4h.csv, eth_30m.csv
│   ├── bnb_1d.csv, bnb_4h.csv, bnb_30m.csv
│   └── size.txt
├── notebook/
│   ├── download/                   # Data collection scripts
│   │   └── download.ipynb
│   ├── for_3_classes/             # 3-class experiments
│   │   ├── btc_3class.ipynb
│   │   ├── eth_3class.ipynb
│   │   ├── bnb_3class.ipynb
│   │   └── *.py                   # Supporting modules
│   ├── for_4_classes/             # 4-class experiments  
│   │   ├── btc_4_class.ipynb
│   │   ├── eth_4class.ipynb
│   │   ├── bnb_4class.ipynb
│   │   └── *_silent.py            # Silent version modules
│   └── for_analysis/              # Statistical analysis
│       ├── ptbtc.ipynb            # Main analysis notebook
│       └── *_trading_analysis.csv # Generated reports
├── results/                       # Experimental results
│   ├── *_results_*_3class.csv    # 3-class results
│   └── *_results_*_4class.csv    # 4-class results
├── requirements.txt               # Python dependencies
└── README.md
```

## Models Tested

| Model | Type | Key Features |
|-------|------|-------------|
| XGBoost | Traditional ML | Gradient boosting, feature importance |
| LSTM | Deep Learning | Long-term temporal dependencies |
| GRU | Deep Learning | Simplified recurrent architecture |
| CNN | Deep Learning | Local pattern recognition |

## Technical Indicators

- **RSI** (Relative Strength Index)
- **EMA** (8, 34, 89 periods)
- **MACD** & Signal Line
- **Stochastic RSI**
- **Volume** indicators (SMA, ROC, OBV)

## Evaluation Metrics

- **Loss Count**: Wrong directional predictions
- **Loss Mean**: Average loss from incorrect predictions  
- **Transaction Count**: Total trading signals
- **Risk Ratio**: Loss count / Transaction count
- **Accuracy**: Standard classification accuracy

## Key Results Summary

### Overall Findings
- **No Universal Winner**: Neither 3-class nor 4-class consistently outperforms
- **Asset Dependent**: Different cryptocurrencies favor different approaches
- **Timeframe Sensitive**: Optimal strategy varies by trading frequency
- **Risk-Return Trade-off**: 4-class often increases both potential returns and risk

### Best Configurations by Asset

| Asset | Best Model | Optimal Lookahead | Peak Accuracy | Recommendation |
|-------|------------|------------------|---------------|----------------|
| BTC | GRU | 3 periods | 96.48% (30M) | Mixed results |
| ETH | Various | Timeframe dependent | ~85-90% | Slight 3-class preference |
| BNB | Mixed | Asset specific | ~85-95% | Context dependent |

### Statistical Significance

| Asset-Timeframe | Significant Metrics | Practical Significance | Confidence |
|-----------------|-------------------|----------------------|------------|
| BTC-1D | 3/4 | Low | Mixed |
| BTC-4H | 4/4 | Medium | Mixed |
| BTC-30M | 3/4 | Low | Mixed |
| ETH-All | 2-6/4 | Low-Medium | Moderate 3-class |
| BNB-All | 2-4/4 | Low-Medium | Mixed |

## Practical Recommendations

### For Traders
1. **Test Both Strategies**: No universal solution exists
2. **Consider Transaction Costs**: 4-class generates more signals
3. **Asset-Specific Testing**: Each crypto has unique patterns
4. **Timeframe Matters**: Higher frequency favors simpler approaches

### For Researchers  
1. **Focus on Risk Metrics**: Beyond accuracy measures
2. **Statistical Rigor**: Use proper multiple testing corrections
3. **Practical Significance**: Economic meaning over statistical significance
4. **Market Regime Analysis**: Performance varies across conditions

## Usage Examples

### Running Bitcoin 3-Class Analysis
```python
# In btc_3class.ipynb
# The notebook handles everything automatically:
# 1. Data loading and preprocessing
# 2. Technical indicator calculation  
# 3. Grid search across all parameters
# 4. Model training and evaluation
# 5. Results saving to CSV
```

### Analyzing Results
```python
# In ptbtc.ipynb  
# Statistical comparison between strategies:
# 1. Load 3-class and 4-class results
# 2. Wilcoxon signed-rank tests
# 3. Effect size calculations
# 4. Multiple testing corrections
# 5. Generate recommendations
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- XGBoost 1.5+
- Scikit-learn
- Pandas, NumPy
- Jupyter Notebook
- GPU recommended for deep learning

See `requirements.txt` for complete dependencies.

## Key Insights

1. **Complexity vs Performance**: More classes don't always mean better results
2. **Risk Management Critical**: Higher granularity increases transaction volume
3. **Market Microstructure**: High-frequency trading favors simpler strategies
4. **Statistical vs Practical**: Significance doesn't guarantee profitability

## Contributing

Contributions welcome! The modular notebook structure makes it easy to:
- Add new assets
- Test additional models
- Modify evaluation metrics
- Extend analysis timeframes

## License

MIT License - See LICENSE file for details

## Disclaimer

This is a research project for educational purposes. Cryptocurrency trading involves significant financial risk. Always conduct your own research and consider consulting financial advisors before making investment decisions.

---

**Note**: All analysis can be reproduced by running the respective Jupyter notebooks. Results are automatically saved to the `results/` directory and can be analyzed using `ptbtc.ipynb`.