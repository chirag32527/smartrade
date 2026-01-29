# NIFTY Naked Options Trading - Optimized PPO Implementation

## Overview

This is a **highly specialized** PPO implementation for NIFTY weekly options naked selling strategy.

### Strategy Specifications

- **Instrument**: NIFTY Index Options (Weekly Expiry)
- **Strategy**: Naked Selling (CE/PE)
- **Strike Selection**: Limited to 5 strikes only
  - 2 OTM (Out of The Money)
  - 1 ATM (At The Money)
  - 2 ITM (In The Money)
- **Position Limit**: Maximum 3 concurrent positions
- **Capital**: ₹1,00,000 (1 lakh)
- **Lot Size**: 50 (NIFTY standard)
- **Expiry**: Weekly (5 trading days per episode)

## Key Features

### 1. NIFTY-Specific Environment (34-dimensional state)

**Basic Features (12)**:
- Account balance (normalized)
- Position utilization (current/max)
- NIFTY spot price
- ATM strike price
- Days to expiry
- India VIX (volatility index)
- Total P&L
- ATM offset
- Max loss today
- Max profit today
- Consecutive losses
- Consecutive wins

**TA-Lib Indicators (12)**:
- ATR, Bollinger Bands, RSI, MACD, Stochastic, SMA, EMA, ADX, OBV, BB Width, VIX Ratio

**Option Premiums (10)**:
- 5 CE premiums (for each strike)
- 5 PE premiums (for each strike)

### 2. Realistic Transaction Costs (Indian Market)

```python
Brokerage: ₹20 flat per executed order
STT: 0.05% on sell side
Exchange Fees: 0.053% (NSE)
GST: 18% on brokerage + exchange fees
```

### 3. Action Space (11 discrete actions)

```
Action 0: Hold (do nothing)
Actions 1-5: Sell CE at strikes [ATM-100, ATM-50, ATM, ATM+50, ATM+100]
Actions 6-10: Sell PE at strikes [ATM-100, ATM-50, ATM, ATM+50, ATM+100]
```

## Optimized Hyperparameters

These hyperparameters are **specifically tuned** for NIFTY naked options trading:

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Learning Rate** | 1e-4 | Conservative - options P&L is high variance |
| **Steps per Update** | 512 | Moderate - balances exploration & learning |
| **Batch Size** | 128 | Large - smooths noisy gradients from volatile P&L |
| **Optimization Epochs** | 20 | More than default (10) - ensures stable convergence |
| **Gamma (Discount)** | 0.99 | Standard for weekly horizon |
| **GAE Lambda** | 0.95 | High - accurate advantage estimation |
| **Clip Range** | 0.2 | Standard PPO clipping |
| **Entropy Coefficient** | 0.02 | **HIGH** - encourages exploration of different strikes/timings |
| **Value Coefficient** | 0.5 | Standard |
| **Max Gradient Norm** | 0.5 | Tight - prevents exploding gradients |
| **Network Architecture** | [128, 128, 64] | Deeper than default - captures complex option dynamics |

### Why These Values?

1. **Low Learning Rate (1e-4)**:
   - Options trading has extreme P&L variance
   - Small steps prevent overshooting optimal policy

2. **High Entropy (0.02 vs default 0.0)**:
   - Must explore different strike prices
   - Different market conditions need different strategies
   - Without entropy bonus, agent may converge to suboptimal strategy

3. **Large Batch Size (128)**:
   - Single bad trade can have huge loss
   - Larger batches average out noise
   - More stable gradient estimates

4. **More Epochs (20)**:
   - Complex state space (34 dimensions)
   - Options require understanding non-linear payoff structures
   - Extra epochs ensure thorough learning

5. **Deeper Network [128, 128, 64]**:
   - Default [64, 64] too shallow for options
   - Need capacity to model:
     - Time decay (theta)
     - Volatility effects (vega)
     - Price movement (delta)
     - Strike selection logic

## Installation & Setup

```bash
# Install dependencies
cd src/q_learning
pip install -r requirements.txt

# Note: Requires TA-Lib C library
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib
```

## Quick Start

### 1. Compare with Baselines (Demo)

```bash
cd src/q_learning/app
python train_nifty.py --compare
```

This shows:
- Random strategy performance
- Always sell ATM strategy performance

### 2. Train Agent

```bash
# Basic training (250k timesteps ~ 50k episodes)
python train_nifty.py --train --timesteps 250000

# With custom save path
python train_nifty.py --train --timesteps 250000 --save-path ./my_models/

# With real data (recommended)
python train_nifty.py --train --data-file nifty_historical.csv
```

**Expected Training Time**: ~25-30 minutes (on modern CPU)

### 3. Monitor Training

```bash
# In separate terminal
tensorboard --logdir=./nifty_ppo_logs/

# Open browser: http://localhost:6006
```

Watch these metrics:
- `rollout/ep_rew_mean`: Average episode reward
- `train/entropy_loss`: Exploration level (should stay positive)
- `train/policy_loss`: Policy improvement
- `train/value_loss`: Value function accuracy

### 4. Evaluate Trained Agent

```bash
# Evaluate default saved model
python train_nifty.py --evaluate --episodes 20

# Evaluate specific model
python train_nifty.py --evaluate --model-path ./nifty_models/best_model --episodes 50

# Evaluate with rendering (see trades)
python train_nifty.py --evaluate --render
```

### 5. Backtest on Historical Data

```bash
python train_nifty.py --backtest --data-file nifty_historical.csv --episodes 50
```

## Data Format

If using real NIFTY data, CSV should have:

```csv
Date,Open,High,Low,Close,Volume,VIX
2024-01-01,19500,19650,19480,19600,1250000,15.2
2024-01-02,19610,19720,19590,19680,1180000,14.8
...
```

**Important**:
- `Close` = NIFTY spot price
- `VIX` = India VIX (if unavailable, will be estimated from volatility)
- Need at least 100 rows for lookback period

## Performance Expectations

### With Synthetic Data (Current)

| Metric | Expected Range | Good Performance |
|--------|----------------|------------------|
| Win Rate | 45-55% | > 55% |
| Avg Weekly P&L | ₹-500 to ₹2,000 | > ₹1,000 |
| Sharpe Ratio | 0.5 - 1.5 | > 1.5 |
| Max Drawdown | 5-15% | < 10% |

### With Real Data (Production)

- **Target Win Rate**: 60-70%
- **Target Weekly P&L**: ₹2,000 - ₹5,000 (2-5% ROI)
- **Target Sharpe**: > 2.0
- **Max Acceptable Drawdown**: < 20%

## Strategy Insights

### What the Agent Learns

1. **Strike Selection**:
   - OTM options for high premium/low risk
   - ITM options for steady income
   - ATM for balance

2. **VIX-Based Decisions**:
   - Sell more when VIX is high (expensive premiums)
   - Reduce positions when VIX is low (cheap premiums)

3. **Time Decay Exploitation**:
   - Sell early in week when time value is high
   - Close/hold near expiry as theta accelerates

4. **Technical Signal Integration**:
   - Avoid selling CE when RSI > 70 (overbought - reversal risk)
   - Avoid selling PE when RSI < 30 (oversold - reversal risk)
   - Use Bollinger Bands to gauge volatility regime

5. **Risk Management**:
   - Limit concurrent positions (max 3)
   - Avoid over-trading (action 0 = hold is valid)
   - Exit on 50% drawdown

## Advanced Configuration

### Modify Risk Parameters

Edit `environment_nifty.py`:

```python
# More aggressive (higher risk/reward)
initial_balance=150000,  # More capital
max_positions=5,         # More concurrent trades
lot_size=75,            # Larger positions

# More conservative
initial_balance=75000,   # Less capital
max_positions=2,         # Fewer concurrent trades
lot_size=25,            # Smaller positions
```

### Tune for Different Market Conditions

**High Volatility Market** (VIX > 20):
```python
ent_coef = 0.03         # More exploration
learning_rate = 5e-5    # More conservative
```

**Low Volatility Market** (VIX < 15):
```python
ent_coef = 0.01         # Less exploration
learning_rate = 2e-4    # Faster learning
```

## Troubleshooting

### Agent Always Takes Action 0 (Hold)

**Problem**: Agent learned that holding is safest
**Solution**:
- Increase `ent_coef` to 0.03 or 0.04
- Add reward shaping: small penalty for consecutive holds
- Ensure VIX and premiums are varying (not static)

### High Variance in P&L

**Problem**: Some episodes +₹10k, some -₹10k
**Solution**:
- Increase `batch_size` to 256
- Reduce `learning_rate` to 5e-5
- Train longer (500k timesteps)

### Agent Loses Money Consistently

**Problem**: Worse than random
**Solution**:
- Check transaction costs (may be too high)
- Verify premium calculation (should decrease near expiry)
- Ensure indicators are normalized correctly
- Try reducing `max_positions` to 2

### Training Crashes / NaN Loss

**Problem**: Gradient explosion
**Solution**:
- Reduce `learning_rate` to 5e-5
- Reduce `max_grad_norm` to 0.3
- Ensure state normalization is enabled (VecNormalize)

## Comparison with DQN

| Aspect | DQN | Optimized PPO |
|--------|-----|---------------|
| State Dimension | 18 | 34 (NIFTY-specific) |
| For NIFTY Options | Not optimized | ✅ Specifically tuned |
| India VIX | No | ✅ Yes |
| Strike Constraints | No | ✅ 5 strikes only |
| Transaction Costs | Generic | ✅ Indian market realistic |
| Training Stability | Can diverge | ✅ Very stable |
| Hyperparameters | Generic | ✅ Tuned for naked selling |

## Files

```
src/q_learning/app/
├── environment_nifty.py        # NIFTY-specific environment (34D state)
├── agent_nifty_ppo.py          # Optimized PPO with tuned hyperparameters
├── train_nifty.py              # Training/evaluation/backtesting script
├── environment_talib.py        # Generic TA-Lib environment (18D state)
└── agent_ppo.py                # Generic PPO agent
```

## Next Steps for Production

1. **Get Real NIFTY Data**:
   - Use NSE API or data provider
   - Include actual option chain data
   - Historical India VIX

2. **Add Greeks Calculation**:
   - Implement Black-Scholes for accurate premiums
   - Include Delta, Gamma, Theta, Vega in state

3. **Paper Trading**:
   - Connect to broker API
   - Test with small capital (₹10k-25k)
   - Monitor for 4-8 weeks

4. **Risk Management**:
   - Add stop-loss (e.g., -₹5000 per position)
   - Position sizing based on VIX
   - Circuit breakers for extreme moves

5. **Continuous Learning**:
   - Retrain monthly with new data
   - A/B test new vs old model
   - Track live performance vs backtest

## Disclaimer

⚠️ **IMPORTANT**: This is for educational purposes only.

- Options trading involves significant risk
- Past performance ≠ future results
- Start with paper trading
- Never risk more than you can afford to lose
- Consult financial advisor before live trading

---

**Ready to train?** Run `python train_nifty.py --train`
