# Q-Learning Module - Options Trading with Deep Reinforcement Learning

This module implements automated options trading using Deep Reinforcement Learning. Two approaches are available:

1. **DQN** (Deep Q-Network) - Original implementation
2. **PPO** (Proximal Policy Optimization) - New, recommended approach

## Key Features

### Enhanced Environment with TA-Lib Indicators

The new `OptionTradingEnvWithTALib` environment includes **18-dimensional state space**:

**Basic Features (6):**
- Strike price difference
- Premium price
- Number of open trades
- Account balance
- Days to expiry
- Current underlying price

**TA-Lib Indicators (12):**
- **Volatility**: ATR, Bollinger Bands (upper, middle, lower, width)
- **Momentum**: RSI, MACD, MACD Signal, MACD Histogram, Stochastic %K
- **Trend**: SMA, EMA, ADX
- **Volume**: OBV (On-Balance Volume)

**Action Space (6 discrete actions):**
- 0: Buy Put (PE)
- 1: Sell Put (PE)
- 2: Hold Put
- 3: Buy Call (CE)
- 4: Sell Call (CE)
- 5: Hold Call

## Installation

```bash
# From the q_learning directory
cd src/q_learning
pip install -r requirements.txt

# Note: TA-Lib requires C library installation first
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib
# See: https://github.com/TA-Lib/ta-lib-python
```

## Quick Start

### 1. Compare Random vs PPO (Demo)

```bash
cd src/q_learning/app
python train_ppo.py
```

### 2. Train PPO Agent

```bash
# Train for 100k timesteps
python train_ppo.py --train --timesteps 100000

# Train with custom save path
python train_ppo.py --train --timesteps 50000 --save-path ./my_models/
```

### 3. Evaluate Trained Agent

```bash
# Evaluate saved model
python train_ppo.py --evaluate --model-path ./ppo_models/ppo_trading_agent_final

# Evaluate for more episodes
python train_ppo.py --evaluate --episodes 50
```

### 4. Compare DQN vs PPO

```bash
# Train both and compare
python compare_agents.py --timesteps 50000
```

## File Structure

```
src/q_learning/
├── requirements.txt              # Dependencies
└── app/
    ├── environment.py           # Original environment (static)
    ├── environment_talib.py     # NEW: Enhanced with TA-Lib indicators
    ├── agent.py                 # Original DQN agent
    ├── agent_ppo.py             # NEW: PPO agent (recommended)
    ├── train_ppo.py             # NEW: Training/evaluation script
    └── compare_agents.py        # NEW: DQN vs PPO comparison
```

## Training Details

### PPO Hyperparameters

```python
learning_rate = 3e-4         # Adam learning rate
n_steps = 2048               # Steps per update
batch_size = 64              # Minibatch size
n_epochs = 10                # Optimization epochs
gamma = 0.99                 # Discount factor
gae_lambda = 0.95           # GAE parameter
clip_range = 0.2            # PPO clipping
ent_coef = 0.01             # Entropy bonus
```

### Monitoring Training

PPO automatically logs to TensorBoard:

```bash
# In a separate terminal
tensorboard --logdir=./ppo_trading_tensorboard/

# Open browser to http://localhost:6006
```

## Architecture Comparison

| Feature | DQN | PPO (New) |
|---------|-----|-----------|
| Algorithm Type | Value-based | Policy-based |
| Sample Efficiency | Lower | Higher |
| Training Stability | Can be unstable | More stable |
| Exploration | Epsilon-greedy | Entropy bonus |
| Memory | Replay buffer | On-policy (no replay) |
| Parallelization | Limited | Easy to parallelize |
| **Recommendation** | Good baseline | **Recommended** |

## Using Your Own Data

To use real market data instead of synthetic:

1. **Modify `environment_talib.py`:**

```python
def __init__(self, data_file=None, ...):
    if data_file:
        # Load your CSV with OHLCV data
        df = pd.read_csv(data_file)
        self.price_history = df['Close'].values[-self.lookback_period:]
        self.high_history = df['High'].values[-self.lookback_period:]
        self.low_history = df['Low'].values[-self.lookback_period:]
        self.volume_history = df['Volume'].values[-self.lookback_period:]
```

2. **Update `_update_price_history()` to step through real data instead of generating synthetic**

## Next Steps

1. **Replace synthetic data** with real options data from your CSV files
2. **Add Black-Scholes pricing** for realistic premium calculations
3. **Include Greeks** (Delta, Gamma, Theta, Vega) in state space
4. **Experiment with different indicators** - TA-Lib has 150+ options
5. **Add risk management** - max drawdown limits, position sizing

## Key Differences from DQN

- **No epsilon decay needed** - PPO handles exploration via entropy
- **Automatic hyperparameter tuning** - Stable-Baselines3 has good defaults
- **Better for complex state spaces** - 18D state with indicators
- **Built-in callbacks** - Easy logging, checkpointing, evaluation
- **TensorBoard integration** - Real-time training visualization

## Troubleshooting

**TA-Lib installation issues:**
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu/Debian
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**"Environment not compatible" error:**
The `check_env()` function validates gym compatibility. Check that your observation/action spaces are correctly defined.

## Performance Tips

1. **Increase training timesteps** - 100k is minimum, try 500k-1M for better results
2. **Tune hyperparameters** - Especially `n_steps`, `batch_size`, `learning_rate`
3. **Use real data** - Synthetic random walk doesn't capture market dynamics
4. **Add more features** - Greeks, implied volatility, order flow
5. **Normalize rewards** - Current implementation normalizes by initial balance
