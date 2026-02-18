## Project Overview

**smartrade** is an automated stock trading bot that combines LSTM neural networks for price prediction with reinforcement learning for trading decisions. The project implements two distinct approaches:

1. **LSTM-based Price Prediction** (`src/lstm/`) - Uses historical stock data and computed features to forecast future stock prices
2. **Reinforcement Learning Options Trading** (`src/q_learning/`) - Multiple RL implementations for options trading strategies

### Key Features by Module

**LSTM Module**: Production-ready price forecasting with configurable feature engineering and extensible architecture.

**Q-Learning Module**: Three progressive implementations:
- **Generic PPO/DQN** (environment_talib.py): 18D state space with TA-Lib indicators
- **NIFTY-Optimized PPO** (environment_nifty.py): 34D state space specialized for NIFTY weekly options, 11 actions (simplified)
- **NIFTY-Corrected PPO** (environment_nifty_corrected.py): 34D state space with 21 actions (realistic buy/sell/close mechanics)

## Architecture

### LSTM Module (`src/lstm/`)

**Design Pattern**: Follows SOLID principles with clear separation of concerns:

- **Configuration Layer** (`config.py`): Single source of truth for all parameters
  - Data extraction settings (Quandl API or local CSV)
  - Feature engineering configuration
  - Neural network hyperparameters
  - Training/testing parameters

- **Data Handler Layer** (`datahandler/`):
  - `DataHandler` (base class in `__init__.py`): Abstract interface
  - `StockDataHandler`: Handles data preprocessing, feature computation, and normalization/standardization
  - Supports custom feature functions via `EXTRA_FEATURES` in config

- **Network Layer** (`network/`):
  - `Network` (base class in `__init__.py`): Abstract interface with template methods
  - `LSTMNetwork`: Concrete implementation for time series forecasting
  - Extensible design allows adding new architectures by inheriting `Network`

- **Utilities** (`utils/`):
  - `compute_stock_features.py`: Predefined features (SMA, EMA, Bollinger Bands, Daily Returns)
  - Data transformations (normalize, standardize, window_transform_series)
  - Quandl API integration

**Data Flow**:
1. `main.py` orchestrates the pipeline
2. Data loaded via `get_datasets()` (Quandl API or local CSV)
3. `StockDataHandler` preprocesses data (computes features, normalizes)
4. `LSTMNetwork` creates windowed sequences for training
5. Model trains/loads weights from `models/` directory
6. Predictions are destandardized back to original scale

### Q-Learning Module (`src/q_learning/`)

**Three Progressive Implementations**:

1. **DQN (Deep Q-Network)** - Original baseline:
   - `environment.py`: Basic environment (5D state)
   - `agent.py`: DQN with experience replay
   - Note: Has compatibility issues, mainly kept for reference

2. **Generic PPO** - TA-Lib enhanced (18D state):
   - `environment_talib.py`: Generic options trading environment
   - `agent_ppo.py`: PPO using Stable-Baselines3
   - `train_ppo.py`: Training/evaluation script
   - State includes: Basic features (6) + TA-Lib indicators (12): ATR, Bollinger Bands, RSI, MACD, Stochastic, SMA, EMA, ADX, OBV
   - Actions: 6 discrete (Buy/Sell/Hold for CE/PE)

3. **NIFTY-Specific PPO** - Production-optimized for NIFTY weekly options (34D state):
   - `environment_nifty.py`: Simplified (11 actions) - faster training
   - `environment_nifty_corrected.py`: Realistic (21 actions) - proper buy/sell/close mechanics
   - `agent_nifty_ppo.py`: Hyperparameters tuned for naked options selling
   - `train_nifty.py`: Complete training/evaluation/backtesting pipeline

   **NIFTY State Space (34 dimensions)**:
   - Basic (12): balance, positions, NIFTY price, ATM strike, days to expiry, India VIX, P&L metrics, win/loss streaks
   - TA-Lib indicators (12): Same as generic PPO
   - Option premiums (10): 5 CE + 5 PE premiums across strikes

   **Strike Selection**: Limited to 5 strikes (2 OTM, ATM, 2 ITM)
   **Transaction Costs**: Indian market realistic (NSE, STT, GST)
   **Optimized Hyperparameters**: lr=1e-4, high entropy (0.02), large batch (128), deep network [128,128,64]
   **Action Masking**: Enabled by default - filters invalid actions at each step for 30-50% faster training

**Choosing an Implementation**:
- Use **Generic PPO** for: Learning RL basics, non-NIFTY instruments, rapid prototyping
- Use **NIFTY Simplified** (11 actions) for: Faster training, initial experiments with NIFTY
- Use **NIFTY Corrected** (21 actions) for: Production, realistic position management, stakeholder demos

**Action Masking Benefits**:
- Prevents agent from attempting invalid actions (e.g., buying when no position exists)
- 30-50% faster convergence compared to unmasked training
- Better sample efficiency and more stable learning
- Already implemented in `environment_nifty.py` via `get_action_mask()`
- Automatically enabled when using `OptimizedNiftyPPO` agent

### Data Scraping (`src/data/`)

Contains web scraping scripts for Indian stock market data from MoneyControl:
- `scrape.py`: Extracts financial statements (P&L, Balance Sheet, Ratios)
- `moneycontrol-scrape.py`, `crawler.py`: Additional data collection utilities

## Common Commands

### Environment Setup

**For LSTM Module:**
```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r src/lstm/requirements.txt
```

**For Q-Learning Module (PPO/DQN):**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (includes TA-Lib and sb3-contrib for action masking)
cd src/q_learning
pip install -r requirements.txt

# Note: TA-Lib requires C library installation first
# macOS: brew install ta-lib
# Ubuntu: sudo apt-get install ta-lib

# Test action masking implementation (optional)
python test_action_masking.py
```

### Running LSTM Price Prediction
```bash
cd src/lstm
python main.py
```

### Running Q-Learning Trading Bots

**Generic PPO (Recommended for learning):**
```bash
cd src/q_learning/app

# Quick demo - compare random vs PPO
python train_ppo.py

# Train new PPO agent
python train_ppo.py --train --timesteps 100000

# Evaluate trained agent
python train_ppo.py --evaluate --model-path ./ppo_models/ppo_trading_agent_final

# Monitor training with TensorBoard
tensorboard --logdir=./ppo_trading_tensorboard/
```

**NIFTY-Specific PPO (Recommended for production):**
```bash
cd src/q_learning/app

# Quick comparison with baselines
python train_nifty.py --compare

# Train NIFTY agent (25-30 min for 250k steps)
python train_nifty.py --train --timesteps 250000

# Train with real data
python train_nifty.py --train --data-file nifty_historical.csv

# Evaluate trained agent
python train_nifty.py --evaluate --episodes 20

# Backtest on historical data
python train_nifty.py --backtest --data-file nifty_historical.csv

# Monitor training
tensorboard --logdir=./nifty_ppo_logs/
```

**Compare DQN vs PPO:**
```bash
cd src/q_learning/app
python compare_agents.py --timesteps 50000
```

**DQN (Legacy, not recommended):**
```bash
cd src/q_learning/app
python agent.py  # Note: Has compatibility issues with environment
```

### Configuration

**LSTM Module** (`src/lstm/config.py`):
- `MODE`: Set to 'LOCAL' or 'QUANDL' for data source
- `QUANDL_KEY`: Add API key to `src/lstm/secrets.txt` if using Quandl: `{'quandl_key': 'YOUR_KEY_HERE'}`
- `STOCKS`: List of tickers to process (format: `'EOD/TICKER'`)
- `TRAIN_NETWORK`: Set to `True` to train new models, `False` to use existing weights
- `VISUALIZE`: Set to `True` to plot predictions

**Q-Learning Module**:
- Generic PPO: Uses synthetic data by default, modify `environment_talib.py` to use real data
- NIFTY PPO: Configure via command-line args or modify parameters in `environment_nifty.py`:
  - `initial_balance`: Starting capital (default: ₹100,000)
  - `max_positions`: Max concurrent positions (default: 3)
  - `lot_size`: NIFTY lot size (default: 50)
  - `strike_offsets`: Strike selection (default: [-2, -1, 0, 1, 2] → [ATM-100, ATM-50, ATM, ATM+50, ATM+100])

### Key Configuration Parameters

**LSTM Feature Engineering** (`src/lstm/config.py`):
- `REL_DATA_COLUMNS`: Raw features from dataset (e.g., `['Adj_Close', 'Volume']`)
- `REL_PREDEFINED_FEATURES`: Built-in features (e.g., `['sma', 'daily_returns']`)
- `EXTRA_FEATURES`: Dict mapping custom feature names to function pointers
- `FEATURE_TO_PREDICT`: Target variable (usually `['Adj_Close']`)

**LSTM Neural Network**:
- `WINDOW_SIZE`: Time steps in each training sequence (default: 5)
- `NUM_LAYERS`: Number of LSTM layers (default: 2)
- `NUM_CELLS_LSTM`: Hidden units per layer (default: 100)
- `LSTM_DROPOUT`: Dropout rate (default: 0.1)
- `LSTM_EPOCHS`: Training epochs (default: 50)
- `LSTM_PATIENCE`: Early stopping patience (default: 15)

**NIFTY PPO Hyperparameters** (tuned for naked options selling):
- `learning_rate`: 1e-4 (conservative for high variance P&L)
- `n_steps`: 512 (moderate exploration)
- `batch_size`: 128 (large to smooth noisy gradients)
- `n_epochs`: 20 (more than default for complex state space)
- `ent_coef`: 0.02 (high entropy to explore different strikes)
- `policy_kwargs`: `{'net_arch': [128, 128, 64]}` (deep network for option dynamics)

## Development Notes

### LSTM Module
- Model weights saved to `src/lstm/models/` with naming pattern `stock_{TICKER}_weights.h5`
- Training data automatically split using `LSTM_TRAIN_TEST_SPLIT` ratio (default: 0.85)
- Visualizations show: black line (actual), blue (training fit), red (testing fit)
- All data is standardized during training; predictions are destandardized before returning
- Add custom features via `EXTRA_FEATURES` dict in config (function pointer mapping)

### Q-Learning Module

**Action Space Clarification**:
- **NIFTY Simplified** (11 actions): Hold + 5 strikes × (Sell CE, Sell PE). Sell action toggles position (simplified).
- **NIFTY Corrected** (21 actions): Hold + 5 strikes × (Sell CE, Buy CE, Sell PE, Buy PE). Realistic open/close mechanics.
- **Only SHORT positions supported**: The "Buy" actions close short positions, not open long positions
- See `src/q_learning/ACTION_SPACE_COMPARISON.md` for detailed comparison

**Action Masking Implementation**:
- NIFTY environments include `get_action_mask()` method that returns valid actions
- `OptimizedNiftyPPO` uses `MaskablePPO` from `sb3-contrib` (not regular PPO)
- ActionMasker wrapper applied automatically when `use_action_masking=True` (default)
- Prevents invalid actions: can't buy without position, can't sell when already holding, can't exceed max_positions
- Expected training improvement: 30-50% faster convergence, more stable learning
- To disable (not recommended): `OptimizedNiftyPPO(env, use_action_masking=False)`

**Data Requirements for NIFTY**:
- CSV format: `Date,Open,High,Low,Close,Volume,VIX`
- Close = NIFTY spot price
- VIX = India VIX (estimated from volatility if unavailable)
- Need at least 100 rows for lookback period

**Performance Expectations** (with real data):
- Target Win Rate: 60-70%
- Target Weekly P&L: ₹2,000-5,000 (2-5% ROI on ₹100k capital)
- Target Sharpe Ratio: > 2.0
- Max Acceptable Drawdown: < 20%

### Adding New Network Architectures (LSTM)
1. Create new class inheriting from `network.Network`
2. Implement abstract methods: `set_train_test_split`, `build_model`, `train_model`, `forecast_model`, `visualize_output`
3. Use in `main.py` instead of `LSTMNetwork`

## File Organization

```
src/
├── lstm/                           # LSTM price prediction module
│   ├── config.py                   # All configuration parameters
│   ├── main.py                     # Entry point
│   ├── secrets.txt                 # API keys (gitignored)
│   ├── data/                       # Stock CSV files
│   ├── models/                     # Saved model weights (.h5)
│   ├── datahandler/                # Data preprocessing
│   │   ├── __init__.py            # DataHandler base class
│   │   └── stock.py               # StockDataHandler implementation
│   ├── network/                    # Neural network implementations
│   │   ├── __init__.py            # Network base class
│   │   └── LSTMNetwork.py         # LSTM implementation
│   └── utils/                      # Helper functions
│       ├── compute_stock_features.py  # Feature engineering
│       └── __init__.py            # Data transformations, Quandl API
│
├── q_learning/                     # Reinforcement learning module
│   ├── requirements.txt            # Dependencies (gym, tensorflow, ta-lib, stable-baselines3, sb3-contrib)
│   ├── test_action_masking.py      # Test script for action masking validation
│   ├── README.md                   # Module documentation
│   ├── NIFTY_README.md            # NIFTY-specific guide
│   ├── HYPERPARAMETERS_EXPLAINED.md  # Deep dive into hyperparameters
│   ├── ACTION_SPACE_COMPARISON.md  # 11 vs 21 action space comparison
│   ├── QUICKSTART.txt             # Quick reference card
│   └── app/
│       ├── environment.py          # Original DQN environment (5D, legacy)
│       ├── agent.py                # DQN agent (legacy, has issues)
│       ├── agent_batch.py          # Batch DQN training
│       ├── environment_talib.py    # Generic PPO environment (18D)
│       ├── agent_ppo.py            # Generic PPO agent
│       ├── train_ppo.py            # Generic PPO training/evaluation
│       ├── environment_nifty.py    # NIFTY environment, 21 actions (34D, with action masking)
│       ├── environment_nifty_corrected.py  # NIFTY environment, 21 actions (34D, alternative)
│       ├── agent_nifty_ppo.py      # NIFTY-optimized MaskablePPO agent
│       ├── train_nifty.py          # NIFTY training/evaluation/backtest
│       └── compare_agents.py       # DQN vs PPO comparison
│
└── data/                           # Web scraping utilities
    ├── scrape.py                   # MoneyControl financial statements scraper
    ├── moneycontrol-scrape.py      # Additional scraping utilities
    └── crawler.py                  # Web crawler

Additional Documentation:
├── CLAUDE.md                       # This file
├── README.md                       # Project overview
└── IMPLEMENTATION_SUMMARY.md       # TA-Lib & PPO implementation details
```

## Important Constraints and Notes

### Security & Best Practices
- **Secrets Management**: Never commit `secrets.txt` with real API keys
- **Data Mode**: Ensure `MODE` in config matches your data source (LOCAL vs QUANDL)
- **Python Version**: Developed with Python 3.x
- **TA-Lib**: Requires C library installation before pip install (see Environment Setup)

### Known Issues
- **DQN Agent** (`agent.py`): Has compatibility issues, mainly kept for reference. Use PPO implementations instead.
- **LSTM API**: Uses older Keras API (pre-TensorFlow 2.0 style in some places)
- **Synthetic Data**: Q-learning modules use random walk by default. Replace with real market data for production.

### Production Readiness
- **LSTM Module**: Production-ready for price prediction tasks
- **Generic PPO**: Suitable for prototyping and learning RL
- **NIFTY PPO**: Requires real data, backtesting, and paper trading before live deployment
- **Risk Warning**: Options trading involves significant risk. Always paper trade first and never risk more than you can afford to lose.

### Documentation References
For deep dives into specific topics:
- General Q-learning setup: `src/q_learning/README.md`
- NIFTY-specific implementation: `src/q_learning/NIFTY_README.md`
- Hyperparameter tuning rationale: `src/q_learning/HYPERPARAMETERS_EXPLAINED.md`
- Action space design decisions: `src/q_learning/ACTION_SPACE_COMPARISON.md`
- Implementation history: `IMPLEMENTATION_SUMMARY.md`
