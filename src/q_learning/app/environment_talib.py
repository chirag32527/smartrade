import gym
import numpy as np
import talib

class OptionTradingEnvWithTALib(gym.Env):
    """
    Enhanced options trading environment with TA-Lib technical indicators.

    State space includes:
    - Basic option data (strike, premium, balance, etc.)
    - Volatility indicators (ATR, Bollinger Bands)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Trend indicators (SMA, EMA, ADX)
    - Volume indicators (OBV, Volume MA)
    """

    def __init__(self, lookback_period=100, initial_balance=10000):
        """
        Args:
            lookback_period: Number of historical bars to maintain for indicator calculation
            initial_balance: Starting account balance
        """
        super(OptionTradingEnvWithTALib, self).__init__()

        # Constants
        self.lookback_period = lookback_period
        self.initial_balance = initial_balance
        self.lot_size = 50
        self.brokerage_penalty = 59
        self.expiry_stt_percentage_penalty = 0.125 / 100

        # State dimension calculation:
        # 6 basic features + 12 TA-Lib indicators = 18 features
        self.state_dim = 18

        # Observation space: [0, inf) for most features, [-100, 100] for some indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space: 0=Buy PE, 1=Sell PE, 2=Hold PE, 3=Buy CE, 4=Sell CE, 5=Hold CE
        self.action_space = gym.spaces.Discrete(6)

        # Initialize price history buffers for TA-Lib
        self.price_history = np.zeros(lookback_period)
        self.high_history = np.zeros(lookback_period)
        self.low_history = np.zeros(lookback_period)
        self.volume_history = np.zeros(lookback_period)

        # Current step in episode
        self.current_step = 0
        self.max_steps = 30  # 30 days to expiry

        # Reset to initialize state
        self.reset()

    def _generate_synthetic_price_data(self):
        """
        Generate synthetic price movements for testing.
        In production, replace this with real market data.
        """
        # Simple random walk with drift
        returns = np.random.randn(self.lookback_period) * 0.02  # 2% daily volatility
        base_price = 100
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC
        self.price_history = prices
        self.high_history = prices * (1 + np.abs(np.random.randn(self.lookback_period) * 0.01))
        self.low_history = prices * (1 - np.abs(np.random.randn(self.lookback_period) * 0.01))
        self.volume_history = np.random.randint(1000, 10000, self.lookback_period)

    def _update_price_history(self):
        """
        Update price history by one step (simulating next bar).
        In production, this would fetch next real data point.
        """
        # Shift history left and add new synthetic price
        new_return = np.random.randn() * 0.02
        new_price = self.price_history[-1] * np.exp(new_return)

        self.price_history = np.roll(self.price_history, -1)
        self.price_history[-1] = new_price

        self.high_history = np.roll(self.high_history, -1)
        self.high_history[-1] = new_price * (1 + abs(np.random.randn() * 0.01))

        self.low_history = np.roll(self.low_history, -1)
        self.low_history[-1] = new_price * (1 - abs(np.random.randn() * 0.01))

        self.volume_history = np.roll(self.volume_history, -1)
        self.volume_history[-1] = np.random.randint(1000, 10000)

    def _calculate_indicators(self):
        """
        Calculate all TA-Lib technical indicators.
        Returns dict with indicator values.
        """
        indicators = {}

        # Current price for reference
        current_price = self.price_history[-1]

        # VOLATILITY INDICATORS
        # ATR - Average True Range (14-period)
        atr = talib.ATR(self.high_history, self.low_history, self.price_history, timeperiod=14)
        indicators['atr'] = atr[-1] if not np.isnan(atr[-1]) else 0

        # Bollinger Bands (20-period, 2 std dev)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            self.price_history, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price
        indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price
        indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price
        indicators['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] if not np.isnan(bb_middle[-1]) and bb_middle[-1] != 0 else 0

        # MOMENTUM INDICATORS
        # RSI - Relative Strength Index (14-period)
        rsi = talib.RSI(self.price_history, timeperiod=14)
        indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50

        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = talib.MACD(
            self.price_history, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
        indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
        indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0

        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            self.high_history, self.low_history, self.price_history,
            fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        indicators['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else 50

        # TREND INDICATORS
        # Simple Moving Average (20-period)
        sma = talib.SMA(self.price_history, timeperiod=20)
        indicators['sma'] = sma[-1] if not np.isnan(sma[-1]) else current_price

        # Exponential Moving Average (20-period)
        ema = talib.EMA(self.price_history, timeperiod=20)
        indicators['ema'] = ema[-1] if not np.isnan(ema[-1]) else current_price

        # ADX - Average Directional Index (trend strength)
        adx = talib.ADX(self.high_history, self.low_history, self.price_history, timeperiod=14)
        indicators['adx'] = adx[-1] if not np.isnan(adx[-1]) else 0

        # VOLUME INDICATORS
        # OBV - On Balance Volume
        obv = talib.OBV(self.price_history, self.volume_history)
        indicators['obv'] = obv[-1] if not np.isnan(obv[-1]) else 0

        return indicators

    def _get_obs(self):
        """
        Construct observation vector from current state + TA-Lib indicators.
        """
        current_price = self.price_history[-1]
        indicators = self._calculate_indicators()

        # Normalize/scale features for better learning
        obs = np.array([
            # Basic features (6)
            self.difference_base_strike_price / 100.0,  # Normalized
            self.premium_price / 10.0,                   # Normalized
            len(self.open_trades) / 10.0,                # Normalized
            self.account_balance / self.initial_balance, # Normalized
            self.days_remaining_to_expiry / 30.0,        # Normalized
            current_price / 100.0,                       # Normalized underlying price

            # Volatility indicators (5)
            indicators['atr'] / current_price,           # Normalized ATR
            (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) if (indicators['bb_upper'] - indicators['bb_lower']) > 0 else 0.5,  # BB position
            indicators['bb_width'],                      # BB width

            # Momentum indicators (4)
            indicators['rsi'] / 100.0,                   # RSI normalized to [0,1]
            indicators['macd'] / current_price,          # MACD normalized
            indicators['macd_hist'] / current_price,     # MACD histogram normalized
            indicators['stoch_k'] / 100.0,               # Stochastic normalized to [0,1]

            # Trend indicators (3)
            (current_price - indicators['sma']) / current_price,  # Distance from SMA
            (current_price - indicators['ema']) / current_price,  # Distance from EMA
            indicators['adx'] / 100.0,                   # ADX normalized

            # Volume indicators (1)
            indicators['obv'] / 1e6,                     # OBV scaled down
        ], dtype=np.float32)

        return obs

    def reset(self):
        """Reset environment to initial state."""
        # Reset account state
        self.difference_base_strike_price = 100
        self.premium_price = 5
        self.open_trades = []
        self.account_balance = self.initial_balance
        self.days_remaining_to_expiry = 30
        self.current_step = 0

        # Generate new price history
        self._generate_synthetic_price_data()

        return self._get_obs()

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Integer from 0-5 representing trading action

        Returns:
            observation, reward, done, info
        """
        # Store previous balance for reward calculation
        previous_balance = self.account_balance

        # Execute the action (same logic as original environment)
        if action == 0:  # buy pe
            sell_order_found = False
            for trade in self.open_trades:
                if trade[0] == "pe" and trade[1] == self.difference_base_strike_price and trade[3] == "sell":
                    profit = (trade[2] - self.premium_price) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    sell_order_found = True
                    break

            if not sell_order_found:
                self.open_trades.append(("pe", self.difference_base_strike_price, self.premium_price, "buy"))

        elif action == 1:  # sell pe
            buy_order_found = False
            for trade in self.open_trades:
                if trade[0] == "pe" and trade[1] == self.difference_base_strike_price and trade[3] == "buy":
                    profit = (self.premium_price - trade[2]) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    buy_order_found = True
                    break
            if not buy_order_found:
                self.open_trades.append(("pe", self.difference_base_strike_price, self.premium_price, "sell"))

        elif action == 3:  # buy ce
            sell_order_found = False
            for trade in self.open_trades:
                if trade[0] == "ce" and trade[1] == self.difference_base_strike_price and trade[3] == "sell":
                    profit = (trade[2] - self.premium_price) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    sell_order_found = True
                    break

            if not sell_order_found:
                self.open_trades.append(("ce", self.difference_base_strike_price, self.premium_price, "buy"))

        elif action == 4:  # sell ce
            buy_order_found = False
            for trade in self.open_trades:
                if trade[0] == "ce" and trade[1] == self.difference_base_strike_price and trade[3] == "buy":
                    profit = (self.premium_price - trade[2]) * self.lot_size
                    self.account_balance += profit
                    self.account_balance -= self.brokerage_penalty
                    self.open_trades.remove(trade)
                    buy_order_found = True
                    break

            if not buy_order_found:
                self.open_trades.append(("ce", self.difference_base_strike_price, self.premium_price, "sell"))

        elif action in [2, 5]:  # hold ce, pe
            pass

        # Update environment state
        self.current_step += 1
        self.days_remaining_to_expiry -= 1
        self._update_price_history()  # Simulate market movement

        # Update premium based on price movement (simplified)
        current_price = self.price_history[-1]
        self.premium_price = max(0.5, self.premium_price + np.random.randn() * 0.5)

        # Handle expiry
        if self.days_remaining_to_expiry == 0 and len(self.open_trades) > 0:
            for trade in self.open_trades:
                if trade[3] == "buy":
                    profit = (self.premium_price - trade[2]) * self.lot_size
                else:  # sell order
                    profit = (trade[2] - self.premium_price) * self.lot_size

                self.account_balance += profit
                self.account_balance -= self.brokerage_penalty
                stt_penalty_amount = profit * self.expiry_stt_percentage_penalty
                self.account_balance -= (1.18 * stt_penalty_amount)
            self.open_trades = []

        # Calculate reward (incremental, not just final balance)
        balance_change = self.account_balance - previous_balance
        reward = balance_change / self.initial_balance  # Normalized reward

        # Check if episode is done
        done = self.days_remaining_to_expiry == 0 or self.account_balance <= 0

        # Additional info for debugging/logging
        info = {
            'account_balance': self.account_balance,
            'num_open_trades': len(self.open_trades),
            'days_remaining': self.days_remaining_to_expiry,
            'current_price': current_price,
        }

        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        """Print current state for debugging."""
        print(f"Step: {self.current_step}")
        print(f"Current Price: {self.price_history[-1]:.2f}")
        print(f"Strike Price Diff: {self.difference_base_strike_price}")
        print(f"Premium Price: {self.premium_price:.2f}")
        print(f"Open Trades: {len(self.open_trades)}")
        print(f"Account Balance: {self.account_balance:.2f}")
        print(f"Days to Expiry: {self.days_remaining_to_expiry}")
        print("-" * 50)
