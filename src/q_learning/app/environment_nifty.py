"""
NIFTY Options Trading Environment
Specialized for naked options trading on NIFTY with limited strike prices.

Strategy Constraints:
- Only 5 strikes: 2 OTM, ATM, 2 ITM (both CE and PE)
- Naked options only (no spreads/combinations)
- NIFTY-specific features (India VIX, weekly expiry)
- Transaction costs: NSE brokerage + STT + exchange fees
"""

import gym
import numpy as np
import talib
from typing import Optional, Dict, List, Tuple


class NiftyOptionsEnv(gym.Env):
    """
    NIFTY Options Trading Environment with realistic constraints.

    Key Features:
    - 5 strike prices only (2 OTM, ATM, 2 ITM)
    - Separate CE and PE for each strike (10 total options)
    - India VIX for volatility
    - Weekly expiry handling (Thursday)
    - Realistic transaction costs
    """

    def __init__(
        self,
        initial_balance: float = 100000,  # 1 lakh starting capital
        max_positions: int = 3,  # Max concurrent positions (risk management)
        lot_size: int = 50,  # NIFTY lot size
        strike_interval: int = 50,  # NIFTY strike intervals
        lookback_period: int = 100,
        data_file: Optional[str] = None,
    ):
        super(NiftyOptionsEnv, self).__init__()

        # Trading parameters
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.lot_size = lot_size
        self.strike_interval = strike_interval
        self.lookback_period = lookback_period

        # Transaction costs (realistic for Indian market)
        self.brokerage_per_trade = 20  # Flat ₹20 per executed order
        self.stt_percentage = 0.05 / 100  # 0.05% on sell side (options)
        self.exchange_fees = 0.053 / 100  # 0.053% NSE charges
        self.gst = 0.18  # 18% GST on brokerage + exchange fees

        # Strike prices (relative to ATM)
        # We'll have 5 strikes for CE and 5 for PE = 10 options total
        self.strike_offsets = [-2, -1, 0, 1, 2]  # In terms of strike_interval

        # State space dimension:
        # Basic: 12 (balance, positions, nifty_price, etc.)
        # TA-Lib: 12 (same as before)
        # Option-specific: 10 (greeks/premiums for each strike)
        # Total: 34 dimensions
        self.state_dim = 34

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space:
        # For each of 5 strikes: Buy CE, Sell CE, Buy PE, Sell PE, Hold
        # Total: 5 actions per strike × 5 strikes = 25 actions
        # But to keep it simpler: 11 actions (one per option + hold)
        # 0: Hold
        # 1-5: Sell CE for strikes [-2, -1, 0, 1, 2]
        # 6-10: Sell PE for strikes [-2, -1, 0, 1, 2]
        # (Naked selling only - most common NIFTY strategy)
        self.action_space = gym.spaces.Discrete(11)

        # Market data buffers
        self.nifty_history = np.zeros(lookback_period)
        self.high_history = np.zeros(lookback_period)
        self.low_history = np.zeros(lookback_period)
        self.volume_history = np.zeros(lookback_period)
        self.vix_history = np.zeros(lookback_period)  # India VIX

        # Episode tracking
        self.current_step = 0
        self.max_steps = 5  # Weekly expiry - 5 trading days

        # Load data if provided
        if data_file:
            self._load_real_data(data_file)

        self.reset()

    def _load_real_data(self, data_file: str):
        """Load real NIFTY data from CSV."""
        import pandas as pd
        df = pd.read_csv(data_file)

        # Expecting columns: Date, Open, High, Low, Close, Volume, VIX
        if len(df) >= self.lookback_period:
            self.nifty_history = df['Close'].values[-self.lookback_period:]
            self.high_history = df['High'].values[-self.lookback_period:]
            self.low_history = df['Low'].values[-self.lookback_period:]
            self.volume_history = df['Volume'].values[-self.lookback_period:]

            # VIX data (India VIX)
            if 'VIX' in df.columns:
                self.vix_history = df['VIX'].values[-self.lookback_period:]
            else:
                # Estimate VIX from price volatility if not available
                self.vix_history = self._estimate_vix_from_prices()

    def _estimate_vix_from_prices(self) -> np.ndarray:
        """Estimate volatility index from price history."""
        returns = np.diff(np.log(self.nifty_history))
        rolling_vol = np.array([
            np.std(returns[max(0, i-20):i+1]) * np.sqrt(252) * 100
            for i in range(len(returns))
        ])
        return np.concatenate([[rolling_vol[0]], rolling_vol])

    def _generate_synthetic_nifty_data(self):
        """Generate synthetic NIFTY price data for testing."""
        # Start at typical NIFTY level
        base_price = 19500

        # Generate with realistic NIFTY volatility (~15-20% annual)
        daily_vol = 0.015  # 1.5% daily
        returns = np.random.randn(self.lookback_period) * daily_vol

        # Add slight positive drift (typical equity index behavior)
        drift = 0.0002
        returns += drift

        prices = base_price * np.exp(np.cumsum(returns))

        self.nifty_history = prices
        self.high_history = prices * (1 + np.abs(np.random.randn(self.lookback_period) * 0.005))
        self.low_history = prices * (1 - np.abs(np.random.randn(self.lookback_period) * 0.005))
        self.volume_history = np.random.randint(100000, 500000, self.lookback_period)

        # Realistic India VIX range: 10-35
        self.vix_history = np.clip(
            15 + np.random.randn(self.lookback_period) * 5,
            10, 35
        )

    def _get_atm_strike(self, nifty_price: float) -> float:
        """Get ATM strike price (rounded to nearest strike interval)."""
        return round(nifty_price / self.strike_interval) * self.strike_interval

    def _calculate_option_premium(
        self,
        strike: float,
        nifty_price: float,
        option_type: str,  # 'CE' or 'PE'
        days_to_expiry: int
    ) -> float:
        """
        Simplified premium calculation.
        In production, use Black-Scholes or real market data.
        """
        time_to_expiry = days_to_expiry / 365.0
        vix = self.vix_history[-1] / 100.0  # Convert to decimal

        # Intrinsic value
        if option_type == 'CE':
            intrinsic = max(0, nifty_price - strike)
        else:  # PE
            intrinsic = max(0, strike - nifty_price)

        # Time value (simplified - in reality use Black-Scholes)
        # Time value decreases as we approach expiry
        moneyness = abs(nifty_price - strike) / nifty_price
        time_value = nifty_price * vix * np.sqrt(time_to_expiry) * (1 - moneyness * 0.5)

        premium = intrinsic + max(0.5, time_value)  # Minimum ₹0.5

        return premium

    def _calculate_transaction_cost(self, premium: float) -> float:
        """Calculate total transaction cost per trade."""
        trade_value = premium * self.lot_size

        # Brokerage
        cost = self.brokerage_per_trade

        # STT (only on sell side, already applied)
        cost += trade_value * self.stt_percentage

        # Exchange fees
        exchange_fee = trade_value * self.exchange_fees
        cost += exchange_fee

        # GST on brokerage + exchange fees
        cost += (self.brokerage_per_trade + exchange_fee) * self.gst

        return cost

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate TA-Lib indicators on NIFTY."""
        indicators = {}

        current_price = self.nifty_history[-1]

        # VOLATILITY
        atr = talib.ATR(self.high_history, self.low_history, self.nifty_history, timeperiod=14)
        indicators['atr'] = atr[-1] if not np.isnan(atr[-1]) else 0

        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.nifty_history, timeperiod=20)
        indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price
        indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price
        indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price

        # MOMENTUM
        rsi = talib.RSI(self.nifty_history, timeperiod=14)
        indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50

        macd, macd_signal, macd_hist = talib.MACD(self.nifty_history)
        indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
        indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0

        slowk, slowd = talib.STOCH(self.high_history, self.low_history, self.nifty_history)
        indicators['stoch'] = slowk[-1] if not np.isnan(slowk[-1]) else 50

        # TREND
        sma = talib.SMA(self.nifty_history, timeperiod=20)
        indicators['sma'] = sma[-1] if not np.isnan(sma[-1]) else current_price

        ema = talib.EMA(self.nifty_history, timeperiod=20)
        indicators['ema'] = ema[-1] if not np.isnan(ema[-1]) else current_price

        adx = talib.ADX(self.high_history, self.low_history, self.nifty_history, timeperiod=14)
        indicators['adx'] = adx[-1] if not np.isnan(adx[-1]) else 0

        # VOLUME
        obv = talib.OBV(self.nifty_history, self.volume_history)
        indicators['obv'] = obv[-1] if not np.isnan(obv[-1]) else 0

        return indicators

    def _get_obs(self) -> np.ndarray:
        """Construct state observation."""
        nifty_price = self.nifty_history[-1]
        atm_strike = self._get_atm_strike(nifty_price)
        vix = self.vix_history[-1]

        # Calculate option premiums for all strikes
        ce_premiums = []
        pe_premiums = []

        for offset in self.strike_offsets:
            strike = atm_strike + (offset * self.strike_interval)

            ce_premium = self._calculate_option_premium(
                strike, nifty_price, 'CE', self.days_to_expiry
            )
            pe_premium = self._calculate_option_premium(
                strike, nifty_price, 'PE', self.days_to_expiry
            )

            ce_premiums.append(ce_premium)
            pe_premiums.append(pe_premium)

        # Calculate TA-Lib indicators
        indicators = self._calculate_indicators()

        # Build state vector (34 dimensions)
        obs = np.array([
            # Basic features (12)
            self.account_balance / self.initial_balance,  # Normalized balance
            len(self.positions) / self.max_positions,     # Position utilization
            nifty_price / 20000.0,                        # Normalized NIFTY
            atm_strike / 20000.0,                         # Normalized ATM
            self.days_to_expiry / 5.0,                    # Normalized days
            vix / 30.0,                                   # Normalized VIX
            self.total_pnl / self.initial_balance,        # Normalized P&L
            (nifty_price - atm_strike) / self.strike_interval,  # ATM offset
            self.max_loss_today / self.initial_balance if hasattr(self, 'max_loss_today') else 0,
            self.max_profit_today / self.initial_balance if hasattr(self, 'max_profit_today') else 0,
            self.consecutive_losses / 5.0 if hasattr(self, 'consecutive_losses') else 0,
            self.consecutive_wins / 5.0 if hasattr(self, 'consecutive_wins') else 0,

            # TA-Lib indicators (12)
            indicators['atr'] / nifty_price,
            (nifty_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) if (indicators['bb_upper'] - indicators['bb_lower']) > 0 else 0.5,
            indicators['rsi'] / 100.0,
            indicators['macd'] / nifty_price,
            indicators['macd_hist'] / nifty_price,
            indicators['stoch'] / 100.0,
            (nifty_price - indicators['sma']) / nifty_price,
            (nifty_price - indicators['ema']) / nifty_price,
            indicators['adx'] / 100.0,
            indicators['obv'] / 1e6,
            (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] if indicators['bb_middle'] > 0 else 0,
            vix / self.vix_history[:-1].mean() if len(self.vix_history) > 1 else 1.0,  # VIX ratio

            # Option premiums (10) - normalized
            *[p / 100.0 for p in ce_premiums],  # 5 CE premiums
            *[p / 100.0 for p in pe_premiums],  # 5 PE premiums
        ], dtype=np.float32)

        return obs

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.account_balance = self.initial_balance
        self.positions: List[Dict] = []
        self.days_to_expiry = 5  # Weekly expiry
        self.current_step = 0
        self.total_pnl = 0
        self.max_loss_today = 0
        self.max_profit_today = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Generate new data
        self._generate_synthetic_nifty_data()

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step."""
        previous_balance = self.account_balance
        reward = 0

        nifty_price = self.nifty_history[-1]
        atm_strike = self._get_atm_strike(nifty_price)

        # Decode action
        if action == 0:  # Hold
            pass
        elif 1 <= action <= 5:  # Sell CE
            strike_idx = action - 1
            strike = atm_strike + (self.strike_offsets[strike_idx] * self.strike_interval)
            self._execute_trade('CE', 'sell', strike, nifty_price)
        elif 6 <= action <= 10:  # Sell PE
            strike_idx = action - 6
            strike = atm_strike + (self.strike_offsets[strike_idx] * self.strike_interval)
            self._execute_trade('PE', 'sell', strike, nifty_price)

        # Update market
        self.current_step += 1
        self.days_to_expiry -= 1
        self._update_market()

        # Update P&L for all positions
        self._update_positions_pnl(self.nifty_history[-1])

        # Handle expiry
        done = False
        if self.days_to_expiry == 0:
            self._settle_positions(self.nifty_history[-1])
            done = True

        # Calculate reward
        balance_change = self.account_balance - previous_balance
        reward = balance_change / self.initial_balance  # Normalized

        # Add penalties/bonuses
        if len(self.positions) > self.max_positions:
            reward -= 0.1  # Penalty for over-trading

        # Bankruptcy check
        if self.account_balance <= self.initial_balance * 0.5:  # 50% drawdown
            done = True
            reward -= 1.0  # Large penalty

        # Track max profit/loss
        self.max_profit_today = max(self.max_profit_today, balance_change)
        self.max_loss_today = min(self.max_loss_today, balance_change)

        info = {
            'account_balance': self.account_balance,
            'num_positions': len(self.positions),
            'days_remaining': self.days_to_expiry,
            'nifty_price': self.nifty_history[-1],
            'total_pnl': self.total_pnl,
        }

        return self._get_obs(), reward, done, info

    def _execute_trade(self, option_type: str, side: str, strike: float, nifty_price: float):
        """Execute a trade (naked sell only)."""
        if len(self.positions) >= self.max_positions:
            return  # Risk limit reached

        premium = self._calculate_option_premium(
            strike, nifty_price, option_type, self.days_to_expiry
        )

        transaction_cost = self._calculate_transaction_cost(premium)

        # For naked sell: receive premium - cost
        if side == 'sell':
            cash_flow = (premium * self.lot_size) - transaction_cost
            self.account_balance += cash_flow

            # Record position
            position = {
                'type': option_type,
                'side': side,
                'strike': strike,
                'entry_premium': premium,
                'entry_nifty': nifty_price,
                'lot_size': self.lot_size,
                'entry_day': self.current_step,
            }
            self.positions.append(position)

    def _update_positions_pnl(self, current_nifty: float):
        """Mark-to-market all positions."""
        total_mtm = 0

        for pos in self.positions:
            current_premium = self._calculate_option_premium(
                pos['strike'], current_nifty, pos['type'], self.days_to_expiry
            )

            # For short positions: profit if premium decreased
            if pos['side'] == 'sell':
                pnl = (pos['entry_premium'] - current_premium) * pos['lot_size']
                total_mtm += pnl

        self.total_pnl = total_mtm

    def _settle_positions(self, final_nifty: float):
        """Settle all positions at expiry."""
        for pos in self.positions:
            # Calculate intrinsic value at expiry
            if pos['type'] == 'CE':
                intrinsic = max(0, final_nifty - pos['strike'])
            else:  # PE
                intrinsic = max(0, pos['strike'] - final_nifty)

            # For short: pay intrinsic value
            if pos['side'] == 'sell':
                payout = intrinsic * pos['lot_size']

                # STT on expiry
                if intrinsic > 0:
                    stt = payout * self.stt_percentage
                    payout += stt * 1.18  # Including GST

                self.account_balance -= payout

        self.positions = []

    def _update_market(self):
        """Update market prices (synthetic)."""
        # Simple random walk
        current_price = self.nifty_history[-1]
        vix = self.vix_history[-1]

        # Daily volatility based on VIX
        daily_vol = (vix / 100.0) / np.sqrt(252)
        price_change = np.random.randn() * daily_vol * current_price

        new_price = current_price + price_change

        # Update histories
        self.nifty_history = np.roll(self.nifty_history, -1)
        self.nifty_history[-1] = new_price

        self.high_history = np.roll(self.high_history, -1)
        self.high_history[-1] = new_price * (1 + abs(np.random.randn() * 0.003))

        self.low_history = np.roll(self.low_history, -1)
        self.low_history[-1] = new_price * (1 - abs(np.random.randn() * 0.003))

        self.volume_history = np.roll(self.volume_history, -1)
        self.volume_history[-1] = np.random.randint(100000, 500000)

        # VIX mean reversion
        mean_vix = 17
        vix_change = (mean_vix - vix) * 0.1 + np.random.randn() * 2
        new_vix = np.clip(vix + vix_change, 10, 35)

        self.vix_history = np.roll(self.vix_history, -1)
        self.vix_history[-1] = new_vix

    def render(self, mode='human'):
        """Render current state."""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step} | Days to Expiry: {self.days_to_expiry}")
        print(f"NIFTY: {self.nifty_history[-1]:.2f} | VIX: {self.vix_history[-1]:.2f}")
        print(f"Balance: ₹{self.account_balance:.2f} | P&L: ₹{self.total_pnl:.2f}")
        print(f"Positions: {len(self.positions)}/{self.max_positions}")

        if self.positions:
            print(f"\nActive Positions:")
            for i, pos in enumerate(self.positions, 1):
                current_premium = self._calculate_option_premium(
                    pos['strike'], self.nifty_history[-1], pos['type'], self.days_to_expiry
                )
                pnl = (pos['entry_premium'] - current_premium) * pos['lot_size']
                print(f"  {i}. {pos['type']} {pos['strike']} {pos['side'].upper()} | "
                      f"Entry: ₹{pos['entry_premium']:.2f} | Current: ₹{current_premium:.2f} | "
                      f"P&L: ₹{pnl:.2f}")
        print(f"{'='*60}\n")
