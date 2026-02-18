"""
NIFTY Options Environment with FULL Position Support (Long + Short)

NEW: Supports BOTH Long and Short positions!

Action Space: 41 actions (8 per strike + 1 hold)
  - For each of 5 strikes:
    - Buy CE to OPEN (go long call)
    - Sell CE to CLOSE (exit long call)
    - Sell CE to OPEN (go short call - naked selling)
    - Buy CE to CLOSE (exit short call)
    - Buy PE to OPEN (go long put)
    - Sell PE to CLOSE (exit long put)
    - Sell PE to OPEN (go short put - naked selling)
    - Buy PE to CLOSE (exit short put)
  - Plus 1 global Hold action

Total: 5 strikes × 8 actions + 1 hold = 41 actions

Position Tracking:
- Each position now has 'direction': 'long' or 'short'
- 'side' indicates the opening action: 'buy' or 'sell'
"""

import gym
import numpy as np
import talib
from typing import Optional, Dict, List, Tuple


class NiftyOptionsEnv(gym.Env):
    """
    NIFTY Options Trading Environment with FULL Long + Short Support.

    Action Space (41 discrete actions):
      0: Hold (do nothing)

      For each strike [-2, -1, 0, +1, +2]:
        Buy CE to OPEN, Sell CE to CLOSE, Sell CE to OPEN, Buy CE to CLOSE,
        Buy PE to OPEN, Sell PE to CLOSE, Sell PE to OPEN, Buy PE to CLOSE

      Example for Strike 0 (ATM):
        Action 1: Buy CE @ ATM to OPEN (go long)
        Action 2: Sell CE @ ATM to CLOSE (exit long)
        Action 3: Sell CE @ ATM to OPEN (go short)
        Action 4: Buy CE @ ATM to CLOSE (exit short)
        Action 5: Buy PE @ ATM to OPEN (go long)
        Action 6: Sell PE @ ATM to CLOSE (exit long)
        Action 7: Sell PE @ ATM to OPEN (go short)
        Action 8: Buy PE @ ATM to CLOSE (exit short)
    """

    def __init__(
        self,
        initial_balance: float = 100000,
        max_positions: int = 3,
        lot_size: int = 50,
        strike_interval: int = 50,
        lookback_period: int = 100,
        data_file: Optional[str] = None,
    ):
        super(NiftyOptionsEnv, self).__init__()

        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.lot_size = lot_size
        self.strike_interval = strike_interval
        self.lookback_period = lookback_period

        # Transaction costs
        # TODO: bound to change with time and events around
        self.brokerage_per_trade = 20
        self.stt_percentage = 0.05 / 100
        self.exchange_fees = 0.053 / 100
        self.gst = 0.18

        # Strike offsets
        self.strike_offsets = [-2, -1, 0, 1, 2]

        # State dimension
        self.state_dim = 34

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # EXPANDED ACTION SPACE: 41 actions
        # 0: Hold
        # For each strike (5 strikes):
        #   1-8: Strike -2 (Buy CE Open, Sell CE Close, Sell CE Open, Buy CE Close,
        #                   Buy PE Open, Sell PE Close, Sell PE Open, Buy PE Close)
        #   9-16: Strike -1 (same 8 actions)
        #   17-24: Strike 0 (same 8 actions)
        #   25-32: Strike +1 (same 8 actions)
        #   33-40: Strike +2 (same 8 actions)
        self.action_space = gym.spaces.Discrete(41)

        # Market data buffers
        self.nifty_history = np.zeros(lookback_period)
        self.high_history = np.zeros(lookback_period)
        self.low_history = np.zeros(lookback_period)
        self.volume_history = np.zeros(lookback_period)
        self.vix_history = np.zeros(lookback_period)

        self.current_step = 0
        self.max_steps = 5

        if data_file:
            self._load_real_data(data_file)

        self.reset()

    def _decode_action(self, action: int) -> Optional[Tuple[str, str, str, int]]:
        """
        Decode action into (option_type, side, direction, strike_index).

        Args:
            action: Integer from 0-40

        Returns:
            Tuple of (option_type, side, direction, strike_index) or None for Hold
            - option_type: 'CE' or 'PE'
            - side: 'buy' or 'sell' (the action being taken NOW)
            - direction: 'long' or 'short' (the position type)
            - strike_index: 0-4 (which strike offset)

        Example:
            action=0  → None (Hold)
            action=1  → ('CE', 'buy', 'long', 0)   # Buy CE to OPEN long at strike -2
            action=2  → ('CE', 'sell', 'long', 0)  # Sell CE to CLOSE long at strike -2
            action=3  → ('CE', 'sell', 'short', 0) # Sell CE to OPEN short at strike -2
            action=4  → ('CE', 'buy', 'short', 0)  # Buy CE to CLOSE short at strike -2
            action=5  → ('PE', 'buy', 'long', 0)   # Buy PE to OPEN long at strike -2
            action=6  → ('PE', 'sell', 'long', 0)  # Sell PE to CLOSE long at strike -2
            action=7  → ('PE', 'sell', 'short', 0) # Sell PE to OPEN short at strike -2
            action=8  → ('PE', 'buy', 'short', 0)  # Buy PE to CLOSE short at strike -2
            action=9  → ('CE', 'buy', 'long', 1)   # Buy CE to OPEN long at strike -1
            ...
        """
        if action == 0:
            return None  # Hold

        # Map action to strike and operation
        action_idx = action - 1  # Convert to 0-based (excluding hold)
        strike_idx = action_idx // 8  # Which strike (0-4)
        operation = action_idx % 8     # Which operation (0-7)

        # Decode operation
        operations = [
            ('CE', 'buy', 'long'),    # 0: Buy CE to open long
            ('CE', 'sell', 'long'),   # 1: Sell CE to close long
            ('CE', 'sell', 'short'),  # 2: Sell CE to open short
            ('CE', 'buy', 'short'),   # 3: Buy CE to close short
            ('PE', 'buy', 'long'),    # 4: Buy PE to open long
            ('PE', 'sell', 'long'),   # 5: Sell PE to close long
            ('PE', 'sell', 'short'),  # 6: Sell PE to open short
            ('PE', 'buy', 'short'),   # 7: Buy PE to close short
        ]

        option_type, side, direction = operations[operation]
        return (option_type, side, direction, strike_idx)

    def _get_position(self, option_type: str, strike: float, direction: str = None) -> Optional[Dict]:
        """
        Find existing position for given option type, strike, and optionally direction.

        Args:
            option_type: 'CE' or 'PE'
            strike: Strike price
            direction: 'long' or 'short' (optional - if None, returns any position at that strike/type)
        """
        for pos in self.positions:
            if pos['type'] == option_type and pos['strike'] == strike:
                if direction is None or pos['direction'] == direction:
                    return pos
        return None

    def _can_execute_trade(
        self,
        option_type: str,
        side: str,
        direction: str,
        strike: float
    ) -> Tuple[bool, str]:
        """
        Check if trade can be executed.

        Args:
            option_type: 'CE' or 'PE'
            side: 'buy' or 'sell' (the action being taken)
            direction: 'long' or 'short' (the position type)
            strike: Strike price

        Returns:
            (can_execute, reason)
        """
        existing_pos = self._get_position(option_type, strike, direction)

        # Determine if this is OPENING or CLOSING a position
        is_opening = (side == 'buy' and direction == 'long') or (side == 'sell' and direction == 'short')
        is_closing = (side == 'sell' and direction == 'long') or (side == 'buy' and direction == 'short')

        if is_opening:
            # OPENING a new position
            if existing_pos is not None:
                return False, f"Already have {direction} {option_type} @ {strike}"

            # Check position limit
            if len(self.positions) >= self.max_positions:
                return False, f"Max positions reached ({self.max_positions})"

            return True, "OK"

        elif is_closing:
            # CLOSING an existing position
            if existing_pos is None:
                return False, f"No existing {direction} position to close"

            return True, "OK"

        else:
            return False, "Invalid side/direction combination"

    def _execute_trade(
        self,
        option_type: str,
        side: str,
        direction: str,
        strike: float,
        nifty_price: float
    ) -> float:
        """
        Execute trade and return P&L.

        Args:
            option_type: 'CE' or 'PE'
            side: 'buy' or 'sell' (the action being taken)
            direction: 'long' or 'short' (the position type)
            strike: Strike price
            nifty_price: Current NIFTY price

        Returns:
            P&L from this trade (positive = profit, negative = loss)
        """
        premium = self._calculate_option_premium(
            strike, nifty_price, option_type, self.days_to_expiry
        )

        transaction_cost = self._calculate_transaction_cost(premium)

        # Determine if OPENING or CLOSING
        is_opening = (side == 'buy' and direction == 'long') or (side == 'sell' and direction == 'short')
        is_closing = (side == 'sell' and direction == 'long') or (side == 'buy' and direction == 'short')

        if is_opening:
            # OPENING a new position
            if direction == 'long':
                # BUY to open long: Pay premium (negative cash flow)
                cash_flow = -(premium * self.lot_size) - transaction_cost
            else:  # direction == 'short'
                # SELL to open short: Receive premium (positive cash flow)
                cash_flow = (premium * self.lot_size) - transaction_cost

            position = {
                'type': option_type,
                'side': side,
                'direction': direction,
                'strike': strike,
                'entry_premium': premium,
                'entry_nifty': nifty_price,
                'lot_size': self.lot_size,
                'entry_day': self.current_step,
            }
            self.positions.append(position)

            return cash_flow

        elif is_closing:
            # CLOSING an existing position
            existing_pos = self._get_position(option_type, strike, direction)

            if direction == 'long':
                # SELL to close long: Receive premium
                # P&L = (current_premium - entry_premium) * lot_size
                pnl = (premium - existing_pos['entry_premium']) * self.lot_size
                pnl -= transaction_cost
            else:  # direction == 'short'
                # BUY to close short: Pay premium
                # P&L = (entry_premium - current_premium) * lot_size
                pnl = (existing_pos['entry_premium'] - premium) * self.lot_size
                pnl -= transaction_cost

            # Remove position
            self.positions.remove(existing_pos)

            return pnl

        else:
            return 0.0  # Should never reach here

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step."""
        previous_balance = self.account_balance
        reward = 0
        pnl = 0

        nifty_price = self.nifty_history[-1]
        atm_strike = self._get_atm_strike(nifty_price)

        # Decode action
        decoded = self._decode_action(action)

        if decoded is not None:
            option_type, side, direction, strike_idx = decoded
            strike = atm_strike + (self.strike_offsets[strike_idx] * self.strike_interval)

            # Check if trade can be executed
            can_execute, reason = self._can_execute_trade(option_type, side, direction, strike)

            if can_execute:
                pnl = self._execute_trade(option_type, side, direction, strike, nifty_price)
                self.account_balance += pnl
            else:
                # Invalid action - small penalty
                reward -= 0.01

        # Update market
        self.current_step += 1
        self.days_to_expiry -= 1
        self._update_market()

        # Mark-to-market all positions
        mtm_pnl = self._update_positions_pnl(self.nifty_history[-1])

        # Handle expiry
        done = False
        if self.days_to_expiry == 0:
            expiry_pnl = self._settle_positions(self.nifty_history[-1])
            self.account_balance += expiry_pnl
            done = True

        # Calculate reward
        balance_change = self.account_balance - previous_balance
        reward += balance_change / self.initial_balance  # Normalized

        # Bankruptcy check
        if self.account_balance <= self.initial_balance * 0.5:
            done = True
            reward -= 1.0

        # Track metrics
        self.total_pnl = self.account_balance - self.initial_balance
        self.max_profit_today = max(self.max_profit_today, balance_change)
        self.max_loss_today = min(self.max_loss_today, balance_change)

        info = {
            'account_balance': self.account_balance,
            'num_positions': len(self.positions),
            'days_remaining': self.days_to_expiry,
            'nifty_price': self.nifty_history[-1],
            'total_pnl': self.total_pnl,
            'trade_pnl': pnl,
        }

        return self._get_obs(), reward, done, info

    def get_action_mask(self) -> np.ndarray:
        """
        Return mask of valid actions (1 = valid, 0 = invalid).

        This can be used with masked PPO for more efficient learning.
        Now supports 41 actions (long + short positions).
        """
        mask = np.zeros(41, dtype=np.float32)
        mask[0] = 1  # Hold is always valid

        nifty_price = self.nifty_history[-1]
        atm_strike = self._get_atm_strike(nifty_price)

        for action in range(1, 41):
            option_type, side, direction, strike_idx = self._decode_action(action)
            strike = atm_strike + (self.strike_offsets[strike_idx] * self.strike_interval)

            can_execute, _ = self._can_execute_trade(option_type, side, direction, strike)
            mask[action] = 1.0 if can_execute else 0.0

        return mask

    # ... (rest of the methods stay the same: _calculate_option_premium,
    #      _calculate_transaction_cost, _calculate_indicators, _get_obs,
    #      _update_market, _settle_positions, etc.)

    # I'll include key methods for completeness:

    def _calculate_option_premium(
        self, strike: float, nifty_price: float,
        option_type: str, days_to_expiry: int
    ) -> float:
        """Calculate option premium (simplified)."""
        time_to_expiry = days_to_expiry / 365.0
        vix = self.vix_history[-1] / 100.0

        if option_type == 'CE':
            intrinsic = max(0, nifty_price - strike)
        else:
            intrinsic = max(0, strike - nifty_price)

        moneyness = abs(nifty_price - strike) / nifty_price
        time_value = nifty_price * vix * np.sqrt(time_to_expiry) * (1 - moneyness * 0.5)

        return intrinsic + max(0.5, time_value)

    def _calculate_transaction_cost(self, premium: float) -> float:
        """Calculate total transaction cost."""
        trade_value = premium * self.lot_size
        cost = self.brokerage_per_trade
        cost += trade_value * self.stt_percentage
        exchange_fee = trade_value * self.exchange_fees
        cost += exchange_fee
        cost += (self.brokerage_per_trade + exchange_fee) * self.gst
        return cost

    def _get_atm_strike(self, nifty_price: float) -> float:
        """Get ATM strike."""
        return round(nifty_price / self.strike_interval) * self.strike_interval

    def _update_positions_pnl(self, current_nifty: float) -> float:
        """Mark-to-market all positions."""
        total_mtm = 0
        for pos in self.positions:
            current_premium = self._calculate_option_premium(
                pos['strike'], current_nifty, pos['type'], self.days_to_expiry
            )
            if pos['side'] == 'sell':
                pnl = (pos['entry_premium'] - current_premium) * pos['lot_size']
                total_mtm += pnl
        return total_mtm

    def _settle_positions(self, final_nifty: float) -> float:
        """Settle all positions at expiry."""
        total_pnl = 0
        for pos in self.positions:
            if pos['type'] == 'CE':
                intrinsic = max(0, final_nifty - pos['strike'])
            else:
                intrinsic = max(0, pos['strike'] - final_nifty)

            if pos['side'] == 'sell':
                payout = intrinsic * pos['lot_size']
                if intrinsic > 0:
                    stt = payout * self.stt_percentage
                    payout += stt * 1.18
                total_pnl -= payout

        self.positions = []
        return total_pnl

    def _generate_synthetic_nifty_data(self):
        """Generate synthetic price data."""
        base_price = 19500
        daily_vol = 0.015
        drift = 0.0002
        returns = np.random.randn(self.lookback_period) * daily_vol + drift
        prices = base_price * np.exp(np.cumsum(returns))

        self.nifty_history = prices
        self.high_history = prices * (1 + np.abs(np.random.randn(self.lookback_period) * 0.005))
        self.low_history = prices * (1 - np.abs(np.random.randn(self.lookback_period) * 0.005))
        self.volume_history = np.random.randint(100000, 500000, self.lookback_period)
        self.vix_history = np.clip(15 + np.random.randn(self.lookback_period) * 5, 10, 35)

    def _update_market(self):
        """Update market prices."""
        current_price = self.nifty_history[-1]
        vix = self.vix_history[-1]
        daily_vol = (vix / 100.0) / np.sqrt(252)
        price_change = np.random.randn() * daily_vol * current_price
        new_price = current_price + price_change

        self.nifty_history = np.roll(self.nifty_history, -1)
        self.nifty_history[-1] = new_price
        self.high_history = np.roll(self.high_history, -1)
        self.high_history[-1] = new_price * (1 + abs(np.random.randn() * 0.003))
        self.low_history = np.roll(self.low_history, -1)
        self.low_history[-1] = new_price * (1 - abs(np.random.randn() * 0.003))
        self.volume_history = np.roll(self.volume_history, -1)
        self.volume_history[-1] = np.random.randint(100000, 500000)

        mean_vix = 17
        vix_change = (mean_vix - vix) * 0.1 + np.random.randn() * 2
        new_vix = np.clip(vix + vix_change, 10, 35)
        self.vix_history = np.roll(self.vix_history, -1)
        self.vix_history[-1] = new_vix

    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate TA-Lib indicators."""
        indicators = {}
        current_price = self.nifty_history[-1]

        atr = talib.ATR(self.high_history, self.low_history, self.nifty_history, timeperiod=14)
        indicators['atr'] = atr[-1] if not np.isnan(atr[-1]) else 0

        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.nifty_history, timeperiod=20)
        indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price
        indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price
        indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price

        rsi = talib.RSI(self.nifty_history, timeperiod=14)
        indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50

        macd, macd_signal, macd_hist = talib.MACD(self.nifty_history)
        indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
        indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0

        slowk, slowd = talib.STOCH(self.high_history, self.low_history, self.nifty_history)
        indicators['stoch'] = slowk[-1] if not np.isnan(slowk[-1]) else 50

        sma = talib.SMA(self.nifty_history, timeperiod=20)
        indicators['sma'] = sma[-1] if not np.isnan(sma[-1]) else current_price

        ema = talib.EMA(self.nifty_history, timeperiod=20)
        indicators['ema'] = ema[-1] if not np.isnan(ema[-1]) else current_price

        adx = talib.ADX(self.high_history, self.low_history, self.nifty_history, timeperiod=14)
        indicators['adx'] = adx[-1] if not np.isnan(adx[-1]) else 0

        obv = talib.OBV(self.nifty_history, self.volume_history)
        indicators['obv'] = obv[-1] if not np.isnan(obv[-1]) else 0

        return indicators

    def _get_obs(self) -> np.ndarray:
        """Construct state observation."""
        nifty_price = self.nifty_history[-1]
        atm_strike = self._get_atm_strike(nifty_price)
        vix = self.vix_history[-1]

        ce_premiums = []
        pe_premiums = []
        for offset in self.strike_offsets:
            strike = atm_strike + (offset * self.strike_interval)
            ce_premium = self._calculate_option_premium(strike, nifty_price, 'CE', self.days_to_expiry)
            pe_premium = self._calculate_option_premium(strike, nifty_price, 'PE', self.days_to_expiry)
            ce_premiums.append(ce_premium)
            pe_premiums.append(pe_premium)

        indicators = self._calculate_indicators()

        obs = np.array([
            self.account_balance / self.initial_balance,
            len(self.positions) / self.max_positions,
            nifty_price / 20000.0,
            atm_strike / 20000.0,
            self.days_to_expiry / 5.0,
            vix / 30.0,
            self.total_pnl / self.initial_balance,
            (nifty_price - atm_strike) / self.strike_interval,
            self.max_loss_today / self.initial_balance if hasattr(self, 'max_loss_today') else 0,
            self.max_profit_today / self.initial_balance if hasattr(self, 'max_profit_today') else 0,
            self.consecutive_losses / 5.0 if hasattr(self, 'consecutive_losses') else 0,
            self.consecutive_wins / 5.0 if hasattr(self, 'consecutive_wins') else 0,
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
            vix / self.vix_history[:-1].mean() if len(self.vix_history) > 1 else 1.0,
            *[p / 100.0 for p in ce_premiums],
            *[p / 100.0 for p in pe_premiums],
        ], dtype=np.float32)

        return obs

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.account_balance = self.initial_balance
        self.positions: List[Dict] = []
        self.days_to_expiry = 5
        self.current_step = 0
        self.total_pnl = 0
        self.max_loss_today = 0
        self.max_profit_today = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        self._generate_synthetic_nifty_data()

        return self._get_obs()

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
                print(f"  {i}. {pos['side'].upper()} {pos['type']} {pos['strike']} | "
                      f"Entry: ₹{pos['entry_premium']:.2f} | Current: ₹{current_premium:.2f} | "
                      f"P&L: ₹{pnl:.2f}")
        print(f"{'='*60}\n")

    def _load_real_data(self, data_file: str):
        """Load real NIFTY data from CSV."""
        import pandas as pd
        df = pd.read_csv(data_file)
        if len(df) >= self.lookback_period:
            self.nifty_history = df['Close'].values[-self.lookback_period:]
            self.high_history = df['High'].values[-self.lookback_period:]
            self.low_history = df['Low'].values[-self.lookback_period:]
            self.volume_history = df['Volume'].values[-self.lookback_period:]
            if 'VIX' in df.columns:
                self.vix_history = df['VIX'].values[-self.lookback_period:]
            else:
                self.vix_history = self._estimate_vix_from_prices()

    def _estimate_vix_from_prices(self) -> np.ndarray:
        """Estimate volatility index from price history."""
        returns = np.diff(np.log(self.nifty_history))
        rolling_vol = np.array([
            np.std(returns[max(0, i-20):i+1]) * np.sqrt(252) * 100
            for i in range(len(returns))
        ])
        return np.concatenate([[rolling_vol[0]], rolling_vol])
