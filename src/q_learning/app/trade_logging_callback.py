"""
Custom callback for logging trades during training.

This callback periodically saves trade history to CSV and prints trade statistics.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback


class TradeLoggingCallback(BaseCallback):
    """
    Callback for logging trades during training.

    Saves trade history to CSV at specified intervals and prints statistics.
    """

    def __init__(
        self,
        save_freq: int = 5000,
        save_path: str = "./trade_logs/",
        log_filename: str = "trades",
        verbose: int = 1
    ):
        """
        Args:
            save_freq: Save logs every N timesteps
            save_path: Directory to save trade logs
            log_filename: Base filename for logs (will append timestamp)
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super(TradeLoggingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.log_filename = log_filename
        self.last_save_timestep = 0

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("Trade Logging Callback Initialized")
            print(f"{'='*70}")
            print(f"Save Frequency: Every {self.save_freq} timesteps")
            print(f"Save Path: {self.save_path}")
            print(f"{'='*70}\n")

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training, False to stop
        """
        # Check if it's time to save
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            self._save_trade_logs()
            self.last_save_timestep = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save final logs
        self._save_trade_logs()

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("Trade Logging Complete")
            print(f"{'='*70}")
            print(f"Final logs saved to: {self.save_path}")
            print(f"{'='*70}\n")

    def _save_trade_logs(self):
        """Save current trade history to CSV and print statistics."""
        # Get the environment (unwrap if necessary)
        env = self.training_env.envs[0]

        # Unwrap VecNormalize if present
        while hasattr(env, 'venv'):
            env = env.venv.envs[0]

        # Unwrap Monitor wrapper
        while hasattr(env, 'env') and not hasattr(env, 'trade_history'):
            env = env.env

        # Unwrap ActionMasker if present
        if hasattr(env, 'env') and hasattr(env.env, 'trade_history'):
            env = env.env

        if not hasattr(env, 'trade_history'):
            if self.verbose >= 1:
                print("⚠️  Warning: Could not access trade_history from environment")
            return

        # Generate filename with timestep
        filepath = os.path.join(
            self.save_path,
            f"{self.log_filename}_{self.num_timesteps}.csv"
        )

        # Export trade history
        env.export_trade_history_to_csv(filepath)

        # Get and print statistics
        stats = env.get_trade_statistics()

        if stats and stats.get('total_trades', 0) > 0:
            if self.verbose >= 1:
                print(f"\n{'='*70}")
                print(f"Trade Statistics @ {self.num_timesteps} timesteps")
                print(f"{'='*70}")
                print(f"Total Trades: {stats['total_trades']}")
                print(f"Win Rate: {stats['win_rate']*100:.1f}%")
                print(f"Total P&L: ₹{stats['total_pnl']:,.2f}")
                print(f"Avg P&L per Trade: ₹{stats['avg_pnl']:,.2f}")
                print(f"Avg Win: ₹{stats['avg_win']:,.2f} | Avg Loss: ₹{stats['avg_loss']:,.2f}")
                print(f"Max Win: ₹{stats['max_win']:,.2f} | Max Loss: ₹{stats['max_loss']:,.2f}")
                print(f"Profit Factor: {stats['profit_factor']:.2f}")
                print(f"Avg Holding Period: {stats['avg_holding_period']:.1f} days")
                print(f"\nStrategy Breakdown:")
                print(f"  Long Trades: {stats['long_trades']} (Win Rate: {stats['long_win_rate']*100:.1f}%)")
                print(f"  Short Trades: {stats['short_trades']} (Win Rate: {stats['short_win_rate']*100:.1f}%)")
                print(f"  CE Trades: {stats['ce_trades']} | PE Trades: {stats['pe_trades']}")
                print(f"{'='*70}\n")


class PeriodicTradeExportCallback(BaseCallback):
    """
    Simpler callback that just exports trade logs at intervals without statistics.
    Useful for minimal overhead during training.
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./trade_logs/",
        clear_after_save: bool = False,
        verbose: int = 0
    ):
        """
        Args:
            save_freq: Save logs every N timesteps
            save_path: Directory to save trade logs
            clear_after_save: Clear trade history after saving (to free memory)
            verbose: Verbosity level
        """
        super(PeriodicTradeExportCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.clear_after_save = clear_after_save
        self.last_save_timestep = 0

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            env = self._get_env()
            if env and hasattr(env, 'trade_history'):
                filepath = os.path.join(self.save_path, f"trades_{self.num_timesteps}.csv")
                env.export_trade_history_to_csv(filepath)

                if self.clear_after_save:
                    # Keep episode number but clear history
                    current_episode = env.episode_number
                    env.clear_trade_history()
                    env.episode_number = current_episode

                self.last_save_timestep = self.num_timesteps

        return True

    def _get_env(self):
        """Get the unwrapped environment."""
        try:
            env = self.training_env.envs[0]

            # Unwrap layers
            while hasattr(env, 'venv'):
                env = env.venv.envs[0]

            while hasattr(env, 'env') and not hasattr(env, 'trade_history'):
                env = env.env

            if hasattr(env, 'env') and hasattr(env.env, 'trade_history'):
                env = env.env

            return env if hasattr(env, 'trade_history') else None
        except:
            return None
