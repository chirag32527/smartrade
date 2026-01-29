"""
Optimized PPO Agent for NIFTY Naked Options Trading

Hyperparameters specifically tuned for:
- Weekly NIFTY options
- Naked selling strategy
- Limited strike prices (5 strikes)
- High risk/reward profile
- Indian market characteristics
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
from typing import Optional


class NiftyTradingCallback(BaseCallback):
    """Custom callback for NIFTY options trading metrics."""

    def __init__(self, verbose=0):
        super(NiftyTradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_balances = []
        self.episode_max_drawdowns = []
        self.win_rate = []

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]

            if 'account_balance' in info:
                self.episode_balances.append(info['account_balance'])

                # Calculate metrics
                initial_balance = 100000
                pnl_pct = (info['account_balance'] - initial_balance) / initial_balance * 100

                if self.verbose > 0:
                    print(f"\nEpisode Complete:")
                    print(f"  Final Balance: ₹{info['account_balance']:,.2f}")
                    print(f"  P&L: {pnl_pct:+.2f}%")
                    print(f"  Positions Taken: {info.get('num_positions', 0)}")

                # Track win rate
                if len(self.episode_balances) >= 10:
                    recent_wins = sum(1 for b in self.episode_balances[-10:] if b > initial_balance)
                    self.win_rate.append(recent_wins / 10.0)

        return True


class OptimizedNiftyPPO:
    """
    PPO agent with hyperparameters optimized for NIFTY naked options trading.

    Key Optimizations:
    1. Higher entropy coefficient for exploration (volatile market)
    2. Larger batch size for stable learning
    3. More training epochs per update
    4. Conservative learning rate
    5. Strong risk penalties
    """

    def __init__(
        self,
        env,
        verbose=1,
        tensorboard_log="./nifty_ppo_logs/",
        seed: Optional[int] = 42,
    ):
        """
        Initialize optimized PPO for NIFTY.

        Args:
            env: NiftyOptionsEnv instance
            verbose: Logging verbosity
            tensorboard_log: Path for tensorboard logs
            seed: Random seed for reproducibility
        """
        self.verbose = verbose

        # Wrap environment for normalization
        # This helps with stability when dealing with varying balance scales
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # HYPERPARAMETERS OPTIMIZED FOR NIFTY NAKED OPTIONS
        # ================================================

        # Learning rate: Conservative for stable learning
        # - Lower than default (3e-4) because of high variance in options P&L
        learning_rate = 1e-4

        # Steps per update: Moderate to balance exploration and learning
        # - Weekly expiry = 5 days max episode
        # - Want ~400-500 steps per update (80-100 episodes)
        n_steps = 512

        # Batch size: Large for stable gradient estimates
        # - Options trading has high variance
        # - Larger batches smooth out noisy gradients
        batch_size = 128

        # Optimization epochs: More epochs for better learning
        # - Default is 10, we increase to 20 for stable convergence
        n_epochs = 20

        # Discount factor: Standard for weekly horizon
        # - 0.99 is good for multi-day episodes
        gamma = 0.99

        # GAE lambda: High value for accurate advantage estimation
        # - 0.95 balances bias-variance tradeoff
        gae_lambda = 0.95

        # Clip range: Standard PPO clipping
        # - 0.2 prevents too large policy updates
        clip_range = 0.2

        # Clip range for value function: Additional stability
        clip_range_vf = None  # Will use same as clip_range

        # Entropy coefficient: HIGHER for more exploration
        # - Options trading requires exploration of different strategies
        # - 0.02 (vs default 0.0) encourages trying different strikes/timings
        ent_coef = 0.02

        # Value function coefficient: Standard
        vf_coef = 0.5

        # Max gradient norm: Tighter for stability
        # - 0.5 (vs default 0.5) prevents exploding gradients
        max_grad_norm = 0.5

        # Network architecture: Deeper for complex state space
        policy_kwargs = dict(
            net_arch=[
                dict(pi=[128, 128, 64], vf=[128, 128, 64])
                # pi: policy network (actor)
                # vf: value network (critic)
                # Deeper than default [64, 64] to capture option dynamics
            ],
            activation_fn=__import__('torch').nn.ReLU,
        )

        if self.verbose > 0:
            print("\n" + "="*70)
            print("NIFTY Naked Options PPO - Optimized Hyperparameters")
            print("="*70)
            print(f"Learning Rate:        {learning_rate}")
            print(f"Steps per Update:     {n_steps}")
            print(f"Batch Size:           {batch_size}")
            print(f"Optimization Epochs:  {n_epochs}")
            print(f"Discount Factor:      {gamma}")
            print(f"GAE Lambda:           {gae_lambda}")
            print(f"Clip Range:           {clip_range}")
            print(f"Entropy Coefficient:  {ent_coef} (HIGH for exploration)")
            print(f"Value Coefficient:    {vf_coef}")
            print(f"Max Gradient Norm:    {max_grad_norm}")
            print(f"Network Architecture: {policy_kwargs['net_arch']}")
            print("="*70 + "\n")

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

    def train(
        self,
        total_timesteps=250000,  # ~500 episodes (5 days each)
        save_path="./nifty_ppo_models/",
        model_name="nifty_naked_options_ppo",
        checkpoint_freq=10000,
        eval_freq=5000,
        eval_episodes=10,
    ):
        """
        Train the optimized NIFTY PPO agent.

        Args:
            total_timesteps: Total training steps (250k recommended for NIFTY)
            save_path: Directory for saving models
            model_name: Base name for saved models
            checkpoint_freq: Save checkpoint every N steps
            eval_freq: Evaluate every N steps
            eval_episodes: Number of eval episodes

        Returns:
            Trained model
        """
        os.makedirs(save_path, exist_ok=True)

        # Setup callbacks
        callbacks = []

        # Custom NIFTY trading callback
        nifty_callback = NiftyTradingCallback(verbose=self.verbose)
        callbacks.append(nifty_callback)

        # Checkpoint callback - save every N steps
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path,
            name_prefix=model_name,
            verbose=1,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback - track best model
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)

        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Starting NIFTY Options Training")
            print(f"{'='*70}")
            print(f"Total Timesteps: {total_timesteps:,}")
            print(f"Expected Episodes: ~{total_timesteps // 5:,} (5 days each)")
            print(f"Checkpoint Frequency: Every {checkpoint_freq:,} steps")
            print(f"Evaluation Frequency: Every {eval_freq:,} steps")
            print(f"Save Path: {save_path}")
            print(f"{'='*70}\n")

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
        )

        # Save final model
        final_path = os.path.join(save_path, f"{model_name}_final")
        self.model.save(final_path)

        # Save normalization stats
        self.env.save(os.path.join(save_path, "vec_normalize.pkl"))

        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Training Complete!")
            print(f"Final model saved: {final_path}")
            print(f"{'='*70}\n")

        return self.model

    def load(self, model_path, norm_path=None):
        """Load trained model and normalization stats."""
        if self.verbose > 0:
            print(f"Loading model from: {model_path}")

        self.model = PPO.load(model_path, env=self.env)

        if norm_path and os.path.exists(norm_path):
            self.env = VecNormalize.load(norm_path, self.env)
            if self.verbose > 0:
                print(f"Loaded normalization from: {norm_path}")

        if self.verbose > 0:
            print("Model loaded successfully!")

    def predict(self, state, deterministic=True):
        """Predict action for given state."""
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action

    def evaluate(
        self,
        num_episodes=20,
        render=False,
        deterministic=True,
    ):
        """
        Evaluate trained agent on NIFTY options.

        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render environment
            deterministic: Use deterministic policy

        Returns:
            Dictionary with comprehensive metrics
        """
        episode_balances = []
        episode_pnls = []
        episode_max_drawdowns = []
        episode_trades = []

        initial_balance = 100000

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_min_balance = initial_balance
            num_trades = 0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)

                # Track metrics (need to unwrap VecNormalize)
                if hasattr(info[0], '__contains__') and 'account_balance' in info[0]:
                    current_balance = info[0]['account_balance']
                    episode_min_balance = min(episode_min_balance, current_balance)

                    if info[0].get('num_positions', 0) > num_trades:
                        num_trades = info[0]['num_positions']

                if render:
                    self.env.render()

            # Unwrap final info
            final_info = info[0] if isinstance(info, (list, tuple)) else info
            final_balance = final_info.get('account_balance', initial_balance)

            episode_balances.append(final_balance)
            episode_pnls.append(final_balance - initial_balance)
            episode_max_drawdowns.append(initial_balance - episode_min_balance)
            episode_trades.append(num_trades)

            if self.verbose > 0:
                pnl_pct = (final_balance - initial_balance) / initial_balance * 100
                dd_pct = episode_max_drawdowns[-1] / initial_balance * 100
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Balance: ₹{final_balance:,.2f} | "
                      f"P&L: {pnl_pct:+.2f}% | "
                      f"Max DD: {dd_pct:.2f}% | "
                      f"Trades: {num_trades}")

        # Calculate comprehensive statistics
        pnls_array = np.array(episode_pnls)
        balances_array = np.array(episode_balances)

        wins = np.sum(pnls_array > 0)
        losses = np.sum(pnls_array < 0)
        win_rate = wins / num_episodes if num_episodes > 0 else 0

        # Sharpe ratio (assuming daily returns)
        if np.std(pnls_array) > 0:
            sharpe = np.mean(pnls_array) / np.std(pnls_array) * np.sqrt(252/5)  # Annualized
        else:
            sharpe = 0

        results = {
            'num_episodes': num_episodes,
            'mean_balance': np.mean(balances_array),
            'std_balance': np.std(balances_array),
            'mean_pnl': np.mean(pnls_array),
            'std_pnl': np.std(pnls_array),
            'mean_pnl_pct': np.mean(pnls_array) / initial_balance * 100,
            'win_rate': win_rate,
            'num_wins': wins,
            'num_losses': losses,
            'max_pnl': np.max(pnls_array),
            'min_pnl': np.min(pnls_array),
            'mean_drawdown': np.mean(episode_max_drawdowns),
            'max_drawdown': np.max(episode_max_drawdowns),
            'sharpe_ratio': sharpe,
            'avg_trades_per_episode': np.mean(episode_trades),
        }

        if self.verbose > 0:
            print("\n" + "="*70)
            print("EVALUATION RESULTS")
            print("="*70)
            print(f"Episodes:              {results['num_episodes']}")
            print(f"Mean Final Balance:    ₹{results['mean_balance']:,.2f}")
            print(f"Mean P&L:              ₹{results['mean_pnl']:,.2f} ({results['mean_pnl_pct']:+.2f}%)")
            print(f"P&L Std Dev:           ₹{results['std_pnl']:,.2f}")
            print(f"Win Rate:              {results['win_rate']*100:.1f}% ({results['num_wins']}W / {results['num_losses']}L)")
            print(f"Best P&L:              ₹{results['max_pnl']:,.2f}")
            print(f"Worst P&L:             ₹{results['min_pnl']:,.2f}")
            print(f"Mean Drawdown:         ₹{results['mean_drawdown']:,.2f}")
            print(f"Max Drawdown:          ₹{results['max_drawdown']:,.2f}")
            print(f"Sharpe Ratio:          {results['sharpe_ratio']:.2f}")
            print(f"Avg Trades/Episode:    {results['avg_trades_per_episode']:.1f}")
            print("="*70 + "\n")

        return results

    def save(self, path):
        """Save model and normalization stats."""
        self.model.save(path)
        self.env.save(path.replace('.zip', '_vec_normalize.pkl'))
        if self.verbose > 0:
            print(f"Model and normalization saved to: {path}")
