"""
PPO (Proximal Policy Optimization) Agent for Options Trading

This implementation uses Stable-Baselines3 for a clean, production-ready PPO agent.
Runs independently from the DQN implementation.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading performance during training.
    """

    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_balances = []

    def _on_step(self) -> bool:
        # Log episode statistics when episode ends
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            if 'account_balance' in info:
                self.episode_balances.append(info['account_balance'])
                if self.verbose > 0:
                    print(f"Episode finished - Balance: {info['account_balance']:.2f}")
        return True


class PPOTradingAgent:
    """
    PPO-based trading agent using Stable-Baselines3.

    Advantages over DQN:
    - Better for continuous-like action spaces (though we use discrete here)
    - More stable training with clipped objective
    - Better sample efficiency
    - Built-in tensorboard logging
    """

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/"
    ):
        """
        Initialize PPO agent.

        Args:
            env: Trading environment (OptionTradingEnvWithTALib)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for gradient clipping
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            tensorboard_log: Path for tensorboard logs
        """
        self.env = env
        self.verbose = verbose

        # Validate environment
        if verbose > 0:
            print("Checking environment compatibility...")
            check_env(env, warn=True)
            print("Environment check passed!")

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",  # Multi-Layer Perceptron policy
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )

    def train(
        self,
        total_timesteps=100000,
        log_interval=10,
        eval_freq=5000,
        eval_episodes=5,
        save_path="./ppo_models/",
        model_name="ppo_trading_agent"
    ):
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total number of timesteps to train
            log_interval: Number of timesteps between logging
            eval_freq: Evaluate the agent every eval_freq timesteps
            eval_episodes: Number of episodes for evaluation
            save_path: Directory to save model checkpoints
            model_name: Name for saved model files

        Returns:
            Trained model
        """
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Setup callbacks
        callbacks = []

        # Custom trading callback
        trading_callback = TradingCallback(verbose=self.verbose)
        callbacks.append(trading_callback)

        # Evaluation callback (saves best model)
        eval_env = Monitor(self.env)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # Train the agent
        if self.verbose > 0:
            print(f"\nStarting PPO training for {total_timesteps} timesteps...")
            print(f"Models will be saved to: {save_path}")
            print(f"Tensorboard logs: {self.model.tensorboard_log}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
        )

        # Save final model
        final_model_path = os.path.join(save_path, f"{model_name}_final")
        self.model.save(final_model_path)

        if self.verbose > 0:
            print(f"\nTraining completed! Final model saved to: {final_model_path}")

        return self.model

    def load(self, model_path):
        """
        Load a pre-trained model.

        Args:
            model_path: Path to saved model file
        """
        if self.verbose > 0:
            print(f"Loading model from: {model_path}")

        self.model = PPO.load(model_path, env=self.env)

        if self.verbose > 0:
            print("Model loaded successfully!")

    def predict(self, state, deterministic=True):
        """
        Predict action given state.

        Args:
            state: Current observation
            deterministic: If True, use deterministic policy (no exploration)

        Returns:
            action, state (for recurrent policies)
        """
        action, _states = self.model.predict(state, deterministic=deterministic)
        return action

    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the trained agent.

        Args:
            num_episodes: Number of episodes to run
            render: Whether to render the environment

        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_balances = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_balances.append(info.get('account_balance', 0))
            episode_lengths.append(episode_length)

            if self.verbose > 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Balance: {info.get('account_balance', 0):.2f}, "
                      f"Length: {episode_length}")

        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_balance': np.mean(episode_balances),
            'std_balance': np.std(episode_balances),
            'mean_length': np.mean(episode_lengths),
            'total_episodes': num_episodes,
        }

        if self.verbose > 0:
            print("\n=== Evaluation Results ===")
            print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
            print(f"Mean Balance: {results['mean_balance']:.2f} +/- {results['std_balance']:.2f}")
            print(f"Mean Episode Length: {results['mean_length']:.2f}")

        return results

    def save(self, path):
        """
        Save the current model.

        Args:
            path: Path to save the model
        """
        self.model.save(path)
        if self.verbose > 0:
            print(f"Model saved to: {path}")
