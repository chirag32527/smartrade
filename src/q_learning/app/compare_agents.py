"""
Comparison script: DQN vs PPO on the same environment.

This script demonstrates both approaches side-by-side.
"""

from environment_talib import OptionTradingEnvWithTALib
from agent_ppo import PPOTradingAgent
from agent import DQNAgent
import numpy as np


def train_and_compare(timesteps_per_agent=50000):
    """
    Train both DQN and PPO agents and compare performance.

    Args:
        timesteps_per_agent: Training timesteps for each agent
    """
    print("="*70)
    print("DQN vs PPO Comparison on Options Trading")
    print("="*70)

    # Create environment
    env = OptionTradingEnvWithTALib(lookback_period=100, initial_balance=10000)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"\nEnvironment Details:")
    print(f"  State Size: {state_size} (includes TA-Lib indicators)")
    print(f"  Action Size: {action_size}")

    # ===== Train PPO Agent =====
    print("\n" + "="*70)
    print("Training PPO Agent")
    print("="*70)

    ppo_agent = PPOTradingAgent(env=env, verbose=1)
    ppo_agent.train(
        total_timesteps=timesteps_per_agent,
        save_path="./comparison_models/ppo/",
        model_name="ppo_comparison"
    )

    # ===== Train DQN Agent =====
    print("\n" + "="*70)
    print("Training DQN Agent")
    print("="*70)

    dqn_agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        replay_memory_size=10000,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=64
    )

    # DQN training loop
    episodes = timesteps_per_agent // 30  # Approximate episodes
    dqn_agent.train(env, episodes=episodes)

    # ===== Evaluate Both Agents =====
    print("\n" + "="*70)
    print("Evaluation Phase")
    print("="*70)

    num_eval_episodes = 20

    # Evaluate PPO
    print("\nEvaluating PPO Agent...")
    ppo_results = ppo_agent.evaluate(num_episodes=num_eval_episodes, render=False)

    # Evaluate DQN
    print("\nEvaluating DQN Agent...")
    dqn_balances = []
    dqn_rewards = []

    for episode in range(num_eval_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = dqn_agent.act(state, epsilon=0.01)  # Small epsilon for evaluation
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            state = next_state

        dqn_balances.append(info['account_balance'])
        dqn_rewards.append(total_reward)
        print(f"  Episode {episode+1}: Balance = {info['account_balance']:.2f}, Reward = {total_reward:.4f}")

    # ===== Comparison Results =====
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\nPPO Results:")
    print(f"  Mean Balance: {ppo_results['mean_balance']:.2f} +/- {ppo_results['std_balance']:.2f}")
    print(f"  Mean Reward: {ppo_results['mean_reward']:.4f} +/- {ppo_results['std_reward']:.4f}")

    print(f"\nDQN Results:")
    print(f"  Mean Balance: {np.mean(dqn_balances):.2f} +/- {np.std(dqn_balances):.2f}")
    print(f"  Mean Reward: {np.mean(dqn_rewards):.4f} +/- {np.std(dqn_rewards):.4f}")

    # Determine winner
    print("\nWinner:")
    if ppo_results['mean_balance'] > np.mean(dqn_balances):
        diff = ppo_results['mean_balance'] - np.mean(dqn_balances)
        print(f"  PPO wins by {diff:.2f} in mean balance!")
    elif np.mean(dqn_balances) > ppo_results['mean_balance']:
        diff = np.mean(dqn_balances) - ppo_results['mean_balance']
        print(f"  DQN wins by {diff:.2f} in mean balance!")
    else:
        print(f"  It's a tie!")

    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DQN vs PPO")
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Training timesteps per agent (default: 50000)'
    )

    args = parser.parse_args()

    train_and_compare(timesteps_per_agent=args.timesteps)
