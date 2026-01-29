"""
Training script for PPO agent with TA-Lib enhanced environment.

This script demonstrates how to:
1. Create the enhanced environment with TA-Lib indicators
2. Train a PPO agent
3. Evaluate the trained agent
4. Save/load models

Usage:
    python train_ppo.py --train          # Train new agent
    python train_ppo.py --evaluate       # Evaluate existing agent
    python train_ppo.py --train --eval   # Train then evaluate
"""

import argparse
from environment_talib import OptionTradingEnvWithTALib
from agent_ppo import PPOTradingAgent


def train_ppo_agent(total_timesteps=100000, save_path="./ppo_models/"):
    """
    Train a new PPO agent.

    Args:
        total_timesteps: Number of timesteps to train
        save_path: Directory to save models

    Returns:
        Trained agent
    """
    print("="*60)
    print("Training PPO Agent with TA-Lib Indicators")
    print("="*60)

    # Create environment
    print("\n1. Creating enhanced environment with TA-Lib indicators...")
    env = OptionTradingEnvWithTALib(
        lookback_period=100,
        initial_balance=10000
    )
    print(f"   State dimension: {env.state_dim}")
    print(f"   Action space: {env.action_space.n} discrete actions")

    # Create PPO agent
    print("\n2. Initializing PPO agent...")
    agent = PPOTradingAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/"
    )

    # Train agent
    print("\n3. Starting training...")
    agent.train(
        total_timesteps=total_timesteps,
        log_interval=10,
        eval_freq=5000,
        eval_episodes=5,
        save_path=save_path,
        model_name="ppo_trading_agent"
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    return agent


def evaluate_ppo_agent(model_path, num_episodes=10, render=False):
    """
    Evaluate a trained PPO agent.

    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment

    Returns:
        Evaluation results dictionary
    """
    print("="*60)
    print("Evaluating PPO Agent")
    print("="*60)

    # Create environment
    print("\n1. Creating environment...")
    env = OptionTradingEnvWithTALib(
        lookback_period=100,
        initial_balance=10000
    )

    # Create and load agent
    print(f"\n2. Loading model from: {model_path}")
    agent = PPOTradingAgent(env=env, verbose=1)
    agent.load(model_path)

    # Evaluate
    print(f"\n3. Running evaluation for {num_episodes} episodes...")
    results = agent.evaluate(num_episodes=num_episodes, render=render)

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)

    return results


def compare_with_random(num_episodes=10):
    """
    Compare PPO agent with random action baseline.

    Args:
        num_episodes: Number of episodes for comparison
    """
    print("="*60)
    print("Comparing PPO vs Random Actions")
    print("="*60)

    env = OptionTradingEnvWithTALib(
        lookback_period=100,
        initial_balance=10000
    )

    # Random baseline
    print("\n1. Testing random action baseline...")
    random_balances = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
        random_balances.append(info['account_balance'])
        print(f"   Episode {episode+1}: Balance = {info['account_balance']:.2f}")

    print(f"\nRandom Agent - Mean Balance: {sum(random_balances)/len(random_balances):.2f}")
    print("\nTrain a PPO agent to see if it can beat this baseline!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Evaluate PPO trading agent")

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new PPO agent'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate existing PPO agent'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with random action baseline'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of timesteps for training (default: 100000)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes for evaluation (default: 10)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='./ppo_models/ppo_trading_agent_final',
        help='Path to saved model for evaluation'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='./ppo_models/',
        help='Directory to save models (default: ./ppo_models/)'
    )

    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )

    args = parser.parse_args()

    # Execute based on arguments
    if args.compare:
        compare_with_random(num_episodes=args.episodes)

    elif args.train:
        agent = train_ppo_agent(
            total_timesteps=args.timesteps,
            save_path=args.save_path
        )

        # Optionally evaluate after training
        if args.evaluate:
            print("\n" + "="*60)
            print("Evaluating trained agent...")
            print("="*60)
            agent.evaluate(num_episodes=args.episodes, render=args.render)

    elif args.evaluate:
        evaluate_ppo_agent(
            model_path=args.model_path,
            num_episodes=args.episodes,
            render=args.render
        )

    else:
        # No arguments provided - show demo
        print("\nNo arguments provided. Running comparison demo...\n")
        compare_with_random(num_episodes=5)

        print("\n" + "="*60)
        print("To train an agent, run:")
        print("  python train_ppo.py --train --timesteps 100000")
        print("\nTo evaluate a trained agent, run:")
        print("  python train_ppo.py --evaluate --model-path ./ppo_models/ppo_trading_agent_final")
        print("\nFor more options, run:")
        print("  python train_ppo.py --help")
        print("="*60)
