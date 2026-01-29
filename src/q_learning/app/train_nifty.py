"""
Training Script for NIFTY Naked Options Trading with Optimized PPO

This script trains a PPO agent specifically for:
- NIFTY weekly options
- Naked selling strategy (CE/PE)
- 5 strike prices (2 OTM, ATM, 2 ITM)
- Indian market conditions

Usage:
    python train_nifty.py --train              # Train new agent
    python train_nifty.py --evaluate           # Evaluate existing agent
    python train_nifty.py --backtest           # Backtest on historical data
"""

import argparse
from environment_nifty import NiftyOptionsEnv
from agent_nifty_ppo import OptimizedNiftyPPO


def train_nifty_agent(
    timesteps=250000,
    save_path="./nifty_models/",
    data_file=None,
):
    """
    Train PPO agent for NIFTY options.

    Args:
        timesteps: Training timesteps (250k recommended)
        save_path: Directory to save models
        data_file: Optional CSV file with real NIFTY data

    Returns:
        Trained agent
    """
    print("\n" + "="*70)
    print("NIFTY NAKED OPTIONS TRADING - PPO TRAINING")
    print("="*70)

    # Create NIFTY environment
    print("\n1. Creating NIFTY Options Environment...")
    env = NiftyOptionsEnv(
        initial_balance=100000,      # ₹1 lakh
        max_positions=3,              # Max 3 concurrent positions
        lot_size=50,                  # NIFTY lot size
        strike_interval=50,           # NIFTY strike intervals
        lookback_period=100,          # Historical data for indicators
        data_file=data_file,          # Optional real data
    )

    print(f"\n   Environment Configuration:")
    print(f"   - Initial Capital: ₹{env.initial_balance:,}")
    print(f"   - Max Positions: {env.max_positions}")
    print(f"   - Lot Size: {env.lot_size}")
    print(f"   - Strike Interval: ₹{env.strike_interval}")
    print(f"   - State Dimension: {env.state_dim}")
    print(f"   - Action Space: {env.action_space.n} (0=Hold, 1-5=Sell CE, 6-10=Sell PE)")
    print(f"   - Strike Prices: 2 OTM + ATM + 2 ITM (5 total)")
    print(f"   - Episode Length: 5 days (weekly expiry)")

    # Create optimized PPO agent
    print("\n2. Initializing Optimized PPO Agent...")
    agent = OptimizedNiftyPPO(
        env=env,
        verbose=1,
        tensorboard_log="./nifty_ppo_logs/",
        seed=42,
    )

    # Train
    print("\n3. Starting Training...")
    print(f"   Total Timesteps: {timesteps:,}")
    print(f"   Expected Episodes: ~{timesteps // 5:,}")
    print(f"   Estimated Training Time: ~{timesteps // 1000} minutes")

    agent.train(
        total_timesteps=timesteps,
        save_path=save_path,
        model_name="nifty_naked_options_ppo",
        checkpoint_freq=10000,
        eval_freq=5000,
        eval_episodes=10,
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"\nTo monitor training progress:")
    print(f"  tensorboard --logdir={agent.model.tensorboard_log}")
    print(f"\nTo evaluate:")
    print(f"  python train_nifty.py --evaluate --model-path {save_path}nifty_naked_options_ppo_final")
    print("="*70 + "\n")

    return agent


def evaluate_nifty_agent(
    model_path,
    num_episodes=20,
    render=False,
    data_file=None,
):
    """
    Evaluate trained NIFTY agent.

    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        render: Whether to render environment
        data_file: Optional real data file

    Returns:
        Evaluation results
    """
    print("\n" + "="*70)
    print("NIFTY NAKED OPTIONS - EVALUATION")
    print("="*70)

    # Create environment
    print("\n1. Creating Environment...")
    env = NiftyOptionsEnv(
        initial_balance=100000,
        max_positions=3,
        lot_size=50,
        strike_interval=50,
        lookback_period=100,
        data_file=data_file,
    )

    # Create and load agent
    print(f"\n2. Loading Model from: {model_path}")
    agent = OptimizedNiftyPPO(env=env, verbose=1)

    # Load model and normalization
    norm_path = model_path.replace('.zip', '_vec_normalize.pkl')
    agent.load(model_path, norm_path)

    # Evaluate
    print(f"\n3. Running Evaluation ({num_episodes} episodes)...")
    results = agent.evaluate(
        num_episodes=num_episodes,
        render=render,
        deterministic=True,
    )

    # Additional analysis
    print("\n" + "="*70)
    print("STRATEGY ANALYSIS")
    print("="*70)

    initial = 100000
    if results['mean_pnl'] > 0:
        roi = (results['mean_pnl'] / initial) * 100
        print(f"Average ROI per Week: {roi:.2f}%")
        print(f"Annualized ROI: {roi * 52:.2f}%")
    else:
        print(f"Average Loss per Week: {results['mean_pnl_pct']:.2f}%")

    max_dd_pct = (results['max_drawdown'] / initial) * 100
    print(f"Maximum Drawdown: {max_dd_pct:.2f}%")

    if results['sharpe_ratio'] > 1.5:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f} ✓ GOOD (>1.5)")
    elif results['sharpe_ratio'] > 1.0:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f} ✓ ACCEPTABLE (>1.0)")
    else:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f} ✗ POOR (<1.0)")

    if results['win_rate'] > 0.6:
        print(f"Win Rate: {results['win_rate']*100:.1f}% ✓ GOOD (>60%)")
    elif results['win_rate'] > 0.5:
        print(f"Win Rate: {results['win_rate']*100:.1f}% ✓ ACCEPTABLE (>50%)")
    else:
        print(f"Win Rate: {results['win_rate']*100:.1f}% ✗ POOR (<50%)")

    print("="*70 + "\n")

    return results


def backtest_strategy(
    model_path,
    data_file,
    num_episodes=50,
):
    """
    Backtest trained agent on historical NIFTY data.

    Args:
        model_path: Path to trained model
        data_file: CSV file with historical NIFTY data
        num_episodes: Number of episodes to backtest

    Returns:
        Backtest results
    """
    print("\n" + "="*70)
    print("NIFTY NAKED OPTIONS - BACKTESTING")
    print("="*70)

    if not data_file:
        print("\n✗ Error: Backtesting requires historical data file")
        print("  Provide data file with: Date, Open, High, Low, Close, Volume, VIX")
        return None

    print(f"\n1. Loading Historical Data from: {data_file}")

    env = NiftyOptionsEnv(
        initial_balance=100000,
        max_positions=3,
        lot_size=50,
        strike_interval=50,
        lookback_period=100,
        data_file=data_file,
    )

    print(f"\n2. Loading Trained Model from: {model_path}")
    agent = OptimizedNiftyPPO(env=env, verbose=1)

    norm_path = model_path.replace('.zip', '_vec_normalize.pkl')
    agent.load(model_path, norm_path)

    print(f"\n3. Running Backtest ({num_episodes} episodes)...")
    results = agent.evaluate(
        num_episodes=num_episodes,
        render=False,
        deterministic=True,
    )

    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    print(f"Total Episodes: {num_episodes}")
    print(f"Total P&L: ₹{results['mean_pnl'] * num_episodes:,.2f}")
    print(f"Average Weekly P&L: ₹{results['mean_pnl']:,.2f}")
    print(f"Best Week: ₹{results['max_pnl']:,.2f}")
    print(f"Worst Week: ₹{results['min_pnl']:,.2f}")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print("="*70 + "\n")

    return results


def compare_with_baseline():
    """Compare PPO agent with simple baseline strategies."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)

    env = NiftyOptionsEnv(
        initial_balance=100000,
        max_positions=3,
        lot_size=50,
    )

    num_episodes = 20

    # Random baseline
    print("\n1. Testing Random Strategy...")
    random_balances = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        random_balances.append(info['account_balance'])

    random_mean = sum(random_balances) / len(random_balances)
    random_pnl = (random_mean - 100000) / 100000 * 100

    # Always sell ATM (common strategy)
    print("\n2. Testing Always Sell ATM Strategy...")
    atm_balances = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        while not done:
            # Action 3 = Sell CE at ATM, Action 8 = Sell PE at ATM
            action = 3 if step_count == 0 else 0  # Sell on first step, hold after
            obs, reward, done, info = env.step(action)
            step_count += 1
        atm_balances.append(info['account_balance'])

    atm_mean = sum(atm_balances) / len(atm_balances)
    atm_pnl = (atm_mean - 100000) / 100000 * 100

    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)
    print(f"Random Strategy:")
    print(f"  Mean Balance: ₹{random_mean:,.2f}")
    print(f"  Mean P&L: {random_pnl:+.2f}%")
    print(f"\nAlways Sell ATM Strategy:")
    print(f"  Mean Balance: ₹{atm_mean:,.2f}")
    print(f"  Mean P&L: {atm_pnl:+.2f}%")
    print("="*70)
    print("\nTrain PPO agent to beat these baselines!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/Evaluate PPO for NIFTY Naked Options Trading"
    )

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
        '--backtest',
        action='store_true',
        help='Backtest on historical data'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with baseline strategies'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=250000,
        help='Number of training timesteps (default: 250000)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes (default: 20)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='./nifty_models/nifty_naked_options_ppo_final',
        help='Path to saved model for evaluation/backtesting'
    )

    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='CSV file with historical NIFTY data (Date,Open,High,Low,Close,Volume,VIX)'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='./nifty_models/',
        help='Directory to save models (default: ./nifty_models/)'
    )

    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )

    args = parser.parse_args()

    # Execute based on arguments
    if args.compare:
        compare_with_baseline()

    elif args.train:
        agent = train_nifty_agent(
            timesteps=args.timesteps,
            save_path=args.save_path,
            data_file=args.data_file,
        )

        # Optionally evaluate after training
        if args.evaluate:
            print("\n" + "="*70)
            print("Evaluating trained agent...")
            print("="*70)
            agent.evaluate(num_episodes=args.episodes, render=args.render)

    elif args.backtest:
        if not args.data_file:
            print("\n✗ Error: --data-file required for backtesting")
            print("  Provide CSV with columns: Date,Open,High,Low,Close,Volume,VIX")
        else:
            backtest_strategy(
                model_path=args.model_path,
                data_file=args.data_file,
                num_episodes=args.episodes,
            )

    elif args.evaluate:
        evaluate_nifty_agent(
            model_path=args.model_path,
            num_episodes=args.episodes,
            render=args.render,
            data_file=args.data_file,
        )

    else:
        # No arguments - show demo
        print("\n" + "="*70)
        print("NIFTY NAKED OPTIONS TRADING - DEMO")
        print("="*70)
        print("\nNo arguments provided. Running baseline comparison...\n")

        compare_with_baseline()

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\nTo train an agent:")
        print("  python train_nifty.py --train --timesteps 250000")
        print("\nTo evaluate a trained agent:")
        print("  python train_nifty.py --evaluate")
        print("\nTo backtest on historical data:")
        print("  python train_nifty.py --backtest --data-file nifty_historical.csv")
        print("\nFor more options:")
        print("  python train_nifty.py --help")
        print("="*70 + "\n")
