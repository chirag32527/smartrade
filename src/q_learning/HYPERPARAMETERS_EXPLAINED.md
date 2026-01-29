# Hyperparameter Tuning Summary for NIFTY Naked Options Trading

## Executive Summary

This document explains **why** each hyperparameter was chosen for NIFTY weekly options naked selling strategy.

## The Challenge: Why Options Trading is Different

Traditional RL hyperparameters (designed for games like Atari, Go) don't work well for options because:

1. **Extreme P&L Variance**: One bad trade can lose ₹25,000, good trade makes ₹3,000
2. **Non-Stationary**: Market conditions change (VIX 12 vs VIX 30 are different worlds)
3. **Time Decay**: Same position has different value each day
4. **Complex Payoffs**: Non-linear profit/loss curves
5. **Limited Data**: Only 5 days per episode (vs thousands of frames in games)

## Optimized Hyperparameters

### 1. Learning Rate: 1e-4 (vs default 3e-4)

**Why Lower?**

```
Default (3e-4):  Agent learns too fast, overfits to recent episodes
                 ₹5000 win → "always sell OTM CE!"
                 Next week: ₹-15000 loss (market rallied)

Optimized (1e-4): Slower learning, more robust
                  Averages over many market conditions
                  Less likely to overfit
```

**Trade-off**:
- ✅ More stable, generalizes better
- ❌ Takes longer to train (250k vs 100k timesteps)

**When to adjust**:
- Increase to 2e-4 if training on 1+ year of real data
- Decrease to 5e-5 if seeing high variance in eval metrics

---

### 2. Entropy Coefficient: 0.02 (vs default 0.0)

**Why Non-Zero?**

```
Default (0.0):  Agent may converge to "always sell ATM CE"
                Works in low VIX, fails in high VIX
                No exploration of other strikes

Optimized (0.02): Forced to try:
                  - Different strikes (OTM vs ITM)
                  - Different timing (day 1 vs day 3)
                  - CE vs PE based on indicators
```

**Effect**:
- Agent maintains 10-15% exploration even after convergence
- Prevents premature convergence to suboptimal strategy
- Critical for discovering VIX-dependent policies

**When to adjust**:
- Increase to 0.03-0.04 if agent only uses 1-2 actions
- Decrease to 0.01 if agent seems too random (not converging)

---

### 3. Batch Size: 128 (vs default 64)

**Why Larger?**

```
Episode P&Ls might be:
Week 1: +₹2000
Week 2: -₹8000
Week 3: +₹1500
Week 4: -₹12000
Week 5: +₹3000

Small batch (64): Gradient dominated by outliers (-₹12000)
                  Learns "never trade" to avoid losses

Large batch (128): Averages across more episodes
                   Sees that wins > losses on average
                   More stable gradient
```

**Trade-off**:
- ✅ Smoother learning, less noise
- ❌ Slower training (fewer updates per epoch)
- ❌ More memory usage

**When to adjust**:
- Increase to 256 if P&L variance > ₹10,000
- Decrease to 64 if memory constrained

---

### 4. N Epochs: 20 (vs default 10)

**Why More?**

```
State Space Complexity:
- 34 dimensions (12 basic + 12 indicators + 10 premiums)
- Non-linear relationships (RSI + VIX → premium)
- Time dependencies (day 1 vs day 5 same price = different action)

More epochs needed to:
1. Learn indicator patterns
2. Understand premium dynamics
3. Discover time-dependent strategies
```

**Effect**:
- Each batch of data is "seen" 20 times
- Network has more opportunity to find patterns
- Especially important early in training

**When to adjust**:
- Increase to 30 if using very complex state (e.g., adding Greeks)
- Decrease to 15 if training already converged (plateaued)

---

### 5. N Steps: 512 (vs default 2048)

**Why Smaller?**

```
Episode Length: 5 days (very short!)

Default (2048): Collects 2048/5 = 409 episodes before update
                Market conditions change across 400+ weeks
                Old data becomes stale

Optimized (512): Collects 512/5 = 102 episodes
                 More frequent updates
                 Adapts faster to policy improvements
```

**Trade-off**:
- ✅ More responsive to learning
- ✅ Better for short episodes
- ❌ Slightly less sample efficient

**When to adjust**:
- Increase to 1024 if episodes are longer (e.g., monthly expiry)
- Decrease to 256 for very short episodes (e.g., intraday)

---

### 6. Gamma: 0.99 (default)

**Why Standard?**

```
Gamma = discount factor for future rewards

Weekly expiry = 5 steps max

Step 1 reward discounted by: 0.99^4 = 0.96 (96% of value)
Step 5 reward discounted by: 0.99^0 = 1.00 (100% of value)

Close to 1.0 means: "all days in the week matter equally"
Makes sense for weekly options
```

**When to adjust**:
- Increase to 0.995 if doing monthly expiry (longer horizon)
- Decrease to 0.95 if only care about immediate P&L

---

### 7. GAE Lambda: 0.95 (default)

**Why Standard?**

```
GAE = Generalized Advantage Estimation
Controls bias-variance tradeoff in advantage calculation

0.95 = good balance for most RL problems
Works well for options trading too
```

**When to adjust**:
- Rarely needs changing
- Increase to 0.98 if high noise in rewards
- Decrease to 0.90 if rewards are very smooth

---

### 8. Network Architecture: [128, 128, 64]

**Why Deeper?**

```
Default [64, 64]:
- Total params: ~10k
- OK for simple Atari games

Optimized [128, 128, 64]:
- Total params: ~40k
- Needed for options because must learn:

  1. Indicator interpretation
     RSI=30 + VIX=25 → sell PE
     RSI=70 + VIX=15 → hold

  2. Premium valuation
     ATM CE premium ₹80 when VIX=20 → expensive
     Same premium when VIX=30 → cheap

  3. Strike selection logic
     OTM when trending
     ITM when ranging

  4. Time decay patterns
     Day 1: slow theta
     Day 5: fast theta
```

**Trade-off**:
- ✅ More capacity for complex patterns
- ❌ Longer training time
- ❌ Risk of overfitting (mitigated by entropy)

---

### 9. Clip Range: 0.2 (default)

**Why Standard?**

```
PPO clips policy updates to prevent catastrophic changes

0.2 = policy can change at most 20% per update
Good for options: prevents "flip-flopping" strategies
```

**When to adjust**:
- Decrease to 0.1 if training very unstable
- Increase to 0.3 if learning too slow

---

### 10. Value Coefficient: 0.5 (default)

**Why Standard?**

```
Balances:
- Policy loss (actor): "what actions to take"
- Value loss (critic): "how good is this state"

0.5 = equal weight to both
Standard and works well
```

**When to adjust**:
- Rarely needs changing
- Increase to 1.0 if value function not learning
- Decrease to 0.25 if focusing on policy

---

## Summary Table

| Hyperparameter | Default | NIFTY Optimized | Key Reason |
|----------------|---------|-----------------|------------|
| Learning Rate | 3e-4 | **1e-4** | High P&L variance |
| Entropy Coef | 0.0 | **0.02** | Need exploration |
| Batch Size | 64 | **128** | Smooth noisy gradients |
| N Epochs | 10 | **20** | Complex state space |
| N Steps | 2048 | **512** | Short episodes |
| Gamma | 0.99 | **0.99** | ✓ Standard works |
| GAE Lambda | 0.95 | **0.95** | ✓ Standard works |
| Network | [64,64] | **[128,128,64]** | More capacity needed |
| Clip Range | 0.2 | **0.2** | ✓ Standard works |
| Value Coef | 0.5 | **0.5** | ✓ Standard works |

## Validation: How to Know It's Working

### During Training (TensorBoard)

✅ **Good Signs**:
```
rollout/ep_rew_mean:  Steadily increasing
train/entropy_loss:   Stays between -2.0 and -1.0
train/policy_loss:    Decreasing then stable
train/value_loss:     Decreasing
train/approx_kl:      < 0.05 (within clip range)
```

❌ **Bad Signs**:
```
rollout/ep_rew_mean:  Flat or decreasing
train/entropy_loss:   → 0 (no exploration)
train/policy_loss:    Exploding / NaN
train/value_loss:     Not decreasing
train/approx_kl:      > 0.1 (updates too large)
```

### During Evaluation

✅ **Good Performance**:
```
Win Rate:      > 55%
Avg Weekly P&L: > ₹1,000
Sharpe Ratio:  > 1.5
Max Drawdown:  < 15%
Avg Trades:    1-3 per episode (not over-trading)
```

❌ **Needs Tuning**:
```
Win Rate:      < 45%  → Increase entropy, train longer
Avg Weekly P&L: < ₹0   → Check transaction costs, premium calc
Sharpe Ratio:  < 1.0  → Increase batch size, reduce LR
Max Drawdown:  > 25%  → Add risk penalties, reduce max positions
Avg Trades:    0 or 10+ → Adjust entropy (0 = too low, 10+ = too high)
```

## Advanced: Hyperparameter Optimization

For automated tuning, use Optuna:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.05)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Train agent
    agent = OptimizedNiftyPPO(env, learning_rate=lr, ...)
    agent.train(total_timesteps=50000)

    # Evaluate
    results = agent.evaluate(num_episodes=20)

    # Optimize for Sharpe ratio
    return results['sharpe_ratio']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_params}")
```

## Conclusion

These hyperparameters are **starting points** tuned for:
- NIFTY weekly options
- Naked selling strategy
- ₹1 lakh capital
- Synthetic data

For **your specific use case**, you may need to adjust based on:
- Your risk tolerance (capital, max positions)
- Data quality (real vs synthetic)
- Market regime (bull, bear, sideways)
- Brokerage costs (different brokers)

**Best practice**: Start with these defaults, train for 250k steps, evaluate for 50 episodes, then fine-tune based on results.
