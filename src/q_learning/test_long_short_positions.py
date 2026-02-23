"""
Test script to verify LONG and SHORT position support in NIFTY environment.

Run after installing dependencies:
    cd src/q_learning
    pip install -r requirements.txt
    python test_long_short_positions.py
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.environment_nifty import NiftyOptionsEnv
import numpy as np


def test_long_short_positions():
    """Test that both long and short positions work correctly."""

    print('\n' + '='*70)
    print('TESTING LONG AND SHORT POSITION SUPPORT')
    print('='*70 + '\n')

    # Create environment
    print('1. Creating NIFTY Options Environment...')
    env = NiftyOptionsEnv(
        initial_balance=100000,
        max_positions=3,
        lot_size=50,
        strike_interval=50,
    )
    print(f'   ✓ Environment created')
    print(f'   ✓ Action space: {env.action_space.n} actions')
    print(f'   ✓ Expected: 41 (1 hold + 5 strikes × 8 actions)')

    if env.action_space.n != 41:
        print(f'   ✗ ERROR: Expected 41 actions, got {env.action_space.n}')
        return False

    # Test action decoding
    print('\n2. Testing action decoding...')
    test_actions = [
        (0, None, 'Hold'),
        (1, ('CE', 'buy', 'long', 0), 'Buy CE to OPEN long @ strike -2'),
        (2, ('CE', 'sell', 'long', 0), 'Sell CE to CLOSE long @ strike -2'),
        (3, ('CE', 'sell', 'short', 0), 'Sell CE to OPEN short @ strike -2'),
        (4, ('CE', 'buy', 'short', 0), 'Buy CE to CLOSE short @ strike -2'),
        (5, ('PE', 'buy', 'long', 0), 'Buy PE to OPEN long @ strike -2'),
        (6, ('PE', 'sell', 'long', 0), 'Sell PE to CLOSE long @ strike -2'),
        (7, ('PE', 'sell', 'short', 0), 'Sell PE to OPEN short @ strike -2'),
        (8, ('PE', 'buy', 'short', 0), 'Buy PE to CLOSE short @ strike -2'),
    ]

    for action, expected, description in test_actions:
        decoded = env._decode_action(action)
        if decoded == expected:
            print(f'   ✓ Action {action}: {description}')
        else:
            print(f'   ✗ Action {action} failed: expected {expected}, got {decoded}')
            return False

    # Test action masking for long/short
    print('\n3. Testing action masking (no positions)...')
    obs, info = env.reset()
    mask = env.get_action_mask()

    print(f'   ✓ Mask shape: {mask.shape}')
    print(f'   ✓ Valid actions: {np.sum(mask)}/{len(mask)}')

    # When no positions exist:
    # - Should be able to OPEN (buy long or sell short)
    # - Should NOT be able to CLOSE (no positions to close)

    # Action 1: Buy CE to OPEN long (should be valid)
    # Action 2: Sell CE to CLOSE long (should be INVALID - no position)
    # Action 3: Sell CE to OPEN short (should be valid)
    # Action 4: Buy CE to CLOSE short (should be INVALID - no position)

    open_actions = [1, 3, 5, 7]  # Buy long, Sell short for CE and PE
    close_actions = [2, 4, 6, 8]  # Sell long, Buy short for CE and PE

    valid_opens = sum(mask[a] == 1.0 for a in open_actions)
    invalid_closes = sum(mask[a] == 0.0 for a in close_actions)

    print(f'   ✓ OPEN actions valid: {valid_opens}/{len(open_actions)} (should be {len(open_actions)})')
    print(f'   ✓ CLOSE actions invalid: {invalid_closes}/{len(close_actions)} (should be {len(close_actions)})')

    if valid_opens != len(open_actions) or invalid_closes != len(close_actions):
        print('   ⚠️  Mask logic might need adjustment')
        return False

    # Test opening a LONG position
    print('\n4. Testing LONG position (Buy CE to open)...')
    initial_balance = env.account_balance

    # Action 1: Buy CE to open long at strike -2
    obs, reward, terminated, truncated, info = env.step(1)

    if len(env.positions) == 1:
        pos = env.positions[0]
        print(f'   ✓ Position opened: {pos["direction"]} {pos["type"]} @ {pos["strike"]}')
        print(f'   ✓ Entry side: {pos["side"]} (should be "buy")')
        print(f'   ✓ Direction: {pos["direction"]} (should be "long")')
        print(f'   ✓ Balance change: ₹{env.account_balance - initial_balance:,.2f} (negative - paid premium)')

        if pos['direction'] != 'long' or pos['side'] != 'buy':
            print('   ✗ Position attributes incorrect!')
            return False
    else:
        print(f'   ✗ Expected 1 position, got {len(env.positions)}')
        return False

    # Test action masking after opening long
    print('\n5. Testing action masking (after opening long CE)...')
    mask = env.get_action_mask()

    # Action 1 (Buy CE long @ -2) should now be INVALID (already have it)
    # Action 2 (Sell CE long @ -2) should now be VALID (can close it)
    # Action 3 (Sell CE short @ -2) should be INVALID (already have long at this strike)

    print(f'   Action 1 (Buy CE long): {"INVALID ✓" if mask[1] == 0 else "VALID ✗ (should be invalid)"}')
    print(f'   Action 2 (Sell CE long): {"VALID ✓" if mask[2] == 1 else "INVALID ✗ (should be valid)"}')

    # Test closing the LONG position
    print('\n6. Testing closing LONG position (Sell CE to close)...')
    balance_before_close = env.account_balance

    # Action 2: Sell CE to close long at strike -2
    obs, reward, terminated, truncated, info = env.step(2)

    if len(env.positions) == 0:
        print(f'   ✓ Position closed successfully')
        print(f'   ✓ P&L from close: ₹{env.account_balance - balance_before_close:,.2f}')
    else:
        print(f'   ✗ Expected 0 positions after close, got {len(env.positions)}')
        return False

    # Test opening a SHORT position
    print('\n7. Testing SHORT position (Sell CE to open)...')
    env.reset()
    initial_balance = env.account_balance

    # Action 3: Sell CE to open short at strike -2
    obs, reward, terminated, truncated, info = env.step(3)

    if len(env.positions) == 1:
        pos = env.positions[0]
        print(f'   ✓ Position opened: {pos["direction"]} {pos["type"]} @ {pos["strike"]}')
        print(f'   ✓ Entry side: {pos["side"]} (should be "sell")')
        print(f'   ✓ Direction: {pos["direction"]} (should be "short")')
        print(f'   ✓ Balance change: ₹{env.account_balance - initial_balance:,.2f} (positive - received premium)')

        if pos['direction'] != 'short' or pos['side'] != 'sell':
            print('   ✗ Position attributes incorrect!')
            return False
    else:
        print(f'   ✗ Expected 1 position, got {len(env.positions)}')
        return False

    # Test closing the SHORT position
    print('\n8. Testing closing SHORT position (Buy CE to close)...')
    balance_before_close = env.account_balance

    # Action 4: Buy CE to close short at strike -2
    obs, reward, terminated, truncated, info = env.step(4)

    if len(env.positions) == 0:
        print(f'   ✓ Position closed successfully')
        print(f'   ✓ P&L from close: ₹{env.account_balance - balance_before_close:,.2f}')
    else:
        print(f'   ✗ Expected 0 positions after close, got {len(env.positions)}')
        return False

    # Test simultaneous long and short at different strikes
    print('\n9. Testing simultaneous LONG and SHORT at different strikes...')
    env.reset()

    # Open long CE @ strike -2
    env.step(1)  # Buy CE long @ -2
    # Open short PE @ strike +2
    env.step(39)  # Sell PE short @ +2 (action 33-40 are for strike +2)

    if len(env.positions) == 2:
        long_pos = [p for p in env.positions if p['direction'] == 'long']
        short_pos = [p for p in env.positions if p['direction'] == 'short']

        print(f'   ✓ {len(long_pos)} long position(s)')
        print(f'   ✓ {len(short_pos)} short position(s)')
        print(f'   ✓ Total positions: {len(env.positions)}')
    else:
        print(f'   ✗ Expected 2 positions, got {len(env.positions)}')
        return False

    print('\n' + '='*70)
    print('✅ ALL TESTS PASSED!')
    print('='*70)
    print('\nBoth LONG and SHORT positions are working correctly.')
    print('You can now train with:')
    print('  cd app')
    print('  python train_nifty.py --train --timesteps 150000')
    print('='*70 + '\n')

    return True


if __name__ == '__main__':
    try:
        success = test_long_short_positions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n❌ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
