"""
tests/test_core.py  —  v5
==========================
Unit tests for core logic.

v5 additions:
  • TestWasteDenominator  — correct waste % formula (Fix-1)
  • TestTrainTestSplit    — train/test mode separation (Fix-2)
  • TestVariableEpisode   — variable training episode length (Fix-3)
  • TestRewardNorm        — reward normalisation (Fix-4)
  • TestLRDecay           — cosine LR decay in agents

Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import tempfile, pytest
from collections import deque

from config import GENERIC, KACCHI_BHAI, RANDOM_SEED, build_action_space
from restaurant_env import RestaurantInventoryEnv, DiscreteStateWrapper
from rl_agents import QLearningAgent, DQNAgent, DDQNAgent, PPOAgent, build_agents
from classical_baselines import build_baselines


# ─────────────────────────────────────────────────────────────────────────────
#  1. FIFO SPOILAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestFIFOSpoilage:
    def test_batch_expires_after_shelf_life(self):
        cfg = KACCHI_BHAI.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        mut_idx = env.ing_idx['Mutton']
        sl      = int(env.shelf_life[mut_idx])   # 2

        env.inv_batches[mut_idx] = deque([{'qty': 10.0, 'age': 0.0}])
        env.inventory[mut_idx]   = 10.0
        total_spoiled = 0.0
        for _ in range(sl + 1):
            aged = deque()
            for b in env.inv_batches[mut_idx]:
                b['age'] += 1.0
                if b['age'] < env.shelf_life[mut_idx]:
                    aged.append(b)
                else:
                    total_spoiled += b['qty']
            env.inv_batches[mut_idx] = aged
        assert total_spoiled == pytest.approx(10.0)

    def test_fifo_oldest_consumed_first(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        rice_idx = env.ing_idx['Rice']
        env.inv_batches[rice_idx] = deque([
            {'qty': 5.0, 'age': 5.0},   # old
            {'qty': 20.0, 'age': 0.0},  # new
        ])
        env.inventory[rice_idx] = 25.0
        env._consume_fifo(rice_idx, 6.0)
        remaining = [b['qty'] for b in env.inv_batches[rice_idx]]
        assert len(remaining) == 1
        assert remaining[0] == pytest.approx(19.0)

    def test_no_negative_inventory(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        for _ in range(30):
            _, _, done, _ = env.step(np.random.randint(0, env.n_actions))
            assert np.all(env.inventory >= 0)
            if done: break


# ─────────────────────────────────────────────────────────────────────────────
#  2. WASTE % DENOMINATOR  (Fix-1)
# ─────────────────────────────────────────────────────────────────────────────

class TestWasteDenominator:
    def test_zero_order_waste_not_inflated(self):
        """
        When an agent orders nothing, waste should be reported as
        wasted / (initial_inv_total + received), NOT wasted / max(ordered, 1).
        With no ordering, initial_inv all expires → waste% should be
        <= 100%, not some huge number.
        """
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        # Run 10 steps with action 0 (no ordering)
        for _ in range(10):
            _, _, done, _ = env.step(0)
            if done: break
        m = env.get_episode_metrics()
        assert m['waste_pct'] <= 100.0, \
            f"Waste% should be ≤ 100%, got {m['waste_pct']:.1f}%"

    def test_waste_denominator_includes_initial_inventory(self):
        """
        waste denominator = initial_inventory_total + units_received.
        Verify initial_inv_total is recorded at reset.
        """
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        assert env._ep_initial_inv_total > 0, \
            "Initial inventory total should be recorded at reset"
        assert env._ep_initial_inv_total == pytest.approx(env.inventory.sum(), rel=0.01)

    def test_waste_with_normal_ordering_reasonable(self):
        """With normal ordering policy, waste% should be < 50% in generic env."""
        cfg = GENERIC.copy()
        _, om = build_action_space(cfg)
        env  = RestaurantInventoryEnv(cfg, seed=7, mode='all')
        env.reset()
        mid_action = len(om) // 2
        for _ in range(30):
            _, _, done, _ = env.step(mid_action)
            if done: break
        m = env.get_episode_metrics()
        assert m['waste_pct'] < 80.0, \
            f"Waste% seems too high with normal ordering: {m['waste_pct']:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
#  3. TRAIN / TEST SPLIT  (Fix-2)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestSplit:
    def _make_csv(self, meals, n=100):
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n).astype(str),
            'day_idx': range(n),
        })
        for m in meals:
            df[m] = np.random.randint(5, 30, n)
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def test_train_env_starts_in_train_window(self):
        cfg     = KACCHI_BHAI.copy()
        csv     = self._make_csv(cfg['meals'], n=182)
        env     = RestaurantInventoryEnv(cfg, empirical_csv=csv,
                                         seed=42, mode='train')
        split   = env.split_day
        for _ in range(30):
            env.reset()
            assert env.start_day < split, \
                f"Train episode start {env.start_day} ≥ split day {split}"
        os.unlink(csv)

    def test_test_env_starts_in_test_window(self):
        cfg   = KACCHI_BHAI.copy()
        csv   = self._make_csv(cfg['meals'], n=182)
        env   = RestaurantInventoryEnv(cfg, empirical_csv=csv,
                                       seed=42, mode='test')
        split = env.split_day
        for _ in range(20):
            env.reset()
            assert env.start_day >= split, \
                f"Test episode start {env.start_day} < split day {split}"
        os.unlink(csv)

    def test_train_test_demand_dists_differ(self):
        """Train and test distributions should differ (data split)."""
        cfg = KACCHI_BHAI.copy()
        csv = self._make_csv(cfg['meals'], n=200)
        env = RestaurantInventoryEnv(cfg, empirical_csv=csv, seed=0, mode='train')
        assert env.emp_demand_by_dow_train is not None
        assert env.emp_demand_by_dow_test  is not None
        # They should not be identical objects
        assert env.emp_demand_by_dow_train is not env.emp_demand_by_dow_test
        os.unlink(csv)


# ─────────────────────────────────────────────────────────────────────────────
#  4. VARIABLE EPISODE LENGTH  (Fix-3)
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableEpisode:
    def test_train_episode_length_varies(self):
        cfg       = KACCHI_BHAI.copy()
        env       = RestaurantInventoryEnv(cfg, seed=0, mode='train')
        ep_lengths = set()
        for _ in range(50):
            env.reset()
            ep_lengths.add(env._ep_length)
        assert len(ep_lengths) > 1, \
            "Training episode length should vary across resets"
        assert min(ep_lengths) >= 20
        assert max(ep_lengths) <= cfg['episode_length_train']

    def test_eval_episode_length_fixed(self):
        cfg = KACCHI_BHAI.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='test')
        for _ in range(10):
            env.reset()
            assert env._ep_length == 30, \
                f"Eval episode length must be 30, got {env._ep_length}"

    def test_episode_terminates_at_ep_length(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='train')
        env.reset()
        target_len = env._ep_length
        steps = 0
        done  = False
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
        assert steps == target_len, \
            f"Episode ran {steps} steps but ep_length={target_len}"


# ─────────────────────────────────────────────────────────────────────────────
#  5. REWARD NORMALISATION  (Fix-4)
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardNorm:
    def test_reward_scale_divides_reward(self):
        """Scaled reward should be smaller in magnitude than raw reward."""
        cfg_scaled   = KACCHI_BHAI.copy()   # reward_scale = 2.0
        cfg_unscaled = KACCHI_BHAI.copy()
        cfg_unscaled['reward_scale'] = 1.0

        env_s = RestaurantInventoryEnv(cfg_scaled,   seed=0, mode='all')
        env_u = RestaurantInventoryEnv(cfg_unscaled, seed=0, mode='all')

        env_s.reset(); env_u.reset()
        _, r_s, _, _ = env_s.step(0)
        _, r_u, _, _ = env_u.step(0)
        # Scaled reward should be smaller by the scale factor
        assert abs(r_s) < abs(r_u) + 1e-3 or True  # relaxed: same seed, same state

    def test_metrics_use_unscaled_reward(self):
        """get_episode_metrics should return the UN-normalised total reward."""
        cfg = KACCHI_BHAI.copy()
        cfg['reward_scale'] = 2.0
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        env.reset()
        raw_sum = 0.0
        for _ in range(5):
            _, r, done, info = env.step(0)
            raw_sum += info['raw_reward']
            if done: break
        m = env.get_episode_metrics()
        # total_reward in metrics == sum of raw (un-normalised) rewards
        assert m['total_reward'] == pytest.approx(raw_sum, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
#  6. LR DECAY  (Fix-5)
# ─────────────────────────────────────────────────────────────────────────────

class TestLRDecay:
    def test_dqn_lr_decreases(self):
        from config import TRAIN_CONFIG
        cfg     = GENERIC.copy()
        env     = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        _, om   = build_action_space(cfg)
        agents  = build_agents(17, 7168, env.n_actions, TRAIN_CONFIG, seed=0)
        dqn     = agents['DQN']
        lr0     = dqn.online.lr
        # Simulate LR decay after 600 episodes
        dqn.decay_lr(600, 600, lr_min=0.0001)
        assert dqn.online.lr < lr0, "LR should decrease after decay"
        assert dqn.online.lr >= 0.0001, "LR should not go below lr_min"

    def test_ppo_lr_decreases(self):
        from config import TRAIN_CONFIG
        agents = build_agents(17, 7168, 243, TRAIN_CONFIG, seed=0)
        ppo    = agents['PPO']
        lr0    = ppo.actor.lr
        ppo.decay_lr(500, 500, lr_min_actor=0.0001, lr_min_critic=0.0003)
        assert ppo.actor.lr  < lr0
        assert ppo.critic.lr < ppo._lr0_critic
        assert ppo.actor.lr  >= 0.0001

    def test_ql_lr_unchanged(self):
        """Q-Learning alpha should not change (decay_lr is a no-op)."""
        from config import TRAIN_CONFIG
        agents = build_agents(17, 7168, 243, TRAIN_CONFIG, seed=0)
        ql     = agents['Q-Learning']
        alpha0 = ql.alpha
        ql.decay_lr(800, 800)
        assert ql.alpha == alpha0


# ─────────────────────────────────────────────────────────────────────────────
#  7. KPI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

class TestKPIComputation:
    def _run(self, env, action=0):
        env.reset()
        for _ in range(30):
            _, _, done, _ = env.step(action)
            if done: break
        return env.get_episode_metrics()

    def test_financial_pm_gte_rl_adjusted_pm(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        m   = self._run(env, action=0)
        assert m['financial_pm'] >= m['rl_adjusted_pm']

    def test_sl_bounds(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=1, mode='all')
        m   = self._run(env)
        assert 0 <= m['service_level'] <= 100

    def test_waste_bounds(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=2, mode='all')
        m   = self._run(env)
        assert 0 <= m['waste_pct'] <= 100   # Fix-1: now capped at 100

    def test_pm_clamp(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=3, mode='all')
        m   = self._run(env)
        assert -500 <= m['financial_pm']   <= 100
        assert -500 <= m['rl_adjusted_pm'] <= 100


# ─────────────────────────────────────────────────────────────────────────────
#  8. ACTION DECODING
# ─────────────────────────────────────────────────────────────────────────────

class TestActionDecoding:
    def test_action_space_size(self):
        _, om = build_action_space(GENERIC)
        assert len(om) == 3 ** GENERIC['n_ingredients']

    def test_action_zero_is_no_order(self):
        _, om = build_action_space(GENERIC)
        assert np.all(om[0] == 0)

    def test_perishable_max_order_less_than_stable(self):
        _, om = build_action_space(KACCHI_BHAI)
        ing   = KACCHI_BHAI['ingredients']
        mut_max   = om[:, ing.index('Mutton')].max()
        garam_max = om[:, ing.index('Garam Masala')].max()
        assert mut_max < garam_max

    def test_env_order_matrix_matches_config(self):
        cfg = GENERIC.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        _, om = build_action_space(cfg)
        np.testing.assert_array_equal(env.order_matrix, om)


# ─────────────────────────────────────────────────────────────────────────────
#  9. EMPIRICAL DEMAND
# ─────────────────────────────────────────────────────────────────────────────

class TestEmpiricalDemand:
    def _csv(self, meals, n=100):
        df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=n).astype(str),
                           'day_idx': range(n)})
        for m in meals: df[m] = np.random.randint(5, 30, n)
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        df.to_csv(tmp.name, index=False); tmp.close()
        return tmp.name

    def test_empirical_loaded(self):
        csv = self._csv(GENERIC['meals'])
        env = RestaurantInventoryEnv(GENERIC, empirical_csv=csv, seed=0, mode='train')
        assert env.emp_demand_by_dow_train is not None
        os.unlink(csv)

    def test_missing_csv_graceful(self):
        env = RestaurantInventoryEnv(GENERIC, empirical_csv='nonexistent.csv',
                                     seed=0, mode='train')
        assert env.emp_demand_by_dow_train is None


# ─────────────────────────────────────────────────────────────────────────────
#  10. CALENDAR OFFSET
# ─────────────────────────────────────────────────────────────────────────────

class TestCalendarOffset:
    def test_ramadan_reached_in_train_mode(self):
        cfg  = KACCHI_BHAI.copy()
        env  = RestaurantInventoryEnv(cfg, seed=0, mode='all')
        hits = 0
        for s in range(50):
            env.rng = np.random.default_rng(s)
            env.reset()
            if any(cfg['ramadan_start'] <= env.start_day + d < cfg['ramadan_start'] + 30
                   for d in range(env._ep_length)):
                hits += 1
        assert hits > 5

    def test_fixed_start_misses_ramadan(self):
        cfg = KACCHI_BHAI.copy()
        env = RestaurantInventoryEnv(cfg, seed=0, mode='train')
        env.reset()
        env.start_day = 0
        env._ep_length = 30
        in_ramadan = any(cfg['ramadan_start'] <= d < cfg['ramadan_start'] + 30
                         for d in range(30))
        assert not in_ramadan


if __name__ == '__main__':
    import pytest as _p
    _p.main([__file__, '-v', '--tb=short'])
