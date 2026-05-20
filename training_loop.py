"""
training_loop.py  —  v5
========================
Universal training and evaluation engine.
"""

from __future__ import annotations
import numpy as np
from restaurant_env import RestaurantInventoryEnv, DiscreteStateWrapper
from rl_agents import QLearningAgent, PPOAgent
from config import EPISODE_LENGTH, STATE_DIM, TRAIN_CONFIG, QL_INV_BINS


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE EPISODE
# ─────────────────────────────────────────────────────────────────────────────

def _run_episode(env, agent, disc_env=None, training=True):
    is_ql  = isinstance(agent, QLearningAgent)
    is_ppo = isinstance(agent, PPOAgent)

    state  = disc_env.reset() if is_ql else env.reset()
    done   = False
    rewards = []

    while not done:
        cont_state = env._get_state()

        if is_ql:
            action = agent.select_action(state, training=training)
        elif is_ppo:
            action, lp = agent.select_action(cont_state, training=training)
        else:
            action = agent.select_action(cont_state, training=training)

        if is_ql:
            next_state, reward, done, info = disc_env.step(action)
        else:
            next_state, reward, done, info = env.step(action)

        rewards.append(reward)

        if training:
            if is_ql:
                agent.update(state, action, reward, next_state, done)
            elif is_ppo:
                agent.store(cont_state, action, reward, lp)
            else:
                agent.store(cont_state, action, reward, next_state, done)
                agent.update()

        state = next_state

    if training:
        if is_ql:
            agent.decay_epsilon()
        elif is_ppo:
            agent.update()
        else:
            agent.end_episode()

    metrics = disc_env.get_episode_metrics() if is_ql else env.get_episode_metrics()
    return float(sum(rewards)), metrics


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING  (mode='train' env, LR decay)
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(env, agent, n_episodes, agent_name='', verbose_every=100):
    is_ql    = isinstance(agent, QLearningAgent)
    disc_env = DiscreteStateWrapper(env, QL_INV_BINS) if is_ql else None
    history  = []
    tc       = TRAIN_CONFIG
    decay_interval = tc.get('lr_decay_interval', 100)

    print(f"\n  Training {agent_name} ({n_episodes} episodes)...")
    for ep in range(n_episodes):
        ep_reward, metrics = _run_episode(env, agent, disc_env, training=True)
        history.append(ep_reward)

        # Cosine LR decay milestone
        if (ep + 1) % decay_interval == 0 and not is_ql:
            lr_min = tc.get('dqn_lr_min', 0.0001)
            agent.decay_lr(ep + 1, n_episodes, lr_min)

        if (ep + 1) % verbose_every == 0:
            avg   = np.mean(history[-verbose_every:])
            eps   = getattr(agent, 'epsilon', '-')
            eps_s = f'{eps:.3f}' if isinstance(eps, float) else '-'
            # Show current LR for neural agents
            cur_lr = getattr(getattr(agent, 'online', None), 'lr',
                     getattr(getattr(agent, 'actor',  None), 'lr', '-'))
            lr_s  = f'{cur_lr:.5f}' if isinstance(cur_lr, float) else '-'
            print(f"    Ep {ep+1:4d}/{n_episodes} | "
                  f"AvgReward: {avg:10.1f} | "
                  f"Eps: {eps_s} | LR: {lr_s} | "
                  f"Waste%: {metrics['waste_pct']:5.1f} | "
                  f"SL%: {metrics['service_level']:5.1f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION  (mode='test' env — OUT-OF-SAMPLE)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(env, agent, n_episodes=100, agent_name=''):
    """Evaluate on mode='test' environment (last 20% of data)."""
    is_ql    = isinstance(agent, QLearningAgent)
    disc_env = DiscreteStateWrapper(env, QL_INV_BINS) if is_ql else None
    all_m, rew = [], []

    for _ in range(n_episodes):
        ep_r, m = _run_episode(env, agent, disc_env, training=False)
        all_m.append(m); rew.append(ep_r)

    keys  = all_m[0].keys()
    mean  = {k: float(np.mean([m[k] for m in all_m])) for k in keys}
    std   = {k: float(np.std( [m[k] for m in all_m])) for k in keys}
    mean['reward_std']   = float(np.std(rew))
    mean['sl_std']       = std.get('service_level', 0.0)
    mean['fin_pm_std']   = std.get('financial_pm',  0.0)

    print(f"  [{agent_name:<14}] "
          f"Reward:{mean['total_reward']:10.1f} | "
          f"FinPM:{mean['financial_pm']:6.1f}±{mean['fin_pm_std']:4.1f}% | "
          f"RLPM:{mean['rl_adjusted_pm']:6.1f}% | "
          f"SL:{mean['service_level']:5.1f}±{mean['sl_std']:4.1f}% | "
          f"Waste:{mean['waste_pct']:4.1f}%")
    return mean, all_m, rew


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-RESTAURANT TRANSFER EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cross_restaurant_transfer(
        source_agents: dict,
        target_cfg: dict,
        target_csv: str | None = None,
        n_episodes: int = 50,
        seed: int = 42) -> dict:
    """
    Zero-shot transfer test: evaluate agents trained on source restaurants
    on the held-out target restaurant's TEST split.

    Parameters
    ----------
    source_agents : dict  — {agent_name: trained_agent}
    target_cfg    : dict  — scenario config of the held-out restaurant
    target_csv    : str   — path to target restaurant's CSV
    n_episodes    : int
    seed          : int

    Returns
    -------
    dict  {agent_name: mean_metrics}
    """
    # Build test-mode env for the target restaurant
    test_env = RestaurantInventoryEnv(
        target_cfg,
        empirical_csv=target_csv,
        seed=seed,
        mode='test',
        init_inv_noise=0.20,
    )

    print(f"\n  [TRANSFER] Evaluating on {target_cfg['restaurant_name']} "
          f"(test split, zero-shot)...")
    results = {}
    for name, agent in source_agents.items():
        mean_m, _, _ = evaluate_agent(test_env, agent,
                                      n_episodes=n_episodes,
                                      agent_name=f"{name}→{target_cfg['restaurant_id']}")
        results[name] = mean_m
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SMOOTH
# ─────────────────────────────────────────────────────────────────────────────

def smooth(values, window=20):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')
