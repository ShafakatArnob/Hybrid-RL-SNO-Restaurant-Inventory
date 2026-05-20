"""
scenario1_generic.py  —  v5
============================
Scenario 1: Generic Restaurant Framework.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import json
import pandas as pd

from config import (GENERIC, TRAIN_CONFIG, STATE_DIM, QL_INV_BINS,
                    RANDOM_SEED, build_action_space)
from restaurant_env import RestaurantInventoryEnv, DiscreteStateWrapper
from rl_agents import build_agents
from classical_baselines import build_baselines
from training_loop import train_agent, evaluate_agent
from data_loader import load_all_scenario1

OUTPUT_DIR = Path('./results/scenario1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _print_dataset_summary(calib, results):
    print("\n" + "=" * 70)
    print("  DATASETS — SCENARIO 1")
    print("=" * 70)
    ms = calib['meal_stats']
    cs = calib['centre_stats']
    pb = calib['price_by_category']
    ti = calib['top_ingredients']
    g_price = next((v for k, v in pb.items() if 'grain' in k.lower()), 2.75)
    p_price = next((v for k, v in pb.items() if 'seafood' in k.lower()), 9.50)
    g_price = g_price if g_price == g_price else 2.75
    p_price = p_price if p_price == p_price else 9.50

    print(f"\n  [CORE — 4 Small Public Datasets]")
    print(f"    1. meal_info.csv        : {ms['n_items']} items, "
          f"{ms['n_categories']} categories, {ms['n_cuisines']} cuisines")
    print(f"    2. Grocery Dataset       : {len(results['grocery']):,} SKUs | "
          f"Grains ${g_price:.2f}, Seafood ${p_price:.2f} per unit")
    print(f"    3. Fulfilment Centres    : {cs['n_centres']} centres | "
          f"mean reorder level {cs['mean_reorder']}")
    print(f"    4. Food Ingredients CSV  : top ingredient: '{ti[0] if ti else 'salt'}'")

    print(f"\n  [EXTENDED — 4 Large Public Datasets]")
    for name, size, use in [
        ("5. Instacart 2017",     "3,421,083 orders",  "Demand + reorder calibration"),
        ("6. Open Food Facts",    "2.7M+ products",    "Shelf-life proxies"),
        ("7. Zomato Restaurants", "9,551 restaurants", "Cuisine demand weights"),
        ("8. UCI Online Retail",  "541,909 transactions","Reorder quantity benchmarking"),
    ]:
        print(f"    {name:<30}: {size:<28} → {use}")

    if calib.get('dow_profile') is not None:
        dp = calib['dow_profile']
        dow_map = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        peak = dow_map[int(dp.loc[dp['relative_demand'].idxmax(), 'order_dow'])]
        print(f"\n  [Instacart insight] Peak grocery day: {peak}")
    if calib.get('top_cuisines'):
        top3 = [c for c, _ in calib['top_cuisines'][:3]]
        print(f"  [Zomato insight]    Top 3 cuisines: {', '.join(top3)}")
    print()


def run_scenario1(train_cfg=None, seed=RANDOM_SEED, verbose_every=160) -> dict:
    tc = train_cfg or TRAIN_CONFIG

    print("\n" + "=" * 70)
    print("  SCENARIO 1 — GENERIC RESTAURANT FRAMEWORK")
    print("  (Calibrated from 4 small + 4 large public datasets)")
    print("=" * 70)

    data = load_all_scenario1(verbose=True)
    _print_dataset_summary(data['calibration'], data)

    _, order_matrix = build_action_space(GENERIC)

    # TRAIN environment (mode='train': days 0–480, variable episode length)
    train_env = RestaurantInventoryEnv(GENERIC, seed=seed,
                                       mode='train', init_inv_noise=0.20)
    # TEST environment (mode='test': days 480–600, fixed 30-day episodes)
    test_env  = RestaurantInventoryEnv(GENERIC, seed=seed+1,
                                       mode='test',  init_inv_noise=0.20)

    disc_train = DiscreteStateWrapper(train_env, QL_INV_BINS)
    n_states   = disc_train.n_states
    n_actions  = train_env.n_actions

    split_pct = int(GENERIC['train_split_ratio'] * 100)
    print(f"\n  Environment : {GENERIC['scenario_name']}")
    print(f"  State dim   : {STATE_DIM} continuous | {n_states:,} discrete (Q-Learning)")
    print(f"  Actions     : {n_actions} joint combinations")
    print(f"  Train split : first {split_pct}% of virtual calendar (days 0–480)")
    print(f"  Test split  : final {100-split_pct}% of virtual calendar (days 480–600)")
    print(f"  Ep len train: variable 20–{GENERIC['episode_length_train']} days")
    print(f"  Ep len eval : fixed {30} days (out-of-sample)")
    print(f"  Reward scale: {GENERIC['reward_scale']} (already USD, no scaling needed)")
    print(f"  Lead time   : {GENERIC['lead_time']} day(s)")
    print(f"\n  [Q-Learning] {GENERIC['ql_note'][:90]}...")

    agents    = build_agents(STATE_DIM, n_states, n_actions, tc, seed=seed)
    baselines = build_baselines(GENERIC, order_matrix)

    n_train = {
        'Q-Learning' : tc['ql_episodes'],
        'DQN'        : tc['dqn_episodes'],
        'Double DQN' : tc['dqn_episodes'],
        'PPO'        : tc['ppo_episodes'],
    }

    print(f"\n  {'─'*60}")
    print("  TRAINING PHASE  (mode=train — first 80% of virtual calendar)")
    print(f"  {'─'*60}")
    reward_curves = {}
    for name, agent in agents.items():
        curves = train_agent(train_env, agent, n_train[name],
                             agent_name=name, verbose_every=verbose_every)
        reward_curves[name] = curves

    print(f"\n  {'─'*60}")
    print(f"  EVALUATION ({tc['eval_episodes']} eps, mode=test — last 20%, out-of-sample)")
    print(f"  {'─'*60}")
    results = {}
    for name, agent in {**agents, **baselines}.items():
        mean_m, all_m, rew_hist = evaluate_agent(
            test_env, agent,
            n_episodes=tc['eval_episodes'],
            agent_name=name)
        results[name] = {'mean_metrics': mean_m, 'reward_history': rew_hist}

    _save_results(results, reward_curves)
    print(f"\n  Results saved → {OUTPUT_DIR}/")
    return {'results': results, 'reward_curves': reward_curves,
            'calibration': data['calibration']}


def _save_results(results, curves):
    rows = []
    for name, d in results.items():
        m = d['mean_metrics']
        rows.append({
            'Agent'          : name,
            'Scenario'       : 'Generic',
            'Total Reward'   : round(m['total_reward'],   2),
            'Financial PM %' : round(m['financial_pm'],   2),
            'RL-Adj PM %'    : round(m['rl_adjusted_pm'], 2),
            'Service Level %': round(m['service_level'],  2),
            'SL Std %'       : round(m.get('sl_std', 0),  2),
            'Waste %'        : round(m['waste_pct'],       2),
            'Stockout %'     : round(m['stockout_rate'],   2),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'scenario1_results.csv', index=False)
    with open(OUTPUT_DIR / 'scenario1_reward_curves.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in curves.items()}, f)
    detailed = {n: {k: round(float(v), 4) for k, v in d['mean_metrics'].items()}
                for n, d in results.items()}
    with open(OUTPUT_DIR / 'scenario1_detailed_metrics.json', 'w') as f:
        json.dump(detailed, f, indent=2)
