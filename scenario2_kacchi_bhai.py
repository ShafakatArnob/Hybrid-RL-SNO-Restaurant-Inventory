"""
scenario2_kacchi_bhai.py  —  v5
================================
Scenario 2: Multi-Restaurant Bangladeshi Study.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import json

from config import (ALL_RESTAURANTS, KACCHI_BHAI, TRAIN_CONFIG,
                    STATE_DIM, QL_INV_BINS, RANDOM_SEED, build_action_space)
from restaurant_env import RestaurantInventoryEnv, DiscreteStateWrapper
from rl_agents import build_agents
from classical_baselines import build_baselines
from training_loop import (train_agent, evaluate_agent,
                            evaluate_cross_restaurant_transfer)

OUTPUT_DIR = Path('./results/scenario2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _wide_csv(cfg):
    base = Path(cfg['data_csv'])
    wide = base.parent / base.name.replace('_daily_orders.csv', '_wide.csv')
    if wide.exists():  return str(wide)
    if base.exists():  return str(base)
    return None


def _print_restaurant_header(cfg, csv_path):
    print(f"\n{'='*70}")
    print(f"  RESTAURANT: {cfg['restaurant_name']}")
    print(f"  {cfg['cuisine_type']} | {cfg['location']}")
    n_days = cfg['total_days']
    split  = int(n_days * cfg['train_split_ratio'])
    print(f"  Dataset: {n_days} days  |  Train: days 0–{split}  "
          f"|  Test: days {split}–{n_days}")
    print(f"  Lead time: {cfg.get('lead_time',1)}d  "
          f"| Reward scale: {cfg.get('reward_scale',1.0):.1f}  "
          f"| Ep-len train: {cfg.get('episode_length_train',30)}d")
    print(f"  Ramadan: day {cfg.get('ramadan_start','N/A')}  "
          f"| Eid: {cfg.get('eid_days','N/A')}")
    print(f"{'='*70}")
    if csv_path:
        df = pd.read_csv(csv_path)
        meals = [m for m in cfg['meals'] if m in df.columns]
        print(f"  Dataset: {len(df):,} rows | "
              f"Empirical demand mode: {'yes' if meals else 'no'}")
        for m in meals:
            print(f"    {m:<26}: mean {df[m].mean():.1f} orders/day")


def run_scenario2(train_cfg=None, seed=RANDOM_SEED,
                  restaurants=None, verbose_every=160) -> dict:
    tc          = train_cfg or TRAIN_CONFIG
    restaurants = restaurants or ALL_RESTAURANTS

    print("\n" + "=" * 70)
    print("  SCENARIO 2 — MULTI-RESTAURANT BANGLADESHI STUDY")
    print("  (Empirical demand | Train/test temporal split | LR decay)")
    print("=" * 70)

    # ── Loading message (no generator reference) ─────────────────────────────
    print("  Loading restaurant datasets...")
    for cfg in ALL_RESTAURANTS:
        p = _wide_csv(cfg)
        if p and Path(p).exists():
            n = len(pd.read_csv(p))
            print(f"    ✓ {cfg['restaurant_name']}: {n:,} rows → data")
        else:
            print(f"    ⚠ {cfg['restaurant_name']}: CSV not found in data/")

    all_results  = {}
    all_agents_by_restaurant = {}   # saved for cross-restaurant transfer

    for cfg in restaurants:
        csv_path = _wide_csv(cfg)
        _print_restaurant_header(cfg, csv_path)

        _, order_matrix = build_action_space(cfg)
        n_actions       = len(list(
            __import__('itertools').product(range(3), repeat=cfg['n_ingredients'])))

        # Train env: mode='train', variable episode length
        train_env = RestaurantInventoryEnv(
            cfg, empirical_csv=csv_path, seed=seed,
            mode='train', init_inv_noise=0.20)

        # Test env: mode='test', fixed 30-day episodes, out-of-sample
        test_env  = RestaurantInventoryEnv(
            cfg, empirical_csv=csv_path, seed=seed+1,
            mode='test',  init_inv_noise=0.20)

        disc_train = DiscreteStateWrapper(train_env, QL_INV_BINS)
        n_states   = disc_train.n_states

        mode_str = ('empirical-train' if train_env.emp_demand_by_dow_train
                    else 'parametric')
        print(f"\n  Demand mode : {mode_str}")
        print(f"  Actions     : {n_actions} | State: {STATE_DIM}-dim")

        agents    = build_agents(STATE_DIM, n_states, n_actions, tc, seed=seed)
        baselines = build_baselines(cfg, order_matrix)

        n_train = {
            'Q-Learning' : tc['ql_episodes'],
            'DQN'        : tc['dqn_episodes'],
            'Double DQN' : tc['dqn_episodes'],
            'PPO'        : tc['ppo_episodes'],
        }

        print(f"\n  {'─'*60}")
        print("  TRAINING  (mode=train — first 80% of dataset)")
        print(f"  {'─'*60}")
        reward_curves = {}
        for name, agent in agents.items():
            curves = train_agent(train_env, agent, n_train[name],
                                 agent_name=name, verbose_every=verbose_every)
            reward_curves[name] = curves

        print(f"\n  {'─'*60}")
        print(f"  EVALUATION  (mode=test — last 20%, {tc['eval_episodes']} eps, out-of-sample)")
        print(f"  {'─'*60}")
        results = {}
        for name, agent in {**agents, **baselines}.items():
            mean_m, all_m, rew_hist = evaluate_agent(
                test_env, agent,
                n_episodes=tc['eval_episodes'],
                agent_name=name)
            results[name] = {'mean_metrics': mean_m, 'reward_history': rew_hist}

        _save_restaurant_results(cfg, results, reward_curves)
        all_results[cfg['restaurant_id']] = {
            'config'        : cfg,
            'results'       : results,
            'reward_curves' : reward_curves,
            'demand_mode'   : mode_str,
            'split_day'     : train_env.split_day,
        }
        all_agents_by_restaurant[cfg['restaurant_id']] = agents

    # ── Cross-restaurant summary ──────────────────────────────────────────────
    _print_cross_summary(all_results)

    # ── Cross-restaurant transfer evaluation (Fix-3: held-out test) ──────────
    if len(restaurants) == len(ALL_RESTAURANTS):
        _run_transfer_evaluation(all_results, all_agents_by_restaurant,
                                 tc, seed)

    _save_combined(all_results)
    return all_results


def _run_transfer_evaluation(all_results, all_agents, tc, seed):
    """
    Zero-shot transfer: use Kachchi Bhai as held-out restaurant.
    Pool agents trained on the other 4 restaurants, evaluate on KB test split.
    """
    print("\n" + "=" * 70)
    print("  CROSS-RESTAURANT TRANSFER TEST")
    print("  Held-out: Kachchi Bhai  |  Source: Star Kabab + Fakruddin + Madchef + Bismillah")
    print("=" * 70)

    # Aggregate (average) source-restaurant agents — pick DQN for transfer
    # (most stable across restaurants in Scenario 2)
    source_rids = [r for r in all_results if r != 'kacchi_bhai']
    if not source_rids:
        return

    # Use the best-performing agent from each source restaurant
    source_agents_by_name = {}
    rl_names = ['DQN', 'Double DQN', 'PPO', 'Q-Learning']
    for agent_name in rl_names:
        # Collect all source agents with this name — pick the one with best FinPM
        best_agent = None
        best_pm    = -9999
        for rid in source_rids:
            ag   = all_agents[rid].get(agent_name)
            pm   = all_results[rid]['results'].get(agent_name, {}).get(
                       'mean_metrics', {}).get('financial_pm', -9999)
            if ag is not None and pm > best_pm:
                best_agent = ag
                best_pm    = pm
        if best_agent is not None:
            source_agents_by_name[agent_name] = best_agent

    # Evaluate on Kachchi Bhai test split
    kb_csv  = _wide_csv(KACCHI_BHAI)
    transfer_results = evaluate_cross_restaurant_transfer(
        source_agents  = source_agents_by_name,
        target_cfg     = KACCHI_BHAI,
        target_csv     = kb_csv,
        n_episodes     = tc.get('eval_episodes', 100) // 2,
        seed           = seed + 99,
    )

    print("\n  Transfer Results (agents trained on 4 other restaurants → Kachchi Bhai test):")
    for name, m in transfer_results.items():
        print(f"  [{name:<14}] FinPM:{m['financial_pm']:6.1f}% | "
              f"SL:{m['service_level']:5.1f}% | Waste:{m['waste_pct']:4.1f}%")

    # Save
    rows = [{'Agent': n, 'Source': 'Star Kabab+Fakruddin+Madchef+Bismillah',
             'Target': 'Kachchi Bhai (test split)',
             **{k: round(float(v),2) for k,v in m.items()}}
            for n, m in transfer_results.items()]
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / 'cross_restaurant_transfer.csv', index=False)
    print(f"\n  Transfer results → {OUTPUT_DIR}/cross_restaurant_transfer.csv")

    # Store in all_results for figures
    all_results['_transfer'] = transfer_results


def _print_cross_summary(all_results):
    print("\n" + "=" * 78)
    print("  CROSS-RESTAURANT SUMMARY — SCENARIO 2")
    print("=" * 78)
    print(f"  {'Restaurant':<28} {'Best RL Agent':<15} {'Best RL FinPM':>14} "
          f"{'Best BL FinPM':>14} {'Split Day':>10}")
    print(f"  {'─'*74}")
    rl_agents = ['Q-Learning','DQN','Double DQN','PPO']
    bl_agents = ['EOQ','Newsvendor','SMA + Reorder','Seasonal Reorder']
    for rid, data in all_results.items():
        if rid.startswith('_'):
            continue
        cfg = data['config']
        res = data['results']
        rl_pms = {a: res[a]['mean_metrics']['financial_pm']
                  for a in rl_agents if a in res}
        bl_pms = {a: res[a]['mean_metrics']['financial_pm']
                  for a in bl_agents if a in res}
        best_rl  = max(rl_pms, key=rl_pms.get) if rl_pms else 'N/A'
        best_rv  = rl_pms.get(best_rl, float('nan'))
        best_bv  = max(bl_pms.values()) if bl_pms else float('nan')
        sd       = data.get('split_day', '-')
        print(f"  {cfg['restaurant_name']:<28} {best_rl:<15} "
              f"{best_rv:>13.1f}%  {best_bv:>13.1f}%  {str(sd):>10}")


def _save_restaurant_results(cfg, results, curves):
    rid   = cfg['restaurant_id']
    r_dir = OUTPUT_DIR / rid
    r_dir.mkdir(exist_ok=True)
    rows = []
    for name, d in results.items():
        m = d['mean_metrics']
        rows.append({
            'Agent'          : name,
            'Restaurant'     : cfg['restaurant_name'],
            'Total Reward'   : round(m['total_reward'],   2),
            'Financial PM %' : round(m['financial_pm'],   2),
            'RL-Adj PM %'    : round(m['rl_adjusted_pm'], 2),
            'Service Level %': round(m['service_level'],  2),
            'SL Std %'       : round(m.get('sl_std',0),   2),
            'Waste %'        : round(m['waste_pct'],       2),
            'Stockout %'     : round(m['stockout_rate'],   2),
        })
    pd.DataFrame(rows).to_csv(r_dir / f'{rid}_results.csv', index=False)
    cs = {k: [float(v) for v in vs] for k, vs in curves.items()}
    with open(r_dir / f'{rid}_reward_curves.json', 'w') as f:
        json.dump(cs, f)
    det = {n: {k: round(float(v),4) for k,v in d['mean_metrics'].items()}
           for n, d in results.items()}
    with open(r_dir / f'{rid}_detailed_metrics.json', 'w') as f:
        json.dump(det, f, indent=2)


def _save_combined(all_results):
    rows = []
    for rid, data in all_results.items():
        if rid.startswith('_'):
            continue
        cfg = data['config']
        for name, d in data['results'].items():
            m = d['mean_metrics']
            rows.append({
                'Restaurant'    : cfg['restaurant_name'],
                'Agent'         : name,
                'Financial PM %': round(m['financial_pm'],   2),
                'RL-Adj PM %'   : round(m['rl_adjusted_pm'], 2),
                'Service Level %': round(m['service_level'], 2),
                'Waste %'       : round(m['waste_pct'],       2),
                'Stockout %'    : round(m['stockout_rate'],   2),
                'Demand Mode'   : data['demand_mode'],
                'Split Day'     : data.get('split_day', '-'),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'scenario2_combined.csv', index=False)
    print(f"\n  Combined results → {OUTPUT_DIR}/scenario2_combined.csv")
