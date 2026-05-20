"""
visualisations.py
=================
Generates all RL figures for both scenarios.

S1-1  Training reward curves (Scenario 1)
S1-2  KPI comparison bar chart — dual PM columns (Scenario 1)
S1-3  Bipartite ingredient-meal network (Scenario 1)
S1-4  Multi-KPI radar chart (Scenario 1)
S2-1  Revenue trend (Kachchi Bhai — loaded from eda_scenario2)
S2-2  Meal breakdown stacked area (Kachchi Bhai)
S2-3  Demand heatmap (Kachchi Bhai)
S2-4  Spoilage (Kachchi Bhai)
S2-5  Training curves (Scenario 2 representative restaurant)
S2-6  KPI comparison — dual PM (Scenario 2 representative)
S2-7  Bipartite network (Kachchi Bhai)
S2-8  Radar chart (Scenario 2 representative)
CS-1  Cross-scenario KPI side-by-side
CS-2  RL improvement over best baseline (both scenarios)
CS-3  Cross-restaurant Financial PM comparison (5 restaurants)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import GENERIC, KACCHI_BHAI, ALL_RESTAURANTS

OUT_S1 = Path('./results/scenario1')
OUT_S2 = Path('./results/scenario2')
OUT_CS = Path('./results')
for d in [OUT_S1, OUT_S2, OUT_CS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
AGENT_COLORS = {
    'Q-Learning'      : '#E74C3C',
    'DQN'             : '#2E75B6',
    'Double DQN'      : '#27AE60',
    'PPO'             : '#8E44AD',
    'EOQ'             : '#F39C12',
    'Newsvendor'      : '#1ABC9C',
    'SMA + Reorder'   : '#95A5A6',
    'Seasonal Reorder': '#E67E22',
}
RL_AGENTS  = ['Q-Learning', 'DQN', 'Double DQN', 'PPO']
BASELINES  = ['EOQ', 'Newsvendor', 'SMA + Reorder', 'Seasonal Reorder']
ALL_AGENTS = RL_AGENTS + BASELINES

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 9,
    'figure.dpi'       : 120,
})


def _save(path: Path, name: str):
    full = path / name
    plt.savefig(full, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


def _smooth(arr, w=20):
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='same')


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(curves: dict, label: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in RL_AGENTS:
        if name not in curves:
            continue
        raw = np.array(curves[name], dtype=float)
        sm  = _smooth(raw, 30)
        c   = AGENT_COLORS[name]
        ax.plot(raw, alpha=0.12, color=c)
        ax.plot(sm,  lw=2.2, color=c,
                label=f'{name} (†)' if name == 'Q-Learning' else name)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title(f'Training Reward Curves — {label}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.axhline(0, color='grey', lw=0.7, ls='--')
    if curves:
        mid = max(len(v) for v in curves.values()) // 2
        ax.axvline(mid, color='grey', lw=0.7, ls=':', alpha=0.6)
        ax.text(mid, ax.get_ylim()[1] * 0.95,
                'Exploration→Exploitation', fontsize=7, color='grey', ha='center')
    # Q-Learning footnote
    ax.annotate('† Q-Learning uses discretised state (low-capacity benchmark)',
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=7, color='#E74C3C', style='italic')
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  KPI COMPARISON BAR CHART — DUAL PM
# ─────────────────────────────────────────────────────────────────────────────

def plot_kpi_comparison(results: dict, label: str, out_path: Path,
                        std_metrics: dict | None = None):
    """
    Grouped bar chart for Financial PM, RL-Adj PM, Service Level, Waste%.
    Dual PM columns implement Major-3 (faculty) fix.
    Error bars shown if std_metrics provided (Minor-2).
    """
    kpis = ['financial_pm', 'rl_adjusted_pm', 'service_level', 'waste_pct']
    klabels = ['Financial PM %\n(excl. stockout)', 'RL-Adjusted PM %\n(incl. stockout)',
               'Service Level %', 'Waste %']
    better  = ['↑ higher', '↑ higher', '↑ higher', '↓ lower']

    agents  = [a for a in ALL_AGENTS if a in results]
    x       = np.arange(len(agents))
    colors  = [AGENT_COLORS.get(a, '#888') for a in agents]
    rl_mask = [a in RL_AGENTS for a in agents]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    for i, (kpi, klab, bet) in enumerate(zip(kpis, klabels, better)):
        vals  = [results[a]['mean_metrics'].get(kpi, 0) for a in agents]
        errs  = ([std_metrics[a].get(kpi, 0) for a in agents]
                 if std_metrics else None)
        bars  = axes[i].bar(x, vals, color=colors, edgecolor='white',
                             width=0.7,
                             yerr=errs, capsize=3 if errs else 0,
                             error_kw=dict(elinewidth=1, ecolor='black'))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([a.replace(' ','\n') for a in agents],
                                  fontsize=7)
        axes[i].set_title(f'{klab}\n({bet})', fontweight='bold', fontsize=9)
        for bar, v in zip(bars, vals):
            ypos = bar.get_height() + (max(vals) * 0.02 if max(vals) != 0 else 0.5)
            axes[i].text(bar.get_x()+bar.get_width()/2, ypos,
                         f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
        # Shade RL region
        axes[i].axvspan(-0.5, sum(rl_mask) - 0.5, alpha=0.05, color='steelblue')
        axes[i].axvline(sum(rl_mask) - 0.5, color='grey', lw=0.5, ls='--')

    rl_p  = mpatches.Patch(color='steelblue', alpha=0.15, label='RL Agents')
    bl_p  = mpatches.Patch(facecolor='#F5F5F5', edgecolor='grey',
                            label='Classical Baselines')
    fig.legend(handles=[rl_p, bl_p], loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle(f'KPI Comparison — {label}', fontsize=12, fontweight='bold', y=1.07)

    if std_metrics:
        fig.text(0.99, 0.01, 'Error bars = ±1 std across seeds',
                 ha='right', fontsize=7, color='grey', style='italic')

    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  BIPARTITE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def plot_bipartite_network(cfg: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    ax.axis('off')

    ings    = cfg['ingredients']
    meals   = cfg['meals']
    network = cfg['meal_ingredient_network']
    price_k = 'meal_prices_usd' if cfg['currency'] == 'USD' else 'meal_prices_bdt'
    prices  = cfg[price_k]
    sym     = '$' if cfg['currency'] == 'USD' else 'BDT '

    ing_y  = np.linspace(0.1, 0.9, len(ings))
    meal_y = np.linspace(0.1, 0.9, len(meals))
    ing_idx  = {n: i for i, n in enumerate(ings)}
    meal_idx = {n: i for i, n in enumerate(meals)}

    # Edges
    for meal, reqs in network.items():
        m = meal_idx[meal]
        for ing, qty in reqs.items():
            if ing in ing_idx:
                ii = ing_idx[ing]
                ax.plot([0.22, 0.78], [ing_y[ii], meal_y[m]],
                        color='grey', lw=0.5 + qty * 0.35, alpha=0.50, zorder=1)

    # Ingredient nodes
    ing_colors = ['#EBF5FB','#FDEBD0','#EAF5F5','#FDFEFE','#F9F0FF']
    for i, (name, y) in enumerate(zip(ings, ing_y)):
        c = plt.Circle((0.22, y), 0.045, color=ing_colors[i % 5],
                        ec='#555', lw=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(0.0, y, name, ha='left', va='center', fontsize=8.5,
                fontweight='bold', color='#1A3A5C')

    # Meal nodes
    meal_colors = ['#FDEDEC','#EBF5FB','#EAFAF1','#FEF9E7','#F4ECF7']
    for m, (name, y) in enumerate(zip(meals, meal_y)):
        c = plt.Circle((0.78, y), 0.045, color=meal_colors[m % 5],
                        ec='#555', lw=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(1.0, y, f'{name}\n({sym}{prices[m]:,.0f})',
                ha='right', va='center', fontsize=8, color='#1A3A5C')

    ax.text(0.22, 1.02, 'Supply Nodes\n(Ingredients)', ha='center',
            fontsize=10, fontweight='bold', color='#1A3A5C')
    ax.text(0.78, 1.02, 'Demand Nodes\n(Meals)', ha='center',
            fontsize=10, fontweight='bold', color='#1E6B2E')
    ax.set_title(f'Bipartite Ingredient–Meal Network — '
                 f'{cfg.get("scenario_name", cfg.get("restaurant_name", ""))}',
                 fontweight='bold', pad=20)
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(results: dict, label: str, out_path: Path):
    categories = ['Financial\nPM', 'RL-Adj\nPM', 'Service\nLevel',
                  'Low\nWaste', 'Low\nStockout']
    N      = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    agents  = [a for a in ALL_AGENTS if a in results]

    for name in agents:
        m   = results[name]['mean_metrics']
        fpm = np.clip(m.get('financial_pm',   0) + 50, 0, 150) / 150
        rpm = np.clip(m.get('rl_adjusted_pm', 0) + 50, 0, 150) / 150
        sl  = m.get('service_level', 0) / 100
        lw  = 1 - np.clip(m.get('waste_pct',    0) / 100, 0, 1)
        ls_ = 1 - np.clip(m.get('stockout_rate', 0) / 100, 0, 1)
        vals = [fpm, rpm, sl, lw, ls_] + [fpm]
        ls  = '-' if name in RL_AGENTS else '--'
        lw_ = 2.2 if name in RL_AGENTS else 1.5
        ax.plot(angles, vals, ls, lw=lw_, color=AGENT_COLORS.get(name,'grey'),
                label=f'{name}{"†" if name=="Q-Learning" else ""}')
        ax.fill(angles, vals, color=AGENT_COLORS.get(name,'grey'), alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(f'Multi-KPI Radar — {label}', fontweight='bold',
                 pad=22, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15),
              fontsize=8, framealpha=0.9)
    ax.annotate('† Q-Learning: low-capacity benchmark',
                xy=(0.01, -0.05), xycoords='axes fraction',
                fontsize=7, color='#E74C3C', style='italic')
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-SCENARIO KPI
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_scenario(s1_results: dict, s2_rep_results: dict,
                        out_path: Path):
    """CS-1: Side-by-side KPI for Scenario 1 vs best Scenario 2 restaurant."""
    kpis   = ['financial_pm', 'service_level', 'waste_pct', 'stockout_rate']
    labels = ['Financial PM %', 'Service Level %', 'Waste %', 'Stockout %']
    bett   = ['↑', '↑', '↓', '↓']
    agents = [a for a in ALL_AGENTS if a in s1_results and a in s2_rep_results]
    x, w   = np.arange(len(agents)), 0.35
    colors = [AGENT_COLORS.get(a, '#888') for a in agents]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, (kpi, lab, bet) in enumerate(zip(kpis, labels, bett)):
        v1 = [s1_results[a]['mean_metrics'].get(kpi, 0)     for a in agents]
        v2 = [s2_rep_results[a]['mean_metrics'].get(kpi, 0) for a in agents]
        axes[i].bar(x - w/2, v1, w, label='Generic (Sc.1)',
                    color=colors, alpha=0.7, edgecolor='white')
        axes[i].bar(x + w/2, v2, w, label='Kachchi Bhai (Sc.2)',
                    color=colors, alpha=1.0, hatch='//', edgecolor='white')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([a.replace(' ','\n') for a in agents], fontsize=7.5)
        axes[i].set_title(f'{lab} ({bet} better)', fontweight='bold')
        if i == 0:
            axes[i].legend(fontsize=9)
        axes[i].axvspan(-0.5, len(RL_AGENTS) - 0.5, alpha=0.04, color='steelblue')

    fig.suptitle('Cross-Scenario KPI Comparison\n'
                 'Generic Restaurant (Sc.1)  vs  Kachchi Bhai (Sc.2)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  RL IMPROVEMENT OVER BEST BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_rl_improvement(s1_results: dict, s2_rep_results: dict,
                        out_path: Path):
    """CS-2: % improvement of RL agents over best classical baseline."""
    kpis = ['financial_pm', 'service_level']
    labs = ['Financial PM Improvement (%)', 'Service Level Improvement (%)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rl_here = [a for a in RL_AGENTS if a in s1_results]
    x  = np.arange(len(rl_here)); w = 0.35

    for ax, kpi, lab in zip(axes, kpis, labs):
        for offset, sc_label, results in [
            (-w/2, 'Generic (Sc.1)',      s1_results),
            ( w/2, 'Kachchi Bhai (Sc.2)', s2_rep_results),
        ]:
            bls = [results[b]['mean_metrics'].get(kpi, 0)
                   for b in BASELINES if b in results]
            if not bls:
                continue
            best_bl = max(bls)
            impr = [((results[a]['mean_metrics'].get(kpi, 0) - best_bl)
                     / (abs(best_bl) + 1e-9) * 100)
                    for a in rl_here]
            ax.bar(x + offset, impr, w, label=sc_label,
                   color=[AGENT_COLORS.get(a, '#888') for a in rl_here],
                   alpha=0.85 if 'Generic' in sc_label else 1.0)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(rl_here, fontsize=9)
        ax.set_title(lab, fontweight='bold')
        ax.set_ylabel('Improvement vs Best Baseline (%)')
        ax.legend(fontsize=9)

    fig.suptitle('RL Improvement over Best Classical Baseline',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-RESTAURANT FINANCIAL PM  (new — CS-3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_restaurant(all_results: dict, out_path: Path):
    """CS-3: Financial PM for each RL agent across all 5 restaurants."""
    rids  = [r for r in all_results if not r.startswith('_')]
    if not rids:
        return
    names = [all_results[r]['config']['restaurant_name'] for r in rids]
    agents = [a for a in RL_AGENTS
              if all(a in all_results[r]['results'] for r in rids)]
    if not agents or not rids:
        return

    x  = np.arange(len(rids))
    w  = 0.8 / len(agents)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, agent in enumerate(agents):
        vals = [all_results[r]['results'][agent]['mean_metrics'].get('financial_pm', 0)
                for r in rids]
        ax.bar(x + i*w - w*len(agents)/2 + w/2, vals, w,
               label=agent, color=AGENT_COLORS[agent], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8.5)
    ax.set_ylabel('Financial Profit Margin (%)')
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title('RL Agent Financial PM Across 5 Bangladeshi Restaurants\n'
                 '(Same framework, different cuisines, schemas, demand patterns)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(out_path.parent, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(s1_results, s1_curves,
                          s2_all,             # {rid: {results, curves, config, …}}
                          s1_std=None):
    print("\n" + "=" * 70)
    print("  GENERATING ALL RL FIGURES")
    print("=" * 70)

    # Pick representative S2 restaurant (Kachchi Bhai if available)
    real_rids  = [r for r in s2_all if not r.startswith('_')]
    s2_rep_id  = ('kacchi_bhai' if 'kacchi_bhai' in real_rids
                  else real_rids[0] if real_rids else None)
    if s2_rep_id is None:
        return
    s2_rep_res = s2_all[s2_rep_id]['results']
    s2_rep_crv = s2_all[s2_rep_id]['reward_curves']

    print("\n  [Scenario 1 — Generic]")
    plot_training_curves(s1_curves, 'Generic Restaurant Framework',
                         OUT_S1 / 'S1_1_training_curves.png')
    plot_kpi_comparison(s1_results, 'Generic Restaurant Framework',
                        OUT_S1 / 'S1_2_kpi_comparison.png',
                        std_metrics=s1_std)
    plot_bipartite_network(GENERIC,  OUT_S1 / 'S1_3_bipartite_network.png')
    plot_radar(s1_results, 'Generic Restaurant Framework',
               OUT_S1 / 'S1_4_radar_chart.png')

    print("\n  [Scenario 2 — Kachchi Bhai / representative restaurant]")
    _gen_s2_from_data()

    plot_training_curves(s2_rep_crv, f"Kachchi Bhai Restaurant",
                         OUT_S2 / 'S2_5_training_curves.png')
    plot_kpi_comparison(s2_rep_res, 'Kachchi Bhai Restaurant',
                        OUT_S2 / 'S2_6_kpi_comparison.png')
    plot_bipartite_network(KACCHI_BHAI, OUT_S2 / 'S2_7_bipartite_network.png')
    plot_radar(s2_rep_res, 'Kachchi Bhai Restaurant',
               OUT_S2 / 'S2_8_radar_chart.png')

    print("\n  [Cross-Scenario]")
    plot_cross_scenario(s1_results, s2_rep_res,
                        OUT_CS / 'CS_1_cross_scenario_kpi.png')
    plot_rl_improvement(s1_results, s2_rep_res,
                        OUT_CS / 'CS_2_rl_improvement.png')
    real_s2 = {r: v for r, v in s2_all.items() if not r.startswith('_')}
    if len(real_s2) > 1:
        plot_cross_restaurant(real_s2, OUT_CS / 'CS_3_cross_restaurant_pm.png')

    print("\n  All RL figures generated!")


def _gen_s2_from_data():
    """Generate S2 figures 1-4 from Kachchi Bhai data."""
    import pandas as pd_
    import matplotlib.pyplot as plt_

    data_dir = Path('./data')
    kb_f     = data_dir / 'kacchi_bhai_daily_orders.csv'
    if not kb_f.exists():
        return

    df = pd_.read_csv(kb_f)
    df['date'] = pd_.to_datetime(df['date'])

    from eda_scenario2 import (
        _fig_revenue_trend, _fig_dow_demand, _fig_heatmap,
        _fig_spoilage, _fig_revenue_mix
    )
    _fig_revenue_trend(df);  plt_.savefig(OUT_S2 / 'S2_1_revenue_trend.png',
                                          bbox_inches='tight'); plt_.close()
    _fig_dow_demand(df);     plt_.savefig(OUT_S2 / 'S2_2_meal_breakdown.png',
                                          bbox_inches='tight'); plt_.close()
    _fig_heatmap(df);        plt_.savefig(OUT_S2 / 'S2_3_demand_heatmap.png',
                                          bbox_inches='tight'); plt_.close()
    _fig_spoilage(df);       plt_.savefig(OUT_S2 / 'S2_4_spoilage.png',
                                          bbox_inches='tight'); plt_.close()
    for name in ['S2_1_revenue_trend.png','S2_2_meal_breakdown.png',
                 'S2_3_demand_heatmap.png','S2_4_spoilage.png']:
        print(f"  Saved: {name}")
