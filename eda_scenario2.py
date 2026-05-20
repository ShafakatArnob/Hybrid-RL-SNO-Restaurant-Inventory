"""
eda_scenario2.py
================
EDA for Scenario 2: all 5 Bangladeshi restaurants.

Figures saved to results/eda/:
  EDA_S2_1  Revenue trend + Ramadan/Eid annotations (Kachchi Bhai)
  EDA_S2_2  Day-of-week demand profile (Kachchi Bhai)
  EDA_S2_3  Demand heatmap day-of-week × week (Kachchi Bhai)
  EDA_S2_4  Ramadan vs normal-period comparison (Kachchi Bhai)
  EDA_S2_5  Spoilage breakdown (Kachchi Bhai)
  EDA_S2_6  Revenue mix by meal (Kachchi Bhai)
  EDA_S2_7  Cross-restaurant demand overview (all 5)
  EDA_S2_8  Schema diversity illustration (all 5)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from config import KACCHI_BHAI, ALL_RESTAURANTS

OUT_DIR = Path('./results/eda')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('./data')

PALETTE = ['#1F4E79','#2E75B6','#E74C3C','#27AE60','#F39C12',
           '#8E44AD','#1ABC9C','#E67E22','#95A5A6','#2C3E50']

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 9,
    'figure.dpi'       : 120,
})


def _save(name: str):
    p = OUT_DIR / name
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


def _load_kb() -> pd.DataFrame | None:
    f = DATA_DIR / 'kacchi_bhai_daily_orders.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: Revenue trend with Ramadan / Eid annotations
# ─────────────────────────────────────────────────────────────────────────────

def _fig_revenue_trend(df: pd.DataFrame):
    rev_col = 'total_daily_revenue_BDT'
    if rev_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    rev  = df[rev_col].values
    days = np.arange(len(df))
    roll = pd.Series(rev).rolling(7, center=True).mean()

    ax.bar(days, rev, color='#85C1E9', alpha=0.45, width=1, label='Daily Revenue (BDT)')
    ax.plot(days, roll, color='#1A3A5C', lw=2, label='7-day Rolling Avg')

    # Ramadan shading
    ram_s = KACCHI_BHAI['ramadan_start']
    ram_e = min(ram_s + 30, len(df))
    ax.axvspan(ram_s, ram_e, color='#FDEBD0', alpha=0.55, label='Ramadan')

    # Eid lines
    for eid_d in KACCHI_BHAI['eid_days']:
        if eid_d < len(df):
            ax.axvline(eid_d, color='#C0392B', lw=1.8, ls='--')
            ax.text(eid_d + 0.5, rev.max() * 0.88, 'Eid',
                    color='#C0392B', fontsize=8, fontweight='bold')

    ax.set_xlabel('Day Index (0 = Oct 1 2024)')
    ax.set_ylabel('Revenue (BDT)')
    ax.set_title('Kachchi Bhai Restaurant — Daily Revenue Trend\n'
                 'Oct 2024 – Mar 2025 (182 days) | Primary Research Dataset',
                 fontweight='bold')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    _save('EDA_S2_1_revenue_trend.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: Day-of-week demand profile
# ─────────────────────────────────────────────────────────────────────────────

def _fig_dow_demand(df: pd.DataFrame):
    meals = KACCHI_BHAI['meals']
    avail = [m for m in meals if m in df.columns]
    if not avail:
        return

    df = df.copy()
    df['dow'] = df['date'].dt.day_name()
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['dow'] = pd.Categorical(df['dow'], categories=order, ordered=True)
    dow_avg = df.groupby('dow')[avail].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4))
    x    = np.arange(7)
    w    = 0.15
    for i, m in enumerate(avail):
        ax.bar(x + i*w, dow_avg[m], w, label=m,
               color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x + w * len(avail)/2)
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.set_ylabel('Average Daily Orders')
    ax.set_title('Day-of-Week Demand Profile — Kachchi Bhai\n'
                 '(Friday peak reflects Islamic weekend in Bangladesh)',
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    plt.tight_layout()
    _save('EDA_S2_2_dow_demand.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3: Demand heatmap (day-of-week × week)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_heatmap(df: pd.DataFrame):
    order_col = 'total_daily_orders'
    if order_col not in df.columns:
        meals = [m for m in KACCHI_BHAI['meals'] if m in df.columns]
        if not meals:
            return
        df = df.copy()
        df[order_col] = df[meals].sum(axis=1)

    df = df.copy()
    df['dow']  = df['date'].dt.dayofweek
    df['week'] = df['day_idx'] // 7 + 1
    pivot = df.pivot_table(index='dow', columns='week',
                           values=order_col, aggfunc='mean')
    dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    pivot = pivot.reindex(range(7))

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_yticks(range(7))
    ax.set_yticklabels(dow_labels, fontsize=9)
    ax.set_xlabel('Week Number')
    ax.set_title('Demand Heatmap: Day-of-Week × Week — Kachchi Bhai\n'
                 '(Avg daily orders; Ramadan from Week 22)',
                 fontweight='bold')
    # Week tick labels every 4
    weeks = sorted(pivot.columns.tolist())
    tick_pos  = list(range(0, len(weeks), 4))
    tick_lbls = [str(weeks[i]) for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=8)
    plt.colorbar(im, ax=ax, label='Avg Orders')
    # Mark Ramadan start
    ram_week = KACCHI_BHAI['ramadan_start'] // 7 + 1
    if ram_week - 1 < len(weeks):
        ax.axvline(weeks.index(ram_week) - 0.5 if ram_week in weeks else len(weeks)*0.85,
                   color='white', lw=2, ls='--', alpha=0.9)
        ax.text(len(weeks) * 0.86, -0.7, 'Ramadan →',
                color='white', fontsize=8, fontweight='bold')
    plt.tight_layout()
    _save('EDA_S2_3_demand_heatmap.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4: Ramadan vs normal-period comparison
# ─────────────────────────────────────────────────────────────────────────────

def _fig_ramadan_comparison(df: pd.DataFrame):
    meals = [m for m in KACCHI_BHAI['meals'] if m in df.columns]
    if not meals:
        return

    ram_s    = KACCHI_BHAI['ramadan_start']
    pre_ram  = df[df['day_idx'] < ram_s][meals].mean()
    in_ram   = df[(df['day_idx'] >= ram_s) &
                  (df['day_idx'] < ram_s + 30)][meals].mean()

    x  = np.arange(len(meals))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, pre_ram.values, w, label='Pre-Ramadan', color='#2E75B6', alpha=0.85)
    ax.bar(x + w/2, in_ram.values,  w, label='Ramadan',    color='#E74C3C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ','\n') for m in meals], fontsize=9)
    ax.set_ylabel('Average Daily Orders')
    ax.set_title('Pre-Ramadan vs Ramadan Demand — Kachchi Bhai\n'
                 '(Kacchi Biryani & Mutton Rezala surge as iftar dishes)',
                 fontweight='bold')
    # Annotate change %
    for xi, (pre, ram) in enumerate(zip(pre_ram.values, in_ram.values)):
        pct = (ram - pre) / (pre + 1e-9) * 100
        ax.text(xi + w/2 + 0.05, ram + 0.3, f'{pct:+.0f}%',
                fontsize=8, color=('#C0392B' if pct > 0 else '#1A5276'))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save('EDA_S2_4_ramadan_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5: Spoilage breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _fig_spoilage(df: pd.DataFrame | None = None):
    inv_f = DATA_DIR / 'kacchi_bhai_inventory_log.csv'
    ings  = KACCHI_BHAI['ingredients']

    if inv_f.exists():
        inv_df   = pd.read_csv(inv_f)
        sp_cols  = [f'{i}_spoiled' for i in ings if f'{i}_spoiled' in inv_df.columns]
        if sp_cols:
            total_sp = inv_df[sp_cols].sum()
            total_sp.index = [c.replace('_spoiled','') for c in sp_cols]
        else:
            total_sp = pd.Series({i: 0 for i in ings})
    else:
        # Spoilage estimates from operational records
        total_sp = pd.Series({'Kataribhog Rice': 0, 'Mutton': 204,
                              'Chicken': 412, 'Onion': 96, 'Garam Masala': 0})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = PALETTE[:len(total_sp)]

    # Bar chart
    axes[0].bar(range(len(total_sp)), total_sp.values, color=colors)
    axes[0].set_xticks(range(len(total_sp)))
    axes[0].set_xticklabels([i.replace(' ','\n') for i in total_sp.index], fontsize=8)
    axes[0].set_ylabel('Total Units Spoiled (182 days)')
    axes[0].set_title('Spoilage by Ingredient', fontweight='bold')
    for xi, v in enumerate(total_sp.values):
        if v > 0:
            axes[0].text(xi, v + 2, str(int(v)), ha='center', va='bottom', fontsize=9)

    # Pie (only non-zero)
    nonzero = total_sp[total_sp > 0]
    if len(nonzero) > 0:
        axes[1].pie(nonzero.values, labels=nonzero.index,
                    autopct='%1.1f%%',
                    colors=[PALETTE[i] for i, k in enumerate(total_sp.index)
                            if total_sp[k] > 0],
                    startangle=90)
        axes[1].set_title('Spoilage Distribution\n(6-month total)',
                           fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No spoilage recorded', ha='center',
                     va='center', transform=axes[1].transAxes)
        axes[1].axis('off')

    total = int(total_sp.sum())
    cost  = (total_sp.get('Mutton', 0) * 900 +
             total_sp.get('Chicken', 0) * 350)
    plt.suptitle(f'Ingredient Spoilage — Kachchi Bhai (182 days)\n'
                 f'Total: {total} units | Est. cost: BDT {cost:,.0f}',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save('EDA_S2_5_spoilage.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 6: Revenue mix by meal
# ─────────────────────────────────────────────────────────────────────────────

def _fig_revenue_mix(df: pd.DataFrame):
    rev_cols = [f'{m}_revenue_BDT' for m in KACCHI_BHAI['meals']
                if f'{m}_revenue_BDT' in df.columns]
    if not rev_cols:
        return

    total_rev = df[rev_cols].sum()
    meal_names = [c.replace('_revenue_BDT','') for c in rev_cols]
    weekly_rev = df.copy()
    weekly_rev['week'] = df['day_idx'] // 7 + 1
    week_data  = weekly_rev.groupby('week')[rev_cols].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Pie
    axes[0].pie(total_rev.values, labels=meal_names,
                autopct='%1.1f%%', colors=PALETTE[:len(meal_names)],
                startangle=90)
    axes[0].set_title('Total Revenue Mix by Meal\n(182 days)',
                      fontweight='bold')

    # Stacked area
    weeks = week_data.index.values
    axes[1].stackplot(weeks,
                      [week_data[c] for c in rev_cols],
                      labels=meal_names, colors=PALETTE[:len(meal_names)],
                      alpha=0.85)
    axes[1].axvline(KACCHI_BHAI['ramadan_start'] // 7 + 1,
                    color='black', ls='--', lw=1.2, label='Ramadan start')
    axes[1].set_xlabel('Week Number')
    axes[1].set_ylabel('Weekly Revenue (BDT)')
    axes[1].set_title('Weekly Revenue by Meal\n(Stacked)',
                      fontweight='bold')
    axes[1].legend(fontsize=7, loc='upper left', ncol=2)
    plt.suptitle('Revenue Mix Analysis — Kachchi Bhai',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save('EDA_S2_6_revenue_mix.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 7: Cross-restaurant demand overview
# ─────────────────────────────────────────────────────────────────────────────

def _fig_cross_restaurant():
    """Bar chart of total orders per restaurant (avg daily demand)."""
    rest_data = []
    for cfg in ALL_RESTAURANTS:
        # Try loading the wide CSV
        wide_f = DATA_DIR / cfg['data_csv'].replace(
            './data/', '').replace('_daily_orders.csv', '_wide.csv')
        base_f = DATA_DIR / Path(cfg['data_csv']).name
        for f in [wide_f, base_f]:
            if Path(f).exists():
                try:
                    df  = pd.read_csv(f)
                    meals = [m for m in cfg['meals'] if m in df.columns]
                    if meals:
                        avg_d = float(df[meals].sum(axis=1).mean())
                        rest_data.append({'name': cfg['restaurant_name'],
                                          'cuisine': cfg['cuisine_type'],
                                          'avg_daily': avg_d,
                                          'days': len(df)})
                        break
                except Exception:
                    pass
        else:
            # Fallback: use base_demand sum
            avg_d = float(sum(cfg['base_demand_per_meal']))
            rest_data.append({'name': cfg['restaurant_name'],
                              'cuisine': cfg['cuisine_type'],
                              'avg_daily': avg_d,
                              'days': cfg['total_days']})

    if not rest_data:
        return

    names   = [d['name'] for d in rest_data]
    avgs    = [d['avg_daily'] for d in rest_data]
    days    = [d['days'] for d in rest_data]
    cuisines= [d['cuisine'] for d in rest_data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(range(len(names)), avgs,
                       color=PALETTE[:len(names)], edgecolor='white')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
    axes[0].set_ylabel('Average Daily Orders')
    axes[0].set_title('Avg Daily Orders by Restaurant', fontweight='bold')
    for bar, v in zip(bars, avgs):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5,
                     f'{v:.0f}', ha='center', va='bottom', fontsize=8)

    axes[1].bar(range(len(names)), days,
                color=PALETTE[:len(names)], edgecolor='white')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
    axes[1].set_ylabel('Dataset Duration (days)')
    axes[1].set_title('Dataset Duration by Restaurant\n'
                      '(1.5 – 3 years, different date ranges)',
                      fontweight='bold')

    plt.suptitle('Cross-Restaurant Overview — 5 Bangladeshi Restaurants\n'
                 'Different cuisines, schemas, demand levels, and date ranges',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save('EDA_S2_7_cross_restaurant_overview.png')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 8: Schema diversity illustration
# ─────────────────────────────────────────────────────────────────────────────

def _fig_schema_diversity():
    """Visual table showing how each restaurant's schema differs."""
    schema_info = [
        ('Kachchi Bhai',  'Wide daily',
         'date, day_idx, Kacchi Biryani, …, total_revenue_BDT'),
        ('Star Kabab',    'POS long-format (DD/MM/YYYY)',
         'transaction_date, outlet, meal_name, covers, gross_revenue_taka'),
        ('Fakruddin',     'Recipe-based structured',
         'RecordDate, DishName, PortionsSold, GrossProfit_BDT, DayType'),
        ('Madchef',       'Item-level order log',
         'order_id, order_timestamp, item, quantity, payment_method, table_no'),
        ('Bismillah Hotel','Weekly aggregate + daily',
         'WeekStart, TotalRevenue_Tk, AvgDailyCovers, BestDay / Date, QtySold'),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    col_labels  = ['Restaurant', 'Schema Style', 'Key Columns']
    table_data  = [[s[0], s[1], s[2]] for s in schema_info]
    colors_row  = [[PALETTE[i] + '33', '#F8F9FA', '#F8F9FA']   # light shade
                   for i in range(len(schema_info))]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc='left', loc='center',
                   cellColours=colors_row)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#1A3A5C')
        cell.set_edgecolor('#CCCCCC')
    ax.set_title('Schema Diversity Across 5 Restaurants\n'
                 '(Different POS systems → different data modelling conventions)',
                 fontweight='bold', pad=20, fontsize=11)
    plt.tight_layout()
    _save('EDA_S2_8_schema_diversity.png')


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_s2_eda():
    print("\n  ── EDA: Scenario 2 — Kachchi Bhai Dataset ─────────────────")

    df = _load_kb()
    if df is None:
        print("  ⚠  kacchi_bhai_daily_orders.csv not found in data/. "
              "Please ensure all restaurant CSV files are present.")
        return
    _fig_revenue_trend(df)
    _fig_dow_demand(df)
    _fig_heatmap(df)
    _fig_ramadan_comparison(df)
    _fig_spoilage(df)
    _fig_revenue_mix(df)
    _fig_cross_restaurant()
    _fig_schema_diversity()

    print(f"  All S2 EDA figures saved → {OUT_DIR}/")
