"""
eda_scenario1.py
================
EDA for all 8 Scenario 1 datasets.

Figures saved to results/eda/:
  EDA_S1_1  Meal category distribution (meal_info.csv)
  EDA_S1_2  Cuisine distribution (meal_info.csv)
  EDA_S1_3  Top ingredients from 13k recipes
  EDA_S1_4  Grocery SKU category + price box
  EDA_S1_5  Stock vs reorder level scatter
  EDA_S1_6  Fulfilment centre type + op area
  EDA_S1_7  Instacart day-of-week order profile
  EDA_S1_8  Zomato cuisine popularity
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_loader import load_all_scenario1

OUT_DIR = Path('./results/eda')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ['#1F4E79','#2E75B6','#3FA3DC','#70B7E0','#A9D4EF',
           '#C00000','#E74C3C','#F5917A','#1E6B2E','#2ECC71',
           '#8E44AD','#D4A017','#95A5A6']

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 9,
    'figure.dpi'       : 120,
})


def _save(name):
    p = OUT_DIR / name
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


def run_s1_eda(data: dict | None = None):
    print("\n  ── EDA: Scenario 1 — Public Datasets ──────────────────────")
    if data is None:
        data = load_all_scenario1(verbose=False)

    # ── Figure 1: Meal category distribution ─────────────────────────────────
    mi = data['meal_info']
    cats = mi['category'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(cats)), cats.values,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(cats))])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats.index, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Number of Meals')
    ax.set_title('Meal Category Distribution\n'
                 f'(meal_info.csv · {len(mi)} items, {mi["category"].nunique()} categories)',
                 fontweight='bold')
    for bar, v in zip(bars, cats.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, str(v),
                ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    _save('EDA_S1_1_meal_category.png')

    # ── Figure 2: Cuisine distribution ───────────────────────────────────────
    cuisines = mi['cuisine'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie(cuisines.values, labels=cuisines.index, autopct='%1.1f%%',
                colors=PALETTE[:len(cuisines)], startangle=90)
    axes[0].set_title('Cuisine Proportions', fontweight='bold')
    axes[1].barh(cuisines.index, cuisines.values,
                 color=PALETTE[:len(cuisines)])
    axes[1].set_xlabel('Number of Meals')
    axes[1].set_title('Cuisine Counts', fontweight='bold')
    plt.suptitle('Meal Cuisine Distribution (meal_info.csv)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save('EDA_S1_2_cuisine_distribution.png')

    # ── Figure 3: Top ingredients ─────────────────────────────────────────────
    top = data.get('top_ingredients', [])
    if not top:
        top = ['salt','oil','pepper','sugar','flour','butter',
               'garlic','onion','water','eggs']
    # Pull frequencies from ingredients df if available
    ing_df = data['ingredients']
    import ast, itertools
    from collections import Counter
    if 'Cleaned_Ingredients' in ing_df.columns:
        def parse(x):
            try: return ast.literal_eval(x)
            except: return []
        all_ing = list(itertools.chain.from_iterable(
            ing_df['Cleaned_Ingredients'].apply(parse)))
        cnt = Counter(all_ing)
        items = cnt.most_common(20)
        labels, freqs = zip(*items) if items else (top[:10], [1]*10)
    else:
        labels, freqs = top[:10], range(len(top[:10]), 0, -1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(list(reversed(labels)), list(reversed(freqs)),
            color=PALETTE[5])
    ax.set_xlabel('Frequency across recipes')
    ax.set_title(f'Top {len(labels)} Ingredients\n'
                 f'(Food Ingredients & Recipe Dataset · {len(ing_df):,} recipes)',
                 fontweight='bold')
    plt.tight_layout()
    _save('EDA_S1_3_top_ingredients.png')

    # ── Figure 4: Grocery category + price ───────────────────────────────────
    g = data['grocery']
    if 'Unit_Price' in g.columns and 'Category' in g.columns:
        cats_g = g['Category'].value_counts().head(10)
        bp_data = [pd.to_numeric(g[g['Category']==c]['Unit_Price'],
                                  errors='coerce').dropna().values
                   for c in cats_g.index]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(range(len(cats_g)), cats_g.values,
                    color=[PALETTE[i % len(PALETTE)] for i in range(len(cats_g))])
        axes[0].set_xticks(range(len(cats_g)))
        axes[0].set_xticklabels(cats_g.index, rotation=40, ha='right', fontsize=8)
        axes[0].set_ylabel('SKU Count')
        axes[0].set_title('SKUs per Category', fontweight='bold')
        bplot = axes[1].boxplot(bp_data, patch_artist=True, notch=False,
                                medianprops=dict(color='black', lw=2))
        for patch, color in zip(bplot['boxes'], PALETTE):
            patch.set_facecolor(color)
        axes[1].set_xticklabels(cats_g.index, rotation=40, ha='right', fontsize=8)
        axes[1].set_ylabel('Unit Price (USD)')
        axes[1].set_title('Price Distribution per Category', fontweight='bold')
        plt.suptitle(f'Grocery Inventory Dataset ({len(g):,} SKUs)',
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        _save('EDA_S1_4_grocery_category_price.png')

    # ── Figure 5: Stock vs reorder ────────────────────────────────────────────
    if all(c in g.columns for c in ['Stock_Quantity','Reorder_Level','Category']):
        sample = g.dropna(subset=['Stock_Quantity','Reorder_Level']).sample(
            min(300, len(g)), random_state=42)
        cats_u = sample['Category'].unique()
        cmap   = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats_u)}
        fig, ax = plt.subplots(figsize=(9, 6))
        for cat in cats_u:
            sub = sample[sample['Category']==cat]
            ax.scatter(sub['Reorder_Level'], sub['Stock_Quantity'],
                       color=cmap[cat], label=cat, alpha=0.6, s=18)
        ax.plot([0, sample['Reorder_Level'].max()],
                [0, sample['Reorder_Level'].max()],
                'k--', lw=1, label='Stock = Reorder')
        ax.set_xlabel('Reorder Level')
        ax.set_ylabel('Current Stock Quantity')
        ax.set_title('Stock vs Reorder Level by Category\n'
                     '(below dashed line = at-risk of stockout)',
                     fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        _save('EDA_S1_5_stock_vs_reorder.png')

    # ── Figure 6: Fulfilment centres ─────────────────────────────────────────
    fc = data['fulfilment']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    type_c = fc['center_type'].value_counts()
    axes[0].pie(type_c.values, labels=type_c.index, autopct='%1.1f%%',
                colors=PALETTE[:len(type_c)], startangle=90)
    axes[0].set_title('Centre Type Distribution', fontweight='bold')
    axes[1].hist(fc['op_area'].dropna(), bins=20, color=PALETTE[2], edgecolor='white')
    axes[1].axvline(fc['op_area'].mean(), color='red', ls='--',
                    label=f"Mean {fc['op_area'].mean():.2f} km²")
    axes[1].set_xlabel('Operational Area (km²)')
    axes[1].set_ylabel('Number of Centres')
    axes[1].set_title('Op Area Distribution', fontweight='bold')
    axes[1].legend(fontsize=8)
    plt.suptitle(f'Fulfilment Centres ({len(fc)} centres)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save('EDA_S1_6_fulfilment_centres.png')

    # ── Figure 7: Instacart day-of-week profile ──────────────────────────────
    inst = data.get('instacart')
    if inst is not None and 'dow_profile' in inst:
        dp = inst['dow_profile']
        dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig, ax = plt.subplots(figsize=(8, 4))
        x = dp['order_dow'].astype(int)
        ax.bar(x, dp['relative_demand'], color=PALETTE[:7])
        ax.axhline(1.0, color='grey', ls='--', label='Average demand')
        ax.set_xticks(range(7))
        ax.set_xticklabels(dow_labels)
        ax.set_ylabel('Relative Order Demand')
        ax.set_title('Instacart 2017 — Day-of-Week Grocery Order Profile\n'
                     '(3.4M orders — calibrates day-of-week demand patterns)',
                     fontweight='bold')
        ax.legend()
        plt.tight_layout()
        _save('EDA_S1_7_instacart_dow_profile.png')
    else:
        # Placeholder noting the dataset
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, 'Instacart 2017 Dataset\n(3,421,083 orders)\n'
                'Place in data/Instacart_2017/ to enable this figure.',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes, color='grey')
        ax.set_title('EDA_S1_7: Instacart Day-of-Week Profile', fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        _save('EDA_S1_7_instacart_dow_profile.png')

    # ── Figure 8: Zomato cuisine popularity ──────────────────────────────────
    zom = data.get('zomato')
    if zom is not None and 'top_cuisines' in zom:
        tc_data = zom['top_cuisines'][:15]
        cuisines_z, counts_z = zip(*tc_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(list(reversed(cuisines_z)), list(reversed(counts_z)),
                color=PALETTE[0])
        ax.set_xlabel('Number of Restaurants')
        ax.set_title('Zomato Restaurant Dataset — Top 15 Cuisines by Popularity\n'
                     '(9,551 restaurants — calibrates cuisine demand weights)',
                     fontweight='bold')
        plt.tight_layout()
        _save('EDA_S1_8_zomato_cuisines.png')
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, 'Zomato Restaurant Dataset\n(9,551 restaurants)\n'
                'Place in data/Zomato_Restaurants/ to enable this figure.',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes, color='grey')
        ax.set_title('EDA_S1_8: Zomato Cuisine Popularity', fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        _save('EDA_S1_8_zomato_cuisines.png')

    print(f"  All S1 EDA figures saved → {OUT_DIR}/")
