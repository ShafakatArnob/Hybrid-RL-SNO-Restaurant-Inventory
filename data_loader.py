"""
data_loader.py
==============
Loads and analyses all 8 datasets for Scenario 1.

Small datasets (in data/):
  1. meal_info.csv
  2. Grocery_Inventory_and_Sales_Dataset.csv
  3. fulfilment_center_info.csv
  4. Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv

Large datasets (in data/ subdirectories):
  5. Instacart_2017/orders.csv + products.csv + aisles.csv
  6. Open_Food_Facts/en.openfoodfacts.org.products.csv
  7. Zomato_Restaurants/zomato.csv
  8. UCI_Online_Retail/Online_Retail.xlsx
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('./data')


# ─────────────────────────────────────────────────────────────────────────────
#  SMALL DATASETS (always present)
# ─────────────────────────────────────────────────────────────────────────────

def load_meal_info():
    """51-item meal catalogue with category and cuisine labels."""
    df = pd.read_csv(DATA_DIR / 'meal_info.csv')
    return df

def load_grocery():
    """990-SKU grocery inventory with price, stock, reorder."""
    df = pd.read_csv(DATA_DIR / 'Grocery_Inventory_and_Sales_Dataset.csv')
    # Normalise column name (some versions spell it 'Catagory')
    if 'Catagory' in df.columns:
        df = df.rename(columns={'Catagory': 'Category'})
    # Clean Unit_Price: strip $ and convert
    if df['Unit_Price'].dtype == object:
        df['Unit_Price'] = (df['Unit_Price'].astype(str)
                            .str.replace('$','',regex=False)
                            .str.strip()
                            .replace('', np.nan))
        df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')
    return df

def load_fulfilment():
    """77 fulfilment centres with type and operational area."""
    return pd.read_csv(DATA_DIR / 'fulfilment_center_info.csv')

def load_food_ingredients(max_rows=20000):
    """13,501-recipe ingredient dataset. Capped at max_rows for speed."""
    import ast, itertools
    from collections import Counter
    df = pd.read_csv(
        DATA_DIR / 'Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv',
        nrows=max_rows)
    if 'Cleaned_Ingredients' in df.columns:
        def parse(x):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        df['_ing_list'] = df['Cleaned_Ingredients'].apply(parse)
        all_ings  = list(itertools.chain.from_iterable(df['_ing_list']))
        top_ings  = [w for w, _ in Counter(all_ings).most_common(30)]
        return df, top_ings
    return df, []


# ─────────────────────────────────────────────────────────────────────────────
#  LARGE DATASET 5: INSTACART 2017
# ─────────────────────────────────────────────────────────────────────────────

def load_instacart(sample_n=500_000):
    """
    Load Instacart 2017 grocery shopping dataset.
    Returns (orders_df, products_df, reorder_stats_df) or None on missing.

    Calibration use:
      • Reorder frequency per product → ingredient reorder interval
      • Day-of-week order patterns    → demand week profile
      • Department-level demand       → category-level demand calibration
    """
    base = DATA_DIR / 'Instacart_2017'
    orders_f = base / 'orders.csv'
    prods_f  = base / 'products.csv'
    prior_f  = base / 'order_products__prior.csv'

    if not orders_f.exists():
        warnings.warn(f"Instacart dataset not found at {base}. "
                      "Skipping — using small-dataset calibration.")
        return None

    print("  Loading Instacart 2017 (orders + products)...")
    orders   = pd.read_csv(orders_f)
    products = pd.read_csv(prods_f)

    # Subsample prior orders for speed
    prior = pd.read_csv(prior_f, nrows=sample_n)

    # Merge
    merged = prior.merge(products[['product_id','product_name',
                                   'department_id']], on='product_id')
    merged = merged.merge(orders[['order_id','order_dow',
                                  'days_since_prior_order']], on='order_id')

    # Reorder stats per department (proxy for ingredient category)
    reorder_stats = (merged.groupby('department_id')
                     .agg(reorder_rate=('reordered','mean'),
                          avg_days_between=('days_since_prior_order','mean'))
                     .reset_index())

    # Day-of-week demand profile
    dow_profile = (orders.groupby('order_dow')['order_id']
                   .count()
                   .rename('order_count')
                   .reset_index())
    dow_profile['relative_demand'] = (dow_profile['order_count']
                                      / dow_profile['order_count'].mean())

    return {
        'orders'        : orders,
        'products'      : products,
        'reorder_stats' : reorder_stats,
        'dow_profile'   : dow_profile,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LARGE DATASET 6: OPEN FOOD FACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_open_food_facts(sample_n=50_000):
    """
    Load Open Food Facts product database.
    Returns shelf-life statistics per food category, or None on missing.

    Calibration use:
      • Median shelf life per food category → shelf_life_days in config
      • Ingredient composition              → meal ingredient network validation
    """
    f = DATA_DIR / 'Open_Food_Facts' / 'en.openfoodfacts.org.products.csv'
    if not f.exists():
        warnings.warn(f"Open Food Facts not found at {f}. Skipping.")
        return None

    print("  Loading Open Food Facts (sample)...")
    # Large file — read only needed columns
    useful_cols = ['product_name', 'categories_en', 'ingredients_text',
                   'food_groups_en', 'conservation_conditions']
    try:
        df = pd.read_csv(f, nrows=sample_n, usecols=useful_cols,
                         on_bad_lines='skip', low_memory=False)
    except Exception:
        df = pd.read_csv(f, nrows=sample_n, on_bad_lines='skip',
                         low_memory=False)

    # Extract shelf-life proxy from conservation_conditions
    if 'conservation_conditions' in df.columns:
        def shelf_proxy(cond):
            if not isinstance(cond, str):
                return np.nan
            cond = cond.lower()
            if 'freeze' in cond or 'frozen' in cond:
                return 180
            if 'refrigerat' in cond or 'chilled' in cond:
                return 7
            if 'dry' in cond or 'room temp' in cond:
                return 90
            return np.nan
        df['shelf_life_proxy'] = df['conservation_conditions'].apply(shelf_proxy)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LARGE DATASET 7: ZOMATO RESTAURANTS
# ─────────────────────────────────────────────────────────────────────────────

def load_zomato():
    """
    Load Zomato restaurant dataset.
    Returns cuisine popularity and average price per cuisine, or None on missing.

    Calibration use:
      • Cuisine-level average cost   → meal price calibration
      • Rating distribution          → demand elasticity proxy
      • City-level restaurant count  → market competition proxy
    """
    f = DATA_DIR / 'Zomato_Restaurants' / 'zomato.csv'
    if not f.exists():
        warnings.warn(f"Zomato dataset not found at {f}. Skipping.")
        return None

    print("  Loading Zomato Restaurants...")
    try:
        df = pd.read_csv(f, encoding='latin-1')
    except Exception:
        df = pd.read_csv(f, encoding='utf-8', errors='replace')

    # Clean average cost column
    cost_col = None
    for c in df.columns:
        if 'cost' in c.lower() or 'price' in c.lower():
            cost_col = c
            break
    if cost_col:
        df[cost_col] = pd.to_numeric(
            df[cost_col].astype(str).str.replace(',','',regex=False),
            errors='coerce')

    # Cuisine popularity
    if 'Cuisines' in df.columns:
        import re
        all_cuisines = []
        for val in df['Cuisines'].dropna():
            all_cuisines.extend([c.strip() for c in str(val).split(',')])
        from collections import Counter
        top_cuisines = Counter(all_cuisines).most_common(20)
        return {'df': df, 'top_cuisines': top_cuisines, 'cost_col': cost_col}

    return {'df': df}


# ─────────────────────────────────────────────────────────────────────────────
#  LARGE DATASET 8: UCI ONLINE RETAIL
# ─────────────────────────────────────────────────────────────────────────────

def load_uci_retail():
    """
    Load UCI Online Retail transaction dataset.
    Returns reorder cycle statistics per product category, or None on missing.

    Calibration use:
      • Reorder quantity distribution  → ORDER_LEVELS calibration
      • Transaction frequency          → demand density calibration
      • Category-level turnover        → inventory turnover benchmarking
    """
    f = DATA_DIR / 'UCI_Online_Retail' / 'Online_Retail.xlsx'
    if not f.exists():
        warnings.warn(f"UCI Online Retail not found at {f}. Skipping.")
        return None

    print("  Loading UCI Online Retail...")
    df = pd.read_excel(f, engine='openpyxl')

    # Clean
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Reorder stats per StockCode
    reorder = (df.groupby('StockCode')
               .agg(n_orders=('InvoiceNo','nunique'),
                    avg_qty=('Quantity','mean'),
                    unit_price=('UnitPrice','mean'))
               .reset_index())

    # Category proxy via Description keyword
    df['Category'] = df['Description'].fillna('').apply(_uci_category)
    cat_stats = (df.groupby('Category')
                 .agg(avg_qty=('Quantity','mean'),
                      avg_price=('UnitPrice','mean'),
                      n_transactions=('InvoiceNo','count'))
                 .reset_index())

    return {'df': df, 'reorder': reorder, 'cat_stats': cat_stats}


def _uci_category(desc: str) -> str:
    desc = desc.lower()
    if any(w in desc for w in ['bag', 'box', 'tin']): return 'Packaging'
    if any(w in desc for w in ['candle', 'light']): return 'Decor'
    if any(w in desc for w in ['mug', 'cup', 'plate', 'bowl']): return 'Kitchenware'
    if any(w in desc for w in ['card', 'gift']): return 'Stationery'
    return 'General'


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER LOADER — loads all datasets and returns calibration summary
# ─────────────────────────────────────────────────────────────────────────────

def load_all_scenario1(verbose=True):
    """
    Load all 8 datasets and derive calibration parameters.
    Returns a dict of loaded dataframes and derived stats.
    """
    results = {}

    if verbose:
        print("\n  Loading Scenario 1 datasets...")

    # Small datasets (always available)
    results['meal_info']     = load_meal_info()
    results['grocery']       = load_grocery()
    results['fulfilment']    = load_fulfilment()
    results['ingredients'], results['top_ingredients'] = load_food_ingredients()

    # Large datasets (graceful fallback)
    results['instacart']     = load_instacart()
    results['openfoodfacts'] = load_open_food_facts()
    results['zomato']        = load_zomato()
    results['uci_retail']    = load_uci_retail()

    # ── Derive calibration stats ──────────────────────────────────────────────
    g = results['grocery']
    price_by_cat = (g.groupby('Category')['Unit_Price']
                    .apply(lambda x: float(pd.to_numeric(x, errors='coerce').median()))
                    .to_dict())

    f = results['fulfilment']
    centre_stats = {
        'n_centres'   : len(f),
        'type_counts' : f['center_type'].value_counts().to_dict(),
        'mean_op_area': round(float(f['op_area'].mean()), 2),
        'mean_reorder': round(float(g['Reorder_Level'].mean()), 1),
    }

    mi = results['meal_info']
    meal_stats = {
        'n_items'    : len(mi),
        'n_categories': mi['category'].nunique(),
        'n_cuisines' : mi['cuisine'].nunique(),
        'top_category': mi['category'].value_counts().index[0],
    }

    dow_profile = None
    if results['instacart'] is not None:
        dow_profile = results['instacart']['dow_profile']

    top_cuisines = None
    if results['zomato'] is not None and 'top_cuisines' in results['zomato']:
        top_cuisines = results['zomato']['top_cuisines']

    calibration = {
        'price_by_category' : price_by_cat,
        'centre_stats'      : centre_stats,
        'meal_stats'        : meal_stats,
        'top_ingredients'   : results['top_ingredients'][:10],
        'dow_profile'       : dow_profile,
        'top_cuisines'      : top_cuisines,
    }

    if verbose:
        print(f"    ✓ Meal info       : {meal_stats['n_items']} items, "
              f"{meal_stats['n_categories']} categories, "
              f"{meal_stats['n_cuisines']} cuisines")
        print(f"    ✓ Grocery         : {len(g):,} SKUs | "
              f"mean reorder level: {centre_stats['mean_reorder']}")
        print(f"    ✓ Fulfilment      : {centre_stats['n_centres']} centres | "
              f"mean op area: {centre_stats['mean_op_area']} km²")
        print(f"    ✓ Food Ingredients: {len(results['ingredients']):,} recipes | "
              f"top: {results['top_ingredients'][0]}")
        for key in ['instacart', 'openfoodfacts', 'zomato', 'uci_retail']:
            status = '✓ loaded' if results[key] is not None else '⚠ not found (graceful skip)'
            print(f"    {status:<15}: {key.replace('_',' ').title()}")

    results['calibration'] = calibration
    return results
