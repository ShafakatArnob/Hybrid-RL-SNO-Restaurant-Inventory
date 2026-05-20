"""
config.py  —  v5
================
Central configuration for BOTH experimental scenarios.
"""

import itertools
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED
# ─────────────────────────────────────────────────────────────────────────────
EPISODE_LENGTH       = 30    # days per evaluation episode (fixed for fair comparison)
EPISODE_LENGTH_TRAIN = 30    # default training episode length; overridden per restaurant
MAX_INVENTORY        = 100   # max stock units per ingredient
RANDOM_SEED          = 42
TRAIN_SPLIT_RATIO    = 0.80  # first 80% of days used for training distributions

# Ingredient-specific action levels (Minor-6: flexible action design)
ORDER_LEVELS_FINE   = [0, 25,  50]   # Mutton, Chicken  (shelf ≤ 2 days, high cost)
ORDER_LEVELS_MEDIUM = [0, 50, 100]   # Rice, Onion      (shelf 5–14 days)
ORDER_LEVELS_COARSE = [0, 75, 150]   # Oil, Spice, Bread (shelf 30+ days)

N_ORDER_LEVELS = 3

# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 1 — GENERIC RESTAURANT
# ─────────────────────────────────────────────────────────────────────────────
GENERIC = {
    'scenario_name' : 'Generic Restaurant Framework',
    'currency'      : 'USD',
    'n_ingredients' : 5,
    'n_meals'       : 5,
    'lead_time'     : 1,
    'total_days'    : 600,    # virtual calendar length for train/test split

    # 80% train split → days 0–480; 20% test → days 480–600
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 30,   # variable 20-40 during training, eval fixed 30
    'reward_scale'         : 1.0,  # USD — already reasonable magnitude

    'ingredients'       : ['Rice', 'Chicken', 'Onion', 'Cooking Oil', 'Spice Mix'],
    'shelf_life_days'   : [7,      2,         5,       30,             60    ],
    'order_levels'      : [
        ORDER_LEVELS_MEDIUM,   # Rice
        ORDER_LEVELS_FINE,     # Chicken  (perishable)
        ORDER_LEVELS_MEDIUM,   # Onion
        ORDER_LEVELS_COARSE,   # Cooking Oil (stable)
        ORDER_LEVELS_COARSE,   # Spice Mix   (stable)
    ],
    'unit_cost_usd'     : [0.50,   3.00,      0.20,    1.00,           0.80  ],
    'initial_inventory' : [60,     40,        50,      60,             50    ],
    'reorder_threshold' : [20,     10,        20,      20,             20    ],

    'meals'             : ['Rice Bowl', 'Grilled Chicken', 'Stir-Fry', 'Curry', 'Veg Soup'],
    'meal_prices_usd'   : [8.0,  12.0,  10.0,  11.0,  6.0],

    'meal_ingredient_network' : {
        'Rice Bowl'       : {'Rice': 3, 'Spice Mix': 0.5},
        'Grilled Chicken' : {'Chicken': 2, 'Spice Mix': 0.5, 'Cooking Oil': 0.3},
        'Stir-Fry'        : {'Chicken': 1, 'Onion': 1, 'Cooking Oil': 0.5, 'Spice Mix': 0.3},
        'Curry'           : {'Chicken': 2, 'Onion': 1, 'Spice Mix': 1, 'Cooking Oil': 0.5},
        'Veg Soup'        : {'Onion': 1, 'Spice Mix': 0.5},
    },

    'base_demand_per_meal' : [20, 18, 15, 22, 12],
    'demand_std_factor'    : 0.25,
    'weekend_boost'        : 1.35,
    'seasonal_amplitude'   : 0.10,

    'waste_cost_multiplier'   : 1.5,
    'stockout_penalty_factor' : 0.8,
    'holding_cost_per_unit'   : 0.02,

    'ql_note' : ('Q-Learning is a low-capacity benchmark. Its 7,168-bin '
                 'discretisation of the 17-dim state intentionally loses '
                 'information vs neural agents. Included for pedagogical '
                 'comparison only.'),
}

# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO 2 — 5 BANGLADESHI RESTAURANTS
# ─────────────────────────────────────────────────────────────────────────────
#
#  reward_scale: normalises daily reward to ~[-100, +100] range regardless
#  of currency magnitude.  Formula: mean_price × mean_daily_demand × 30 / 1000
#  This allows neural nets to learn on gradients of similar scale across
#  all five restaurants despite very different BDT price levels.
#
#  episode_length_train: training uses variable-length episodes drawn from
#  [20, episode_length_train] so agents don't overfit to exactly 30 days.
#  Evaluation always uses fixed EPISODE_LENGTH=30 for fair comparison.
#
#  train_split_ratio = 0.80:
#    Empirical demand distributions are built ONLY from the first 80% of
#    the historical CSV rows.  Evaluation episodes are constrained to start
#    in the final 20% window — ensuring the agent is tested on data it has
#    never seen during training (out-of-sample generalisation).

KACCHI_BHAI = {
    'restaurant_id'        : 'kacchi_bhai',
    'restaurant_name'      : 'Kachchi Bhai Restaurant',
    'location'             : 'Dhanmondi, Dhaka',
    'cuisine_type'         : 'Biryani / Mughlai',
    'currency'             : 'BDT',
    'n_ingredients'        : 5,
    'n_meals'              : 5,
    'lead_time'            : 1,
    'total_days'           : 182,
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 30,
    # reward_scale = mean_price(370) × mean_demand(179/day) × 30 / 1e6 ≈ 2.0
    'reward_scale'         : 2.0,
    'ramadan_start'        : 151,
    'eid_days'             : [180, 181],

    'ingredients'     : ['Kataribhog Rice', 'Mutton', 'Chicken', 'Onion', 'Garam Masala'],
    'shelf_life_days' : [10,  2,   2,   5,   90],
    'order_levels'    : [
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_COARSE,
    ],
    'unit_cost_bdt'   : [80,  900, 350,  40,  120],
    'initial_inventory': [60,  20,  30,  60,   50],
    'reorder_threshold': [20,   8,  12,  20,   15],

    'meals'           : ['Kacchi Biryani', 'Chicken Tehari', 'Chicken Roast',
                         'Mutton Rezala', 'Mixed Platter'],
    'meal_prices_bdt' : [480, 220, 280, 350, 520],

    'meal_ingredient_network' : {
        'Kacchi Biryani' : {'Kataribhog Rice': 3, 'Mutton': 2, 'Garam Masala': 1, 'Onion': 1},
        'Chicken Tehari' : {'Kataribhog Rice': 3, 'Chicken': 2, 'Garam Masala': 0.5, 'Onion': 0.5},
        'Chicken Roast'  : {'Chicken': 3, 'Garam Masala': 1, 'Onion': 0.5},
        'Mutton Rezala'  : {'Mutton': 2, 'Garam Masala': 1, 'Onion': 0.5},
        'Mixed Platter'  : {'Kataribhog Rice': 2, 'Mutton': 1, 'Chicken': 1,
                            'Garam Masala': 0.5, 'Onion': 0.5},
    },

    'base_demand_per_meal' : [35, 45, 30, 20, 15],
    'demand_std_factor'    : 0.30,
    'friday_boost'         : 1.60,
    'saturday_boost'       : 1.40,
    'ramadan_demand_shift' : {
        'peak_multiplier' : 1.80,
        'off_peak_factor' : 0.50,
        'iftar_meals'     : ['Kacchi Biryani', 'Mutton Rezala'],
    },
    'eid_spike_multiplier' : 2.50,
    'seasonal_amplitude'   : 0.12,

    'waste_cost_multiplier'   : 1.5,
    'stockout_penalty_factor' : 0.8,
    'holding_cost_per_unit'   : 0.05,
    'data_csv' : 'data/kacchi_bhai_daily_orders.csv',
}

STAR_KABAB = {
    'restaurant_id'        : 'star_kabab',
    'restaurant_name'      : 'Star Kabab & Restaurant',
    'location'             : 'Purana Paltan, Old Dhaka',
    'cuisine_type'         : 'Kebab / Grilled',
    'currency'             : 'BDT',
    'n_ingredients'        : 5,
    'n_meals'              : 5,
    'lead_time'            : 1,
    'total_days'           : 912,
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 45,   # longer episodes — 2.5yr dataset
    # reward_scale = mean_price(280) × mean_demand(230/day) × 30 / 1e6 ≈ 1.9
    'reward_scale'         : 1.9,
    'ramadan_start'        : 100,
    'eid_days'             : [112, 113, 468, 469],

    'ingredients'     : ['Beef', 'Chicken', 'Onion', 'Spices', 'Bread'],
    'shelf_life_days' : [2,   2,   5,   60,   2],
    'order_levels'    : [
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_COARSE,
        ORDER_LEVELS_FINE,
    ],
    'unit_cost_bdt'   : [750, 320,  35, 200,  15],
    'initial_inventory': [25,  30,  50,  40,  20],
    'reorder_threshold': [ 8,  10,  15,  10,   6],

    'meals'           : ['Beef Shami Kabab', 'Chicken Tikka', 'Mixed Grill Platter',
                         'Kabab Roll', 'Chicken Boti'],
    'meal_prices_bdt' : [320, 250, 480, 150, 200],

    'meal_ingredient_network' : {
        'Beef Shami Kabab'   : {'Beef': 2, 'Spices': 0.5, 'Onion': 0.5},
        'Chicken Tikka'      : {'Chicken': 2, 'Spices': 0.5, 'Onion': 0.5},
        'Mixed Grill Platter': {'Beef': 1, 'Chicken': 1, 'Spices': 1, 'Onion': 1},
        'Kabab Roll'         : {'Beef': 1, 'Bread': 2, 'Onion': 0.5, 'Spices': 0.3},
        'Chicken Boti'       : {'Chicken': 2, 'Spices': 0.5},
    },

    'base_demand_per_meal' : [40, 35, 20, 55, 45],
    'demand_std_factor'    : 0.28,
    'friday_boost'         : 1.70,
    'saturday_boost'       : 1.50,
    'ramadan_demand_shift' : {
        'peak_multiplier' : 2.00,
        'off_peak_factor' : 0.30,
        'iftar_meals'     : ['Beef Shami Kabab', 'Mixed Grill Platter'],
    },
    'eid_spike_multiplier' : 2.20,
    'seasonal_amplitude'   : 0.08,

    'waste_cost_multiplier'   : 1.6,
    'stockout_penalty_factor' : 0.8,
    'holding_cost_per_unit'   : 0.06,
    'data_csv' : 'data/star_kabab_daily_orders.csv',
}

FAKRUDDIN = {
    'restaurant_id'        : 'fakruddin',
    'restaurant_name'      : 'Fakruddin Restaurant',
    'location'             : 'Gulshan, Dhaka',
    'cuisine_type'         : 'Traditional Mughlai',
    'currency'             : 'BDT',
    'n_ingredients'        : 5,
    'n_meals'              : 5,
    'lead_time'            : 2,
    'total_days'           : 1095,
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 45,   # longer episodes — 3yr dataset
    # reward_scale = mean_price(350) × mean_demand(156/day) × 30 / 1e6 ≈ 1.6
    'reward_scale'         : 1.6,
    'ramadan_start'        : 60,
    'eid_days'             : [72, 73, 427, 428, 792, 793],

    'ingredients'     : ['Basmati Rice', 'Mutton', 'Hilsha Fish', 'Lentils', 'Spice Blend'],
    'shelf_life_days' : [14,  2,   1,   30,   90],
    'order_levels'    : [
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_COARSE,
        ORDER_LEVELS_COARSE,
    ],
    'unit_cost_bdt'   : [120, 950, 600,  80,  180],
    'initial_inventory': [70,  20,  10,  60,   40],
    'reorder_threshold': [25,   6,   3,  20,   12],

    'meals'           : ['Mutton Biryani', 'Hilsha Fry', 'Dal Makhani',
                         'Mughlai Paratha', 'Mutton Korma'],
    'meal_prices_bdt' : [550, 480, 180, 120, 420],

    'meal_ingredient_network' : {
        'Mutton Biryani'  : {'Basmati Rice': 3, 'Mutton': 2, 'Spice Blend': 1},
        'Hilsha Fry'      : {'Hilsha Fish': 2, 'Spice Blend': 0.5},
        'Dal Makhani'     : {'Lentils': 3, 'Spice Blend': 0.5},
        'Mughlai Paratha' : {'Lentils': 1, 'Spice Blend': 0.3},
        'Mutton Korma'    : {'Mutton': 2, 'Spice Blend': 1},
    },

    'base_demand_per_meal' : [30, 20, 25, 40, 22],
    'demand_std_factor'    : 0.22,
    'friday_boost'         : 1.55,
    'saturday_boost'       : 1.35,
    'ramadan_demand_shift' : {
        'peak_multiplier' : 1.60,
        'off_peak_factor' : 0.40,
        'iftar_meals'     : ['Mutton Biryani', 'Mutton Korma'],
    },
    'eid_spike_multiplier' : 3.00,
    'seasonal_amplitude'   : 0.15,

    'waste_cost_multiplier'   : 1.8,
    'stockout_penalty_factor' : 0.9,
    'holding_cost_per_unit'   : 0.04,
    'data_csv' : 'data/fakruddin_daily_orders.csv',
}

MADCHEF = {
    'restaurant_id'        : 'madchef',
    'restaurant_name'      : 'Madchef',
    'location'             : 'Banani, Dhaka',
    'cuisine_type'         : 'Modern / Fast Food',
    'currency'             : 'BDT',
    'n_ingredients'        : 5,
    'n_meals'              : 5,
    'lead_time'            : 1,
    'total_days'           : 548,
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 40,
    # reward_scale = mean_price(244) × mean_demand(308/day) × 30 / 1e6 ≈ 2.3
    'reward_scale'         : 2.3,
    'ramadan_start'        : 245,
    'eid_days'             : [275, 276],

    'ingredients'     : ['Chicken Breast', 'Buns', 'Cheese', 'Lettuce', 'Sauce'],
    'shelf_life_days' : [2,   2,   7,   4,   30],
    'order_levels'    : [
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_COARSE,
    ],
    'unit_cost_bdt'   : [320,  25, 180,  30,  90],
    'initial_inventory': [30,  40,  25,  20,  30],
    'reorder_threshold': [10,  12,   8,   6,  10],

    'meals'           : ['Zinger Burger', 'Chicken Sandwich', 'Cheese Fries',
                         'Grilled Chicken Box', 'Chicken Wrap'],
    'meal_prices_bdt' : [280, 220, 150, 320, 250],

    'meal_ingredient_network' : {
        'Zinger Burger'       : {'Chicken Breast': 1, 'Buns': 2, 'Lettuce': 1, 'Sauce': 0.5},
        'Chicken Sandwich'    : {'Chicken Breast': 1, 'Buns': 2, 'Cheese': 1, 'Lettuce': 1},
        'Cheese Fries'        : {'Cheese': 2, 'Sauce': 0.5},
        'Grilled Chicken Box' : {'Chicken Breast': 2, 'Sauce': 0.5},
        'Chicken Wrap'        : {'Chicken Breast': 1, 'Lettuce': 1, 'Sauce': 0.5, 'Cheese': 0.5},
    },

    'base_demand_per_meal' : [60, 45, 70, 40, 50],
    'demand_std_factor'    : 0.35,
    'friday_boost'         : 1.45,
    'saturday_boost'       : 1.60,
    'ramadan_demand_shift' : {
        'peak_multiplier' : 1.30,
        'off_peak_factor' : 0.60,
        'iftar_meals'     : ['Zinger Burger', 'Chicken Wrap'],
    },
    'eid_spike_multiplier' : 1.80,
    'seasonal_amplitude'   : 0.06,

    'waste_cost_multiplier'   : 1.5,
    'stockout_penalty_factor' : 0.75,
    'holding_cost_per_unit'   : 0.05,
    'data_csv' : 'data/madchef_daily_orders.csv',
}

BISMILLAH = {
    'restaurant_id'        : 'bismillah',
    'restaurant_name'      : 'Bismillah Hotel & Restaurant',
    'location'             : 'Mirpur-10, Dhaka',
    'cuisine_type'         : 'Local / Thali / Rice-based',
    'currency'             : 'BDT',
    'n_ingredients'        : 5,
    'n_meals'              : 5,
    'lead_time'            : 1,
    'total_days'           : 730,
    'train_split_ratio'    : TRAIN_SPLIT_RATIO,
    'episode_length_train' : 30,
    # reward_scale = mean_price(102) × mean_demand(263/day) × 30 / 1e6 ≈ 0.8
    # slightly boosted to 1.2 to give agents a fighting chance on low-margin dishes
    'reward_scale'         : 1.2,
    'ramadan_start'        : 100,
    'eid_days'             : [112, 113, 467, 468],

    'ingredients'     : ['Rice', 'Lentils', 'Vegetables', 'Mustard Oil', 'Dry Fish'],
    'shelf_life_days' : [30,  14,   3,   60,   30],
    'order_levels'    : [
        ORDER_LEVELS_COARSE,
        ORDER_LEVELS_MEDIUM,
        ORDER_LEVELS_FINE,
        ORDER_LEVELS_COARSE,
        ORDER_LEVELS_MEDIUM,
    ],
    'unit_cost_bdt'   : [55,  90,  30, 160,  200],
    'initial_inventory': [100, 60,  30,  80,   40],
    'reorder_threshold': [ 30, 20,   8,  20,   12],

    'meals'           : ['Rice & Dal Thali', 'Fish Curry Plate', 'Vegetable Khichuri',
                         'Dry Fish & Rice', 'Special Thali'],
    'meal_prices_bdt' : [80, 120, 70, 100, 160],

    'meal_ingredient_network' : {
        'Rice & Dal Thali'    : {'Rice': 3, 'Lentils': 2, 'Mustard Oil': 0.3},
        'Fish Curry Plate'    : {'Rice': 3, 'Dry Fish': 1, 'Mustard Oil': 0.5, 'Vegetables': 1},
        'Vegetable Khichuri'  : {'Rice': 2, 'Lentils': 2, 'Vegetables': 2, 'Mustard Oil': 0.3},
        'Dry Fish & Rice'     : {'Rice': 3, 'Dry Fish': 2, 'Mustard Oil': 0.3},
        'Special Thali'       : {'Rice': 3, 'Lentils': 1, 'Dry Fish': 1, 'Vegetables': 1,
                                 'Mustard Oil': 0.5},
    },

    'base_demand_per_meal' : [80, 50, 40, 45, 30],
    'demand_std_factor'    : 0.18,
    'friday_boost'         : 1.30,
    'saturday_boost'       : 1.20,
    'ramadan_demand_shift' : {
        'peak_multiplier' : 1.40,
        'off_peak_factor' : 0.55,
        'iftar_meals'     : ['Rice & Dal Thali', 'Special Thali'],
    },
    'eid_spike_multiplier' : 2.00,
    'seasonal_amplitude'   : 0.10,

    'waste_cost_multiplier'   : 1.4,
    'stockout_penalty_factor' : 0.7,
    'holding_cost_per_unit'   : 0.03,
    'data_csv' : 'data/bismillah_daily_orders.csv',
}

ALL_RESTAURANTS = [KACCHI_BHAI, STAR_KABAB, FAKRUDDIN, MADCHEF, BISMILLAH]

# ─────────────────────────────────────────────────────────────────────────────
#  ACTION SPACE
# ─────────────────────────────────────────────────────────────────────────────
def build_action_space(scenario_cfg):
    n_ing   = scenario_cfg['n_ingredients']
    levels  = scenario_cfg['order_levels']
    combos  = list(itertools.product(range(N_ORDER_LEVELS), repeat=n_ing))
    order_matrix = np.array(
        [[levels[i][lvl] for i, lvl in enumerate(combo)] for combo in combos],
        dtype=float
    )
    return combos, order_matrix

N_ACTIONS_GENERIC = N_ORDER_LEVELS ** GENERIC['n_ingredients']   # 243

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    'ql_episodes'        : 800,
    'ql_alpha'           : 0.15,
    'ql_gamma'           : 0.95,
    'ql_epsilon_start'   : 1.0,
    'ql_epsilon_end'     : 0.05,
    'ql_epsilon_decay'   : 0.995,

    'dqn_episodes'       : 600,
    'dqn_lr'             : 0.001,
    'dqn_lr_min'         : 0.0001,  # LR decay floor
    'dqn_gamma'          : 0.95,
    'dqn_batch_size'     : 64,
    'dqn_buffer_size'    : 8000,
    'dqn_target_update'  : 20,
    'dqn_epsilon_start'  : 1.0,
    'dqn_epsilon_end'    : 0.05,
    'dqn_epsilon_decay'  : 0.995,
    'dqn_hidden_dim'     : 128,

    'ppo_episodes'       : 500,
    'ppo_lr_actor'       : 0.001,
    'ppo_lr_actor_min'   : 0.0001,  # LR decay floor
    'ppo_lr_critic'      : 0.003,
    'ppo_lr_critic_min'  : 0.0003,
    'ppo_gamma'          : 0.95,
    'ppo_clip_epsilon'   : 0.20,
    'ppo_update_epochs'  : 6,
    'ppo_entropy_coef'   : 0.01,
    'ppo_hidden_dim'     : 128,

    'eval_episodes'      : 100,
    'eval_seeds'         : [42, 123, 777],

    # LR cosine decay: applied every lr_decay_interval episodes
    'lr_decay_interval'  : 100,
}

# ─────────────────────────────────────────────────────────────────────────────
#  STATE SPACE
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM   = 17
QL_INV_BINS = 4
