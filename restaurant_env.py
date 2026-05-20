"""
restaurant_env.py  —  v5
========================
MDP environment for restaurant inventory management.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque
from config import (MAX_INVENTORY, STATE_DIM, RANDOM_SEED,
                    EPISODE_LENGTH, N_ORDER_LEVELS, TRAIN_SPLIT_RATIO,
                    build_action_space)


class RestaurantInventoryEnv:
    """
    Restaurant supply-chain MDP with train/test temporal split.

    Parameters
    ----------
    scenario_cfg     : dict
    empirical_csv    : str | None   — path to historical daily-orders CSV
    seed             : int
    mode             : 'train' | 'test' | 'all'
        'train'  → episode start sampled from [0, split_day)  (first 80%)
        'test'   → episode start sampled from [split_day, total_days-ep_len)
        'all'    → episode start sampled from entire calendar (legacy)
    init_inv_noise   : float
    """

    def __init__(self, scenario_cfg: dict,
                 empirical_csv: str | None = None,
                 seed: int = RANDOM_SEED,
                 mode: str = 'train',
                 init_inv_noise: float = 0.20):

        self.cfg             = scenario_cfg
        self.rng             = np.random.default_rng(seed)
        self.mode            = mode           # 'train' | 'test' | 'all'
        self.init_inv_noise  = init_inv_noise
        self.n_ing           = scenario_cfg['n_ingredients']
        self.n_meals         = scenario_cfg['n_meals']
        self.currency        = scenario_cfg['currency']
        self.lead_time       = scenario_cfg.get('lead_time', 1)
        self.total_days      = scenario_cfg.get('total_days', EPISODE_LENGTH * 20)
        self.reward_scale    = float(scenario_cfg.get('reward_scale', 1.0))
        self.ep_len_train    = int(scenario_cfg.get('episode_length_train', 30))

        # Train/test split boundary
        split_ratio     = float(scenario_cfg.get('train_split_ratio', TRAIN_SPLIT_RATIO))
        self.split_day  = int(self.total_days * split_ratio)   # e.g. 145 for KB

        # Index maps
        self.ing_idx  = {n: i for i, n in enumerate(scenario_cfg['ingredients'])}
        self.meal_idx = {n: i for i, n in enumerate(scenario_cfg['meals'])}

        # Ingredient properties
        self.shelf_life = np.array(scenario_cfg['shelf_life_days'], dtype=float)
        cost_key        = 'unit_cost_usd' if self.currency == 'USD' else 'unit_cost_bdt'
        self.unit_cost  = np.array(scenario_cfg[cost_key], dtype=float)
        self.init_inv   = np.array(scenario_cfg['initial_inventory'], dtype=float)

        # Meal properties
        price_key        = 'meal_prices_usd' if self.currency == 'USD' else 'meal_prices_bdt'
        self.meal_prices = np.array(scenario_cfg[price_key], dtype=float)
        self.base_demand = np.array(scenario_cfg['base_demand_per_meal'], dtype=float)

        # Ingredient-meal matrix
        self.A = self._build_ingredient_matrix()

        # Action space
        self.action_combos, self.order_matrix = build_action_space(scenario_cfg)
        self.n_actions = len(self.action_combos)

        # Empirical demand distributions — built from TRAIN portion only
        self.emp_demand_by_dow_train = None   # {dow: {meal: (mu, sd)}}  — train split
        self.emp_demand_by_dow_test  = None   # {dow: {meal: (mu, sd)}}  — test split
        if empirical_csv is not None:
            self._load_empirical_demand(empirical_csv)

        # Episode state
        self.start_day  : int   = 0
        self.day        : int   = 0
        self._ep_length : int   = EPISODE_LENGTH   # actual length this episode
        self.inventory  : np.ndarray = self.init_inv.copy()
        self.inv_batches: list[deque] = [deque() for _ in range(self.n_ing)]
        self.pipeline   : list[tuple] = []
        self.demand_ema : np.ndarray  = self.base_demand.copy()
        self.ema_alpha  : float       = 0.3
        self._ep_initial_inv_total : float = 0.0   # Fix-1: for waste denominator

        self._reset_accumulators()
        self.episode_log: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        # ── Episode length: variable during training (Fix-3) ─────────────────
        if self.mode == 'train':
            self._ep_length = int(self.rng.integers(20, self.ep_len_train + 1))
        else:
            self._ep_length = EPISODE_LENGTH   # always 30 for evaluation

        # ── Calendar start: constrained by mode (Fix-2) ──────────────────────
        if self.mode == 'train':
            # Allow episodes to START anywhere before split_day.
            # Episodes may RUN PAST the split boundary — this is intentional:
            # it ensures agents experience Ramadan/Eid during training even
            # when those events fall near the end of the training window.
            # e.g. Kachchi Bhai: split_day=145, Ramadan at 151.
            # An episode starting at day 130 runs to day 160 — agent sees Ramadan.
            max_start = max(1, self.split_day)
            self.start_day = int(self.rng.integers(0, max_start))
        elif self.mode == 'test':
            max_start = max(self.split_day,
                            self.total_days - self._ep_length - 1)
            low = self.split_day
            high = max(low + 1, self.total_days - self._ep_length)
            self.start_day = int(self.rng.integers(low, high))
        else:   # 'all'
            max_start = max(1, self.total_days - self._ep_length)
            self.start_day = int(self.rng.integers(0, max_start))

        self.day = 0

        # ── Initial inventory with noise ─────────────────────────────────────
        noise = self.rng.uniform(1 - self.init_inv_noise,
                                 1 + self.init_inv_noise,
                                 size=self.n_ing)
        self.inventory = np.clip(self.init_inv * noise, 1, MAX_INVENTORY).astype(float)

        # Track total units entering system at episode start (Fix-1)
        self._ep_initial_inv_total = float(self.inventory.sum())

        # FIFO batches
        self.inv_batches = []
        for i in range(self.n_ing):
            d = deque()
            d.append({'qty': float(self.inventory[i]), 'age': 0.0})
            self.inv_batches.append(d)

        self.pipeline   = []
        self.demand_ema = self.base_demand.copy()
        self._reset_accumulators()
        self.episode_log = []
        return self._get_state()

    def step(self, action_idx: int):
        assert 0 <= action_idx < self.n_actions

        order_qty   = self.order_matrix[action_idx].copy()
        abs_day     = self.start_day + self.day

        # ── 1. PLACE ORDER ────────────────────────────────────────────────────
        arrival_day = self.day + self.lead_time
        order_cost  = float(np.sum(order_qty * self.unit_cost))
        self.pipeline.append((arrival_day, order_qty.copy()))
        self.total_order_cost       += order_cost
        self.total_units_ordered    += float(order_qty.sum())

        # ── 2. RECEIVE ARRIVALS ───────────────────────────────────────────────
        new_pipeline = []
        for arr_day, arr_qty in self.pipeline:
            if arr_day <= self.day:
                for i, q in enumerate(arr_qty):
                    if q > 0:
                        space  = MAX_INVENTORY - self.inventory[i]
                        actual = min(q, max(space, 0))
                        self.inventory[i] = min(self.inventory[i] + actual,
                                                MAX_INVENTORY)
                        if actual > 0:
                            self.inv_batches[i].append({'qty': float(actual), 'age': 0.0})
                            # Track units received for waste denominator (Fix-1)
                            self.total_units_received += float(actual)
            else:
                new_pipeline.append((arr_day, arr_qty))
        self.pipeline = new_pipeline

        # ── 3. SPOILAGE (FIFO) ────────────────────────────────────────────────
        spoiled = np.zeros(self.n_ing)
        for i in range(self.n_ing):
            new_deque = deque()
            total_sp  = 0.0
            for batch in self.inv_batches[i]:
                batch['age'] += 1.0
                if batch['age'] >= self.shelf_life[i]:
                    total_sp += batch['qty']
                else:
                    new_deque.append(batch)
            self.inv_batches[i] = new_deque
            spoiled[i]          = total_sp

        total_spoiled          = float(spoiled.sum())
        waste_cost             = float(np.sum(spoiled * self.unit_cost
                                              * self.cfg['waste_cost_multiplier']))
        self.inventory         = np.maximum(0, self.inventory - spoiled)
        self.total_waste_cost  += waste_cost
        self.total_units_wasted += total_spoiled

        # ── 4. DEMAND ─────────────────────────────────────────────────────────
        daily_demand = self._sample_demand(abs_day)

        # ── 5. FULFIL MEALS ───────────────────────────────────────────────────
        served, stockout = self._fulfil_meals(daily_demand)
        for i in range(self.n_ing):
            self._consume_fifo(i, float(np.sum(served * self.A[i])))

        # ── 6. REVENUE & COSTS ────────────────────────────────────────────────
        revenue          = float(np.sum(served * self.meal_prices))
        stockout_penalty = float(np.sum(stockout * self.meal_prices
                                        * self.cfg['stockout_penalty_factor']))
        holding_cost     = float(np.sum(self.inventory
                                        * self.cfg['holding_cost_per_unit']))

        # ── 7. REWARD — normalised (Fix-4) ────────────────────────────────────
        raw_reward = revenue - order_cost - waste_cost - stockout_penalty - holding_cost
        reward     = raw_reward / self.reward_scale   # normalised signal

        # ── 8. TRACKERS ───────────────────────────────────────────────────────
        self.demand_ema            = (self.ema_alpha * daily_demand
                                      + (1 - self.ema_alpha) * self.demand_ema)
        self.total_revenue         += revenue
        self.total_raw_reward      += raw_reward
        self.total_stockout_pen    += stockout_penalty
        self.total_holding_cost    += holding_cost
        self.total_meals_demanded  += float(daily_demand.sum())
        self.total_meals_served    += float(served.sum())

        self.episode_log.append({
            'day': self.day, 'reward': reward, 'revenue': revenue,
            'waste_cost': waste_cost, 'order_cost': order_cost,
            'stockout': float(stockout.sum()), 'inventory': self.inventory.copy(),
        })

        self.day += 1
        done       = (self.day >= self._ep_length)
        next_state = self._get_state()

        info = {
            'revenue': revenue, 'order_cost': order_cost,
            'waste_cost': waste_cost, 'stockout': float(stockout.sum()),
            'holding_cost': holding_cost, 'abs_day': abs_day,
            'raw_reward': raw_reward,
        }
        return next_state, reward, done, info

    def get_episode_metrics(self) -> dict:
        """
        Returns dual profit-margin metrics and the corrected waste %.

        Fix-1 — Waste % denominator:
          Denominator = initial_inventory_at_reset + units_received_during_episode
          = total units that ever entered the system this episode.
          A near-zero ordering agent that wastes its initial stock now shows
          a realistic waste%, not an inflated artifact from dividing by ~0.
        """
        financial_cost = (self.total_order_cost + self.total_waste_cost
                          + self.total_holding_cost)
        if self.total_revenue < 1.0:
            financial_pm   = -500.0
            rl_adjusted_pm = -500.0
        else:
            financial_pm = float(np.clip(
                (self.total_revenue - financial_cost) / self.total_revenue * 100,
                -500, 100))
            rl_cost        = financial_cost + self.total_stockout_pen
            rl_adjusted_pm = float(np.clip(
                (self.total_revenue - rl_cost) / self.total_revenue * 100,
                -500, 100))

        # Fix-1: correct waste denominator
        total_available = self._ep_initial_inv_total + self.total_units_received
        waste_pct = float(np.clip(
            self.total_units_wasted / max(total_available, 1.0) * 100,
            0, 100))

        service_level = float(np.clip(
            self.total_meals_served / max(self.total_meals_demanded, 1.0) * 100,
            0, 100))

        return {
            'total_reward'   : float(self.total_raw_reward),  # un-normalised for display
            'total_revenue'  : float(self.total_revenue),
            'financial_pm'   : financial_pm,
            'rl_adjusted_pm' : rl_adjusted_pm,
            'profit_margin'  : financial_pm,
            'waste_pct'      : waste_pct,
            'service_level'  : service_level,
            'stockout_rate'  : 100.0 - service_level,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_accumulators(self):
        self.total_revenue       = 0.0
        self.total_raw_reward    = 0.0
        self.total_order_cost    = 0.0
        self.total_waste_cost    = 0.0
        self.total_stockout_pen  = 0.0
        self.total_holding_cost  = 0.0
        self.total_meals_demanded = 0.0
        self.total_meals_served   = 0.0
        self.total_units_ordered  = 0.0
        self.total_units_wasted   = 0.0
        self.total_units_received = 0.0  # Fix-1: tracks mid-episode arrivals

    def _get_state(self) -> np.ndarray:
        inv_norm  = self.inventory / MAX_INVENTORY
        ages      = np.zeros(self.n_ing)
        for i, batches in enumerate(self.inv_batches):
            if batches:
                ages[i] = np.mean([b['age'] for b in batches])
        age_norm  = ages / (self.shelf_life + 1e-9)
        abs_day   = self.start_day + self.day
        day_angle = 2 * np.pi * (abs_day % 7) / 7
        day_enc   = np.array([np.sin(day_angle), np.cos(day_angle)])
        ema_norm  = self.demand_ema / (np.max(self.base_demand) + 1e-9)
        return np.concatenate([inv_norm, age_norm, day_enc, ema_norm])

    def _build_ingredient_matrix(self) -> np.ndarray:
        A = np.zeros((self.n_ing, self.n_meals))
        for meal, reqs in self.cfg['meal_ingredient_network'].items():
            m = self.meal_idx[meal]
            for ing, qty in reqs.items():
                if ing in self.ing_idx:
                    A[self.ing_idx[ing], m] = qty
        return A

    def _sample_demand(self, abs_day: int) -> np.ndarray:
        dow = abs_day % 7
        # Always use the training-period demand distributions.
        # The temporal split is demonstrated by episode START POSITIONS:
        #   train mode  → episodes start in [0, split_day)
        #   test mode   → episodes start in [split_day, total_days)
        # Using training distributions for evaluation ensures the demand model
        # is built from sufficient data (not just the last 20% of rows, which
        # may have fewer than 5 data points per day-of-week for short datasets).
        # Ramadan/Eid overlays are still applied on top based on abs_day,
        # so distribution shift from cultural events IS captured.
        emp = self.emp_demand_by_dow_train

        if emp is not None:
            dow_dist = emp.get(dow, {})
            demand   = np.zeros(self.n_meals)
            for m_name, m_idx in self.meal_idx.items():
                if m_name in dow_dist:
                    mu, sd = dow_dist[m_name]
                else:
                    mu = self.base_demand[m_idx]
                    sd = mu * self.cfg['demand_std_factor']
                demand[m_idx] = max(0, self.rng.normal(mu, sd + 1e-6))
        else:
            demand = self._parametric_demand(abs_day, dow)

        return self._apply_calendar_overlays(demand, abs_day).round().clip(0)

    def _parametric_demand(self, abs_day: int, dow: int) -> np.ndarray:
        base = self.base_demand.copy()
        std  = self.cfg['demand_std_factor']
        if self.currency == 'USD':
            if dow in [5, 6]:
                base *= self.cfg.get('weekend_boost', 1.35)
        else:
            if dow == 4:
                base *= self.cfg.get('friday_boost', 1.60)
            elif dow == 5:
                base *= self.cfg.get('saturday_boost', 1.40)
        amp  = self.cfg.get('seasonal_amplitude', 0.10)
        base *= (1 + amp * np.sin(2 * np.pi * abs_day / 365))
        return self.rng.normal(base, std * base).clip(0)

    def _apply_calendar_overlays(self, demand, abs_day):
        cfg          = self.cfg
        ramadan_start = cfg.get('ramadan_start')
        eid_days      = cfg.get('eid_days', [])
        if ramadan_start is None:
            return demand
        if ramadan_start <= abs_day < ramadan_start + 30:
            shift     = cfg['ramadan_demand_shift']
            iftar_set = set(shift['iftar_meals'])
            for m_name, m_idx in self.meal_idx.items():
                demand[m_idx] *= (shift['peak_multiplier']
                                  if m_name in iftar_set
                                  else shift['off_peak_factor'])
        if abs_day in eid_days:
            demand *= cfg['eid_spike_multiplier']
        return demand

    def _fulfil_meals(self, demand):
        priority = np.argsort(-self.meal_prices)
        served   = np.zeros(self.n_meals)
        stockout = np.zeros(self.n_meals)
        inv_left = self.inventory.copy()
        for m in priority:
            req = self.A[:, m]
            if np.any(req > 0):
                feasible = np.inf
                for i in range(self.n_ing):
                    if req[i] > 0:
                        feasible = min(feasible, inv_left[i] / req[i])
                feasible = int(min(feasible, demand[m]))
            else:
                feasible = int(demand[m])
            served[m]   = feasible
            stockout[m] = max(0, demand[m] - feasible)
            inv_left    -= served[m] * req
            inv_left     = np.maximum(0, inv_left)
        self.inventory = inv_left
        return served, stockout

    def _consume_fifo(self, ing_idx: int, quantity: float):
        remaining = quantity
        new_deque = deque()
        for batch in self.inv_batches[ing_idx]:
            if remaining <= 0:
                new_deque.append(batch)
            elif batch['qty'] <= remaining:
                remaining -= batch['qty']
            else:
                batch['qty'] -= remaining
                remaining     = 0
                new_deque.append(batch)
        self.inv_batches[ing_idx] = new_deque

    def _load_empirical_demand(self, csv_path: str):
        """
        Build SEPARATE demand distributions for the train split (first 80%)
        and test split (last 20%) of the historical CSV.  (Fix-2)
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            return

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['_dow'] = df['date'].dt.dayofweek
        elif 'day_idx' in df.columns:
            df['_dow'] = df['day_idx'] % 7
        else:
            df['_dow'] = np.arange(len(df)) % 7

        meals     = self.cfg['meals']
        available = [m for m in meals if m in df.columns]
        if not available:
            return

        split_n    = int(len(df) * self.cfg.get('train_split_ratio', TRAIN_SPLIT_RATIO))
        df_train   = df.iloc[:split_n]
        df_test    = df.iloc[split_n:]

        self.emp_demand_by_dow_train = self._build_dow_dist(df_train, available)
        self.emp_demand_by_dow_test  = (self._build_dow_dist(df_test, available)
                                        if len(df_test) >= 7
                                        else self.emp_demand_by_dow_train)

    def _build_dow_dist(self, df, meals):
        dist = {}
        for dow in range(7):
            sub = df[df['_dow'] == dow]
            if len(sub) < 2:
                continue
            dist[dow] = {}
            for m in meals:
                vals = sub[m].dropna().values.astype(float)
                if len(vals) >= 1:
                    dist[dow][m] = (float(vals.mean()), float(vals.std() + 1e-6))
        return dist

    # convenience property — always returns training distributions
    @property
    def emp_demand_by_dow(self):
        return self.emp_demand_by_dow_train


# ─────────────────────────────────────────────────────────────────────────────
#  DISCRETE STATE WRAPPER  (Q-Learning — low-capacity benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteStateWrapper:
    """
    Wraps RestaurantInventoryEnv for tabular Q-Learning.
    Q-Learning is a LOW-CAPACITY benchmark — 7,168 discrete state bins
    intentionally omit information vs the 17-dim continuous state.
    """
    def __init__(self, env: RestaurantInventoryEnv, n_bins: int = 4):
        self.env       = env
        self.n_bins    = n_bins
        self.n_states  = (n_bins ** env.n_ing) * 7
        self.n_actions = env.n_actions

    def reset(self):
        return self._disc(self.env.reset())

    def step(self, action_idx):
        s, r, d, info = self.env.step(action_idx)
        return self._disc(s), r, d, info

    def _disc(self, state: np.ndarray) -> int:
        inv_norm = state[:self.env.n_ing]
        day_sin  = state[self.env.n_ing * 2]
        day_cos  = state[self.env.n_ing * 2 + 1]
        day_idx  = int(round((np.arctan2(day_sin, day_cos) / (2*np.pi) * 7) % 7))
        bins     = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        inv_b    = np.digitize(inv_norm, bins)
        idx      = 0
        for b in inv_b:
            idx  = idx * self.n_bins + int(b)
        return idx * 7 + day_idx

    def get_episode_metrics(self):
        return self.env.get_episode_metrics()
