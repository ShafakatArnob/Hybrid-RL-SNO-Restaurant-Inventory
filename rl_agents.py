"""
rl_agents.py  —  v5
====================
Four RL agents in pure NumPy.
"""

from __future__ import annotations
import numpy as np
from collections import deque


# ═════════════════════════════════════════════════════════════════════════════
#  NUMPY MLP
# ═════════════════════════════════════════════════════════════════════════════

class NumpyMLP:
    """He-init feedforward MLP with Adam optimiser and LR setter."""

    def __init__(self, layer_sizes, lr=0.001, seed=42):
        rng        = np.random.default_rng(seed)
        self.lr    = lr
        self._lr0  = lr   # initial LR for decay reference
        self.sizes = layer_sizes
        self.L     = len(layer_sizes) - 1

        self.W  = [rng.standard_normal((layer_sizes[i], layer_sizes[i+1]))
                   * np.sqrt(2.0 / layer_sizes[i]) for i in range(self.L)]
        self.b  = [np.zeros(layer_sizes[i+1]) for i in range(self.L)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0

    def set_lr(self, new_lr: float):
        """Update the learning rate (used by cosine LR decay)."""
        self.lr = float(new_lr)

    def forward(self, x):
        a, zs, acs = x.copy(), [], [x.copy()]
        for l in range(self.L):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = np.maximum(0, z) if l < self.L - 1 else z
            acs.append(a)
        return a, zs, acs

    def predict(self, x):
        out, _, _ = self.forward(x)
        return out

    def train_step(self, x, target):
        if x.ndim == 1:
            x, target = x[None], target[None]
        batch = x.shape[0]
        out, zs, acs = self.forward(x)
        delta = 2.0 * (out - target) / batch
        dWs, dbs = [None]*self.L, [None]*self.L
        for l in range(self.L - 1, -1, -1):
            dWs[l] = acs[l].T @ delta
            dbs[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (zs[l-1] > 0).astype(float)
        self._adam(dWs, dbs)
        return float(np.mean((out - target) ** 2))

    def train_step_with_grad(self, x, output_grad):
        if x.ndim == 1:
            x, output_grad = x[None], output_grad[None]
        _, zs, acs = self.forward(x)
        delta = output_grad
        dWs, dbs = [None]*self.L, [None]*self.L
        for l in range(self.L - 1, -1, -1):
            dWs[l] = acs[l].T @ delta
            dbs[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (zs[l-1] > 0).astype(float)
        self._adam(dWs, dbs)

    def _adam(self, dWs, dbs, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for l in range(self.L):
            self.mW[l] = b1*self.mW[l] + (1-b1)*dWs[l]
            self.vW[l] = b2*self.vW[l] + (1-b2)*dWs[l]**2
            self.W[l] -= self.lr * (self.mW[l]/(1-b1**self.t)) / (
                          np.sqrt(self.vW[l]/(1-b2**self.t)) + eps)
            self.mb[l] = b1*self.mb[l] + (1-b1)*dbs[l]
            self.vb[l] = b2*self.vb[l] + (1-b2)*dbs[l]**2
            self.b[l] -= self.lr * (self.mb[l]/(1-b1**self.t)) / (
                          np.sqrt(self.vb[l]/(1-b2**self.t)) + eps)

    def copy_weights_from(self, other: 'NumpyMLP'):
        for l in range(self.L):
            self.W[l] = other.W[l].copy()
            self.b[l] = other.b[l].copy()


def _cosine_lr(lr0, lr_min, current_ep, total_eps):
    """Cosine annealing: lr decreases smoothly from lr0 → lr_min."""
    frac = current_ep / max(total_eps, 1)
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * frac))


# ═════════════════════════════════════════════════════════════════════════════
#  1. Q-LEARNING  (low-capacity benchmark)
# ═════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning — LOW-CAPACITY BENCHMARK.
    7,168 discrete state bins intentionally lose information vs 17-dim
    continuous state. Included for pedagogical comparison only.
    Reference: Watkins & Dayan (1992).
    """
    def __init__(self, n_states, n_actions,
                 alpha=0.15, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, seed=42):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng          = np.random.default_rng(seed)
        self.Q            = np.random.default_rng(seed).uniform(
                                -0.01, 0.01, (n_states, n_actions))
        self.name = "Q-Learning"
        self._ep = 0

    def select_action(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_lr(self, current_ep, total_eps, lr_min=None):
        pass  # Q-Learning uses fixed alpha

    def end_episode(self):
        self._ep += 1


# ═════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=8000, seed=42):
        self.buffer = deque(maxlen=capacity)
        self.rng    = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx    = self.rng.integers(0, len(self.buffer), batch_size)
        batch  = [self.buffer[i] for i in idx]
        states  = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int32)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        nexts   = np.stack([t[3] for t in batch])
        dones   = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, nexts, dones

    def __len__(self): return len(self.buffer)


# ═════════════════════════════════════════════════════════════════════════════
#  2. DQN
# ═════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """DQN (Mnih et al. 2015) with cosine LR decay."""

    def __init__(self, state_dim, n_actions,
                 hidden_dim=128, lr=0.001, gamma=0.95,
                 batch_size=64, buffer_size=8000,
                 target_update=20,
                 epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, seed=42):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)
        self.episode_count = 0
        self.name          = "DQN"
        self._lr0          = lr

        sizes = [state_dim, hidden_dim, hidden_dim, n_actions]
        self.online = NumpyMLP(sizes, lr=lr, seed=seed)
        self.target = NumpyMLP(sizes, lr=lr, seed=seed+1)
        self.target.copy_weights_from(self.online)
        self.replay = ReplayBuffer(buffer_size, seed=seed)

    def select_action(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.online.predict(state)))

    def store(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        q_next   = self.target.predict(s2).max(axis=1)
        q_target = r + self.gamma * q_next * (1 - d)
        targets  = self.online.predict(s)
        for i, (ai, qt) in enumerate(zip(a, q_target)):
            targets[i, ai] = qt
        self.online.train_step(s, targets)

    def end_episode(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.episode_count % self.target_update == 0:
            self.target.copy_weights_from(self.online)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_lr(self, current_ep, total_eps, lr_min=0.0001):
        """Cosine LR annealing — call at lr_decay_interval milestones."""
        new_lr = _cosine_lr(self._lr0, lr_min, current_ep, total_eps)
        self.online.set_lr(new_lr)


# ═════════════════════════════════════════════════════════════════════════════
#  3. DOUBLE DQN
# ═════════════════════════════════════════════════════════════════════════════

class DDQNAgent(DQNAgent):
    """Double DQN (van Hasselt et al. 2016) with cosine LR decay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Double DQN"

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        best_a   = self.online.predict(s2).argmax(axis=1)
        q_tgt_s2 = self.target.predict(s2)
        q_sel    = q_tgt_s2[np.arange(self.batch_size), best_a]
        y        = r + self.gamma * q_sel * (1 - d)
        targets  = self.online.predict(s)
        for i, (ai, yt) in enumerate(zip(a, y)):
            targets[i, ai] = yt
        self.online.train_step(s, targets)


# ═════════════════════════════════════════════════════════════════════════════
#  4. PPO
# ═════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """PPO Actor-Critic (Schulman et al. 2017) with entropy bonus + LR decay."""

    def __init__(self, state_dim, n_actions,
                 hidden_dim=128, lr_actor=0.001, lr_critic=0.003,
                 gamma=0.95, clip_epsilon=0.20, update_epochs=6,
                 entropy_coef=0.01, seed=42):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.clip_epsilon  = clip_epsilon
        self.update_epochs = update_epochs
        self.entropy_coef  = entropy_coef
        self.rng           = np.random.default_rng(seed)
        self.name          = "PPO"
        self._lr0_actor    = lr_actor
        self._lr0_critic   = lr_critic

        sizes_a = [state_dim, hidden_dim, hidden_dim, n_actions]
        sizes_c = [state_dim, hidden_dim, hidden_dim, 1]
        self.actor  = NumpyMLP(sizes_a, lr=lr_actor,  seed=seed)
        self.critic = NumpyMLP(sizes_c, lr=lr_critic, seed=seed+100)

        self.states:    list = []
        self.actions:   list = []
        self.rewards:   list = []
        self.log_probs: list = []

    def _softmax(self, x):
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def select_action(self, state, training=True):
        logits = self.actor.predict(state)
        probs  = self._softmax(logits)
        probs  = np.clip(probs, 1e-8, 1.0);  probs /= probs.sum()
        action = (int(self.rng.choice(self.n_actions, p=probs))
                  if training else int(np.argmax(probs)))
        return action, float(np.log(probs[action] + 1e-9))

    def store(self, s, a, r, lp):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(lp)

    def update(self):
        T = len(self.rewards)
        if T == 0:
            return
        states        = np.stack(self.states)
        actions       = np.array(self.actions)
        log_probs_old = np.array(self.log_probs)

        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self.rewards[t] + self.gamma * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.update_epochs):
            self.critic.train_step(states, returns.reshape(-1, 1))
            values     = self.critic.predict(states).flatten()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            logits    = self.actor.predict(states)
            probs_all = np.vstack([self._softmax(logits[t]) for t in range(T)])
            probs_all = np.clip(probs_all, 1e-9, 1.0)
            log_new   = np.log(probs_all[np.arange(T), actions] + 1e-9)
            ratio     = np.exp(np.clip(log_new - log_probs_old, -5, 5))

            not_clipped = (
                ((advantages >= 0) & (ratio <= 1 + self.clip_epsilon)) |
                ((advantages <  0) & (ratio >= 1 - self.clip_epsilon))
            )
            eff_adv  = advantages * not_clipped.astype(float)
            entropy  = -float(np.mean(np.sum(probs_all * np.log(probs_all + 1e-9), axis=1)))

            one_hot     = np.zeros_like(logits)
            one_hot[np.arange(T), actions] = 1.0
            pg_grad     = (eff_adv[:, None] * (probs_all - one_hot)) / T
            ent_grad    = -self.entropy_coef * (np.log(probs_all + 1e-9) + 1) / T
            self.actor.train_step_with_grad(states, pg_grad + ent_grad)

        self.states = []; self.actions = []
        self.rewards = []; self.log_probs = []

    def end_episode(self): pass
    def decay_epsilon(self): pass

    def decay_lr(self, current_ep, total_eps, lr_min_actor=0.0001, lr_min_critic=0.0003):
        new_lr_a = _cosine_lr(self._lr0_actor,  lr_min_actor,  current_ep, total_eps)
        new_lr_c = _cosine_lr(self._lr0_critic, lr_min_critic, current_ep, total_eps)
        self.actor.set_lr(new_lr_a)
        self.critic.set_lr(new_lr_c)


# ─────────────────────────────────────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_agents(state_dim, n_states_disc, n_actions, cfg, seed=42):
    tc = cfg
    return {
        'Q-Learning': QLearningAgent(
            n_states=n_states_disc, n_actions=n_actions,
            alpha=tc['ql_alpha'], gamma=tc['ql_gamma'],
            epsilon=tc['ql_epsilon_start'], epsilon_min=tc['ql_epsilon_end'],
            epsilon_decay=tc['ql_epsilon_decay'], seed=seed),
        'DQN': DQNAgent(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dim=tc['dqn_hidden_dim'], lr=tc['dqn_lr'],
            gamma=tc['dqn_gamma'], batch_size=tc['dqn_batch_size'],
            buffer_size=tc['dqn_buffer_size'],
            target_update=tc['dqn_target_update'],
            epsilon=tc['dqn_epsilon_start'], epsilon_min=tc['dqn_epsilon_end'],
            epsilon_decay=tc['dqn_epsilon_decay'], seed=seed),
        'Double DQN': DDQNAgent(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dim=tc['dqn_hidden_dim'], lr=tc['dqn_lr'],
            gamma=tc['dqn_gamma'], batch_size=tc['dqn_batch_size'],
            buffer_size=tc['dqn_buffer_size'],
            target_update=tc['dqn_target_update'],
            epsilon=tc['dqn_epsilon_start'], epsilon_min=tc['dqn_epsilon_end'],
            epsilon_decay=tc['dqn_epsilon_decay'], seed=seed),
        'PPO': PPOAgent(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dim=tc['ppo_hidden_dim'],
            lr_actor=tc['ppo_lr_actor'], lr_critic=tc['ppo_lr_critic'],
            gamma=tc['ppo_gamma'], clip_epsilon=tc['ppo_clip_epsilon'],
            update_epochs=tc['ppo_update_epochs'],
            entropy_coef=tc['ppo_entropy_coef'], seed=seed),
    }
