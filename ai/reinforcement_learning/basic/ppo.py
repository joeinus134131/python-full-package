"""
ppo_debug_demo.py
Simple, educational implementation of PPO-Clip using numpy only.
Environment: 1D point trying to reach target (discrete actions: left, stay, right).
Policy: linear logits -> softmax
Value: linear -> scalar
No external ML libs so all gradients computed manually (analytic).
Heavy debug prints included.

Run: python3 ppo.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt  # optional for final plot

np.random.seed(0)


# ---------------------------
# Simple 1D env (custom)
# ---------------------------
class Simple1DEnv:
    """
    State: [position, velocity]
    Actions: 0 (left), 1 (stay), 2 (right)
    Action forces: -f, 0, +f
    Dynamics: pos += vel; vel = vel * damp + force + noise
    Reward: -|pos - target|  (higher when closer)
    Episode length: max_steps
    """
    def __init__(self, max_steps=50, target=1.0, force=0.06, damp=0.9, noise_std=0.01):
        self.max_steps = max_steps
        self.target = target
        self.force = force
        self.damp = damp
        self.noise_std = noise_std
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.t = 0
        return np.array([self.pos, self.vel], dtype=np.float32)

    def step(self, action):
        # action -> force
        if action == 0:
            a = -self.force
        elif action == 1:
            a = 0.0
        else:
            a = self.force

        noise = np.random.randn() * self.noise_std
        self.vel = self.vel * self.damp + a + noise
        self.pos = self.pos + self.vel
        self.t += 1

        # reward: negative absolute distance (so higher if close to target)
        dist = abs(self.pos - self.target)
        reward = -dist

        done = (self.t >= self.max_steps)
        state = np.array([self.pos, self.vel], dtype=np.float32)
        return state, reward, done, {'dist': dist}

    def sample_action(self):
        return np.random.choice(3)


# ---------------------------
# Helpers: softmax, one-hot
# ---------------------------
def softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=-1, keepdims=True)


def onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------
# Policy and Value models (linear for clarity)
# ---------------------------
class LinearPolicy:
    def __init__(self, state_dim, action_dim, lr=1e-2):
        # weights shape: (action_dim, state_dim), bias shape: (action_dim,)
        self.W = np.random.randn(action_dim, state_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.lr = lr

    def logits(self, states):
        # states: (N, state_dim) or (state_dim,)
        return states.dot(self.W.T) + self.b  # (N, action_dim)

    def get_action_and_prob(self, state):
        # state: (state_dim,)
        logits = self.logits(state.reshape(1, -1))[0]
        probs = softmax(logits)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action], probs

    def get_probs(self, states):
        return softmax(self.logits(states))

    # Manual gradient step using collected batch grads (we'll call in PPO update)
    def apply_gradients(self, dW, db):
        # simple SGD
        self.W += self.lr * dW
        self.b += self.lr * db


class LinearValue:
    def __init__(self, state_dim, lr=1e-2):
        self.w = np.random.randn(state_dim) * 0.1
        self.b = 0.0
        self.lr = lr

    def predict(self, states):
        # states: (N, state_dim) -> returns (N,)
        return states.dot(self.w) + self.b

    def apply_gradients(self, dw, db):
        self.w += self.lr * dw
        self.b += self.lr * db


# ---------------------------
# PPO core functions
# ---------------------------
def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """
    rewards: list length T
    values: list length T (value at each state)
    last_value: value(s) for next state (usually 0 if done)
    returns: advantages (T,), returns_to_go (T,)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_nonterminal = 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * last_gae * next_nonterminal
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def ppo_update(policy, valuef, states, actions, old_probs, advantages, returns,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, epochs=4, batch_size=64, debug=False):
    """
    states: (N, state_dim)
    actions: (N,) ints
    old_probs: (N,) probability of the taken action under old policy
    advantages: (N,)
    returns: (N,) target values
    We'll do multiple epochs and minibatch updates.
    """
    N = len(states)
    action_dim = policy.W.shape[0]
    state_dim = policy.W.shape[1]

    # Normalize advantages (standard trick)
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    indices = np.arange(N)

    # Accumulate debug metrics
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    count = 0

    for ep in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, N, batch_size):
            mb_idx = indices[start:start + batch_size]
            mb_states = states[mb_idx]  # (M, state_dim)
            mb_actions = actions[mb_idx]
            mb_old_probs = old_probs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            # forward current policy
            logits = policy.logits(mb_states)  # (M, action_dim)
            probs = softmax(logits)  # (M, action_dim)
            # probability of the taken actions (M,)
            probs_taken = probs[np.arange(len(mb_actions)), mb_actions]

            # ratio r = pi_theta(a|s) / pi_old(a|s)
            ratio = probs_taken / (mb_old_probs + 1e-12)

            # clipped surrogate objective
            surrogate1 = ratio * mb_adv
            surrogate2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -np.mean(np.minimum(surrogate1, surrogate2))  # negative because we do gradient ascent

            # entropy (encourage exploration)
            entropy = -np.mean(np.sum(probs * np.log(probs + 1e-12), axis=1))

            # value loss (MSE)
            values_pred = valuef.predict(mb_states)
            value_loss = np.mean((mb_returns - values_pred) ** 2)

            # --- Compute gradients manually ---

            # Value net gradients (simple linear: values = states.dot(w) + b)
            # d/dw MSE = (2/M) * states^T * (values_pred - returns)
            M = len(mb_states)
            value_error = values_pred - mb_returns  # (M,)
            dw = (2.0 / M) * (mb_states.T.dot(value_error))  # (state_dim,)
            db = (2.0 / M) * np.sum(value_error)

            # Policy gradients:
            # For a softmax linear policy: logits = W x + b
            # grad_W = (1/M) * sum_over_batch ( ( (one_hot(a) - pi) * advantage * weight ) outer state )
            # But we must account for clipping: surrogate chooses min between unclipped and clipped.
            # We'll compute "effective advantage" using selected surrogate (this gives us a simple surrogate gradient).
            chosen_surrogate = np.where(np.abs(ratio - 1.0) > clip_eps,
                                       surrogate2, surrogate1)  # elementwise choose which used
            # However for derivation we approximate gradient as grad log pi * effective_adv
            # grad_logpi wrt logits: one_hot(a) - pi
            # so weight = effective_adv / probs_taken (since d log pi = d pi / pi)
            # But better and simpler: use grad of negative loss: -mean(min(...)) -> so gradient ascend on min(...)
            eff_adv = chosen_surrogate / (mb_old_probs + 1e-12)  # approximate (dimension M)
            # Now compute gradient wrt logits: (one_hot - pi) * eff_adv[:, None]
            grad_logits = (onehot_batch(mb_actions, action_dim) - probs) * eff_adv[:, None]  # (M, action_dim)
            # dW = (1/M) * grad_logits^T dot states
            dW = (1.0 / M) * (grad_logits.T.dot(mb_states))  # (action_dim, state_dim)
            db_policy = (1.0 / M) * np.sum(grad_logits, axis=0)  # (action_dim,)

            # Apply gradients: ascend on policy (we already had negative sign in policy_loss).
            # Note: our parameter convention: apply_gradients does W += lr * dW
            policy.apply_gradients(dW, db_policy)
            valuef.apply_gradients(-dw * 0.5, -db * 0.5)  # coefficient to slow value updates (vf_coef ~ 0.5)

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            count += 1

    # average metrics
    avg_pol_loss = total_policy_loss / max(1, count)
    avg_val_loss = total_value_loss / max(1, count)
    avg_ent = total_entropy / max(1, count)

    if debug:
        # compute approximate mean KL: KL(old || new) approx mean(log(old_probs) - log(new_probs_of_taken))
        # For this we need new probs on full batch
        new_probs = policy.get_probs(states)
        new_taken = new_probs[np.arange(len(actions)), actions]
        mean_kl = np.mean(np.log(old_probs + 1e-12) - np.log(new_taken + 1e-12))
        ratio = new_taken / (old_probs + 1e-12)
        print(f"[DEBUG PPO UPDATE] avg_policy_loss={avg_pol_loss:.6f}, avg_value_loss={avg_val_loss:.6f}, avg_entropy={avg_ent:.6f}")
        print(f"  Approx KL={mean_kl:.6f}, ratio mean={ratio.mean():.6f}, ratio std={ratio.std():.6f}, ratio min={ratio.min():.6f}, ratio max={ratio.max():.6f}")

    return avg_pol_loss, avg_val_loss, avg_ent


# helper to create batch of onehots
def onehot_batch(actions, action_dim):
    M = len(actions)
    oh = np.zeros((M, action_dim), dtype=np.float32)
    oh[np.arange(M), actions] = 1.0
    return oh


# ---------------------------
# Training loop
# ---------------------------
def train_ppo(env,
              policy,
              valuef,
              epochs=200,
              steps_per_epoch=2048,
              gamma=0.99,
              lam=0.95,
              clip_eps=0.2,
              debug_every=10):
    episode_rewards = []
    avg_rewards_history = []

    for epoch in range(1, epochs + 1):
        # collect trajectories until we have steps_per_epoch timesteps
        states = []
        actions = []
        rewards = []
        old_probs = []
        values = []
        ep_rews = []
        steps = 0

        while steps < steps_per_epoch:
            s = env.reset()
            ep_reward = 0.0
            done = False
            traj_states = []
            traj_actions = []
            traj_rewards = []
            traj_old_probs = []
            traj_values = []

            while not done and steps < steps_per_epoch:
                # choose action under current policy
                a, p_a, p_all = policy.get_action_and_prob(s)
                v = valuef.predict(s.reshape(1, -1))[0]

                next_s, r, done, info = env.step(a)

                # store
                traj_states.append(s.copy())
                traj_actions.append(a)
                traj_rewards.append(r)
                traj_old_probs.append(p_a)
                traj_values.append(v)

                s = next_s
                steps += 1
                ep_reward += r

                if done:
                    # compute last value for GAE (0 if done)
                    last_val = 0.0
                    advs, rets = compute_gae(traj_rewards, traj_values, last_val, gamma=gamma, lam=lam)
                    # append to big buffers
                    states.extend(traj_states)
                    actions.extend(traj_actions)
                    rewards.extend(traj_rewards)
                    old_probs.extend(traj_old_probs)
                    values.extend(traj_values)
                    ep_rews.append(ep_reward)
                    break

        # convert to arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        # Need advantages & returns for entire buffer; we computed per-episode but only appended values.
        # For simplicity, recompute advantages per-episode using a pass through buffer:
        # We'll recompute by slicing episodes from the ep_rews lengths.
        # But simpler approach: recompute GAE by walking buffer backward assuming episode boundaries every env.max_steps.
        # For this educational script, do a straightforward approach: recompute per-episode with the same logic above.

        # Re-scan the buffer to split episodes: we can use env.max_steps to approximate splits when episodes ended.
        # To keep it robust, we reconstruct episodes from the rewards sequence by detecting done via lengths (since env fixed max_steps)
        # We'll break the buffer into episodes of length <= env.max_steps, from sequential filling earlier.
        advs_all = []
        returns_all = []
        idx = 0
        while idx < len(rewards):
            # find next episode length: up to env.max_steps or until we run out
            L = min(env.max_steps, len(rewards) - idx)
            r_segment = rewards[idx: idx + L]
            v_segment = values[idx: idx + L]
            last_val = 0.0
            advs, rets = compute_gae(r_segment, v_segment, last_val, gamma=gamma, lam=lam)
            advs_all.extend(advs.tolist())
            returns_all.extend(rets.tolist())
            idx += L

        advantages = np.array(advs_all, dtype=np.float32)
        returns = np.array(returns_all, dtype=np.float32)

        # Debug: sample stats
        mean_reward = np.mean(ep_rews)
        avg_rewards_history.append(mean_reward)
        episode_rewards.extend(ep_rews)

        # PPO update
        pol_loss, val_loss, ent = ppo_update(policy, valuef, states, actions, old_probs, advantages, returns,
                                             clip_eps=clip_eps, epochs=4, batch_size=64, debug=(epoch % debug_every == 0))

        if epoch % debug_every == 0 or epoch == 1:
            print(f"Epoch {epoch} | samples {len(states)} | mean_ep_reward {mean_reward:.4f} | policy_loss {pol_loss:.6f} | val_loss {val_loss:.6f} | entropy {ent:.6f}")

    return avg_rewards_history


# ---------------------------
# Main: initialize and run
# ---------------------------
if __name__ == "__main__":
    env = Simple1DEnv(max_steps=50, target=1.0)
    state_dim = 2
    action_dim = 3

    policy = LinearPolicy(state_dim=state_dim, action_dim=action_dim, lr=2e-2)
    valuef = LinearValue(state_dim=state_dim, lr=1e-2)

    # debug: play one random episode and print transitions
    def debug_episode():
        s = env.reset()
        print("=== Debug Episode Start ===")
        for t in range(20):
            a, p_a, p_all = policy.get_action_and_prob(s)
            v = valuef.predict(s.reshape(1, -1))[0]
            ns, r, done, info = env.step(a)
            print(f"t={t:02d} | s={s.round(3)} | action={a} | prob={p_a:.3f} | value={v:.3f} | reward={r:.3f} | dist={info['dist']:.3f}")
            s = ns
            if done:
                break
        print("=== Debug Episode End ===\n")

    print("Initial debug rollout (random-ish policy):")
    debug_episode()

    print("Training PPO... (this can take ~ <1-2 minutes depending on machine)")
    rewards = train_ppo(env, policy, valuef, epochs=80, steps_per_epoch=1024, gamma=0.99, lam=0.95, clip_eps=0.2)

    print("Final debug rollout (after training):")
    debug_episode()

    # Plot average rewards over epochs (optional)
    plt.plot(rewards)
    plt.xlabel("Epoch (per debug interval)")
    plt.ylabel("Mean episode reward")
    plt.title("Training progress (mean episode reward per epoch)")
    plt.show()
