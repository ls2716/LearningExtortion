"""Find the extortion strategy that extorts the learning agent to cooperate."""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
OUTPUT_DIR = "outputs/extortion_strategy"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_rewards(prices, A, C, mu, a_0):
    """
    Compute the rewards for both agents given their prices.

    Args:
        prices (np.ndarray): Array of shape (n, 2) where each row is (p_f, p_l).
        A (np.ndarray): Quality coefficients for both agents.
        C (np.ndarray): Cost coefficients for both agents.
        mu (float): Scale parameter for the logit model.
        a_0 (float): Outside option utility.
    Returns:
        np.ndarray: Array of shape (n, 2) where each row is (r_f, r_l).
    """
    exp_utilities = np.exp((A - prices) / mu)
    sum_exp_utilities = np.sum(exp_utilities, axis=1, keepdims=True)
    exp_0 = np.exp(a_0 / mu)
    sum_exp_utilities += exp_0
    demands = exp_utilities / sum_exp_utilities
    rewards = demands * (prices - C)
    return rewards

def compute_best_response(A, C, p_opponent, price_bounds, mu, a_0, num_points=201):
    """
    Compute the best response price given the opponent's price.

    Args:
        A: Quality coefficients for both agents.
        C: Cost coefficients for both agents.
        p_opponent: Fixed price of the opponent.
        price_bounds (tuple): Bounds for the agent's price search.
        mu: Scale parameter for the logit model.
        a_0: Outside option utility.
        num_points: Number of points to evaluate in the price range.

    Returns:
        Tuple: (best_response_price, reward_at_br, all_prices, all_rewards)
    """
    prices = np.linspace(price_bounds[0], price_bounds[1], num_points)
    prices_all = np.vstack((prices, np.full_like(prices, p_opponent))).T
    rewards = get_rewards(prices_all, A, C, mu, a_0)
    best_index = np.argmax(rewards[:, 0])
    return prices_all[best_index], rewards[best_index], prices, rewards


A = np.array([[1, 1]])
C = np.array([[1, 1]])
mu = 0.25
a_0 = -1.0

price_bounds = [1.0, 2.8]

# Configure plotting
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 17})


# ============================================================================
# Phase 1: Find the minimum reward (constraint line starting point)
# ============================================================================
p_opponent = C[0, 1]
prices_min, rewards_min, prices, rewards = compute_best_response(
    A, C, p_opponent=p_opponent, price_bounds=price_bounds, mu=mu, a_0=a_0
)
r_f_min = rewards_min[0]
p_f_min = prices_min[0]

print(f"Minimum reward of the follower: {r_f_min:.4f} at price {p_f_min:.4f}")

# ============================================================================
# Phase 2: Find constraint line that maximizes leader reward
# ============================================================================
prices_leader = np.linspace(price_bounds[0], price_bounds[1], 201)
rewards_leader = []
price_follower = prices_min[0]
prices_follower = []
for p_leader in prices_leader[1:]:
    rewards = get_rewards(np.array([[p_leader, price_follower]]), A, C, mu, a_0)
    r_f = rewards[0, 1]
    while r_f > r_f_min:
        price_follower += 0.0001
        rewards = get_rewards(np.array([[p_leader, price_follower]]), A, C, mu, a_0)
        r_f = rewards[0, 1]
    rewards_leader.append(rewards[0, 0])
    prices_follower.append(price_follower)
    print(f"Leader price: {p_leader:.4f}, Follower price: {price_follower:.4f}, Leader reward: {rewards[0,0]:.4f}, Follower reward: {rewards[0,1]:.4f}")
rewards_leader = np.array(rewards_leader)
max_index = np.argmax(rewards_leader)
best_price_leader = prices_leader[1:][max_index]

print(f"Best leader price: {best_price_leader:.4f} with reward {rewards_leader[max_index]:.4f}")

# Plot constraint line results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_leader[1:], rewards_leader, label='Leader', linewidth=2)
axs[0].plot(prices_leader[1:], r_f_min*np.ones_like(prices_leader[1:]), label='Follower', linewidth=2)
axs[0].set_xlabel("Leader price $p^L$")
axs[0].set_ylabel("Rewards")
axs[0].grid()
axs[0].legend()
axs[1].plot(prices_leader[1:], prices_follower, linewidth=2)
axs[1].set_xlabel("Leader price $p^L$")
axs[1].set_ylabel("Follower price $p^F$")
axs[1].grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leader_follower_prices_constraint_line.png"), dpi=300)
plt.show()



# ============================================================================
# Phase 3: Define extortion mapping using optimal prices
# ============================================================================
p_L_star = best_price_leader
p_F_star = prices_follower[max_index]

def extortion_mapping(p_F):
    """Leader's price in response to follower's price."""
    if p_F < p_F_star:
        return price_bounds[0]
    else:
        return p_L_star + 0.05

# Compute rewards along the extortion mapping
prices_follower = np.linspace(price_bounds[0], price_bounds[1], 201)
prices_leader = [extortion_mapping(p_F) for p_F in prices_follower]
rewards = get_rewards(np.array(list(zip(prices_leader, prices_follower))), A, C, mu, a_0)
rewards_leader = rewards[:, 0]
rewards_follower = rewards[:, 1]

# Save the mapping
np.savetxt(os.path.join(OUTPUT_DIR, "extortion_mapping.txt"), np.array(list(zip(prices_follower, prices_leader))))

# Plot extortion mapping results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_follower, rewards_leader, label='Leader', linewidth=2)
axs[0].plot(prices_follower, rewards_follower, label='Follower', linewidth=2)
axs[0].set_xlabel("Follower price $p^F$")
axs[0].set_ylabel("Rewards")
axs[0].legend()
axs[0].grid()
axs[1].plot(prices_follower, prices_leader, linewidth=2)
axs[1].set_xlabel("Follower price $p^F$")
axs[1].set_ylabel("Leader price $p^L$")
axs[1].grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "extortion_mapping.png"), dpi=300)
plt.show()


# ============================================================================
# Phase 4: Find unimodal extortion mapping
# ============================================================================
prices_follower = np.linspace(price_bounds[0], price_bounds[1], 201)
prices_leader = []
rewards_leader = []
rewards_follower = []
p_l = price_bounds[0]
for p_f in prices_follower:
    if p_f < p_f_min:
        prices_leader.append(price_bounds[0])
        rewards = get_rewards(np.array([[price_bounds[0], p_f]]), A, C, mu, a_0)
        rewards_leader.append(rewards[0,0])
        rewards_follower.append(rewards[0,1])
        r_f = rewards[0,1]
    else:
        new_r_f = r_f_min - 1.
        while new_r_f < r_f+0.0001:
            p_l += 0.0001
            rewards = get_rewards(np.array([[p_l, p_f]]), A, C, mu, a_0)
            new_r_f = rewards[0, 1]
        if rewards[0,0] < rewards_leader[-1]:
            while len(prices_leader)< len(prices_follower):
                prices_leader.append(prices_leader[-1])
            break
        else:
            prices_leader.append(p_l)
            rewards_leader.append(rewards[0,0])
            rewards_follower.append(rewards[0,1])
            r_f = rewards[0,1]

prices_leader = np.array(prices_leader)
rewards = get_rewards(np.array(list(zip(prices_leader, prices_follower))), A, C, mu, a_0)
rewards_leader = rewards[:, 0]
rewards_follower = rewards[:, 1]

# Plot unimodal mapping results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_follower, rewards_leader, label='Leader', linewidth=2)
axs[0].plot(prices_follower, rewards_follower, label='Follower', linewidth=2)
axs[0].set_xlabel("Follower price $p^F$")
axs[0].set_ylabel("Rewards")
axs[0].legend()
axs[0].grid()
axs[1].plot(prices_follower, prices_leader, linewidth=2)
axs[1].set_xlabel("Follower price $p^F$")
axs[1].set_ylabel("Leader price $p^L$")
axs[1].grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "unimodal_extortion_mapping.png"), dpi=300)
plt.show()

# Save the mapping
np.savetxt(os.path.join(OUTPUT_DIR, "unimodal_extortion_mapping.txt"), np.array(list(zip(prices_follower, prices_leader))))
