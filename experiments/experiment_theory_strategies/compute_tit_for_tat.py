"""Compute tit-for-tat strategy."""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
OUTPUT_DIR = "outputs/tit_for_tat"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_rewards(prices, A, C, mu, a_0):
    """
    Compute the rewards for both agents given their prices.

    Args:
        prices (np.ndarray): Array of shape (n, 2) where each row is (p_f, p_l).
        A (np.ndarray): Quality coefficients for both agents.
        C (np.ndarray): Cost coefficients for both agents.
        mu (float): Substitution coefficient.
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
        mu: Substitution coefficient.
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
plt.rcParams.update({'font.size': 16})

# ============================================================================
# Compute tit-for-tat strategy
# ============================================================================
prices = np.linspace(price_bounds[0], price_bounds[1], 1001)
tft_prices = []
max_reward = -1
max_price = 0

for p in prices:
    # Leader plays with offset, follower matches
    rewards = get_rewards(np.array([[p + 0.002, p]]), A, C, mu, a_0)
    if rewards[0, 0] > max_reward:
        max_reward = rewards[0, 0]
        max_price = p + 0.002
        tft_prices.append(p + 0.002)
    else:
        tft_prices.append(max_price)

tft_prices = np.array(tft_prices)

# ============================================================================
# Plot and save results
# ============================================================================
rewards = get_rewards(np.vstack((prices, tft_prices)).T, A, C, mu, a_0)

# Plot reward curve
plt.figure(figsize=(8, 6))
plt.plot(prices, rewards[:, 0], linewidth=2)
plt.xlabel("Follower Price")
plt.ylabel("Leader Reward")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tit_for_tat_rewards.png"))
plt.close()

# Plot tit-for-tat strategy
plt.figure(figsize=(8, 6))
plt.plot(prices, tft_prices, linewidth=2)
plt.xlabel("Follower Price")
plt.ylabel("Leader Price (Tit-for-Tat)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tit_for_tat_prices.png"))
plt.close()

# ============================================================================
# Save data to files
# ============================================================================
# Normalize prices to [0, 1]
tft_params = (tft_prices - price_bounds[0]) / (price_bounds[1] - price_bounds[0])
np.savetxt(os.path.join(OUTPUT_DIR, "tit_for_tat_up_params.txt"), tft_params)

# Save tit-for-tat price mapping
np.savetxt(os.path.join(OUTPUT_DIR, "tit_for_tat_up_prices.txt"), np.vstack((prices, tft_prices)).T)

# Create and save y=x mapping
y_values = np.linspace(0, 1, 101)
y_prices = y_values * (price_bounds[1] - price_bounds[0]) + price_bounds[0]
x_prices = y_prices
np.savetxt(os.path.join(OUTPUT_DIR, "y=x.txt"), np.array(list(zip(x_prices, y_prices))))


# ============================================================================
# Print results
# ============================================================================
leader_price = max_price
follower_price = max_price - 0.002
result_rewards = get_rewards(np.array([[leader_price, follower_price]]), A, C, mu, a_0)

print(f"Leader Price: {leader_price:.6f}")
print(f"Follower Price: {follower_price:.6f}")
print(f"Leader Reward: {result_rewards[0, 0]:.6f}")
print(f"Follower Reward: {result_rewards[0, 1]:.6f}")