"""Compute the best response mapping for every price"""

import numpy as np
import matplotlib.pyplot as plt


def get_rewards(prices, A, C, mu, a_0):
    """
    Compute the rewards for both agents given their prices.

    Args:
        prices (np.ndarray): Array of shape (n, 2) where each row is (p_f, p_l).
        A (np.ndarray): Demand intercepts for both agents.
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



A = np.array([[1, 1]])
C = np.array([[1, 1]])
mu = 0.25
a_0 = -1.0

price_bounds = [1.0, 2.8]

prices = np.linspace(price_bounds[0], price_bounds[1], 100)
rewards = []

for p in prices:
    rew = get_rewards(np.array([[p, p]]), A, C, mu, a_0)
    rewards.append(rew[0, 0])

rewards = np.array(rewards)
# Plot the best response mapping
plt.figure()
plt.plot(prices, rewards)
plt.title("Reward as a function of Joint Price")
plt.xlabel("Agent Price")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get the Pareto price and reward
pareto_index = np.argmax(rewards)
pareto_price = prices[pareto_index]
pareto_reward = rewards[pareto_index]

print(f"Pareto Price: {pareto_price}, Pareto Reward: {pareto_reward}")