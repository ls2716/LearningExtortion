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

def compute_best_response(A, C, p_opponent, price_bounds, mu, a_0, num_points=401):
    """
    Compute the best response price for the learning agent given the fixed price of the extortion agent.

    Args:
        a_f (float): Demand intercept for the extortion agent.
        a_l (float): Demand intercept for the learning agent.
        p_f (float): Fixed price of the extortion agent.
        price_bounds (tuple): Bounds for the learning agent's price.
        mu (float): Scale parameter for the logit model.
        a_0 (float): Outside option utility.
        num_points (int): Number of points to evaluate in the price range.

    Returns:
        float: Best response price for the learning agent.
        float: Maximum reward for the learning agent.
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

prices = np.linspace(price_bounds[0], price_bounds[1], 100)
best_responses = []

for p in prices:
    best_price, rewards, _, _ = compute_best_response(A, C, p, price_bounds, mu, a_0)
    best_responses.append(best_price[0])


# Plot the best response mapping
plt.figure()
plt.plot(prices, best_responses)
plt.ylim(price_bounds[0]-0.1, price_bounds[1]+0.1)
plt.xlim(price_bounds[0]-0.1, price_bounds[1]+0.1)
plt.title("Best Response Mapping for Learning Agent")
plt.xlabel("Agent Price")
plt.ylabel("Best Response Learning Agent Price")
plt.grid(True)
plt.tight_layout()
plt.show()

best_responses = (np.array(best_responses)-price_bounds[0])/(price_bounds[1]-price_bounds[0])
# Save to a text file in the same directory as this script
output_path = "best_response_mapping.txt"
np.savetxt(output_path, best_responses)