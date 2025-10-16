"""Find the extortion strategy that extorts the learning agent to cooperate."""

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

def compute_best_response(A, C, p_opponent, price_bounds, mu, a_0, num_points=4001):
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

plt.rcParams.update({'font.size': 16})

prices = np.linspace(price_bounds[0], price_bounds[1], 201)
best_response_prices = []
for p in prices:
    br_price, br_reward, prices_plot, rewards_plot = compute_best_response(
        A, C, p, price_bounds, mu, a_0)
    best_response_prices.append(br_price[0])


# Plot the best response function
plt.figure(figsize=(8, 6))
plt.plot(prices, best_response_prices, label='Best Response')
plt.plot(prices, prices, 'k--', label='y=x')
plt.xlim(price_bounds)
plt.ylim(price_bounds)
plt.xlabel("Price")
plt.ylabel("Best Response Price")
plt.title("Best Response Function")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("best_response_function.png")
plt.show()

# Save the best response data
# Normalise by the range
best_response_params = (np.array(best_response_prices) - price_bounds[0]) / (price_bounds[1] - price_bounds[0])
np.savetxt("best_response_params.txt", best_response_params)

br_price, br_reward, prices_plot, rewards_plot = compute_best_response(
    A, C, 1.4604315082785075, price_bounds, mu, a_0)

print("Best response to 1.4604315082785075:", br_price, br_reward)

# # Plot the reward function for both agents
# plt.figure(figsize=(10, 6))
# plt.plot(prices_plot, rewards_plot[:, 0], label='Extortion Agent Reward', color='blue')
# plt.plot(prices_plot, rewards_plot[:, 1], label='Learning Agent Reward', color='orange')
# plt.axvline(x=1.4604315082785075, color='red', linestyle='--', label='Extortion Agent Price')
# plt.scatter(br_price[0], br_reward[1], color='green', zorder=5, label='Best Response')
# plt.xlim(price_bounds)
# plt.ylim(0, 0.3)
# plt.xlabel("Learning Agent Price")
# plt.ylabel("Reward")
# plt.title("Reward Functions")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()


# Plot rewards for best response
prices_all = np.vstack((prices, best_response_prices)).T
rewards = get_rewards(prices_all, A, C, mu, a_0)    
plt.figure(figsize=(10, 6))
plt.plot(prices, rewards[:, 0], label='Other Reward')
plt.plot(prices, rewards[:, 1], label='Best Response Reward')
plt.xlim(price_bounds)
plt.ylim(0, 0.3)
plt.xlabel("Price")
plt.ylabel("Reward")
plt.title("Rewards at Best Response")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


print(get_rewards(np.array([[1.4604315082785075, 1.4604315082785075]]), A, C, mu, a_0))
print(get_rewards(np.array([[1.4614315082785075, 1.4604315082785075]]), A, C, mu, a_0))
print(get_rewards(np.array([[1.4624315082785075, 1.4604315082785075]]), A, C, mu, a_0))
print(get_rewards(np.array([[1.4634315082785075, 1.4604315082785075]]), A, C, mu, a_0))