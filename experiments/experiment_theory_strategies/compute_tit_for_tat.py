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

def compute_best_response(A, C, p_opponent, price_bounds, mu, a_0, num_points=201):
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

prices = np.linspace(price_bounds[0], price_bounds[1], 1001)
plt.rcParams.update({'font.size': 16})

tit_for_tat_prices = []
monopoly_reward = -1
monopoly_price = 0
for p in prices:
    rewards = get_rewards(np.array([[p+0.002, p]]), A, C, mu, a_0)
    if rewards[0, 0] > monopoly_reward:
        monopoly_reward = rewards[0, 0]
        monopoly_price = p + 0.002
        tit_for_tat_prices.append(p + 0.002)
    else:
        tit_for_tat_prices.append(monopoly_price)

# Plot the reward curve
tit_for_tat_prices = np.array(tit_for_tat_prices)

rewards = get_rewards(np.vstack((prices, tit_for_tat_prices)).T, A, C, mu, a_0)

plt.figure(figsize=(8, 6))
plt.plot(prices, rewards[:,0])
plt.xlabel("Price")
plt.ylabel("Reward")
plt.grid()
plt.tight_layout()
plt.savefig("tit_for_tat_rewards.png")
plt.close()


# Plot the prices
plt.figure(figsize=(8, 6))
plt.plot(prices, tit_for_tat_prices)
plt.xlabel("Price")
plt.ylabel("Tit-for-Tat price")
plt.grid()
plt.tight_layout()
plt.savefig("tit_for_tat_prices.png")
plt.close()


# Save the prices to a file
# Normalise
tit_for_tat_params = (tit_for_tat_prices - price_bounds[0]) / (price_bounds[1] - price_bounds[0])
np.savetxt("tit_for_tat_up_params.txt", tit_for_tat_params)


# Save the tit-for-tat prices for plotting
np.savetxt("tit_for_tat_up_prices.txt", np.vstack((prices, tit_for_tat_prices)).T)


# Create y=x mapping
y = np.linspace(0, 1, 101)
y = y * (price_bounds[1] - price_bounds[0]) + price_bounds[0]
x = y
np.savetxt("y=x.txt", np.array(list(zip(x, y))))


# Compute the rewards at the monopoly price
monopoly_rewards = get_rewards(np.array([[monopoly_price, monopoly_price-0.002]]), A, C, mu, a_0)

# Print the monopoly rewards and the prices
print("Monopoly Price Leader:", monopoly_price)
print("Monopoly Price Follower:", monopoly_price-0.002)
print("Monopoly Rewards:", monopoly_rewards)