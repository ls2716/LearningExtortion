import numpy as np


def get_probs(prices, a, c, mu, a_0):
    """
    Calculate the probabilities of choosing each action based on a logistic function.

    Args:
        prices (np.ndarray): Array of prices (shape: [batch, n_agents]).
        a (float): Parameter for the logistic function.
        c (float): Cost parameter (not used directly here).
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.

    Returns:
        np.ndarray: Probabilities for each action (shape: [batch, n_agents]).
    """
    exps = np.exp((a - prices) / mu)
    probs = exps / (np.sum(exps, axis=1, keepdims=True) + np.exp(a_0 / mu))
    return probs


def get_rewards(prices, a, c, mu, a_0, covariance_matrix):
    """
    Calculate the rewards for each agent, using the logistic function and adding multivariate Gaussian noise.

    Args:
        prices (np.ndarray): Array of prices (shape: [batch, n_agents]).
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.
        covariance_matrix (np.ndarray): Covariance matrix for the noise (shape: [n_agents, n_agents]).

    Returns:
        np.ndarray: Rewards for each agent (shape: [batch, n_agents]).
    """
    probs = get_probs(prices, a, c, mu, a_0)
    random_noise = np.random.multivariate_normal(
        mean=np.zeros(prices.shape[1]), cov=covariance_matrix, size=prices.shape[0]
    )*0
    rewards = (prices - c) * probs + random_noise
    return rewards


def find_nash(N, a, c, mu, a_0):
    """
    Find the Nash equilibrium price for a symmetric game.

    Args:
        N (int): Number of agents.
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.

    Returns:
        float: Nash equilibrium price (for the first agent, symmetric case).
    """
    print("Finding Nash equilibrium...")
    p = np.zeros(shape=(1, N)) + c + np.random.rand(1, N) * 0.01
    for iteration in range(100):
        p_new = (c + mu / (1 - get_probs(p, a, c, mu, a_0))).reshape(1, N)
        p_new = p + (p_new - p) * 0.2  # Smooth the update
        if np.all(np.abs(p_new - p) < 1e-5):
            break
        p = p_new
    if iteration == 19:
        print("Warning: find_nash did not converge within 20 iterations.")
    # Find the nash reward
    p = p.reshape(1, N)
    reward = (p - c) * get_probs(p, a, c, mu, a_0)

    return p[0, 0], reward[0, 0]


a = 1
c = 1
N = 2
mu = 0.25
a_0 = -1.0

nash_price, nash_reward = find_nash(N, a, c, mu, a_0)

print(f"Nash price {nash_price} with reward {nash_reward}")

# Compute the Pareto prices
price_bounds = [1.0, 2.8]

prices = np.linspace(price_bounds[0], price_bounds[1], 1001)

tit_for_tat_prices = []
monopoly_reward = -1
monopoly_price = 0
for p in prices:
    rewards = get_rewards(np.array([[p, p]]), a, c, mu, a_0, np.eye(N)*0)
    if rewards[0, 0] > monopoly_reward:
        monopoly_reward = rewards[0, 0]
        monopoly_price = p
    else:
        break

print(f"Pareto price {monopoly_price} with reward {monopoly_reward}")
