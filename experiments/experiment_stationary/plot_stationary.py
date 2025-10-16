import matplotlib.pyplot as plt
import numpy as np


def plot_strategy(params, reward_fn, price_bounds, save_path, nash_price=None, nash_reward=None):
    """
    Plot the extortionate strategy defined by the given parameters.
    The parameters correspond to discrete probabilities for different prices.

    Args:
        params (list or np.ndarray): Parameters defining the extortionate strategy.
        price_bounds (tuple): A tuple (min_price, max_price) defining the price bounds.
        save_path (str): Path to save the generated plot.
    """
    # Set font size for plots
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['legend.fontsize'] = 17

    min_price, max_price = price_bounds
    
    # Plot a bar chart of the probabilities for each discrete price
    params = np.array(params)
    # Normalise using softmax
    exp_params = np.exp(params - np.max(params))  # Subtract max for numerical stability
    params = exp_params / exp_params.sum()

    discrete_prices = np.linspace(min_price, max_price, len(params))
    # plt.figure(figsize=(10, 6))
    # plt.bar(discrete_prices, params, width=(max_price - min_price) / len(params), align='center', alpha=0.7)
    # plt.xlabel('Price')
    # plt.ylabel('Probability')
    # plt.title('Extortionate Strategy Distribution')
    # plt.xticks(discrete_prices)
    # plt.grid(axis='y')
    # plt.savefig(save_path + '/strategy_distribution.png')
    # plt.close()

    # Compute the induced reward function
    prices = np.linspace(min_price, max_price, 51)
    follower_rewards = []
    leader_rewards = []
    for price in prices:
        follower_reward = 0
        leader_reward = 0
        for i, opponent_price in enumerate(discrete_prices):
            rewards = reward_fn(np.array([[price, opponent_price]]))
            follower_reward += rewards[0, 0] * params[i]
            leader_reward += rewards[0, 1] * params[i]
        leader_rewards.append(leader_reward)
        follower_rewards.append(follower_reward)
    follower_rewards = np.array(follower_rewards)
    leader_rewards = np.array(leader_rewards)


    # Plot both plots side by side for better visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].bar(discrete_prices, params, width=(max_price - min_price) / len(params), align='center', alpha=1.)
    axs[0].set_xlabel('Leader price $p^L$')
    axs[0].set_ylabel('Probability')
    # axs[0].set_xticks(discrete_prices)
    

    axs[1].plot(prices, leader_rewards, label='Leader')
    axs[1].plot(prices, follower_rewards, label='Follower')
    if nash_price is not None:
        axs[1].axvline(nash_price, color='red', linestyle='--', label='Nash')
        axs[1].axhline(nash_reward, color='red', linestyle='--',)

    axs[1].legend()
    axs[1].set_xlabel('Follower price $p^F$')
    axs[1].set_ylabel('Induced rewards')
    axs[1].grid()

    plt.tight_layout()

    plt.savefig(save_path + '/strategy_and_reward.png', dpi=300)
    plt.close()
