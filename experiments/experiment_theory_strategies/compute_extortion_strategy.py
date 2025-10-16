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

plt.rcParams.update({'font.size': 20})
# Set legends font size
plt.rcParams['legend.fontsize'] = 17


# Find the minimum reward of the follower
p_opponent = C[0, 1]
prices_min, rewards_min, prices, rewards = compute_best_response(
    A, C, p_opponent=p_opponent, price_bounds=price_bounds, mu=mu, a_0=a_0
)
r_f_min = rewards_min[0]
p_f_min = prices_min[0]

print(f"Minimum reward of the follower: {r_f_min:.4f} at price {p_f_min:.4f}")

# Find the line and maximise the leader reward
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

# Plot both the reward and follower price on subplots with two columns
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_leader[1:], rewards_leader, label='Leader')
axs[0].plot(prices_leader[1:], r_f_min*np.ones_like(prices_leader[1:]), label='Follower')
axs[0].set_xlabel("Leader price $p^L$")
axs[0].set_ylabel("Rewards")
# axs[0].set_title("Rewards along the Constraint Line")
axs[0].grid()
axs[0].legend()
axs[1].plot(prices_leader[1:], prices_follower)
axs[1].set_xlabel("Leader price $p^L$")
axs[1].set_ylabel("Follower price $p^F$")
# axs[1].set_title("Extortion Constraint Line")
axs[1].grid()
plt.tight_layout()
plt.savefig("leader_follower_prices_constraint_line.png", dpi=300)
plt.show()



# Find the best leader price
p_L_star = best_price_leader
p_F_star = prices_follower[max_index]


def leader_mapping(p_F):
    if p_F < p_F_star:
        return price_bounds[0]
    else:
        return p_L_star+0.05

# Find the rewards and plot the mapping
prices_follower = np.linspace(price_bounds[0], price_bounds[1], 201)
prices_leader = [leader_mapping(p_F) for p_F in prices_follower]
rewards = get_rewards(np.array(list(zip(prices_leader, prices_follower))), A, C, mu, a_0)
rewards_leader = rewards[:, 0]
rewards_follower = rewards[:, 1]

# Save the mapping
np.savetxt("extortion_mapping.txt", np.array(list(zip(prices_follower, prices_leader))))

# Plot both the reward and follower price on subplots with two columns
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_follower, rewards_leader, label='Leader')
axs[0].plot(prices_follower, rewards_follower, label='Follower')
axs[0].set_xlabel("Follower price $p^F$")
axs[0].set_ylabel("Rewards")
# axs[0].set_title("Rewards given Extortion Mapping")
axs[0].legend()
axs[0].grid()
axs[1].plot(prices_follower, prices_leader)
axs[1].set_xlabel("Follower price $p^F$")
axs[1].set_ylabel("Leader price $p^L$")
# axs[1].set_title("Extortion Mapping")
axs[1].grid()
plt.tight_layout()
plt.savefig("extortion_mapping.png", dpi=300)
plt.show()


# Finding unimodal extortion mapping
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

# Plot both the reward and follower price on subplots with two columns
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_follower, rewards_leader, label='Leader')
axs[0].plot(prices_follower, rewards_follower, label='Follower')
axs[0].set_xlabel("Follower price $p^F$")
axs[0].set_ylabel("Rewards")
# axs[0].set_title("Rewards given Unimodal Extortion Mapping")
axs[0].legend()
axs[0].grid()
axs[1].plot(prices_follower, prices_leader)
axs[1].set_xlabel("Follower price $p^F$")
axs[1].set_ylabel("Leader price $p^L$")
# axs[1].set_title("Unimodal Extortion Mapping")
axs[1].grid()
plt.tight_layout()
plt.savefig("unimodal_extortion_mapping.png", dpi=300)
plt.show()

    
# Save the mapping
np.savetxt("unimodal_extortion_mapping.txt", np.array(list(zip(prices_follower, prices_leader))))
