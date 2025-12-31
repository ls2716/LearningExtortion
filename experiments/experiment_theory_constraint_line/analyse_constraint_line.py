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
    prices_all = np.vstack((np.full_like(prices, p_opponent), prices)).T
    rewards = get_rewards(prices_all, A, C, mu, a_0)
    best_index = np.argmax(rewards[:, 1])
    return (
        prices[best_index],
        rewards[best_index, 1],
        prices,
        rewards[:, 1],
        rewards[best_index, 0],
    )


A = np.array([[1, 1]])
C = np.array([[1, 1]])
mu = 0.25
a_0 = -1.0

price_bounds = [1.0, 3.0]

# Plot the best response
p_opponent = 1.0

p_fmin, r_fmin, prices, rewards, r_lmin = compute_best_response(
    A, C, p_opponent=p_opponent, price_bounds=price_bounds, mu=mu, a_0=a_0
)


# Increase p_opponent and find appriopriate p_f that ensures r_f(p_f, p_opponent) = r_fmin
p_l = p_opponent
p_f = p_fmin
prices_leader = [p_l]
prices_follower = [p_f]
while p_l < price_bounds[1]:
    p_l += 0.01
    rewards = get_rewards(np.array([[p_f, p_l]]), A, C, mu, a_0)[0]
    r_f = rewards[0]
    r_l = rewards[1]
    while r_f > r_fmin and p_f < price_bounds[1]:
        p_f += 0.0001
        rewards = get_rewards(np.array([[p_f, p_l]]), A, C, mu, a_0)[0]
        r_f = rewards[0]
        r_l = rewards[1]
    prices_leader.append(p_l)
    prices_follower.append(p_f)
    print(f"p_L {p_l} with reward {r_l} vs p_F {p_f} with reward {r_f}")

plt.plot(prices_leader, prices_follower)
plt.xlabel("Leader Price")
plt.ylabel("Follower Price")
plt.grid()
# plt.show()
plt.close()

# Compute monopoly price
p_best, r_best, prices, rewards, _ = compute_best_response(
    A, C, p_opponent=np.inf, price_bounds=price_bounds, mu=mu, a_0=a_0
)
print(f"Monopoly price {p_best} with reward {r_best}")


# Plot surface of the rewards for the follower


p_f = np.linspace(price_bounds[0], price_bounds[1], 101)
p_l = np.linspace(price_bounds[0], price_bounds[1], 101)
P_F, P_L = np.meshgrid(p_f, p_l)
prices_all = np.vstack((P_L.ravel(), P_F.ravel())).T
rewards = get_rewards(prices_all, A, C, mu, a_0)
R_F = rewards[:, 1].reshape(P_F.shape)
R_L = rewards[:, 0].reshape(P_F.shape)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(P_F, P_L, R_F, cmap="viridis")
ax.set_xlabel("Follower price $r^F$")
ax.set_ylabel("Leader price $p^L$")
ax.set_zlabel("Follower Reward")
ax.set_title("Follower Reward Surface")
ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(P_F, P_L, R_L, cmap="viridis")
ax.set_xlabel("Follower Price")
ax.set_ylabel("Leader Price")
ax.set_zlabel("Leader Reward")
ax.set_title("Leader Reward Surface")
# plt.show()
plt.close()


# Plot just the follower reward and overlay a line prices_leader, prices_follower
# Set font size
plt.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(P_F, P_L, R_F, cmap="viridis", alpha=0.7)
ax.plot(
    prices_follower,
    prices_leader,
    +np.array([r_fmin] * len(prices_follower)),
    color="r",
)
ax.set_xlabel("Follower price $p^F$")
ax.set_ylabel("Leader price $p^L$")
ax.set_zlabel("Follower reward $r^F$")
# ax.set_title('Follower Reward Surface with Constraint Line')
plt.tight_layout()
plt.savefig("follower_reward_surface_with_constraint_line.png", dpi=300)
plt.show()
