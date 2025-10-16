import matplotlib.pyplot as plt
import numpy as np

from tools.agents.extortion_maps import make_monotone_equispaced_map
from tools.environment import RewardFunction, EnvironmentParameters

price_bounds = (1.0, 2.8)
A = np.array([1, 1])
C = np.array([1, 1])
mu = 0.25
a_0 = -1.0
env_params = EnvironmentParameters(N=2, sigma=0, A=A, C=C, mu_0=mu, a_0=a_0,)
reward_fn = RewardFunction(env_params)

# Load theoretical extortion mapping
theoretical_mapping = np.loadtxt("extortion_mapping.txt")


# Load learned extortion mapping
params = np.loadtxt("optimised_params.txt")
map_fn, map_dict = make_monotone_equispaced_map(params)

prices_follower = np.linspace(price_bounds[0], price_bounds[1], 201)
normalised_prices_follower = (prices_follower - price_bounds[0]) / (price_bounds[1] - price_bounds[0])
normalised_prices_leader = np.array([map_fn(p) for p in normalised_prices_follower])
prices_leader = normalised_prices_leader * (price_bounds[1] - price_bounds[0]) + price_bounds[0]

    
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(prices_follower, prices_leader, label='Learned Mapping', color='tab:blue')
axs[0].plot(theoretical_mapping[:, 0], theoretical_mapping[:, 1], label='Theoretical Mapping', color='tab:blue', linestyle='--')
axs[0].set_xlabel("Follower's Price $p^F$")
axs[0].set_ylabel("Leader's Price $p^L$")
axs[0].legend()
axs[0].grid()
rewards = reward_fn(np.array(list(zip(prices_leader, prices_follower))))
rewards_leader = rewards[:, 0]
rewards_follower = rewards[:, 1]
theoretical_rewards = reward_fn(theoretical_mapping)
theoretical_rewards_leader = theoretical_rewards[:, 0]
theoretical_rewards_follower = theoretical_rewards[:, 1]
axs[1].plot(prices_follower, rewards_leader, label='Leader Reward $r^L$', color='tab:orange')
axs[1].plot(prices_follower, rewards_follower, label='Follower Reward $r^F$', color='tab:blue')
axs[1].plot(theoretical_mapping[:, 0], theoretical_rewards_leader, label='Theoretical Leader Reward $r^L$', color='tab:blue', linestyle='--')
axs[1].plot(theoretical_mapping[:, 0], theoretical_rewards_follower, label='Theoretical Follower Reward $r^F$', color='tab:orange', linestyle='--')
axs[1].set_xlabel("Follower's Price $p^F$")
axs[1].set_ylabel("Reward")
axs[1].grid()
axs[1].legend()
plt.tight_layout()
plt.savefig("eps_greedy_extortion_mapping.png", dpi=300)
plt.show()
plt.close()