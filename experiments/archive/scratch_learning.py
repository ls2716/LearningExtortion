import numpy as np, cma
from functools import lru_cache
from tools.agents.extortion_agent import make_monotone_mapper, ExtortionAgent, plot_map
from tools.competition import run_multiple_competitions, run_competition
from tools.environment import EnvironmentParameters, get_reward_function
from tools.agents.eps_greedy import EpsGreedyBandit


delay = 0.5
T = 400

# Define environment parameters
env_params = EnvironmentParameters(
    A = [1.0, 1.0],
    C = [1.0, 1.0],
    mu_0 = 0.25,
    a_0 = -1.0,
    N = 2,
    sigma = 0.0
)

reward_fn = get_reward_function(env_params)

def score(params: np.ndarray) -> float:
    extortion_agent = ExtortionAgent(map_type="monotone", params=params)
    learning_agent = EpsGreedyBandit(n_actions=21, epsilon=0.5, decay=0.99)
    results = run_multiple_competitions(
        n_competitions=10,
        reward_function=reward_fn,
        extortion_agent=extortion_agent,
        learning_agent=learning_agent,
        T=T,
        price_bounds=(1.0, 2.4),
        extortion_agent_initial_price=1.5,
        learning_agent_initial_price=1.5,
        delay=delay,
        skip_step_sum=int(T * 0.5)  # Skip first 50% of steps in cumulative reward
    )
    # Compute average loss (negative reward) over competitions
    avg_cum_reward = np.mean(results['extortion_agent_cumulative_rewards'])
    loss = -avg_cum_reward
    return float(loss)

@lru_cache(maxsize=4096)
def score_cached(key): return score(np.frombuffer(key, dtype=np.float64))

def objective(x):
    arr = np.array(x, dtype=np.float64)
    return score_cached(arr.tobytes())

n_params = 22
x0 = np.zeros(n_params)  # start (all zeros => uniform mapper)
sigma0 = 0.8             # initial step size; increase if exploration stalls

es = cma.CMAEvolutionStrategy(
    x0.tolist(),
    sigma0,
    {
        "bounds": [-6.0, 6.0],    # soft-ish box; CMA handles well
        "popsize": 8 + int(3*np.log(n_params)),  # default heuristic
        "seed": 42,
        "tolfun": 1e-6,
        "tolx": 1e-6,
        "maxfevals": 100,        # overall budget cap (optional)
        "verb_disp": 1,
    },
)
while not es.stop():
    X = es.ask()
    # (Optional) evaluate in parallel with joblib / multiprocessing here
    es.tell(X, [objective(x) for x in X])
    es.disp()

es.result_pretty()

best_params = np.array(es.result.xbest)
best_score = es.result.fbest

print("Best parameters:", best_params)
print("Best score (negative cumulative reward):", best_score)

# Visualize the best mapping function
plot_map(best_params)


# Run and plot a competition
extortion_agent = ExtortionAgent(map_type="monotone", params=best_params)
learning_agent = EpsGreedyBandit(n_actions=21, epsilon=0.5, decay=0.99)

results = run_competition(
    reward_function=reward_fn,
    extortion_agent=extortion_agent,
    learning_agent=learning_agent,
    T=T,
    price_bounds=(1.0, 2.4),
    extortion_agent_initial_price=1.5,
    learning_agent_initial_price=1.5,
    delay=delay,
    return_history=True,
    skip_step_sum=int(T * 0.5)  # Skip first 50% of steps in cumulative reward
)

# Print cumulative rewards
print(f"Cumulative Learning Agent Reward: {results['learning_agent_cumulative_reward']}")
print(f"Cumulative Extortion Agent Reward: {results['extortion_agent_cumulative_reward']}")
from tools.competition import plot_price_history
plot_price_history(results)


# Plot the reward function of the learning agent against the extortion agent's price
import matplotlib.pyplot as plt

prices_01 = np.linspace(0, 1, 100)
extortion_prices_01 = np.array([extortion_agent.mapping_function(p) for p in prices_01])

prices = 1.0 + prices_01 * 1.4
extortion_prices = 1.0 + extortion_prices_01 * 1.4
all_prices = np.vstack((extortion_prices, prices)).T
rewards = reward_fn(all_prices)

plt.plot(prices, rewards[:,1], label="Learning Agent Reward")
plt.plot(prices, rewards[:,0], label="Extortion Agent Reward")
plt.xlabel("Learning Agent Price")
plt.ylabel("Reward")
plt.title("Reward vs Learning Agent Price")
plt.grid()
plt.legend()
plt.show()
