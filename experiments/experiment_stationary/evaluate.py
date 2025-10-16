"""Run single experiment for learning mapping against eps_greedy bandit."""

from tools.config import load_config
import numpy as np
import os
from tools.competition import MultiScoreFn
from tools.competition_plot import run_competition_and_plot
from tools.environment import EnvironmentParameters, RewardFunction
from pprint import pprint

from plot_stationary import plot_strategy


def main(training_config, evaluation_config, directory):
    seed = evaluation_config["optimization"].get("seed", None)
    rng = np.random.default_rng(seed)
    print("Random seed:", seed)
    env_config = evaluation_config["environment"]
    price_bounds = evaluation_config["price_bounds"]
    extortion_agent_config = training_config["extortion_agent"]
    optimization_cases = evaluation_config["optimization_cases"]
    optimization_config = evaluation_config["optimization"]
    for opt_case in optimization_cases:
        opt_case["agent"]["params"]["action_bounds"] = price_bounds
    extortion_agent_config["params"]["action_bounds"] = price_bounds

    weights = optimization_config.get("weights", None)

    optimization_config["weights"] = weights

    # Create the objective function with reproducible RNG
    objective_function = MultiScoreFn(
        optimization_cases=optimization_cases,
        env_config=env_config,
        extortion_config=extortion_agent_config,
        rng=rng,
        weights=weights)
    
    # Load parameters from training
    optimised_params = np.loadtxt(directory + "/optimised_params.txt").tolist()

    score, result = objective_function.evaluate_params(optimised_params, return_rewards=True)

    print("Detailed results with best parameters:")
    pprint(result, indent=4)
    print(f"Total score: {score}")

    extortion_agent_config["params"]["map_params"] = optimised_params

    env_params = EnvironmentParameters(**env_config)
    reward_fn = RewardFunction(env_params)

    print(env_config)

    # Run competitions for every learning agent and plot results
    image_dir = directory + "/images"
    os.makedirs(image_dir, exist_ok=True)
    for i, opt_case in enumerate(optimization_cases):
        suffix = f"_{opt_case["name"]}"
        save_dir = os.path.join(image_dir, opt_case["name"])
        os.makedirs(save_dir, exist_ok=True)
        if opt_case["name"] == "self_play":
            agent_config = extortion_agent_config
        else:
            agent_config = opt_case["agent"]
        learning_agent = run_competition_and_plot(
            reward_fn=reward_fn,
            learning_agent_config=agent_config,
            extortion_agent_config=extortion_agent_config,
            competition_config=opt_case,
            save_dir=save_dir,
            suffix=suffix,
            nash_price=env_config.get("nash_price", None),
            nash_reward=env_config.get("nash_reward", None),
            pareto_reward=env_config.get("pareto_reward", None),
        )

    # Plot the q_values
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(learning_agent.action_space, learning_agent.q_values, alpha=0.7)
    # plt.xlabel('Actions')
    # plt.ylabel('Q-Values')
    # plt.title('Q-Values for Each Action')
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()

    plot_strategy(
        params=optimised_params,
        reward_fn=reward_fn,
        price_bounds=price_bounds,
        save_path=image_dir,
        nash_price=env_config.get("nash_price", None),
        nash_reward=env_config.get("nash_reward", None)
    )



if __name__ == "__main__":
    
    directory = "eps_greedy"
    evaluation_config_path = "evaluation_config.yaml"
    evaluation_config = load_config(evaluation_config_path)

    training_config_path = directory + "/config.yaml"
    training_config = load_config(training_config_path)

    main(training_config, evaluation_config, directory=directory)