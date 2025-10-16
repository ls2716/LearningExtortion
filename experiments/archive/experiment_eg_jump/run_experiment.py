"""Run single experiment for learning mapping against eps_greedy bandit."""

from tools.config import load_config
from tools.optimise import run_cma, ScoreFn
import pickle
from tools.visualize import run_competition_and_plot, plot_map
import numpy as np

def main(config):
    seed = config["optimization"].get("seed", None)
    rng = np.random.default_rng(seed)
    print("Random seed:", seed)
    # Separate infor agent, environment, competition and optimisation parameters
    agent_config = config["learning_agent"]
    env_config = config["environment"]
    competition_config = config["competition"]
    optimization_config = config["optimization"]
    extortion_agent_config = config["extortion_agent"]

    # Create the objective function with reproducible RNG
    objective_function = ScoreFn(
        agent_config=agent_config,
        env_config=env_config,
        competition_config=competition_config,
        extortion_config=extortion_agent_config,
        rng=rng)
    
    params = [0.5, 0.5]
    print("Initial score (negative cumulative reward):", objective_function(params))

    with open("data.pkl", "wb") as f:
        pickle.dump(objective_function, f)

    best_params, best_score = run_cma(
        objective=objective_function,
        extortion_cfg = extortion_agent_config,
        cma_cfg=optimization_config,
        seed=optimization_config.get("seed", None),
    )

    print("Best Parameters:", best_params)
    print("Best Score (negative cumulative reward):", best_score)

    # Plot the mapping function
    plot_map(best_params, extortion_agent_config=extortion_agent_config, save_dir="./")

    # Run a competition and plot results
    run_competition_and_plot(
        params=best_params,
        learning_agent_config=agent_config,
        reward_fn=objective_function.reward_fn,
        competition_config=competition_config,
        extortion_agent_config=extortion_agent_config,
        save_dir="./",
    )




if __name__ == "__main__":
    config = load_config("config.yaml")
    main(config)