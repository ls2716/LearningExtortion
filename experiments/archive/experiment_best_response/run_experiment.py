"""Run single experiment for learning mapping against eps_greedy bandit."""

from tools.config import load_config
from tools.optimise import run_cma
import numpy as np
from tools.competition import NaiveScoreFn
from tools.competition_plot import run_competition_and_plot
from tools.visualize import plot_map


def main(config, directory="./"):
    seed = config["optimization"].get("seed", None)
    rng = np.random.default_rng(seed)
    print("Random seed:", seed)
    # Separate infor agent, environment, competition and optimisation parameters
    agent_config = config["learning_agent"]
    env_config = config["environment"]
    competition_config = config["competition"]
    optimization_config = config["optimization"]
    extortion_agent_config = config["extortion_agent"]
    agent_config["params"]["action_bounds"] = competition_config["price_bounds"]
    extortion_agent_config["params"]["action_bounds"] = competition_config["price_bounds"]

    # Best Response Mapping
    # Load the best response mapping from file
    best_responses = np.loadtxt(directory + "/best_response_mapping.txt")
    agent_config["params"]["params"] = best_responses.tolist()

    # Create the objective function with reproducible RNG
    objective_function = NaiveScoreFn(
        agent_config=agent_config,
        env_config=env_config,
        competition_config=competition_config,
        extortion_config=extortion_agent_config,
        rng=rng)
    
    params = [0.1] * 12 # Initial parameters for the mapping function
    print("Initial score (negative cumulative reward):", objective_function(params))
    

    best_params, best_score = run_cma(
        objective=objective_function,
        extortion_cfg=extortion_agent_config,
        cma_cfg=optimization_config,
        seed=optimization_config.get("seed", None),
    )

    print("Best Parameters:", best_params)
    print("Best Score (negative cumulative reward):", best_score)

    # Print best cumulative score to a text file
    with open(directory + "/best_score.txt", "w") as f:
        f.write(f"Best Score (negative cumulative reward): {best_score}\n")
        f.write("Best Parameters:\n")
        f.write(", ".join([f"{p:.6f}" for p in best_params]) + "\n")

    # Plot the mapping function
    plot_map(best_params, extortion_agent_config=extortion_agent_config, save_dir=directory)

    # Run a competition and plot results
    run_competition_and_plot(
        params=best_params,
        learning_agent_config=agent_config,
        reward_fn=objective_function.reward_fn,
        competition_config=competition_config,
        extortion_agent_config=extortion_agent_config,
        save_dir=directory,
    )




if __name__ == "__main__":
    
    directory = "monotone_equispaced"
    config_path = directory + "/config.yaml"
    config = load_config(config_path)
    main(config, directory=directory)