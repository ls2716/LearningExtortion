"""Run single experiment for learning mapping against eps_greedy bandit."""

from tools.config import load_config
from tools.optimise import run_cma
import numpy as np
from tools.competition import MultiScoreFn
import json
from pprint import pprint

def main(config, directory="./"):
    seed = config["optimization"].get("seed", None)
    rng = np.random.default_rng(seed)
    print("Random seed:", seed)
    env_config = config["environment"]
    price_bounds = config["price_bounds"]
    extortion_agent_config = config["extortion_agent"]
    optimization_cases = config["optimization_cases"]
    optimization_config = config["optimization"]
    for opt_case in optimization_cases:
        opt_case["agent"]["params"]["action_bounds"] = price_bounds
    extortion_agent_config["params"]["action_bounds"] = price_bounds

    weights = optimization_config.get("weights", None)

    optimization_config["weights"] = weights

    # Create the objective function with reproducible RNG
    objective_function = MultiScoreFn(
        env_config=env_config,
        optimization_cases=optimization_cases,
        extortion_config=extortion_agent_config,
        rng=rng,
        weights=weights)
    
    # Trial evaluation
    trial_params = [0.1] * extortion_agent_config["params"]["n_params"]
    trial_score, trial_result = objective_function.evaluate_params(trial_params, return_rewards=True)
    print("Trial score (negative cumulative reward):", trial_score)
    
    # Run the CMA-ES optimisation
    best_params, best_score = run_cma(
        objective=objective_function,
        extortion_cfg=extortion_agent_config,
        cma_cfg=optimization_config,
        seed=optimization_config.get("seed", None),
    )

    print("Best Parameters:", best_params)
    print("Best Score (negative cumulative reward):", best_score)

    score, result = objective_function.evaluate_params(best_params, return_rewards=True)

    print("Detailed results with best parameters:")
    pprint(result, indent=4)

    # Save the optimisation results to a json file
    result_dict = {
        "results": {
            "best_params": best_params.tolist(),
            "best_score": best_score,
            "final_score": score,
            "detailed_results": result,
        }
    }
    with open(directory + "/results.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    # Save params to txt file
    np.savetxt(directory + "/optimised_params.txt", np.array(best_params))
    




if __name__ == "__main__":
    
    directory = "eps_greedy_d1.00"
    config_path = directory + "/config.yaml"
    config = load_config(config_path)

    main(config, directory=directory)