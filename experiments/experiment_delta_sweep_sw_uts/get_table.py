"""Evaluate all strategies with all delays and return a Latex table of results."""

from tools.config import load_config
import json



def get_mean_rewards(delays, evaluation_config):
    
    delay_results = {}
    for training_delay in delays:
        directory = f"sw_uts_d{training_delay:.2f}"

        result = json.load(open(directory + "/evaluation_results.json"))
        delay_results[training_delay] = {}
        for evaluation_delay in delays:
            case_result = result[f"case_sw_uts_delay_{evaluation_delay:.2f}"]
            delay_results[training_delay][evaluation_delay] = case_result
    return delay_results


def create_latex_table(delay_results):
    delays = list(delay_results.keys())

    print("LaTeX Table:")
    print("\\begin{tabular}{l|" + "c" * len(delays) + "}")
    header = " & " + " & ".join([f"{delay:.2f}" for delay in delays]) + " \\\\ \\hline"
    print(header)
    print("\\hline")
    for train_delay in delays:
        row = f"{train_delay:.2f} & " + " & ".join([f"{delay_results[train_delay][evaluation_delay]['extortion_reward']:.3f} - {delay_results[train_delay][evaluation_delay]['opponent_reward']:.3f}" for evaluation_delay in delays]) + " \\\\"
        print(row)
    print("\\end{tabular}")
    


if __name__ == "__main__":
    evaluation_config_path = "evaluation_config.yaml"
    evaluation_config = load_config(evaluation_config_path)

    delays = [0, 0.25, 0.5, 0.75, 1.0]
    delay_results = get_mean_rewards(delays, evaluation_config)
    create_latex_table(delay_results)