"""Evaluate all strategies and return a Latex table of results."""

from tools.config import load_config
import json


name_dict = {
    "eps_greedy": "$\\varepsilon$-greedy",
    "sw_uts": "SW-UTS",
    "best_response": "Best-response",
    "best_response_penalised": "Best-response (penalised)",
    "self_play": "Self-play",
    "multi_brp+sp": "BR+SP",
    "multi_brp+swuts": "BR+SWUTS",
    "multi_all": "ALL",
    "multi_all_weighted": "ALL (weighted)",
    "tit_for_tat_up": "Tit-for-Tat"
}


def get_mean_rewards(training_cases, evaluation_cases):
    
    training_results = {}
    for training_case in training_cases:
        directory = training_case

        result = json.load(open(directory + "/evaluation_results.json"))
        training_results[training_case] = result
        for evaluation_case in evaluation_cases:
            case_result = result[f"case_{evaluation_case}"]
            training_results[training_case][f"case_{evaluation_case}"] = case_result
    return training_results


def create_latex_table(evaluation_results, training_cases, evaluation_cases):

    print("LaTeX Table:")
    print("\\begin{tabular}{l|" + "c" * len(evaluation_cases) + "}")
    header = " & " + " & ".join([f"{name_dict[case]}" for case in evaluation_cases]) + " \\\\ \\hline"
    print(header)
    print("\\hline")
    for training_case in training_cases:
        row = f"{name_dict[training_case]} & " + " & ".join([f"{evaluation_results[training_case][f'case_{evaluation_case}']['extortion_reward']:.3f} - {evaluation_results[training_case][f'case_{evaluation_case}']['opponent_reward']:.3f}" for evaluation_case in evaluation_cases]) + " \\\\"
        print(row)
    print("\\end{tabular}")
    



if __name__ == "__main__":
    evaluation_config_path = "evaluation_config.yaml"
    evaluation_config = load_config(evaluation_config_path)

    training_cases = [
        "eps_greedy",
        "sw_uts",
        "best_response",
        "best_response_penalised",
        "self_play",
        "multi_brp+sp",
        "multi_brp+swuts",
        "multi_all",
        # "multi_all_weighted",
        "tit_for_tat_up"
        ]
    
    evaluation_cases = [
        "eps_greedy",
        "sw_uts",
        "best_response",
        "self_play",
    ]


    evaluation_results = get_mean_rewards(training_cases, evaluation_cases)
    create_latex_table(evaluation_results, training_cases, evaluation_cases)