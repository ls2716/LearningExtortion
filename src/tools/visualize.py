import matplotlib.pyplot as plt
from tools.agents.extortion_agent import mapping_dict

import os


def plot_map(params, extortion_agent_config, save_dir=None):
    """
    Plot the monotone mapping function. If save_dir is given, saves as map_plot.png in that directory.
    """
    mapping_type = extortion_agent_config["params"]["map_type"]
    make_map = mapping_dict[mapping_type]
    extortion_mapping, mapping_info = make_map(params)
    # x_vals = np.linspace(0, 1, 100)
    # y_vals = [extortion_mapping(x) for x in x_vals]
    x_pos = mapping_info["x_pos"]
    y_pos = mapping_info["y_pos"]
    plt.figure()
    plt.plot(x_pos, y_pos)
    plt.ylim(-0.1, 1.1)
    plt.title("Monotone Increasing Piecewise-Linear Mapping Function")
    plt.xlabel("Input (Learning Agent Action)")
    plt.ylabel("Output (Extortion Agent Action)")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        map_plot_path = os.path.join(save_dir, "map_plot.png")
        plt.savefig(map_plot_path)
        plt.close()
    else:
        plt.show()
