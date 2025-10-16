"""Implementation of agents for pricing in a competition."""

import numpy as np

from tools.agents.base import LearningAgentBase
from tools.agents.extortion_maps import mapping_dict

class ExtortionAgent(LearningAgentBase):
    def __init__(self, action_bounds, mapping_type, map_params, rng=None):
        """Initialize the extortion agent.

        Args:
            map_type (str): Type of mapping function ('monotone' or 'jump').
            params: Parameters for the mapping function.
            kwargs: Additional keyword arguments for mapping function.
        """
        self.mapping_function, self.mapping_function_info = mapping_dict[mapping_type](map_params, rng=rng)
        self.action_bounds = action_bounds

    def reset(self):
        """Reset the agent's state."""
        pass

    def get_action(self, opponent_action: float) -> float:
        """Get the extortion agent's action based on the learning agent's action.

        Args:
            learning_agent_action_01 (float): Learning agent's action in [0, 1].

        Returns:
            float: Extortion agent's action in [0, 1].
        """
        # Map the action from [action_bounds[0], action_bounds[1]] to [0, 1]
        learning_agent_action_01 = (opponent_action - self.action_bounds[0]) / (self.action_bounds[1] - self.action_bounds[0])
        if self.mapping_function is None:
            raise ValueError(
                "Mapping function not set. Call set_mapping(params) first."
            )
        # Get the extortion agent's action in [0, 1]
        # and map it back to [action_bounds[0], action_bounds[1]]
        extortion_action_01 = self.mapping_function(learning_agent_action_01)
        return self.action_bounds[0] + extortion_action_01 * (self.action_bounds[1] - self.action_bounds[0])


def get_extortion_agent(extortion_config, rng=None):
    """Get an extortion agent based on the configuration.

    Args:
        mapping_type (str): Type of mapping function ('monotone_equispaced', 'equispaced', 'monotone_knot', 'stationary').
        params: Parameters for the mapping function.
        action_bounds (tuple): Bounds for the actions.
        rng (np.random.Generator, optional): Random number generator for reproducibility.

    Returns:
        ExtortionAgent: An instance of ExtortionAgent.
    """
    mapping_type = extortion_config["map_type"]
    action_bounds = extortion_config["action_bounds"]
    if "map_params_path" in extortion_config:
        map_params_path = extortion_config["map_params_path"]
        map_params = np.loadtxt(map_params_path).tolist()
    else:
        map_params = extortion_config["map_params"]
    if mapping_type not in mapping_dict:
        raise ValueError(
            f"Unknown mapping type: {mapping_type}. Supported types: {list(mapping_dict.keys())}"
        )
    return ExtortionAgent(
        action_bounds=action_bounds, mapping_type=mapping_type, map_params=map_params, rng=rng
    )


if __name__ == "__main__":
    agent = get_extortion_agent("monotone_equispaced", (0, 1), [0.1] * 12, rng=np.random.default_rng(0))
    for action in [0.0, 0.1, 0.5, 0.9, 1.0]:
        extortion_action = agent.get_action(action)
        print(
            f"Learning agent action: {action:.2f}, Extortion agent action: {extortion_action:.2f}"
        )

    # Plot the points
    import matplotlib.pyplot as plt

    x_pos = agent.mapping_function_info["x_pos"]
    y_pos = agent.mapping_function_info["y_pos"]
    plt.plot(x_pos, y_pos, marker="o")
    plt.title("Mapping Function Knot Points")
    plt.xlabel("Learning Agent Action")
    plt.ylabel("Extortion Agent Action")
    plt.grid()
    plt.show()
