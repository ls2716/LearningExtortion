"""Implement base class for learning agent."""


from typing import Tuple


class LearningAgentBase:
    """Base class for learning agents."""

    def __init__(self):
        """Initialize the agent."""
        pass

    def reset(self):
        """Reset the agent's state."""
        pass

    def get_action(self) -> Tuple[float, int]:
        """Get the agent's action.

        Returns:
            Tuple[float, int]: A tuple of (action in [0,1], action index).
        """
        pass

    def set_initial_action(self, initial_action: float):
        """Set the initial action of the agent.

        Args:
            initial_action (float): Initial action to set.
        """
        pass

    def update(self, agent_action, opponent_action, agent_reward, opponent_reward):
        """Update the agent's state based on the received reward.

        Args:
            reward (float): Reward received after taking the action.
            action_index (int): Index of the action taken.
        """
        pass

    @classmethod
    def from_config(cls, **kwargs):
        """Create an agent instance from configuration parameters.

        Args:
            **kwargs: Configuration parameters for the agent.
        Returns:
            LearningAgentBase: An instance of a learning agent.
        """
        return cls(**kwargs)



def get_agent(agent_config, rng=None):
    """Get the learning agent based on the configuration.

    Args:
        agent_config (dict): Configuration dictionary with keys 'name' and 'params'.
        rng (np.random.Generator, optional): Random number generator for reproducibility.

    Returns:
        LearningAgentBase: An instance of a learning agent.
    """
    if agent_config["type"] == "EpsGreedyBandit":
        from tools.agents.eps_greedy import EpsGreedyBandit
        
        params = dict(agent_config["params"])
        if rng is not None:
            params["rng"] = rng
        return EpsGreedyBandit(**params)
    elif agent_config["type"] == "SW_UTS":
        from tools.agents.sw_uts import SW_UTS
        params = dict(agent_config["params"])
        if rng is not None:
            params["rng"] = rng
        return SW_UTS(**params)
    elif agent_config["type"] == "Logistic2D":
        from tools.agents.logistic_2d import Logistic2D
        params = dict(agent_config["params"])
        if rng is not None:
            params["rng"] = rng
        return Logistic2D(**params)
    elif agent_config["type"] == "ExtortionAgent":
        params = dict(agent_config["params"])
        from tools.agents.extortion_agent import get_extortion_agent
        return get_extortion_agent(extortion_config=params, rng=rng)
    else:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")