"""Implement baseline eps-greedy learning agent."""

import numpy as np
from typing import Tuple
from tools.agents.base import LearningAgentBase


class EpsGreedyBandit(LearningAgentBase):
    """Epsilon-greedy multi-armed bandit agent with reproducible RNG."""

    def __init__(
        self,
        action_bounds: Tuple[float, float],
        n_actions: int,
        epsilon: float = 0.2,
        decay: float = 1.0,
        rng: "np.random.Generator" = None,
        verbose: bool = False,
    ):
        """Initialize the agent.

        Args:
            n_actions (int): Number of possible actions.
            epsilon (float): Probability of choosing a random action.
            decay (float): Epsilon decay rate.
            rng (np.random.Generator, optional): Random number generator for reproducibility.
        """
        super().__init__()
        self.n_actions = n_actions
        self.starting_epsilon = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.action_space = np.linspace(action_bounds[0], action_bounds[1], n_actions)
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.verbose = verbose
        self.action_bounds = action_bounds

    def _get_action_index(self, action: float) -> int:
        """Get the index of the closest action to the given action.

        Uses the fact that actions are evenly spaced.

        Args:
            action (float): action in action bounds.

        Returns:
            int: Index of the closest action.
        """
        index = int(
            round(
                (action - self.action_space[0])
                / (self.action_space[-1] - self.action_space[0])
                * (self.n_actions - 1)
            )
        )
        index = max(0, min(self.n_actions - 1, index))  # Clamp to valid range
        return index

    def get_action(self, *args, **kwargs) -> float:
        """Return action and its index."""
        if self.rng.random() < self.epsilon:
            action_index = self.rng.integers(self.n_actions)
        else:
            # Exploit: choose the action with the highest estimated value if random tie breaking
            max_q = np.max(self.q_values)
            candidates = np.where(self.q_values == max_q)[0]
            action_index = self.rng.choice(candidates)
        action = self.action_space[action_index]
        return action

    def reset(self):
        """Reset the agent's state."""
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.epsilon = self.starting_epsilon

    def set_initial_action(self, action: float):
        """Set the initial action of the agent.

        Args:
            action (float): Initial action to set.
        """
        pass  # No specific action needed for epsilon-greedy agent

    def update(self, agent_action, opponent_action, agent_reward, opponent_reward):
        """Update the agent's state based on the received reward.

        Args:
            action_index (int): Index of the action taken.
            reward (float): Reward received after taking the action.
        """
        action_index = self._get_action_index(agent_action)
        self.action_counts[action_index] += 1
        n = self.action_counts[action_index]
        q = self.q_values[action_index]
        # Incremental update to avoid storing all rewards
        self.q_values[action_index] += (agent_reward - q) / n
        self.epsilon *= self.decay  # Optional: decay epsilon over time

