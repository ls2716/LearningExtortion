import numpy as np
from typing import Tuple

class SW_UTS:
    def __init__(
        self,
        action_bounds: Tuple[float, float],
        n_actions=11,
        n_neighbors=2,
        starting_action=0,
        sliding_window=100,
        std=0.1,
        rng: "np.random.Generator" = None,
        verbose: bool = False,
    ):
        self.n_actions = int(n_actions)
        self.n_neighbors = int(n_neighbors)
        self.sliding_window = int(sliding_window)
        self.variance = std**2

        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = (np.zeros((self.sliding_window)) - 1).astype(
            int
        )  # Initialize with -1 to indicate no action was chosen

        self.average_rewards = np.zeros(self.n_actions)
        self.counts = np.zeros(
            self.n_actions
        )  # Count of how many times each action was chosen
        self.action_space = np.linspace(action_bounds[0], action_bounds[1], n_actions)
        self.action_bounds = action_bounds

        self._buffer_idx = 0  # Circular buffer index
        self.set_initial_action(starting_action)
        self.incumbent = self.starting_action_index
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_action(self, *args, **kwargs) -> float:
        # Get the neighbors of the incumbent action
        left = max(0, self.incumbent - self.n_neighbors)
        right = min(self.n_actions, self.incumbent + self.n_neighbors + 1)
        variances = self.variance / (self.counts[left:right] + 1e-5)
        values_neighbors = self.rng.normal(
            self.average_rewards[left:right], np.sqrt(variances)
        ) + self.rng.normal(0, 0.0001, size=(right - left))

        action_index = (
            np.argmax(values_neighbors) + left
        )  # Adjust index to the full action space
        action = self.action_space[action_index]

        return action

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
                (action - self.action_bounds[0])
                / (self.action_bounds[1] - self.action_bounds[0])
                * (self.n_actions - 1)
            )
        )
        index = max(0, min(self.n_actions - 1, index))  # Clamp to valid range
        return index

    def update(self, agent_action, opponent_action, agent_reward, opponent_reward):
        # Get the index of the action taken
        action_index = self._get_action_index(agent_action)
        reward = agent_reward
        # Get the index of the action that was chosen that goes out of the sliding window
        outgoing_action_index = self.was_chosen[self._buffer_idx]
        outgoing_reward = self.rewards[self._buffer_idx]

        # Update the average reward for the action that goes out of the sliding window
        if outgoing_action_index != -1:  # If an action was chosen
            count = self.counts[outgoing_action_index]
            if count > 1:
                self.average_rewards[outgoing_action_index] = (
                    self.average_rewards[outgoing_action_index] * count
                    - outgoing_reward
                ) / (count - 1)
            else:
                self.average_rewards[outgoing_action_index] = 0
            # Decrease the count for the outgoing action
            self.counts[outgoing_action_index] -= 1

        # Update the rewards and was_chosen matrices
        self.rewards[self._buffer_idx] = agent_reward
        self.was_chosen[self._buffer_idx] = action_index
        self.average_rewards[action_index] = (
            self.average_rewards[action_index] * self.counts[action_index] + reward
        ) / (self.counts[action_index] + 1)
        self.counts[action_index] += 1

        # Update the circular buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self.sliding_window

        self.incumbent = np.argmax(self.average_rewards)

    def reset(self):
        """Reset the agent's state."""
        self.average_rewards = np.zeros(self.n_actions)
        self.counts = np.zeros(self.n_actions)  # Reset counts
        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = np.zeros((self.sliding_window)).astype(int) - 1  #
        self._buffer_idx = 0
        self.incumbent = self.starting_action_index

    def set_initial_action(self, action:float):
        """Set the starting action for the agent."""
        action_index = self._get_action_index(action)
        self.starting_action_index = action_index
        self.incumbent = action_index
