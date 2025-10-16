"""Implement a logistic environment including environment parameter object.

The environment simulates a logit demand environment for agents to interact with.
The parameters for the environment are encapsulated in the EnvironmentParameters class.
"""

from pydantic import BaseModel
import numpy as np
from typing import List


class EnvironmentParameters(BaseModel):
    """Parameters for the environment."""

    A: List[float]
    C: List[float]
    mu_0: float
    a_0: float
    N: int
    sigma: float

    def __init__(self, **data):
        """Initialize the environment parameters.

        Args:
            A (List[float]): List of demand intercepts for each agent.
            C (List[float]): List of marginal costs for each agent.
            mu_0 (float): Scale parameter for the logit model.
            a_0 (float): Outside option utility.
        """
        super().__init__(**data)
        assert len(self.A) == self.N, "Length of A must be equal to N"
        self.A = np.array(self.A).reshape(-1, self.N)
        self.C = np.array(self.C).reshape(-1, self.N)
        self.sigma = float(self.sigma)
        self.a_0 = float(self.a_0)


class RewardFunction:
    """
    Reward function for the environment.

    Attributes:
        env_params (EnvironmentParameters): Parameters of the environment.
    """

    def __init__(self, env_params: "EnvironmentParameters"):
        """
        Initialize the reward function with environment parameters.

        Args:
            env_params (EnvironmentParameters): Parameters of the environment.
        """
        self.env_params = env_params

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate the rewards for each agent given their prices.

        Args:
            prices (np.ndarray): Array of prices set by each agent.

        Returns:
            np.ndarray: Array of rewards for each agent.
        """
        prices = prices.reshape(-1, self.env_params.N)

        exp_utility = np.exp((self.env_params.A - prices) / self.env_params.mu_0)
        exp_outside = np.exp(self.env_params.a_0 / self.env_params.mu_0)
        sum_exp_utility = np.sum(exp_utility, axis=1, keepdims=True) + exp_outside
        demand = exp_utility / sum_exp_utility
        rewards = demand * (prices - self.env_params.C)
        return rewards


# Example usage:
if __name__ == "__main__":
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[1.0, 1.0], mu_0=0.25, a_0=-1.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([1.5, 1.5])
    rewards = reward_fn(prices)
    print("Rewards:", rewards)
