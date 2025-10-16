"""Agent competitions in a logit demand environment."""

from typing import Callable
import numpy as np

from tools.environment import EnvironmentParameters, RewardFunction
from tools.agents.base import get_agent


def run_competition(
    reward_function: Callable,
    extortion_agent,
    learning_agent,
    T: int,
    extortion_agent_initial_price: float,
    learning_agent_initial_price: float,
    delay=0,
    return_history=False,
    skip_steps=0,
    verbose=False,
):
    """Run a competition among agents in the given environment."""
    extortion_agent.reset()
    learning_agent.reset()
    learning_agent.set_initial_action(learning_agent_initial_price)
    last_extortion_price = extortion_agent_initial_price

    learning_agent_prices = []
    extortion_agent_prices = []
    learning_agent_rewards = []
    extortion_agent_rewards = []

    learning_agent_cumulative_reward = 0
    extortion_agent_cumulative_reward = 0

    for t in range(T):
        learning_agent_price = learning_agent.get_action(
            opponent_action=last_extortion_price
        )
        extortion_agent_price = extortion_agent.get_action(
            opponent_action=learning_agent_price
        )

        rewards_pre = reward_function(
            np.array([last_extortion_price, learning_agent_price])
        )[0]
        rewards_post = reward_function(
            np.array([extortion_agent_price, learning_agent_price])
        )[0]
        total_rewards = delay * rewards_pre + (1 - delay) * rewards_post

        learning_agent.update(
            agent_action=learning_agent_price,
            opponent_action=extortion_agent_price,
            agent_reward=total_rewards[1],
            opponent_reward=total_rewards[0],
        )
        if verbose:
            print(
                f"Step {t}: Learning Agent Price: {learning_agent_price:.3f}, "
                f"Extortion Agent Price: {extortion_agent_price:.3f}, "
                f"Learning Agent Reward: {total_rewards[1]:.3f}, "
                f"Extortion Agent Reward: {total_rewards[0]:.3f}"
            )
            # print(f"Learning Agent Q-values: {learning_agent.q_values}")

        last_extortion_price = extortion_agent_price

        if t >= skip_steps:
            extortion_agent_cumulative_reward += total_rewards[0]
            learning_agent_cumulative_reward += total_rewards[1]

        if return_history:
            learning_agent_prices.append(learning_agent_price)
            extortion_agent_prices.append(extortion_agent_price)
            learning_agent_rewards.append(total_rewards[1])
            extortion_agent_rewards.append(total_rewards[0])

    n_considered_steps = T - skip_steps
    learning_agent_cumulative_reward /= n_considered_steps
    extortion_agent_cumulative_reward /= n_considered_steps

    return {
        "learning_agent_prices": learning_agent_prices,
        "extortion_agent_prices": extortion_agent_prices,
        "learning_agent_rewards": learning_agent_rewards,
        "extortion_agent_rewards": extortion_agent_rewards,
        "delay": delay,
        "extortion_agent_initial_price": extortion_agent_initial_price,
        "learning_agent_cumulative_reward": learning_agent_cumulative_reward,
        "extortion_agent_cumulative_reward": extortion_agent_cumulative_reward,
    }


def run_multiple_competitions(
    n_competitions: int,
    reward_function: Callable,
    extortion_agent,
    learning_agent,
    T,
    extortion_agent_initial_price: float,
    learning_agent_initial_price: float,
    delay,
    skip_steps,
):
    """Run multiple competitions among agents in the given environment."""
    learning_agent_cumulative_rewards = []
    extortion_agent_cumulative_rewards = []
    for _ in range(n_competitions):
        result = run_competition(
            reward_function,
            extortion_agent,
            learning_agent,
            T,
            extortion_agent_initial_price,
            learning_agent_initial_price,
            delay,
            return_history=False,
            skip_steps=skip_steps,
        )
        learning_agent_cumulative_rewards.append(
            result["learning_agent_cumulative_reward"]
        )
        extortion_agent_cumulative_rewards.append(
            result["extortion_agent_cumulative_reward"]
        )
    return {
        "learning_agent_cumulative_rewards": learning_agent_cumulative_rewards,
        "extortion_agent_cumulative_rewards": extortion_agent_cumulative_rewards,
    }


class NaiveScoreFn:
    """Callable scoring function for agent parameter optimization."""

    def __init__(
        self, agent_config, env_config, competition_config, extortion_config, rng=None
    ):
        self.agent_config = agent_config
        self.extortion_config = extortion_config
        self.competition_config = competition_config
        self.rng = rng

        env_params = EnvironmentParameters(**env_config)
        self.reward_fn = RewardFunction(env_params)

    def __call__(self, params):
        """Evaluate the negative mean cumulative reward for the extortion agent."""
        learning_agent = get_agent(agent_config=self.agent_config, rng=self.rng)
        extortion_config = self.extortion_config.copy()
        extortion_config["params"]["params"] = params
        extortion_agent = get_agent(agent_config=extortion_config, rng=self.rng)

        results = run_multiple_competitions(
            n_competitions=self.competition_config["n_competitions"],
            reward_function=self.reward_fn,
            extortion_agent=extortion_agent,
            learning_agent=learning_agent,
            T=self.competition_config["T"],
            extortion_agent_initial_price=self.competition_config[
                "extortion_agent_initial_price"
            ],
            learning_agent_initial_price=self.competition_config[
                "learning_agent_initial_price"
            ],
            delay=self.competition_config["delay"],
            skip_steps=self.competition_config["skip_steps"],
        )
        mean_extortion_reward = np.mean(results["extortion_agent_cumulative_rewards"])
        mean_opponent_reward = np.mean(results["learning_agent_cumulative_rewards"])
        # Compute the penalty if opponent's reward is higher
        penalty = 1000 * max(0, mean_opponent_reward - mean_extortion_reward)
        print(
            f"Mean extortion reward: {mean_extortion_reward}, Mean opponent reward: {mean_opponent_reward}, Penalty: {penalty}"
        )
        return -float(mean_extortion_reward - penalty)


class SelfPlayScoreFn:
    """Callable scoring function for agent parameter optimization."""

    def __init__(
        self, agent_config, env_config, competition_config, extortion_config, rng=None
    ):
        self.agent_config = agent_config
        self.extortion_config = extortion_config
        self.competition_config = competition_config
        self.rng = rng

        env_params = EnvironmentParameters(**env_config)
        self.reward_fn = RewardFunction(env_params)

    def __call__(self, params):
        """Evaluate the negative mean cumulative reward for the extortion agent."""
        # learning_agent = get_agent(agent_config=self.agent_config, rng=self.rng)
        extortion_config = self.extortion_config.copy()
        extortion_config["params"]["params"] = params
        extortion_agent = get_agent(agent_config=extortion_config, rng=self.rng)
        learning_agent = get_agent(agent_config=extortion_config, rng=self.rng)

        results = run_multiple_competitions(
            n_competitions=self.competition_config["n_competitions"],
            reward_function=self.reward_fn,
            extortion_agent=extortion_agent,
            learning_agent=learning_agent,
            T=self.competition_config["T"],
            extortion_agent_initial_price=self.competition_config[
                "extortion_agent_initial_price"
            ],
            learning_agent_initial_price=self.competition_config[
                "learning_agent_initial_price"
            ],
            delay=self.competition_config["delay"],
            skip_steps=self.competition_config["skip_steps"],
        )
        mean_extortion_reward = np.mean(results["extortion_agent_cumulative_rewards"])
        return -float(mean_extortion_reward)


class MultiScoreFn:
    """Callable scoring function for agent parameter optimization."""

    def __init__(
        self,
        env_config,
        optimization_cases,
        extortion_config,
        rng=None,
        weights=None
    ):
        self.extortion_config = extortion_config
        self.optimization_cases = optimization_cases
        self.rng = rng
        self.weights = weights

        env_params = EnvironmentParameters(**env_config)
        self.reward_fn = RewardFunction(env_params)

        if weights is None:
            self.weights = [1.0] * len(optimization_cases)

        if len(optimization_cases) != len(self.weights):
            raise ValueError(
                "When selfplay is disabled, weights must be provided for all agents excluding self-play."
            )

    def __call__(self, map_params):
        """Evaluate the negative mean cumulative reward for the extortion agent."""
        return self.evaluate_params(map_params)

    def evaluate_params(self, map_params, return_rewards=False):
        """Evaluate the negative mean cumulative reward for the extortion agent."""
        extortion_config = self.extortion_config.copy()
        extortion_config["params"]["map_params"] = map_params

        scores = []
        detailed_results = {}

        for i, opt_case in enumerate(self.optimization_cases):
            penalty_coeff = opt_case.get("penalty_coeff", 0)
            penalty_slack = opt_case.get("penalty_slack", 0)

            if opt_case["name"] == "self_play":
                agent_config = extortion_config
            else:
                agent_config = opt_case["agent"]
            extortion_agent = get_agent(agent_config=extortion_config, rng=self.rng)
            learning_agent = get_agent(agent_config=agent_config, rng=self.rng)
            results = run_multiple_competitions(
                n_competitions=opt_case["n_competitions"],
                reward_function=self.reward_fn,
                extortion_agent=extortion_agent,
                learning_agent=learning_agent,
                T=opt_case["T"],
                extortion_agent_initial_price=opt_case["extortion_agent_initial_price"],
                learning_agent_initial_price=opt_case["learning_agent_initial_price"],
                delay=opt_case["delay"],
                skip_steps=opt_case["skip_steps"],
            )
            mean_extortion_reward = np.mean(results["extortion_agent_cumulative_rewards"])
            mean_opponent_reward = np.mean(results["learning_agent_cumulative_rewards"])
            # Compute the penalty if opponent's reward is higher
            penalty = penalty_coeff * max(0, mean_opponent_reward-penalty_slack - mean_extortion_reward)
            scores.append(mean_extortion_reward - penalty)
            detailed_results[f"case_{opt_case["name"]}"] = {
                "agent_type": agent_config["type"],
                "extortion_reward": mean_extortion_reward,
                "opponent_reward": mean_opponent_reward,
                "penalty": penalty,
            }

        weighted_sum = sum(w * r for w, r in zip(self.weights, scores))

        if return_rewards:
            return -float(weighted_sum), detailed_results

        return -float(weighted_sum)