
import numpy as np
from collections import deque


class Logistic2DModel:
    """
    A class to represent a 2D logistic model for bandit problems.

    Attributes:
        n_actions (int): Number of discrete price points.
        left_bound (float): Minimum value of the price range.
        right_bound (float): Maximum value of the price range.
    """

    def __init__(
        self,
        n_agents,
        n_actions,
        left_bound,
        right_bound,
        param_bounds,
        C,
        a_0=-1.0,
        n_global_samples=2000,
        n_local_samples=200,
        local_sample_ratio=0.2,
        opt_mode="rewards",  # 'rewards' or 'probabilities'
    ):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.n_actions = n_actions
        self.prices = np.linspace(left_bound, right_bound, n_actions, endpoint=True)
        self.no_agents = n_agents
        self.param_bounds = param_bounds
        for key in ["A", "mu", "BD"]:
            if key not in param_bounds:
                raise ValueError(f"Parameter bounds must include '{key}'")
            # Check that the bounds are tuples with two elements
            print(f"Checking parameter bounds for '{key}': {param_bounds[key]}")
            if (
                not (
                    isinstance(param_bounds[key], list)
                    or isinstance(param_bounds[key], tuple)
                )
                or len(param_bounds[key]) != 2
            ):
                raise ValueError(
                    f"Parameter bounds for '{key}' must be a tuple with two elements"
                )
            # If the bounds are lists, make into a tuple
            if isinstance(param_bounds[key], list):
                param_bounds[key] = (param_bounds[key][0], param_bounds[key][1])

        # Check that A is a tuple with lists of length n_agents
        if (
            not isinstance(self.param_bounds["A"][0], list)
            or len(self.param_bounds["A"][0]) != n_agents
            or len(self.param_bounds["A"][1]) != n_agents
        ):
            raise ValueError(
                f"Parameter bounds for 'A' must be a tuple with two lists of length {n_agents}"
            )
        self.a_0 = a_0  # Intercept for the logistic function
        self.C = np.array(C).reshape(n_agents)  # Cost for each agent
        if n_agents != 2:
            raise ValueError("This model is designed for exactly 2 agents.")
        self.n_global_samples = n_global_samples
        self.n_local_samples = n_local_samples
        self.local_sample_ratio = local_sample_ratio
        self.opt_mode = opt_mode
        self.reset()

        # logger.info("Setup logistic 2D model with parameters:")
        # logger.info(
        #     f"n_agents: {n_agents}, n_actions: {n_actions},\n"
        #     f"left_bound: {left_bound}, right_bound: {right_bound},\n"
        #     f"param_bounds: {param_bounds}, \nC: {C},\n"
        #     f"n_global_samples: {n_global_samples}, \nn_local_samples: {n_local_samples},\n"
        #     f"local_sample_ratio: {local_sample_ratio}, \nopt_mode: {opt_mode}"
        # )

    def generate_parameter_samples(self, n_samples, param_bounds=None):
        """
        Generate random parameter samples within the specified bounds.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with random parameters.
        """
        if param_bounds is None:
            param_bounds = self.param_bounds
        else:
            # Ensure param_bounds is a dictionary with the required keys
            for key in ["A", "mu", "BD"]:
                if key not in param_bounds:
                    raise ValueError(f"Parameter bounds must include '{key}'")
        # Make sure that bounds are withing the self.param_bounds
        param_bounds["A"] = (
            [
                max(param_bounds["A"][0][i], self.param_bounds["A"][0][i])
                for i in range(self.no_agents)
            ],
            [
                min(param_bounds["A"][1][i], self.param_bounds["A"][1][i])
                for i in range(self.no_agents)
            ],
        )
        param_bounds["mu"] = (
            max(param_bounds["mu"][0], self.param_bounds["mu"][0]),
            min(param_bounds["mu"][1], self.param_bounds["mu"][1]),
        )
        param_bounds["BD"] = (
            max(param_bounds["BD"][0], self.param_bounds["BD"][0]),
            min(param_bounds["BD"][1], self.param_bounds["BD"][1]),
        )

        A_bounds = param_bounds["A"]
        mu_bounds = param_bounds["mu"]
        BD_bounds = param_bounds["BD"]

        A_samples = np.random.uniform(
            A_bounds[0], A_bounds[1], size=(n_samples, self.no_agents)
        )
        mu_samples = np.random.uniform(mu_bounds[0], mu_bounds[1], n_samples)
        BD_samples = np.random.uniform(BD_bounds[0], BD_bounds[1], n_samples)

        return {"A": A_samples, "mu": mu_samples, "BD": BD_samples}
    


    def get_probs(self, prices, param_samples):
        """
        Calculate the probabilities for given prices using the logistic function.

        Args:
            prices (np.ndarray): Array of shape (no_history, no_agents) with prices.
            A (np.ndarray): Array of shape (no_samples, no_agents) with agent parameters.
            mu (np.ndarray): Array of shape (no_samples, 1) with mean parameters.
            a_0 (np.ndarray): Array of shape (no_samples, 1) with intercept parameters.

        Returns:
            np.ndarray: Array of shape (no_samples, no_history, no_agents) with probabilities.
        """
        prices = prices.reshape(1, -1, self.no_agents)
        A = param_samples["A"].reshape(-1, 1, self.no_agents)
        mu = param_samples["mu"].reshape(-1, 1, 1)
        BD = param_samples["BD"].reshape(-1, 1, 1)

        A_minus_p = A - prices
        exps = np.exp(A_minus_p / mu)
        exp_0 = np.exp(-1 / mu)
        denum = np.sum(exps, axis=2, keepdims=True) + exp_0
        probs = exps / denum * BD
        return probs

    def get_rewards(self, prices, param_samples):
        probs = self.get_probs(prices, param_samples)
        # Compute rewards for the agent
        prices_tmp = prices.reshape(1, -1, self.no_agents)
        C_tmp = self.C.reshape(1, 1, self.no_agents)
        rewards = probs * (prices_tmp - C_tmp)
        return rewards

    def compute_loss(self, true_y, y):
        """
        Compute the loss between true probabilities and predicted probabilities.

        Args:
            true_y (np.ndarray): True probabilities of shape (no_history, no_agents).
            y (np.ndarray): Predicted probabilities of shape (no_samples, no_history, no_agents).

        Returns:
            float: The computed loss.
        """
        error = true_y - y
        loss = np.mean(
            np.square(error)[:, :, 0], axis=1
        )  # We only get the first agent loss
        assert loss.shape == (y.shape[0],), "Loss shape mismatch"
        return loss

    def get_best_parameter(self, param_samples):
        """
        Get the best parameter sample based on the computed loss.

        Args:
            param_samples (dict): Dictionary containing parameter samples.

        Returns:
            dict: The best parameter sample.
        """
        # Get the rewards for the agent
        rewards = self.get_rewards(self.price_history, param_samples)
        true_rewards = self.reward_history.reshape(1, -1, self.no_agents)
        # Check if the rewards and true_rewards have the same shape
        # logger.debug(
        #     f"Rewards shape: {rewards.shape}, True rewards shape: {true_rewards.shape}"
        # )
        # Compute the loss for each parameter sample
        loss = self.compute_loss(self.conversion_history, true_rewards, rewards)
        # Find the index of the minimum loss
        best_index = np.argmin(loss)
        # Return the best parameters
        best_parameters = {
            "A": param_samples["A"][best_index],
            "mu": param_samples["mu"][best_index],
            "BD": param_samples["BD"][best_index],
        }
        return best_parameters, best_index

    def compute_max_action(self, opponent_price):
        """Compute the max action (price) for the agent given the opponent's price.
        Args:
            opponent_price (float): The price set by the opponent.
        Returns:
            tuple: (best_price (float), best_index (int)) corresponding to max Q-value.
        """
        if self.best_parameters is None:
            random_action_index = np.random.randint(self.n_actions)
            return self.prices[random_action_index], random_action_index
        prices_ar = np.zeros((self.n_actions, self.no_agents))
        prices_ar[:, 0] = self.prices
        prices_ar[:, 1] = opponent_price
        # Get the probabilities for the agent given the opponent's price
        probs = self.get_probs(prices_ar, self.best_parameters)
        # Find the index of the maximum probability for the agent
        rewards = probs[0, :, 0] * (self.prices - self.C[0])  # Rewards for the agent
        max_action_index = np.argmax(rewards)
        best_price = self.prices[max_action_index]
        return best_price, max_action_index

    def update(
        self, agent_price, opponent_price, agent_reward, opponent_reward
    ):
        """
        Update the model with the agent's and opponent's price and reward.

        Args:
            agent_price (float): The price set by the agent.
            agent_reward (float): The reward received by the agent.
            opponent_price (float): The price set by the opponent.
        """
        # Update the price history
        if self.price_history is None:
            self.price_history = np.zeros((1, self.no_agents))
            # self.conversion_history = np.zeros((1, self.no_agents))
            self.reward_history = np.zeros((1, self.no_agents))
        else:
            self.price_history = np.vstack(
                (self.price_history, np.zeros((1, self.no_agents)))
            )
            # self.conversion_history = np.vstack(
            #     (self.conversion_history, np.zeros((1, self.no_agents)))
            # )
            self.reward_history = np.vstack(
                (self.reward_history, np.zeros((1, self.no_agents)))
            )
        # Update the global losses
        self.price_history[-1, 0] = agent_price
        self.price_history[-1, 1] = opponent_price
        # self.conversion_history[-1, 0] = agent_conversion
        # self.conversion_history[-1, 1] = opponent_conversion
        self.reward_history[-1, 0] = agent_reward
        self.reward_history[-1, 1] = opponent_reward

        # Compute the losses for the global samples
        rewards = self.get_rewards(self.price_history[-1, :], self.param_samples)
        true_rewards = self.reward_history[-1, :].reshape(1, 1, self.no_agents)

        losses = self.compute_loss(true_y=true_rewards, y=rewards)

        # Update the mse
        n = self.price_history.shape[0]
        self.mse = self.mse * (n - 1) / n + losses / n

        # Get the best parameter sample based on the computed losses
        best_index = np.argmin(self.mse)
        self.best_parameters = {
            "A": self.param_samples["A"][best_index],
            "mu": self.param_samples["mu"][best_index],
            "BD": self.param_samples["BD"][best_index],
        }

        # Print best mse and parameters
        # logger.info(f"Best MSE: {self.mse[best_index]} with index {best_index}")

        # Generate local parameter samples
        local_param_bounds = {
            "A": (
                (1 - self.local_sample_ratio) * self.best_parameters["A"],
                (1 + self.local_sample_ratio) * self.best_parameters["A"],
            ),
            "mu": (
                (1 - self.local_sample_ratio) * self.best_parameters["mu"],
                (1 + self.local_sample_ratio) * self.best_parameters["mu"],
            ),
            "BD": (
                (1 - self.local_sample_ratio) * self.best_parameters["BD"],
                (1 + self.local_sample_ratio) * self.best_parameters["BD"],
            ),
        }
        # print("Local parameter bounds", local_param_bounds)

        local_samples = self.generate_parameter_samples(
            self.n_local_samples, param_bounds=local_param_bounds
        )
        # Put the best parameters in the first position of the local samples
        local_samples["A"][0] = self.best_parameters["A"]
        local_samples["mu"][0] = self.best_parameters["mu"]
        local_samples["BD"][0] = self.best_parameters["BD"]

        # Get the best local parameter sample based on the computed losses
        true_rewards = self.reward_history.reshape(1, -1, self.no_agents)
        local_rewards = self.get_rewards(self.price_history, local_samples)
        local_losses = self.compute_loss(true_y=true_rewards, y=local_rewards)
        local_best_index = np.argmin(local_losses, axis=0)

        # Print the new mse
        # logger.info(f"New MSE: {local_losses[local_best_index]}")

        if self.mse[best_index] > local_losses[local_best_index]:
            # Print parameters
            # print("Old parameters", self.best_parameters)
            self.best_parameters = {
                "A": local_samples["A"][local_best_index],
                "mu": local_samples["mu"][local_best_index],
                "BD": local_samples["BD"][local_best_index],
            }
            # print("New parameters", self.best_parameters)

        self.param_samples["A"][0] = self.best_parameters["A"]
        self.param_samples["mu"][0] = self.best_parameters["mu"]
        self.param_samples["BD"][0] = self.best_parameters["BD"]
        # Update the losses for the global samples
        self.mse[0] = local_losses[local_best_index]

    def reset(self):
        """Reset the model to its initial state."""
        self.price_history = None
        self.reward_history = None
        # self.conversion_history = None
        self.best_parameters = None
        self.global_losses = None

        # Generate initial parameter samples
        self.param_samples = self.generate_parameter_samples(
            self.n_global_samples, self.param_bounds
        )
        self.mse = np.zeros(self.n_global_samples)



class Opponent:
    """
    A simple opponent model that simulates a stationary opponent in a 2D pricing competition.
    
    This opponent always selects the same price, which is the average of the left and right bounds.
    """
    
    def __init__(self, window_size, trim_size=0):
        """
        Initialize the opponent with a fixed price.
        
        Args:
            window_size (int): Size of the sliding window for price history (not used in this simple model).
            trim_size (int): Number of extreme values to trim from the price history
        """
        self.window_size = window_size
        self.trim_size = trim_size
        if trim_size < 0:
            raise ValueError("trim_size must be non-negative")
        if trim_size >= window_size/2:
            raise ValueError("trim_size must be less than window_size")
        self.last_prices = deque(maxlen=window_size)
    
    def get(self):
        """
        Get the current price of the opponent.
        
        Returns:
            float: The fixed price of the opponent.
        """
        if len(self.last_prices) < self.window_size:
            return None
        prices = np.array(self.last_prices)
        # Sort prices and trim the extreme values
        if self.trim_size > 0:
            prices = np.sort(prices)[self.trim_size:-self.trim_size]
        # Return the average of the remaining prices
        return np.mean(prices)
    
    def reset(self):
        self.last_prices.clear()
    
    def update(self, price):
        self.last_prices.append(price)
    


class Logistic2D():
    """
    Epsilon-greedy bandit for a pricing competition which assumes
      a logistic model of the environment.
    """

    def __init__(
        self,
        action_bounds,
        n_actions,
        epsilon,
        decay,
        model_params,
        # opponent: Opponent,
        rng=None
    ):
        """
        Initialize the epsilon-greedy bandit.

        Args:
            agent_index (int): Index of this agent in a multi-agent setting.
            n_agents (int): Total number of agents (not actively used in this base model).
            n_actions (int): Number of discrete price actions.
            left_bound (float): Minimum price value.
            right_bound (float): Maximum price value.
            epsilon (float): Probability of taking a random action (exploration).
        """
        self.prices = np.linspace(action_bounds[0], action_bounds[1], n_actions, endpoint=True)
        self.n_prices = n_actions
        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.action_bounds = action_bounds
        self.model = Logistic2DModel(
            n_agents=2,
            n_actions=n_actions,
            left_bound=action_bounds[0],
            right_bound=action_bounds[1],
            **model_params,
        )
        # self.opponent = opponent
        # self.opponent.reset()
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    
    def select_local_random_action(self):
        """Select a random action from the local price space.
        """
        opponent_price = self.opponent.get()
        if opponent_price is None:
            action_index = self.rng.randint(self.n_prices)
            return self.prices[action_index], action_index
        else:
            price, price_index = self.model.compute_max_action(opponent_price)
            # Get prices from self.prices that are within 0.1 of the selected price
            distances = np.abs(self.prices - price)
            close_indices = np.where(distances < 0.1)[0]
            if len(close_indices) == 0:
                # If no close prices, raise an error
                raise ValueError("No close prices found within 0.1 of the selected price.")
            # Randomly select one of the close prices
            action_index = self.rng.choice(close_indices)
            return self.prices[action_index]


    def get_action(self, opponent_action):
        """
        Select a price to offer using epsilon-greedy strategy.

        Returns:
            tuple: (price (float), index (int)) of the selected action.
        """
        # opponent_price = self.opponent.get()
        opponent_price = opponent_action
        if self.rng.random() < self.epsilon or opponent_price is None:
            # # MODIFICATION: Use local random action selection
            # return self.select_local_random_action()
            # Original code:
            action_index = self.rng.integers(self.n_prices)
            return self.prices[action_index]
        else:
            price, price_index = self.model.compute_max_action(opponent_price)
            return price

    def update(self, agent_action, opponent_action, agent_reward, opponent_reward):
        """
        Update the model with observed prices and rewards.

        Args:
            prices (list or np.ndarray): List of prices selected by all agents.
            rewards (list or np.ndarray): List of rewards received by all agents.
            conversions (list or np.ndarray): List of conversions
        """
        # agent_price = prices[self.agent_index]
        # opponent_price = prices[1 - self.agent_index]  # Assuming 2 agents
        # agent_conversion = conversions[self.agent_index]
        # opponent_conversion = conversions[1 - self.agent_index]
        self.model.update(
            agent_price=agent_action,
            opponent_price=opponent_action,
            agent_reward=agent_reward,
            opponent_reward=opponent_reward,
        )
        self.epsilon *= self.decay
        # self.opponent.update(opponent_price)

    def reset(self):
        """
        Reset the internal model to its initial state.
        """
        self.model.reset()
        self.epsilon = self.epsilon_0
        # self.opponent.reset()

    def set_initial_action(self, action:float):
        pass

    # @classmethod
    # def from_config(cls, config: dict, rng=None):
    #     """
    #     Create an instance of Logistic2D from a configuration dictionary.

    #     Args:
    #         config (dict): Configuration parameters for the bandit.

    #     Returns:
    #         Logistic2D: An initialized Logistic2D bandit instance.
    #     """
    #     # Extract or validate required parameters
    #     agent_index = config["agent_index"]
    #     n_agents = config["n_agents"]
    #     n_actions = config["n_actions"]
    #     left_bound = config["left_bound"]
    #     right_bound = config["right_bound"]
    #     epsilon = config["epsilon"]
    #     model_params = config["model_params"]

    #     # The Opponent object must be passed or constructed externally
    #     opponent = Opponent(
    #         window_size=config["opponent_window_size"],
    #         trim_size=config["opponent_trim_size"],
    #     )
    #     print(model_params)
    #     return cls(
    #         agent_index=agent_index,
    #         n_agents=n_agents,
    #         n_actions=n_actions,
    #         left_bound=left_bound,
    #         right_bound=right_bound,
    #         epsilon=epsilon,
    #         model_params=model_params,
    #         opponent=opponent,
    #     )
    


if __name__=="__main__":
    
    def get_rewards(prices):
        prices = prices.reshape(-1, 2)
        exps = np.exp((1 - prices) / 0.25)
        exp_0 = np.exp(-1 / 0.25)
        denum = np.sum(exps, axis=1, keepdims=True) + exp_0
        probs = exps / denum
        rewards = (prices - 1) * probs
        return rewards
    

    import matplotlib.pyplot as plt
    n_actions = 101
    action_bounds = [1.0, 2.8]
    prices = np.linspace(action_bounds[0], action_bounds[1], n_actions, endpoint=True)
    price_opponent = 1.5
    prices_ar = np.zeros((n_actions, 2))
    prices_ar[:, 0] = prices
    prices_ar[:, 1] = price_opponent
    rewards = get_rewards(prices_ar)
    plt.plot(prices, rewards[:, 0], label="Agent reward")
    plt.plot(prices, rewards[:, 1], label="Opponent reward")
    plt.xlabel("Price")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    # Generate random samples from range [1., 2.8] of shape (n_samples, 2)
    price_samples = np.random.uniform(1.0, 2.8, size=(500, 2))
    rewards = get_rewards(price_samples)

    logistic_model = Logistic2DModel(
        n_agents=2,
        n_actions=51,
        left_bound=1.,
        right_bound=2.8,
        C= [1., 1.],
        param_bounds={
            "A": [[0.5, 0.5], [2.0, 2.0]],
            "mu": [0.1, 0.4],
            "BD": [0.2, 1.0],
        }
    )
    for i in range(50):
        logistic_model.update(
            agent_price=price_samples[i, 0],
            opponent_price=price_samples[i, 1],
            agent_reward=rewards[i, 0],
            opponent_reward=rewards[i, 1],
        )
        print(f"Iteration {i}, best parameters: {logistic_model.best_parameters}")
        print(f"Iteration {i}, mse: {logistic_model.mse.min()}")


