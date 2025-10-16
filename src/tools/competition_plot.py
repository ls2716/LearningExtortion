import numpy as np
import os
import matplotlib.pyplot as plt

from tools.agents.base import get_agent
from tools.competition import run_competition


def plot_price_history(results, save_path=None, steps_per_period=20):
    """Plot the price history of the agents."""
    delay = results["delay"]

    learning_agent_prices = np.repeat(
        results["learning_agent_prices"][:-1], steps_per_period
    )
    extortion_agent_prices = np.repeat(
        results["extortion_agent_prices"][:-1], steps_per_period
    )

    pre = int(steps_per_period * delay)
    extortion_agent_prices = np.concatenate(
        (np.full(pre, results["extortion_agent_initial_price"]), extortion_agent_prices)
    )
    extortion_agent_prices = extortion_agent_prices[: learning_agent_prices.shape[0]]

    x = np.arange(learning_agent_prices.shape[0]) / steps_per_period
    plt.figure()
    plt.plot(x, learning_agent_prices, label="Learning Agent Price", color="tab:blue")
    plt.plot(
        x, extortion_agent_prices, label="Extortion Agent Price", color="tab:orange"
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Price History (response delay={delay})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_reward_history(results, save_path=None, steps_per_period=20):
    """Plot the reward history of the agents."""
    delay = results["delay"]

    learning_agent_rewards = np.repeat(
        results["learning_agent_rewards"][:-1], steps_per_period
    )
    extortion_agent_rewards = np.repeat(
        results["extortion_agent_rewards"][:-1], steps_per_period
    )

    x = np.arange(learning_agent_rewards.shape[0]) / steps_per_period
    plt.figure()
    plt.plot(x, learning_agent_rewards, label="Learning Agent Reward", color="tab:blue")
    plt.plot(
        x, extortion_agent_rewards, label="Extortion Agent Reward", color="tab:orange"
    )
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title(f"Reward History (response delay={delay})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_prices_rewards(
    results,
    steps_per_period=20,
    nash_price=None,
    nash_reward=None,
    pareto_price=None,
    pareto_reward=None,
    save_path=None,
):
    """Plot the scatter of prices and rewards."""
    plt.rcParams.update({"font.size": 20})
    plt.rcParams['legend.fontsize'] = 17

    delay = results["delay"]

    learning_agent_prices = np.repeat(
        results["learning_agent_prices"][:-1], steps_per_period
    )
    extortion_agent_prices = np.repeat(
        results["extortion_agent_prices"][:-1], steps_per_period
    )

    pre = int(steps_per_period * delay)
    extortion_agent_prices = np.concatenate(
        (np.full(pre, results["extortion_agent_initial_price"]), extortion_agent_prices)
    )
    extortion_agent_prices = extortion_agent_prices[: learning_agent_prices.shape[0]]

    x = np.arange(learning_agent_prices.shape[0]) / steps_per_period

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(
        x, learning_agent_prices, label="Follower", color="tab:orange", alpha=0.8
    )
    axs[0].plot(x, extortion_agent_prices, label="Leader", color="tab:blue", alpha=0.8)
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Price")
    if nash_price is not None:
        axs[0].axhline(nash_price, color="red", linestyle="--", label="Nash")
    if pareto_price is not None:
        axs[0].axhline(pareto_price, color="green", linestyle="--", label="Pareto")
    axs[0].grid()
    axs[0].legend()

    learning_agent_rewards = np.repeat(
        results["learning_agent_rewards"][:-1], steps_per_period
    )
    extortion_agent_rewards = np.repeat(
        results["extortion_agent_rewards"][:-1], steps_per_period
    )

    x = np.arange(learning_agent_rewards.shape[0]) / steps_per_period

    axs[1].plot(
        x, learning_agent_rewards, label="Follower", color="tab:orange", alpha=0.8
    )
    axs[1].plot(x, extortion_agent_rewards, label="Leader", color="tab:blue", alpha=0.8)
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Reward")
    if nash_reward is not None:
        axs[1].axhline(nash_reward, color="red", linestyle="--", label="Nash")
    if pareto_reward is not None:
        axs[1].axhline(pareto_reward, color="green", linestyle="--", label="Pareto")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def run_competition_and_plot(
    reward_fn,
    learning_agent_config,
    extortion_agent_config,
    competition_config,
    save_dir=None,
    suffix="",
    nash_price=None,
    nash_reward=None,
    pareto_price=None,
    pareto_reward=None,
):
    """Run a competition and plot results."""
    extortion_agent = get_agent(agent_config=extortion_agent_config)
    learning_agent = get_agent(agent_config=learning_agent_config)

    results = run_competition(
        reward_function=reward_fn,
        extortion_agent=extortion_agent,
        learning_agent=learning_agent,
        T=competition_config["T"],
        extortion_agent_initial_price=competition_config[
            "extortion_agent_initial_price"
        ],
        learning_agent_initial_price=competition_config["learning_agent_initial_price"],
        delay=competition_config["delay"],
        return_history=True,
        skip_steps=competition_config["skip_steps"],
    )

    comp_evolution_path = None
    if save_dir:
        comp_evolution_path = os.path.join(
            save_dir, f"competition_evolution{suffix}.png"
        )



    plot_prices_rewards(
        results,
        save_path=comp_evolution_path,
        nash_price=nash_price,
        nash_reward=nash_reward,
        pareto_price=pareto_price,
        pareto_reward=pareto_reward,
    )

    return learning_agent


def plot_response_rewards(
    reward_fn, extortion_agent_config, price_bounds, save_dir=None, 
):
    """Plot the response and induced rewards of the extortion agent."""
    plt.rcParams.update({"font.size": 20})
    plt.rcParams['legend.fontsize'] = 17
    extortion_agent = get_agent(agent_config=extortion_agent_config)
    prices = np.linspace(price_bounds[0], price_bounds[1], 100)
    extortion_prices = np.array([extortion_agent.get_action(p) for p in prices])

    # Compute induced rewards
    all_prices = np.vstack((extortion_prices, prices)).T
    rewards = reward_fn(all_prices)

    theoretical_mapping_path = extortion_agent_config.get(
        "theoretical_mapping_path", None
    )
    theoretical_mapping_name = extortion_agent_config.get(
        "theoretical_mapping_name", "Theory"
    )
    plot_yx = extortion_agent_config.get("plot_yx", False)

    if theoretical_mapping_path is not None:
        theoretical_mapping = np.loadtxt(theoretical_mapping_path)
        theo_prices = theoretical_mapping[:, 0]
        theo_ext_prices = theoretical_mapping[:, 1]
        theo_all_prices = np.vstack((theo_ext_prices, theo_prices)).T
        theo_rewards = reward_fn(theo_all_prices)

    nash_price = extortion_agent_config.get("nash_price", None)
    pareto_price = extortion_agent_config.get("pareto_price", None)

    # Plot response function and induced rewards side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(prices, extortion_prices, label="Learned", color="tab:blue")
    # Plot theoretical mapping if available
    if theoretical_mapping_path is not None:
        axs[0].plot(
            theo_prices,
            theo_ext_prices,
            label=theoretical_mapping_name,
            color="tab:blue",
            linestyle="--",
        )
    if plot_yx:
        axs[0].plot(prices, prices, label="y=x", color="gray", linestyle=":")
    if nash_price:
        # axs[0].axhline(nash_price, color="red", linestyle="--", label="Nash")
        # axs[0].axvline(nash_price, color="red", linestyle="--")
        axs[0].scatter(nash_price, nash_price, marker='+', color='red', s=200, label='Nash')
    if pareto_price:
        axs[0].scatter(pareto_price, pareto_price, marker='+', color='green', s=200, label='Pareto')
    axs[0].set_xlabel("Follower price $p^F$")
    axs[0].set_ylabel("Leader price $p^L$")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(prices, rewards[:, 0], label="Leader", color="tab:blue")
    axs[1].plot(prices, rewards[:, 1], label="Follower", color="tab:orange")
    # Plot theoretical induced rewards if available
    if theoretical_mapping_path is not None:
        axs[1].plot(
            theo_prices,
            theo_rewards[:, 0],
            label=f"Leader ({theoretical_mapping_name})",
            color="tab:blue",
            linestyle="--",
        )
        axs[1].plot(
            theo_prices,
            theo_rewards[:, 1],
            label=f"Follower ({theoretical_mapping_name})",
            color="tab:orange",
            linestyle="--",
        )
    axs[1].set_xlabel("Follower price $p^F$")
    axs[1].set_ylabel("Induced rewards")
    axs[1].grid()
    axs[1].legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "response_and_induced_rewards.png"), dpi=300)
        plt.close()
    else:
        plt.show()


def run_competition_final_step(
    reward_fn,
    learning_agent_config,
    extortion_agent_config,
    competition_config,
):
    """Run a competition and plot results."""
    extortion_agent = get_agent(agent_config=extortion_agent_config)
    learning_agent = get_agent(agent_config=learning_agent_config)

    results = run_competition(
        reward_function=reward_fn,
        extortion_agent=extortion_agent,
        learning_agent=learning_agent,
        T=competition_config["T"],
        extortion_agent_initial_price=competition_config[
            "extortion_agent_initial_price"
        ],
        learning_agent_initial_price=competition_config["learning_agent_initial_price"],
        delay=competition_config["delay"],
        return_history=True,
        skip_steps=competition_config["skip_steps"],
    )

    # Return final prices
    return (
        results["learning_agent_prices"][-1],
        results["extortion_agent_prices"][-1],
        results["learning_agent_rewards"][-1],
        results["extortion_agent_rewards"][-1],
    )

