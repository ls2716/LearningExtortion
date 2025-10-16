"""Compare strategies corresponding to different delays.

Specifically, we will plot strategies for delays 0, 0.5, and 1.0 for clarity.

Additionally, the plot will have the theoretical best response curve for reference."""

import matplotlib.pyplot as plt
import numpy as np

from tools.agents.extortion_maps import make_monotone_equispaced_map
from tools.environment import RewardFunction, EnvironmentParameters


def get_mapped_prices(mapping, prices_follower, price_bounds):
    """
    Map follower prices through the given mapping.
    """
    prices_normalised = (prices_follower - price_bounds[0]) / (
        price_bounds[1] - price_bounds[0]
    )
    mapped_prices = np.array([mapping(p) for p in prices_normalised])
    return mapped_prices * (price_bounds[1] - price_bounds[0]) + price_bounds[0]


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 16})
    price_bounds = (1.0, 2.8)
    prices_follower = np.linspace(price_bounds[0], price_bounds[1], 101, endpoint=True)

    # Load theoretical strategy
    theoretical_path = "../experiment_theory_strategies/unimodal_extortion_mapping.txt"
    theoretical_strategy = np.loadtxt(theoretical_path)

    delay_path_template = "sw_uts_d{:.2f}/optimised_params.txt"
    delays = [0.0,  1.0]
    strategies = {}

    for delay in delays:
        delay_path = delay_path_template.format(delay)
        params = np.loadtxt(delay_path)
        strategies[delay], _ = make_monotone_equispaced_map(params)

    # Plot
    # plt.figure(figsize=(7, 5))
    # for delay in delays:
    #     prices_leader = get_mapped_prices(
    #         strategies[delay], prices_follower, price_bounds
    #     )
    #     plt.plot(prices_follower, prices_leader, label=f"$\delta={delay:.1f}$")
    # plt.plot(
    #     theoretical_strategy[:, 0],
    #     theoretical_strategy[:, 1],
    #     label="$\delta=0.0$ theory",
    #     linestyle="--",
    #     color="tab:blue",
    # )
    # plt.xlabel("Follower Price $p^F$")
    # plt.ylabel("Leader Price $p^L$")
    # plt.legend()
    # plt.grid()
    # plt.savefig("strategies_delay.png", dpi=300)
    # plt.close()

    N=2
    A = [1.0, 1.0]
    C = [1.0, 1.0]
    sigma = 0
    mu = 0.25
    a_0 = -1

    reward_fn = RewardFunction(
        EnvironmentParameters(N=2, A=A, C=C, mu_0=mu, a_0=a_0, sigma=sigma)
    )

    # Plot strategies and induced rewards on subplots
    plt.rcParams.update({"font.size": 20})
    plt.rcParams['legend.fontsize'] = 17
    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    for delay in delays:
        prices_leader = get_mapped_prices(
            strategies[delay], prices_follower, price_bounds
        )
        axs[0].plot(prices_follower, prices_leader, label=f"$\delta={delay:.1f}$")
        prices_all = np.vstack((prices_follower, prices_leader)).T
        rewards = reward_fn(prices_all)
        axs[1].plot(
            prices_follower, rewards[:, 1], label=f"Leader $\delta={delay:.1f}$"
        )
        axs[1].plot(
            prices_follower,
            rewards[:, 0],
            label=f"Follower $\delta={delay:.1f}$",
        )
    axs[0].plot(
        theoretical_strategy[:, 0],
        theoretical_strategy[:, 1],
        label="Theory $\delta=0.0$",
        linestyle="--",
        color="tab:blue",
    )
    axs[0].set_xlabel("Follower price $p^F$")
    axs[0].set_ylabel("Leader price $p^L$")
    axs[0].grid()
    axs[0].legend()

    axs[1].set_xlabel("Follower price $p^F$")
    axs[1].set_ylabel("Induced rewards")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()

    plt.savefig("comparison_strategies.png", dpi=300)
    plt.show()