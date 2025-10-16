# File: src/tools/test_environment.py
import pytest
import numpy as np

from src.tools.environment import EnvironmentParameters, RewardFunction

def test_environment_parameters_initialization_valid():
    params = EnvironmentParameters(
        A=[1.0, 2.0], C=[0.5, 1.5], mu_0=0.25, a_0=-1.0, N=2, sigma=0.1
    )
    assert params.A.shape == (1, 2)
    assert params.C.shape == (1, 2)
    assert isinstance(params.sigma, float)
    assert isinstance(params.a_0, float)
    assert params.N == 2

def test_environment_parameters_initialization_invalid_length():
    with pytest.raises(AssertionError):
        EnvironmentParameters(
            A=[1.0], C=[0.5, 1.5], mu_0=0.25, a_0=-1.0, N=2, sigma=0.1
        )

def test_reward_function_single_input():
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[0.5, 0.5], mu_0=0.5, a_0=0.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([1.0, 1.0])
    rewards = reward_fn(prices)
    assert rewards.shape == (1, 2)
    # Check rewards are finite and positive (since price > cost)
    assert np.all(np.isfinite(rewards))
    assert np.all(rewards >= 0)

def test_reward_function_batch_input():
    params = EnvironmentParameters(
        A=[2.0, 2.0], C=[1.0, 1.0], mu_0=1.0, a_0=0.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([[1.5, 1.5], [2.0, 2.0]])
    rewards = reward_fn(prices)
    assert rewards.shape == (2, 2)
    assert np.all(np.isfinite(rewards))

def test_reward_function_high_prices_zero_demand():
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[0.5, 0.5], mu_0=0.25, a_0=0.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([100.0, 100.0])
    rewards = reward_fn(prices)
    # Rewards should be very close to zero
    assert np.all(rewards < 1e-6)

def test_reward_function_negative_prices():
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[0.5, 0.5], mu_0=0.5, a_0=0.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([-1.0, -1.0])
    rewards = reward_fn(prices)
    assert rewards.shape == (1, 2)
    assert np.all(np.isfinite(rewards))

def test_reward_function_small_mu0():
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[0.5, 0.5], mu_0=1e-6, a_0=0.0, N=2, sigma=0.0
    )
    reward_fn = RewardFunction(params)
    prices = np.array([1.0, 1.0])
    rewards = reward_fn(prices)
    assert rewards.shape == (1, 2)
    assert np.all(np.isfinite(rewards))

def test_sigma_and_a0_type_conversion():
    params = EnvironmentParameters(
        A=[1.0, 1.0], C=[1.0, 1.0], mu_0=0.25, a_0="-1.0", N=2, sigma="0.0"
    )
    assert isinstance(params.sigma, float)
    assert isinstance(params.a_0, float)