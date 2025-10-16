# File: src/tools/test_extortion_agent.py
import pytest
import numpy as np

from tools.agents.extortion_agent import ExtortionAgent, make_map

def test_extortion_agent_initialization_and_action():
    params = [0.2, 0.3, 0.5, 0.1, 0.2, 0.3]
    mapping_function, info = make_map(params)
    agent = ExtortionAgent(mapping_function)
    # Test several values in [0, 1]
    for x in np.linspace(0, 1, 5):
        action = agent.get_action(x)
        assert 0.0 <= action <= 1.0

def test_extortion_agent_action_without_mapping():
    agent = ExtortionAgent(None)
    with pytest.raises(ValueError):
        agent.get_action(0.5)

def test_make_map_returns_callable_and_info():
    params = [0.2, 0.3, 0.5, 0.1, 0.2, 0.3]
    mapping_function, info = make_map(params)
    assert callable(mapping_function)
    assert "x_pos" in info and "y_pos" in info

def test_make_map_raises_for_odd_length_params():
    params = [0.2, 0.3, 0.5]
    with pytest.raises(ValueError):
        make_map(params)

def test_mapping_function_edges_and_interpolation():
    params = [0.5, 0.5, 0.5, 0.5]
    mapping_function, info = make_map(params)
    # x=0.0 returns first y_pos
    assert mapping_function(0.0) == pytest.approx(info["y_pos"][0])
    # x=1.0 returns last y_pos
    assert mapping_function(1.0) == pytest.approx(info["y_pos"][-1])
    # Test interpolation between knots
    assert mapping_function(0.5) == pytest.approx((info["y_pos"][0] + info["y_pos"][1]) / 2)


def test_reset_method_noop():
    params = [0.2, 0.3, 0.5, 0.1, 0.2, 0.3]
    mapping_function, _ = make_map(params)
    agent = ExtortionAgent(mapping_function)
    agent.reset()  # Should not raise