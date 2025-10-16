"""Implementation of agents for pricing in a competition."""
import numpy as np



class ExtortionAgent:
    def __init__(self, price_mapping):
        """Initialize the extortion agent.

        Args:
            map_type (str): Type of mapping function ('monotone' or 'jump').
            params: Parameters for the mapping function.
            kwargs: Additional keyword arguments for mapping function.
        """
        self.mapping_function = price_mapping


    def reset(self):
        """Reset the agent's state."""
        pass

    def get_action(self, learning_agent_action_01: float) -> float:
        """Get the extortion agent's action based on the learning agent's action.

        Args:
            learning_agent_action_01 (float): Learning agent's action in [0, 1].

        Returns:
            float: Extortion agent's action in [0, 1].
        """
        if self.mapping_function is None:
            raise ValueError("Mapping function not set. Call set_mapping(params) first.")
        return self.mapping_function(learning_agent_action_01)
    


def make_monotone_mapper(params, eps: float = 0.0):
    """
    
    """
    # Softmax-like normalization (stable) → y_diffs sum to 1
    params = np.asarray(params, dtype=float)
    exp_params = np.exp(params - params.max())
    y_diffs = exp_params / exp_params.sum()

    # Cumulative y at interior knots (exclude final 1.0)
    y_knots = np.cumsum(y_diffs)[:-1]
    n_knots = y_knots.shape[0]
    m = max(n_knots - 1, 1)  # number of segments used below (avoid zero division)

    def f(x):
        """Monotone piecewise-linear interpolation."""
        x = np.clip(x, 0.0, 1.0)
        xm = x * m
        # segment index j in [0, m-1]; handles x=1 by clamping
        j = np.minimum(xm.astype(int), m - 1)

        # y at endpoints of the segment
        y0 = y_knots[j]
        y1 = y_knots[j + 1]

        # since knots are uniform: x0 = j/m, x1 = (j+1)/m → t = xm - j (no division)
        t = xm - j
        return y0 + t * (y1 - y0)

    return f




def make_jump_mapper(params):
    """
    Create a piecewise constant jump function from parameters.
    """
    # First param is the jump location in (0, 1)
    jump_loc = params[0]
    # Second param is the jump height in (0, 1)
    jump_height = params[1]

    def f(x):
        """Piecewise constant jump function."""
        x = np.clip(x, 0.0, 1.0)
        return np.where(x < jump_loc, 0.0, jump_height)
    
    return f





mapping_dict = {
    "monotone": make_monotone_mapper,
    "jump": make_jump_mapper
}
