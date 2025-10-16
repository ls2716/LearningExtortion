import numpy as np

def make_monotone_knot_map(map_params, rng=None):
    """
    Create a mapping function based on the provided parameters.
    The parameters provide increments in the x and y directions for knot points.

    Args:
        map_params (list or np.ndarray): Parameters defining the mapping function. Divisible by 2.
        First half are x increments, second half are y increments.
        For the y_increments first and last values correspond to initial and final y values at 0 and 1.

    Returns:
        function: A mapping function that takes a float input in [0, 1] and returns a float in [0, 1].
    """
    map_params = np.asarray(map_params, dtype=float).flatten()
    if map_params.shape[0] % 2 != 0:
        raise ValueError("Length of map_params must be divisible by 2.")
    n = (map_params.shape[0]-2) // 2
    x_increments = map_params[:n]
    y_increments = map_params[n:]
    # Clip the increments to be non-negative
    x_increments = np.clip(x_increments, 0.0, None)
    y_increments = np.clip(y_increments, 0.0, None)
    # Normalize increments to sum to 1
    x_diffs = x_increments / x_increments.sum() if x_increments.sum() > 0 else np.ones_like(x_increments) / n
    y_diffs = y_increments / y_increments.sum() if y_increments.sum() > 0 else np.ones_like(y_increments) / (n+2)
    # Compute the positions of the knots (0.0 to 1.0)
    x_pos = np.concatenate(([0.0], np.cumsum(x_diffs))) # Prepend 0.0 - finish with 1.0
    y_pos = np.cumsum(y_diffs)[:-1] # Exclude final 1.0 - first value corresponds to first increment

    def mapping_function(x: float) -> float:
        """Mapping function from [0, 1] to [0, 1] based on knot points.

        Args:
            x (float): Input value in [0, 1].

        Returns:
            float: Mapped value in [0, 1].
        """
        if x <= 0.0:
            return y_pos[0]
        elif x >= 1.0:
            return y_pos[-1]
        else:
            # Find the interval for x
            idx = np.searchsorted(x_pos, x) - 1
            idx = np.clip(idx, 0, len(x_pos) - 2)  # Ensure idx is within bounds
            # Linear interpolation
            x0, x1 = x_pos[idx], x_pos[idx + 1]
            y0, y1 = y_pos[idx], y_pos[idx + 1]
            if x1 == x0:
                return y0  # Avoid division by zero
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    
    return mapping_function, {"x_pos": x_pos, "y_pos": y_pos}



def make_equispaced_map(map_params, rng=None):
    """
    Create a mapping function based on the provided parameters.
    The parameters provide only the y values for knot points.
    The x positions are equispaced.

    Args:
        map_params (list or np.ndarray): Parameters defining the mapping function. Length n.
        Corresponds to y values at equispaced x positions.
    Returns:
        function: A mapping function that takes a float input in [0, 1] and returns a float in [0, 1].
    """

    map_params = np.asarray(map_params, dtype=float).flatten()
    n = map_params.shape[0]
    if n < 2:
        raise ValueError("Length of map_params must be at least 2.")
    # Clip the y values to be in [0, 1]
    y_pos = np.clip(map_params, 0.0, 1.0)
    # Compute equispaced x positions
    x_pos = np.linspace(0.0, 1.0, n)

    def mapping_function(x: float) -> float:
        """Mapping function from [0, 1] to [0, 1] based on knot points.

        Args:
            x (float): Input value in [0, 1].

        Returns:
            float: Mapped value in [0, 1].
        """
        if x <= 0.0:
            return y_pos[0]
        elif x >= 1.0:
            return y_pos[-1]
        else:
            # Find the interval for x
            idx = np.searchsorted(x_pos, x) - 1
            idx = np.clip(idx, 0, len(x_pos) - 2)  # Ensure idx is within bounds
            # Linear interpolation
            x0, x1 = x_pos[idx], x_pos[idx + 1]
            y0, y1 = y_pos[idx], y_pos[idx + 1]
            if x1 == x0:
                return y0  # Avoid division by zero
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    
    return mapping_function, {"x_pos": x_pos, "y_pos": y_pos}

def make_monotone_equispaced_map(map_params, rng=None):
    """
    Create a monotone increasing mapping function based on the provided parameters.
    The parameters provide only the y increments for knot points.
    The x positions are equispaced.

    Args:
        map_params (list or np.ndarray): Parameters defining the mapping function. Length n-2.
        Corresponds to y increments at equispaced x positions (excluding first and last which are fixed at 0 and 1).
    Returns:
        function: A mapping function that takes a float input in [0, 1] and returns a float in [0, 1].
    """

    map_params = np.asarray(map_params, dtype=float).flatten()
    n = map_params.shape[0]
    if n < 2:
        raise ValueError("Length of map_params must be at least 0.")
    # Clip the y increments to be non-negative
    y_increments = np.clip(map_params, 0.0, None)
    # Normalize increments to sum to 1
    y_diffs = y_increments / y_increments.sum() if y_increments.sum() > 0 else np.ones_like(y_increments) / (n-2)
    # Compute the positions of the knots (0.0 to 1.0)
    x_pos = np.linspace(0.0, 1.0, n-1)
    y_pos = np.cumsum(y_diffs)[:-1]

    def mapping_function(x: float) -> float:
        """Mapping function from [0, 1] to [0, 1] based on knot points.

        Args:
            x (float): Input value in [0, 1].

        Returns:
            float: Mapped value in [0, 1].
        """
        if x <= 0.0:
            return y_pos[0]
        elif x >= 1.0:
            return y_pos[-1]
        else:
            # Find the interval for x
            idx = np.searchsorted(x_pos, x) - 1
            idx = np.clip(idx, 0, len(x_pos) - 2)  # Ensure idx is within bounds
            # Linear interpolation
            x0, x1 = x_pos[idx], x_pos[idx + 1]
            y0, y1 = y_pos[idx], y_pos[idx + 1]
            if x1 == x0:
                return y0  # Avoid division by zero
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    
    return mapping_function, {"x_pos": x_pos, "y_pos": y_pos}


def make_stationary_map(map_params, rng=None):
    """
    Create a stationary mapping function that returns a random action according to the distribution defined by map_params.

    Args:
        map_params (list or np.ndarray): Parameters defining stationary action distribution.

    Returns:
        function: A mapping function that takes a float input in [0, 1] and returns a float in [0, 1].
    """
    map_params = np.asarray(map_params, dtype=float).flatten()
    # Define the equispaced grid 
    n = map_params.shape[0]
    x_pos = np.linspace(0, 1, n, endpoint=True)
    y = map_params
    # Normalise y using softmax
    exp_y = np.exp(y - np.max(y))  # Subtract max for numerical stability
    y_pos = exp_y / exp_y.sum()

    # Get random generator
    rng = rng if rng is not None else np.random.default_rng()

    def get_action(x: float) -> float:
        """Mapping function that always returns the constant value.

        Args:
            x (float): Input value in [0, 1].

        Returns:
            float: random action according to probability distribution
        """
        # Find the correct value
        action_index = rng.choice(x_pos.shape[0], p=y_pos)
        return x_pos[action_index]

    
    return get_action, {"x_pos": x_pos, "y_pos": y_pos}


def make_custom(map_params, rng=None):
    """
    Define custom mapping function by y_pos

    Args:
        map_params (list or np.ndarray): Parameters defining the mapping function. Length n.
        Corresponds to y values at equispaced x positions.
    Returns:
        function: A mapping function that takes a float input in [0, 1] and returns a float in [0, 1].
    """
    x_pos = np.linspace(0.0, 1.0, len(map_params))
    y_pos = np.clip(map_params, 0.0, 1.0)

    def get_action(x: float) -> float:
        """Mapping function from [0, 1] to [0, 1] based on knot points.

        Args:
            x (float): Input value in [0, 1].

        Returns:
            float: Mapped value in [0, 1].
        """
        if x <= 0.0:
            return y_pos[0]
        elif x >= 1.0:
            return y_pos[-1]
        else:
            # Find the interval for x
            idx = np.searchsorted(x_pos, x) - 1
            idx = np.clip(idx, 0, len(x_pos) - 2)  # Ensure idx is within bounds
            # Linear interpolation
            x0, x1 = x_pos[idx], x_pos[idx + 1]
            y0, y1 = y_pos[idx], y_pos[idx + 1]
            if x1 == x0:
                return y0  # Avoid division by zero
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    
    return get_action, {"x_pos": x_pos, "y_pos": y_pos}


mapping_dict = {
    "monotone_equispaced": make_monotone_equispaced_map,
    "equispaced": make_equispaced_map,
    "monotone_knot": make_monotone_knot_map,
    "stationary": make_stationary_map,
    "custom": make_custom,
}



