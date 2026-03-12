def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    """Start a new episode.

    Args:
        seed: Random seed for reproducible episodes
        options: Additional configuration (unused in this example)

    Returns:
        tuple: (observation, info) for the initial state
    """
    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)

    # Randomly place the agent anywhere on the grid
    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    # Randomly place target, ensuring it's different from agent position
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
        self._target_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )

    observation = self._get_obs()
    info = self._get_info()

    return observation, info