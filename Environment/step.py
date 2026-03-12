def step(self, action):
    """Execute one timestep within the environment.

    Args:
        action: The action to take (0-3 for directions)

    Returns:
        tuple: (observation, reward, terminated, truncated, info)
    """
    # Map the discrete action (0-3) to a movement direction
    direction = self._action_to_direction[action]

    # Update agent position, ensuring it stays within grid bounds
    # np.clip prevents the agent from walking off the edge
    self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
    )

    # Check if agent reached the target
    terminated = np.array_equal(self._agent_location, self._target_location)

    # We don't use truncation in this simple environment
    # (could add a step limit here if desired)
    truncated = False

    # Simple reward structure: +1 for reaching target, 0 otherwise
    # Alternative: could give small negative rewards for each step to encourage efficiency
    reward = 1 if terminated else 0

    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, terminated, truncated, info