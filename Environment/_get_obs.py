def _get_obs(self):
    """Convert internal state to observation format.

    Returns:
        dict: Observation with agent and target positions
    """
    return {"agent": self._agent_location, "target": self._target_location}