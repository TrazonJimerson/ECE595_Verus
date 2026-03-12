def _get_info(self):
    """Compute auxiliary information for debugging.

    Returns:
        dict: Info with distance between agent and target
    """
    return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }