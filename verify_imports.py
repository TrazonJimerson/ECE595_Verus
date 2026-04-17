# Test basic imports
import safe_autonomy_sims
import safe_autonomy_sims.gym
import safe_autonomy_sims.pettingzoo

# Test Gymnasium environment creation
import gymnasium

env = gymnasium.make("Docking-v0")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test PettingZoo environment creation
multi_env = safe_autonomy_sims.pettingzoo.MultiDockingEnv()
print(f"Agents: {multi_env.agents}")