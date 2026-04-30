import os
import typing
import numpy as np
import gymnasium as gym
import copy
import numpy as np
import safe_autonomy_simulation.sims.inspection as sim
import safe_autonomy_sims.gym.inspection.reward as r
import safe_autonomy_sims.gym.inspection.utils as utils

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray import tune

class InspectionEnv(gym.Env):
    def __init__(
        self,
        success_threshold: float = 99,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 12236,
    ) -> None:
        # Each spacecraft obs = [x, y, z, v_x, v_y, v_z, theta_sun, n, x_ups, y_ups, z_ups]
        self.observation_space = gym.spaces.Box(
            np.concatenate(
                (
                    [-np.inf] * 3,  # position
                    [-np.inf] * 3,  # velocity
                    [0],  # sun angle
                    [0],  # num inspected
                    [-1] * 3,  # nearest cluster
                )
            ),
            np.concatenate(
                (
                    [np.inf] * 3,  # position
                    [np.inf] * 3,  # velocity
                    [2 * np.pi],  # sun angle
                    [100],  # num inspected
                    [1] * 3,  # nearest cluster
                )
            ),
            shape=(11,),
        )

        self.num_bins = 21
        self.action_space = gym.spaces.Discrete(self.num_bins ** 3)


        # Environment parameters
        self.crash_radius = crash_radius
        self.max_distance = max_distance
        self.max_time = max_time
        self.success_threshold = success_threshold

        # Episode level information
        self.prev_state = None
        self.prev_num_inspected = 0
        self.reward_components = {}
        self.status = "Running"

    def _map_action(self, action):

        raw_actions = np.array(np.unravel_index(
            int(action),
            (self.num_bins, self.num_bins, self.num_bins),
            )
        )

        values = np.linspace(-1.0, 1.0, self.num_bins, dtype=np.float32)
        return values[raw_actions]

    def reset(
        self, *, seed: int | None = None, options: dict[str, typing.Any] | None = None
    ) -> tuple[typing.Any, dict[str, typing.Any]]:
        super().reset(seed=seed, options=options)
        self._init_sim()  # sim is light enough we just reconstruct it
        self.simulator.reset()
        self.reward_components = {}
        self.status = "Running"
        obs, info = self._get_obs(), self._get_info()
        self.prev_state = None
        self.prev_num_inspected = 0
        return obs, info

    def step(
        self, action: typing.Any
    ) -> tuple[typing.Any, typing.SupportsFloat, bool, bool, dict[str, typing.Any]]:
        assert self.action_space.contains(
            action
        ), f"given action {action} is not contained in action space {self.action_space}"

        # Remap the action space to [-1.0, 1.0] with 20 steps between
        action = self._map_action(action)

        # Store previous simulator state
        self.prev_state = self.sim_state.copy()
        if self.simulator.sim_time > 0:
            self.prev_num_inspected = (
                self.chief.inspection_points.get_num_points_inspected()
            )

        # Update simulator state
        self.deputy.add_control(action)
        self.simulator.step()

        # Get info from simulator
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        if terminated or truncated:
            print(
                "END:",
                self.status,
                "reward:",
                reward,
                "time:",
                self.simulator.sim_time,
                "num inspected:",
                self.chief.inspection_points.get_num_points_inspected(),
                "distance:",
                utils.rel_dist(
                    pos1=self.chief.position,
                    pos2=self.deputy.position,
                ),
        )

        # print("OBS:", observation, type(observation), observation.shape)

        return observation, reward, terminated, truncated, info

    def _init_sim(self):
        # Initialize spacecraft, sun, and simulator
        self.chief = sim.Target(
            name="chief",
            num_points=100,
            radius=10,
        )
        self.deputy = sim.Inspector(
            name="deputy",
            position=utils.polar_to_cartesian(
                r=self.np_random.uniform(50, 100),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            velocity=utils.polar_to_cartesian(
                r=self.np_random.uniform(0, 0.8),
                theta=self.np_random.uniform(0, 2 * np.pi),
                phi=self.np_random.uniform(-np.pi / 2, np.pi / 2),
            ),
            fov=np.pi,
            focal_length=1,
        )
        self.sun = sim.Sun(theta=self.np_random.uniform(0, 2 * np.pi))
        self.simulator = sim.InspectionSimulator(
            frame_rate=0.1,
            inspectors=[self.deputy],
            targets=[self.chief],
            sun=self.sun,
        )

    # def _get_obs(self):
    #     obs = self.observation_space.sample()
    #     obs[:3] = self.deputy.position
    #     obs[3:6] = self.deputy.velocity
    #     obs[6] = self.sun.theta % (2 * np.pi)
    #     obs[7] = self.chief.inspection_points.get_num_points_inspected()
    #     obs[8:11] = self.chief.inspection_points.kmeans_find_nearest_cluster(
    #         camera=self.deputy.camera, sun=self.sun
    #     )
    #     return obs

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[:3] = np.asarray(self.deputy.position, dtype=np.float32)
        obs[3:6] = np.asarray(self.deputy.velocity, dtype=np.float32)
        obs[6] = np.float32(self.sun.theta % (2 * np.pi))
        obs[7] = np.float32(self.chief.inspection_points.get_num_points_inspected())

        cluster = self.chief.inspection_points.kmeans_find_nearest_cluster(
            camera=self.deputy.camera,
            sun=self.sun,
        )
        obs[8:11] = np.asarray(cluster, dtype=np.float32)

        return obs

    def _get_info(self):
        return {"reward_components": copy.copy(self.reward_components), "status": copy.copy(self.status),
                "sim_time": self.simulator.sim_time,
                "num_inspected": self.chief.inspection_points.get_num_points_inspected(),
                "distance": utils.rel_dist(
                    pos1=self.chief.position,
                    pos2=self.deputy.position,
            ),
        }

    def _get_reward(self):
        reward = 0

        # Dense rewards
        points_reward = r.observed_points_reward(
            chief=self.chief, prev_num_inspected=self.prev_num_inspected
        )
        self.reward_components["observed_points"] = points_reward
        reward += points_reward

        delta_v_reward = r.delta_v_reward(
            v=self.deputy.velocity,
            prev_v=self.prev_state["deputy"][3:6],
        )
        self.reward_components["delta_v"] = delta_v_reward
        reward += delta_v_reward

        # Sparse rewards
        success_reward = r.inspection_success_reward(
            chief=self.chief,
            total_points=self.success_threshold,
        )
        self.reward_components["success"] = success_reward
        reward += success_reward

        crash_reward = r.crash_reward(
            chief=self.chief,
            deputy=self.deputy,
            crash_radius=self.crash_radius,
        )
        self.reward_components["crash"] = crash_reward
        reward += crash_reward

        # TODO: add another reward based on how long simulation runs

        return reward

    def _get_terminated(self):
        # Get state info
        d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)

        # Determine if in terminal state
        crash = d < self.crash_radius
        all_inspected = (
            self.chief.inspection_points.get_num_points_inspected()
            >= self.success_threshold
        )

        # Update Status
        if crash:
            self.status = "Crash"
        elif all_inspected:
            self.status = "Success"

        return crash or all_inspected

    def _get_truncated(self):
        d = utils.rel_dist(pos1=self.chief.position, pos2=self.deputy.position)
        timeout = self.simulator.sim_time > self.max_time
        oob = d > self.max_distance

        # Update Status
        if oob:
           self.status = "Out of Bounds"
        elif timeout:
            self.status = "Timeout"

        return timeout or oob

    @property
    def sim_state(self) -> dict:
        state = {
            "deputy": self.deputy.state,
            "chief": self.chief.state,
        }
        return state


def env_creator(env_config):
    return InspectionEnv(**env_config)

register_env("inspection_env", env_creator)

config = (
    DQNConfig()
    .environment(
        env="inspection_env",
        env_config={
            "success_threshold": 99,
            "crash_radius": 15,
            "max_distance": 800,
            "max_time": 1223,
        },
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0,
              rollout_fragment_length=64,
              )
    .training(
        lr=1e-4,
        train_batch_size=256,
        dueling=False,
        n_step=1,
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 5000,
            "prioritized_replay": False,
        },
        double_q=True,
        hiddens=[],
        num_steps_sampled_before_learning_starts=500,
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
    )
    .resources(
        num_gpus=0,
    )
)

# algo = config.build()

# for i in range(50):
#     result = algo.train()
#     print(i, result["episode_reward_mean"])
#
# checkpoint = algo.save()
# print("Saved checkpoint:", checkpoint)
#
# algo.stop()
# ray.shutdown()


tune.run(
    "DQN",
    name="DQN",
    stop={"timesteps_total": 100000 if not os.environ.get("CI") else 50000},
    checkpoint_freq=10,
    config=config.to_dict(),
)
