import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from requests import get
import wandb
from baloo_lib import (
    detect_box_touch,
    get_box_position,
    get_box_vel,
    get_elevator_height,
    get_elevator_vel,
    get_joint_angles,
    get_joint_vel,
    get_tactile_image,
    set_elevator_cmd,
    set_joint_pressure_commands,
)

from force_reward_wrapper import ForceRewardWrapper
from gymnasium import spaces
from gymnasium.wrappers import TimeAwareObservation
from gymnasium.wrappers import RecordVideo, TimeLimit, AutoResetWrapper
from mujoco_rendering import MujocoRenderer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# from gymnasium.utils.env_checker import check_env
from wandb.integration.sb3 import WandbCallback


class Observation:
    # to help with bookkeeping, this class is used to store the observation vector
    def __init__(
        self,
        object_pos,
        object_vel,
        elevator_pos,
        elevator_vel,
        left_pos,
        right_pos,
        left_vel,
        right_vel,
    ):
        self.object_pos = object_pos
        self.object_vel = object_vel
        self.elevator_pos = elevator_pos
        self.elevator_vel = elevator_vel
        self.left_j0_pos = left_pos[0:2]
        self.left_j1_pos = left_pos[2:4]
        self.left_j2_pos = left_pos[4:6]
        self.right_j0_pos = right_pos[0:2]
        self.right_j1_pos = right_pos[2:4]
        self.right_j2_pos = right_pos[4:6]
        self.left_j0_vel = left_vel[0:2]
        self.left_j1_vel = left_vel[2:4]
        self.left_j2_vel = left_vel[4:6]
        self.right_j0_vel = right_vel[0:2]
        self.right_j1_vel = right_vel[2:4]
        self.right_j2_vel = right_vel[4:6]

        self.obs_lower_bound = np.asarray(
            [-2, -2, 0]
            + [-2] * 3
            + [-1.5]
            + [-5]
            + [-np.pi] * 6
            + [-np.pi] * 6
            + [-2 * np.pi] * 6
            + [-2 * np.pi] * 6
        )

        self.obs_upper_bound = np.asarray(
            [2, 2, 2]
            + [2] * 3
            + [0]
            + [5]
            + [np.pi] * 6
            + [np.pi] * 6
            + [2 * np.pi] * 6
            + [2 * np.pi] * 6
        )

    def to_array(self):
        return np.hstack(
            [
                self.object_pos,
                self.object_vel,
                self.elevator_pos,
                self.elevator_vel,
                self.left_j0_pos,
                self.left_j1_pos,
                self.left_j2_pos,
                self.right_j0_pos,
                self.right_j1_pos,
                self.right_j2_pos,
                self.left_j0_vel,
                self.left_j1_vel,
                self.left_j2_vel,
                self.right_j0_vel,
                self.right_j1_vel,
                self.right_j2_vel,
            ]
        )

    def __repr__(self):
        return f"{self.to_array()}"

    def normalize_and_center(self):
        return (
            2
            * (self.to_array() - self.obs_lower_bound)
            / (self.obs_upper_bound - self.obs_lower_bound)
            - 1
        )


class IncrementalAction:
    """
    This class is used to store the action vector.
    """

    def __init__(self, normalized_action_vector):
        self.elevator_height = np.asarray(normalized_action_vector[0])
        self.left_j0_pressure = np.asarray(normalized_action_vector[1:5])
        self.left_j1_pressure = np.asarray(normalized_action_vector[5:9])
        self.left_j2_pressure = np.asarray(normalized_action_vector[9:13])
        self.right_j0_pressure = np.asarray(normalized_action_vector[13:17])
        self.right_j1_pressure = np.asarray(normalized_action_vector[17:21])
        self.right_j2_pressure = np.asarray(normalized_action_vector[21:25])

        self.action_lower_bound = np.asarray([-1] + [0] * 24)
        self.action_upper_bound = np.asarray([0] + [300] * 24)

        # add flag to declare if action is normalized or not.
        self.is_normalized = True

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack(
            [
                self.elevator_height,
                self.left_j0_pressure,
                self.left_j1_pressure,
                self.left_j2_pressure,
                self.right_j0_pressure,
                self.right_j1_pressure,
                self.right_j2_pressure,
            ]
        )

    def _saturate(self):
        np.clip(
            self.elevator_height,
            self.action_lower_bound[0],
            self.action_upper_bound[0],
            out=self.elevator_height,
        )
        np.clip(
            self.left_j0_pressure,
            self.action_lower_bound[1:5],
            self.action_upper_bound[1:5],
            out=self.left_j0_pressure,
        )
        np.clip(
            self.left_j1_pressure,
            self.action_lower_bound[5:9],
            self.action_upper_bound[5:9],
            out=self.left_j1_pressure,
        )
        np.clip(
            self.left_j2_pressure,
            self.action_lower_bound[9:13],
            self.action_upper_bound[9:13],
            out=self.left_j2_pressure,
        )
        np.clip(
            self.right_j0_pressure,
            self.action_lower_bound[13:17],
            self.action_upper_bound[13:17],
            out=self.right_j0_pressure,
        )
        np.clip(
            self.right_j1_pressure,
            self.action_lower_bound[17:21],
            self.action_upper_bound[17:21],
            out=self.right_j1_pressure,
        )
        np.clip(
            self.right_j2_pressure,
            self.action_lower_bound[21:25],
            self.action_upper_bound[21:25],
            out=self.right_j2_pressure,
        )

    def increment(self, increment_directions):
        """
        increment_directions is a 25 element vector of +1, 0, or -1.
        Each on is scaled by 20kpa for pressures and .1m for height.
        """
        self.elevator_height += increment_directions[0] * 0.05
        self.left_j0_pressure += increment_directions[1:5] * 10
        self.left_j1_pressure += increment_directions[5:9] * 10
        self.left_j2_pressure += increment_directions[9:13] * 10
        self.right_j0_pressure += increment_directions[13:17] * 10
        self.right_j1_pressure += increment_directions[17:21] * 10
        self.right_j2_pressure += increment_directions[21:25] * 10

        self._saturate()

        return self


class BalooV1(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode=None,
        camera_id=None,
        camera_name=None,
    ):
        super().__init__()

        # action space is elevator height, pressure commands for each joint (h, left [0,1,2,3], right [0,1,2,3])
        self.action_space = spaces.MultiDiscrete([3] * 25)

        self.observation_space = spaces.Box(
            -1, 1, shape=(6 + 6 + 6 + 6 + 3 + 3 + 2,), dtype=np.float32
        )

        self.camera_id = camera_id
        self.camera_name = camera_name
        self.render_mode = render_mode

        self._reinitialize_states()

        self.simulation_timestep = self.model.opt.timestep
        self.control_timestep = 0.01  # seconds

        assert int(self.control_timestep % self.simulation_timestep) == 0
        self.sim_steps_per_control_step = int(
            self.control_timestep / self.simulation_timestep
        )

    def _reinitialize_states(self):
        self.current_actions = IncrementalAction(
            [1.0] + [-1.0] * 24
        )  # highest command for elevator, lowest for pressures.
        self._initialize_model_from_xml()

    def _initialize_model_from_xml(self):
        xml_path = "/home/curtis/curtis_ws/src/curtis_sandbox/src/mujoco/models/baloo/baloo.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.mujoco_renderer = MujocoRenderer(self.model, self.data)

    def _buffer_to_array(self, buffer):
        # convert deque of jangles objects to a single numpy array
        return np.array([jangle.as_array() for jangle in buffer]).flatten()

    def get_obs(self):
        rawObs = Observation(**self._get_sensor_data(self.model, self.data))

        return rawObs.normalize_and_center().astype(self.observation_space.dtype)

    def _get_sensor_data(self, model, data):
        left_pos = get_joint_angles(model, data, "left")
        left_vel = get_joint_vel(model, data, "left")
        right_pos = get_joint_angles(model, data, "right")
        right_vel = get_joint_vel(model, data, "right")

        object_pos = get_box_position(model, data)
        object_vel = get_box_vel(model, data)
        elevator_pos = get_elevator_height(model, data)
        elevator_vel = get_elevator_vel(model, data)

        return {
            "object_pos": object_pos,
            "object_vel": object_vel,
            "elevator_pos": elevator_pos,
            "elevator_vel": elevator_vel,
            "left_pos": left_pos,
            "right_pos": right_pos,
            "left_vel": left_vel,
            "right_vel": right_vel,
        }

    def set_commands_from_action(self, action):
        # ! gym 0.29.0 added start parameter for multidiscrete action space [-1,1] and sb3 2.1.0 I don't think follows it. [0,2]
        # ! downgrading to 0.28.1 and 2.0.0 eliminates the start param issue, but now actions can only be 0,1,or2.

        action = np.asarray(action) - 1  # shift to be between -1 and 1.

        # action is +1,0,or 1 for each command.
        self.current_actions = self.current_actions.increment(action)

        # apply action to the model
        set_elevator_cmd(self.model, self.data, self.current_actions.elevator_height)

        left_pressures = [
            self.current_actions.left_j0_pressure,
            self.current_actions.left_j1_pressure,
            self.current_actions.left_j2_pressure,
        ]
        right_pressures = [
            self.current_actions.right_j0_pressure,
            self.current_actions.right_j1_pressure,
            self.current_actions.right_j2_pressure,
        ]

        for i in range(3):
            set_joint_pressure_commands(
                self.model, self.data, "left", i, left_pressures[i]
            )
            set_joint_pressure_commands(
                self.model, self.data, "right", i, right_pressures[i]
            )

    def _calc_reward(self):
        # calculate reward based on number of active taxels
        taxel_left_l0 = get_tactile_image(self.model, self.data, "left", 0)
        taxel_left_l1 = get_tactile_image(self.model, self.data, "left", 1)
        taxel_right_l0 = get_tactile_image(self.model, self.data, "right", 0)
        taxel_right_l1 = get_tactile_image(self.model, self.data, "right", 1)
        taxel_chest = get_tactile_image(self.model, self.data, "chest", None)

        reward_left_l0 = np.count_nonzero(taxel_left_l0)
        reward_left_l1 = np.count_nonzero(taxel_left_l1)
        reward_right_l0 = np.count_nonzero(taxel_right_l0)
        reward_right_l1 = np.count_nonzero(taxel_right_l1)
        reward_chest = np.count_nonzero(taxel_chest)

        total_reward = (
            reward_left_l0
            + reward_left_l1
            + reward_right_l0
            + reward_right_l1
            + reward_chest
        )
        print("old calc reward")

        return (
            total_reward - 1
        )  # penalize if total_reward is 0, hopefully to push arms to move

    def step(self, action):
        # map action to elevator height and joint pressure commands
        self.set_commands_from_action(action)

        # step the model forward in time however many steps are needed to match the control timestep
        mujoco.mj_step(self.model, self.data, nstep=self.sim_steps_per_control_step)
        # print(self.data.ctrl)

        # get observation, reward, done, info
        observation = self.get_obs()
        info = {}

        reward = self._calc_reward()
        # reward = -1
        # terminated = detect_box_touch(self.model, self.data)
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self._setup_visualization()
        self._reinitialize_states()
        # todo: randomly set position (and size) of box eventually.

        observation = self.get_obs()
        info = {}

        return observation, info

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


if __name__ == "__main__":
    # deprecation warning https://github.com/pytorch/pytorch/issues/84712

    config = {
        "total_timesteps": 1500000,
        "env_name": "BalooGymEnv",
        "time_limit_sec": 5,
        "time_aware_obs": True,
    }

    run = wandb.init(
        project="ppo_baloo",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    def make_env():
        env = BalooV1("rgb_array", camera_name="fixedcam")

        env = ForceRewardWrapper(env)
        print("Checking custom environment for compatibility with SB3")
        check_env(env)
        print("Done.")

        env = TimeLimit(
            env, max_episode_steps=config["time_limit_sec"] / env.control_timestep
        )  # must come before TimeAwareObservationV0
        #
        if config["time_aware_obs"]:
            env = TimeAwareObservation(env)  #! causes chagne to float64
        #
        env = RecordVideo(
            env, f"./rollout_videos/{run.id}", episode_trigger=lambda x: x % 100 == 0
        )  #!causes change to float64
        return env

    env = make_env()

    #     obs, info = env.reset()
    #     env.render()
    # #
    #     while True:
    #         action = env.action_space.sample()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         env.render()

    rl_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    rl_model.save("ppo_baloo_v1")
    run.finish()
