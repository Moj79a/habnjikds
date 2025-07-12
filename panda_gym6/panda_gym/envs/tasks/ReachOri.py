import numpy as np
from panda_gym.envs.core import Task
from typing import Any, Dict
import os
from pyquaternion import Quaternion
from panda_gym.utils import *


class ReachOri(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        get_ee_orientation,
        reward_type: str = "dense",
        distance_threshold: float = 0.05,
        ori_distance_threshold: float = 0.0873,
    ) -> None:
        super().__init__(sim)
        self.get_ee_position = get_ee_position
        self.get_ee_orientation = get_ee_orientation
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold  # 5cm
        self.ori_distance_threshold = ori_distance_threshold  # 5 degrees
        self.goal_range_low = np.array([0.3, -0.5, 0])  # table width, table length, height
        self.goal_range_high = np.array([0.75, 0.5, 0.2])
        self.action_weight = -1
        self.collision_weight = -500
        self.success_reward = 200
        self.distance_weight = -70
        self.orientation_weight = -30
        self.delta = 0.2
        self.collision = False
        self.link_dist = np.zeros(5)
        self.test_goal = np.zeros(6)
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=2.0, yaw=60, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-1.04)
        self.sim.create_table(length=1.1, width=1.8, height=0.92, x_offset=0.5, z_offset=-0.12)
        self.sim.create_track(length=0.2, width=1.1, height=0.12, x_offset=0.0, z_offset=0.0)
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * 0.05 / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=np.array([0.225, 0.5, 0.1]),
            mass=0.0,
            ghost=True,
            position=np.array([0.525, 0.0, 0.1]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array(self.goal)


    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        ee_orientation = np.array(self.get_ee_orientation())
        #ee_orientation =np.zeros(3)
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])

    def set_goal(self, test_goal: np.ndarray) -> None:
        self.goal = test_goal
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])

    def _sample_goal(self) -> np.ndarray:
        goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        goal_rot = sample_euler_constrained()
        goal = np.concatenate((goal_pos, goal_rot))
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        distance_success = distance2(achieved_goal[:3], desired_goal[:3]) < self.distance_threshold
        orientation_success = angular_distance(achieved_goal, desired_goal) < self.ori_distance_threshold
        return np.array(distance_success & orientation_success, dtype=np.bool_)

    """
    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        return self.collision
    """

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        dist = distance2(achieved_goal[:3], desired_goal[:3])
        ori_dist = angular_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return np.where(self.is_success(achieved_goal, desired_goal), 0.0, -1.0)
        else:  # dense
            reward = np.float64(0.0)
            reward += np.where(self.is_success(achieved_goal, desired_goal), self.success_reward, 0)
            reward += dist * self.distance_weight
            reward += ori_dist * self.orientation_weight
            #reward += self.collision_weight if self.collision else 0
            return reward
