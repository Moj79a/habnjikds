import numpy as np
from panda_gym.envs.core import Task
from typing import Any, Dict
import os
from pyquaternion import Quaternion
from panda_gym.utils import *


class Reach2(Task):
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
        self.obs_range_low = np.array([0.5, -0.8, 0.25])  # table width, table length, height
        self.obs_range_high = np.array([1.2, 0.8, 0.75])
        
        # margin and weight
        self.distance_threshold = 0.05  # 5cm
        self.ori_distance_threshold = 0.0873  # 5 degrees
        self.action_weight = -1
        self.collision_weight = -600
        self.distance_weight = -70
        self.orientation_weight = -30
        link_weights = [ 2.7, 2, 2, 3 , 1.3 , 1.2]
        self.dist_change_weight = np.array(link_weights) / np.sum(link_weights) * 50
        self.success_weight = 450
        self.joint_velocity_limits = np.array([2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14])
        self.joint_acceleration_limits = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self.velocity_penalty_weight = -0.1
        self.acceleration_penalty_weight = -0.0005
        
        # Stored values
        self.last_estimated_velocity = np.zeros(7) # برای ۷ مفصل
        self.obstacle_start = np.zeros(6)
        self.obstacle_end = np.zeros(6)
        self.velocity = np.zeros(6)
        self.collision = False
        self.link_dist = np.zeros(6)
        self.last_dist = np.zeros(6)
        self.step_num = 0

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
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        self.sim.create_cylinder(
            body_name="obstacle",
            radius=0.05,
            height=0.4,
            mass=0.0,
            ghost=False,
            position=np.array([0.0, 0.0, 1.0]),
            rgba_color=np.array([1.0, 0.92, 0.8, 1.0]),
        )
        self.sim.create_box(
            body_name="zone_goal",
            half_extents=(self.goal_range_high - self.goal_range_low) / 2,
            mass=0.0,
            ghost=True,
            position=(self.goal_range_high + self.goal_range_low) / 2,
            rgba_color=np.array([1.0, 1.0, 1.0, 0.3]),
        )
        self.sim.create_box(
            body_name="zone_obs",
            half_extents=(self.obs_range_high - self.obs_range_low) / 2,
            mass=0.0,
            ghost=True,
            position=(self.obs_range_high + self.obs_range_low) / 2,
            rgba_color=np.array([1.0, 1.0, 1.0, 0.2]),
        )


    def get_obs(self) -> np.ndarray:
        obstacle_position = self.sim.get_base_position("obstacle")
        obstacle_rotation = self.sim.get_base_rotation("obstacle")
        obstacle_current = np.concatenate((obstacle_position, obstacle_rotation))
        return np.concatenate((self.goal, obstacle_current, self.velocity, self.link_dist))

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        ee_orientation = np.array(self.get_ee_orientation())
        #ee_orientation =np.zeros(3)
        return np.concatenate((ee_position, ee_orientation))

    def reset(self) -> None:
        self.collision = False
        self.step_num = 0
        distance_fail = True
        while distance_fail:
            self.goal = self._sample_goal()
            self.obstacle_start = self._sample_obstacle()
            self.obstacle_end = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
            self.sim.set_base_pose("obstacle", self.obstacle_end[:3], self.obstacle_end[3:])
            start_end_dist = distance(self.obstacle_end, self.obstacle_start)
            distance_fail = (self.sim.get_target_to_obstacle_distance() < 0.1) or (start_end_dist < 1)

        # set obstacle to start position after checking
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist
        if self.collision:
            print("Collision after reset, this should not happen")
    
    def reset_generate(self, i, j, k) -> None:
        self.collision = False
        distance_fail = True
        while distance_fail:
            goal_pos = np.array([self.goal_range_low[0] + i * 0.05, self.goal_range_low[1] + j * 0.05, self.goal_range_low[2] + k * 0.05])

            goal_rot = sample_euler_constrained()
            goal = np.concatenate((goal_pos, goal_rot))
            self.goal = goal

            self.obstacle_start = self._sample_obstacle()
            self.obstacle_end = self._sample_obstacle()
            self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
            self.sim.set_base_pose("obstacle", self.obstacle_end[:3], self.obstacle_end[3:])
            start_end_dist = distance(self.obstacle_end, self.obstacle_start)
            distance_fail = (self.sim.get_target_to_obstacle_distance() < 0.1) or (start_end_dist < 0.3)


    def set_goal_and_obstacle(self, test_data):
        self.goal = test_data[:6]
        self.obstacle_start = test_data[6:12]
        self.obstacle_end = test_data[12:]

        # set the rotation for obstacle to same value for linear movement
        self.sim.set_base_pose("target", self.goal[:3], self.goal[3:])
        # set final pose to keep the environment constant
        self.sim.set_base_pose("obstacle", self.obstacle_start[:3], self.obstacle_start[3:])
        self.collision = self.sim.check_collision()
        self.link_dist = self.sim.get_link_distances()
        self.last_dist = self.link_dist

    def _sample_goal(self) -> np.ndarray:
        goal_pos = np.array(self.np_random.uniform(self.goal_range_low, self.goal_range_high))
        goal_rot = sample_euler_constrained()
        goal = np.concatenate((goal_pos, goal_rot))
        return goal

    def _sample_obstacle(self):
        obstacle_pos = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
        obstacle_rot = sample_euler_obstacle()
        obstacle = np.concatenate((obstacle_pos, obstacle_rot))
        return obstacle

    def set_velocity(self):
        """
        This function is used to control the movement of obstacle
        The obstacle moves in a constant speed from start pose to end pose
        When end pose is achieved, speed is set to zero
        """

        if self.step_num < 25:
            time_duration = 2
            linear_velocity = (self.obstacle_end[:3] - self.obstacle_start[:3]) / time_duration
            # Calculating the relative rotation from start to end orientation
            rot_end = self.sim.euler_to_quaternion(self.obstacle_end[3:])
            rot_start = self.sim.euler_to_quaternion(self.obstacle_start[3:])
            relative_rotation = self.sim.get_quaternion_difference(rot_start, rot_end)
            # Convert the relative rotation quaternion to axis-angle representation
            axis, angle = self.sim.get_axis_angle(relative_rotation)
            # Calculate the angular velocity required to achieve the rotation in 1 second
            angular_velocity = np.array(axis) * angle / time_duration
            self.sim.set_velocity("obstacle", linear_velocity, angular_velocity)
            self.velocity = np.concatenate((linear_velocity, angular_velocity))
        else:
            linear_velocity = np.zeros(3)
            angular_velocity = np.zeros(3)
            self.sim.set_velocity("obstacle", linear_velocity, angular_velocity)
            self.velocity = np.concatenate((linear_velocity, angular_velocity))
        self.step_num += 1


    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        distance_success = distance2(achieved_goal[:3], desired_goal[:3]) < self.distance_threshold
        orientation_success = angular_distance(achieved_goal, desired_goal) < self.ori_distance_threshold
        return np.array(distance_success & orientation_success, dtype=np.bool_)


    def check_collision(self) -> bool:
        self.collision = self.sim.check_collision()
        return self.collision


    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {} , **kwargs) -> np.ndarray:
        estimated_velocity = kwargs.get("estimated_velocity", np.zeros(7))
        dist = distance2(achieved_goal[:3], desired_goal[:3])
        ori_dist = angular_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return np.where(self.is_success(achieved_goal, desired_goal), 0.0, -1.0)
        else:  # dense
            """"collision reward"""
            if self.collision:
                return np.float64(self.collision_weight)
            """success reward"""
            if self.is_success(achieved_goal, desired_goal):
                return np.float64(self.success_weight)
            """distance reward"""
            dist = distance2(achieved_goal, desired_goal)
            ori_dist = angular_distance(achieved_goal, desired_goal)
            reward = np.float64(0.0)
            reward += self.distance_weight * dist

            """orientation reward"""
            reward += self.orientation_weight * ori_dist

            """obstacle distance reward"""
            self.link_dist = self.sim.get_link_distances()
            dist_change = self.link_dist - self.last_dist
            self.last_dist = self.link_dist
            reward_changes = np.where(self.link_dist < 0.2, self.dist_change_weight * dist_change, 0)
            reward += reward_changes.sum()

            velocity_over_limit = np.maximum(0, np.abs(estimated_velocity) - self.joint_velocity_limits)
            velocity_penalty = self.velocity_penalty_weight * np.sum(velocity_over_limit)
            reward += velocity_penalty
            if self.step_num > 0:
                dt = 1.0 / 125
                acceleration = (estimated_velocity - self.last_estimated_velocity) / dt
                
                acceleration_over_limit = np.maximum(0, np.abs(acceleration) - self.joint_acceleration_limits)
                acceleration_penalty = self.acceleration_penalty_weight * np.sum(acceleration_over_limit)
                reward += acceleration_penalty
            # به‌روزرسانی سرعت قبلی برای گام بعدی
            self.last_estimated_velocity = estimated_velocity.copy()
            print(f"Vel Penalty: {velocity_penalty:.4f} | Accel Penalty: {acceleration_penalty:.4f} | Distance Reward: {float(self.distance_weight * dist):.4f} | Ori Reward: {float(self.orientation_weight * ori_dist):.4f}")

            return reward
