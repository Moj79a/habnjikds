from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import os
import csv
from panda_gym.pybullet import PyBullet


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_space: spaces.Space,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)


    def get_link_orientation(self, link: int) -> np.ndarray:
        """Returns the orientation of a link as (x, y, z, w)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Rotation as (x, y, z, w)
        """
        return self.sim.get_link_orientation(self.body_name, link, "euler")

    def get_link_angular_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (wx, wy, wz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Angular Velocity as (wx, wy, wz)
        """
        return self.sim.get_link_angular_velocity(self.body_name, link)


    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {} , **kwargs) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: PyBulletRobot,
        task: Task,
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.render_mode = self.sim.render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task


        # === START: FINAL FIX FOR ValueError ===
        self.log_filepath = "observer_log_cumulative.csv"
        file_exists = os.path.exists(self.log_filepath)
        self.log_file_handler = open(self.log_filepath, "a", newline="")
        # The header now includes ALL keys that will be logged in the step method.
        header = ["time", "episode"] + [f"pos_true_j{i}" for i in range(7)] +  [f"pos_est_j{i}" for i in range(7)] +  [f"vel_true_j{i}" for i in range(7)] + [f"vel_est_j{i}" for i in range(7)]
        self.csv_writer = csv.DictWriter(self.log_file_handler, fieldnames=header)
        if not file_exists:
            self.csv_writer.writeheader()
        self.time = 0.0
        self.episode_num = 0
        # === END: FINAL FIX FOR ValueError ===
        
        
        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
            )
        )
        
        self.action_space = self.robot.action_space

        print("--- DEBUG: 'log_data' has been successfully initialized. ---")
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
                # We no longer open/close the file here. We just reset the timers and counters.
        self.time = 0.0
        self.episode_num += 1 # Increment episode counter
        # === END: MODIFIED RESET LOGIC ===
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        if hasattr(self, 'spec') and self.spec and self.spec.id == "PandaReach2-v3": self.task.set_velocity()
        elif hasattr(self, 'spec') and self.spec and self.spec.id == "PandaReach1-v3" and (hasattr(self.task, 'obstacle_end') and not np.array_equal(self.task.obstacle_end, np.zeros(6))): self.task.set_velocity()
        self.sim.step()
        
        # LOGGING LOGIC IS PERFORMED FIRST
        self.robot._update_observer()
        num_movable_joints = 7
        robot_id = self.sim.body_name_to_id[self.robot.body_name]
        joint_states = self.sim.physics_client.getJointStates(robot_id, self.robot.joint_indices[:num_movable_joints])
        true_pos = [s[0] for s in joint_states]
        true_vel = [s[1] for s in joint_states]
        est_pos = self.robot.get_estimated_joint_positions() 
        est_vel = self.robot.get_estimated_joint_velocities()
        step_log = {'time': self.time, 'episode': self.episode_num}
        for j in range(num_movable_joints):
            step_log[f'pos_true_j{j}'] = true_pos[j]
            step_log[f'pos_est_j{j}'] = est_pos[j]
            step_log[f'vel_true_j{j}'] = true_vel[j]
            step_log[f'vel_est_j{j}'] = est_vel[j]
        if self.csv_writer: self.csv_writer.writerow(step_log)
        self.time += self.sim.dt

        # THEN WE COMPUTE THE RETURN VALUES
        observation = self._get_obs()
        collision = self.task.check_collision()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()) or collision)
        info = {"is_success": not collision if terminated else terminated}
        estimated_velocity = np.array(self.robot.get_estimated_joint_velocities())
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info , estimated_velocity=estimated_velocity ))
        
        return observation, reward, terminated, False, info
    def close(self) -> None:
        super().close()
        if self.log_file_handler is not None:
            self.log_file_handler.close()
            self.log_file_handler = None

    def render(self) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
