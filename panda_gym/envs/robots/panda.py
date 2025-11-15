from typing import Optional
import pybullet as p
import numpy as np
from gymnasium import spaces
import panda_py as panda
from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        real_robot_ip: Optional[str] = None, # پارامتر جدید
    ) -> None:
        self.real_robot = None
        if real_robot_ip:
            self.real_robot = panda.Panda(172.16.0.2) # اتصال به ربات واقعی
            joint_speed_factor = 0.2
            self.real_robot.move_to_start() # حرکت به حالت اولیه
            print(f"Successfully connected to real robot at {172.16.0.2}")
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        #self.robot_id = p.loadURDF("franka_panda/panda.urdf", base_position, useFixedBase=True)
        #self.ee_link = 11
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.dt = sim.dt
        epsilon = 0.03
        self.H1 = 6 / epsilon  #10 10
        self.H2 = 12 / epsilon**2  #75 0.1
    
        num_movable_joints = 7
        self.observer_z1 = [0.0] * num_movable_joints
        self.observer_z2 = [0.0] * num_movable_joints
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),

        )
        self.position_gains = np.array([0.3, 0.3, 0.25, 0.33, 0.25, 0.25, 0.22]) 
        self.velocity_gains = np.array([1.40, 1.40, 1.3, 1.33, 1.30, 1.30, 1.30]) 
        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)


    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7] * np.pi # map joint velocity from -1~+1 to -pi~+pi
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
            #arm_joint_ctrl = action[:7] * np.pi  # map joint velocity from -1~+1 to -pi~+pi
            #d_joint_angles = arm_joint_ctrl * 0.05 
            #current_arm_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
            #target_arm_angles = current_arm_angles + d_joint_angles

        """
        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)
        """
        real_robot.move_to_joint_posiotion(target_arm_angles)
        self.set_joint_angles(angles=target_arm_angles)
        #self.control_joints(target_angles=target_arm_angles)


    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        state = real_robot.get_state() 
        O_T_EE = np.array(state.O_T_EE).reshape(4, 4)
        ee_position = O_T_EE[:3, 3]
        # ee_velocity را از O_dP_EE_d می‌گیریم
        ee_velocity = np.array(state.O_dP_EE_d[:3])  # فقط سرعت خطی
        joint_angles = np.array(state.q)

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, joint_angles , [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity, joint_angles))
        clipped_observation = np.clip(observation, -10.0, 10.0)
        return observation

    def reset(self) -> None:
        real_joint_angles = self.real_robot.get_joint_positions()
        for i, angle in enumerate(real_joint_angles):
            self.sim.reset_joint_state(self.body_name, self.joint_indices[i], angle)
        num_movable_joints = 7
        self.observer_z1 = [0.0] * num_movable_joints
        self.observer_z2 = [0.0] * num_movable_joints
        initial_joint_angles = [self.get_joint_angle(joint=i) for i in range(num_movable_joints)]
        self.observer_z1 = initial_joint_angles.copy()


    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
        
    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as euler(x, y, z)"""
        return self.sim.get_link_orientation(self.body_name, self.ee_link)
        
    def get_joint_angles(self) -> np.ndarray:
        """Returns the angles of the all 7 joints as (j1, j2, j3, j4, j5, j6, j7)"""
        angles = np.zeros(9)
        for i, ind in enumerate(self.joint_indices):
            angles[i] = self.get_joint_angle(ind)
        return angles

    def get_action(self):
        """Returns the action of the all 7 joints as (a1, a2, a3, a4, a5, a6 ,a7)"""
        return self.action

    # following functions are for OMPL
    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = self.sim.physics_client.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return self.get_joint_angles()

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self.set_joint_angles(state)
        self.state = state

    def _update_observer(self) -> None:
        """Updates the high-gain observer states."""
        # This logic is from 2d.py
        num_movable_joints = 7
        robot_id = self.sim.body_name_to_id[self.body_name]
        joint_states = self.sim.physics_client.getJointStates(robot_id, self.joint_indices[:num_movable_joints])
        current_joint_angles = [s[0] for s in joint_states]
    
        for j in range(num_movable_joints):
            y_measured = current_joint_angles[j]
            z1, z2 = self.observer_z1[j], self.observer_z2[j]
            error = y_measured - z1
            z1_dot = z2 + self.H1 * error
            z2_dot = self.H2 * error
            self.observer_z1[j] += z1_dot * self.dt
            self.observer_z2[j] += z2_dot * self.dt

    def get_estimated_joint_velocities(self) -> list:
        """Returns the current estimated joint velocities from the observer."""
        return self.observer_z2
        
    def get_estimated_joint_positions(self) -> list:
        """Returns the current estimated joint positions from the observer (z1)."""
        return self.observer_z1
        
    def control_joints(self, target_angles: np.ndarray) -> None:
        """
        This method OVERRIDES the generic method in the base class.
        It correctly controls only the 7 arm joints.
        """

        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices[:7],      # Correctly pass only 7 indices
            target_angles=target_angles,        # The target angles are also length 7
            #forces=self.joint_forces[:7],       # The forces are also for 7 joints
            position_gains=self.position_gains,
            velocity_gains=self.velocity_gains,
        )
