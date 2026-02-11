"""Robot client for Franka arm control and state retrieval."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from franka_env.utils.rotations import euler_2_quat, quat_2_euler


DEFAULT_SERVER_URL = "http://127.0.0.2:5000"


class RobotClient:
    """Client for interacting with the Franka robot server."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: float = 1.0):
        """Initialize robot client.

        Args:
            server_url: URL of the robot server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def get_state(self) -> Dict:
        """Get full robot state from server.

        Returns:
            Dict with keys: pose (7D quat), vel, force, torque, q (joints), dq, jacobian, gripper_pos
        """
        url = f"{self.server_url}/getstate"
        try:
            resp = requests.post(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get robot state: {e}") from e

    def get_observation_state(self) -> np.ndarray:
        """Get robot state in the observation format.

        Returns:
            14D state vector:
                state[0:7]  - joint positions (q)
                state[7]    - gripper (0-1, 1=open, 0=closed)
                state[8:14] - EEF pose (x, y, z, roll, pitch, yaw)
        """
        state = self.get_state()

        # Joint positions (7D)
        q = np.array(state["q"], dtype=np.float64)

        # Gripper position already normalized to 0-1 by gripper server
        # Franka: gripper_pos = np.sum(msg.position) / 0.08
        # Robotiq: gripper_pos = 1 - msg.gPO / 255
        # Both: 1=open, 0=closed
        gripper = np.clip(state["gripper_pos"], 0.0, 1.0)

        # EEF pose: convert quaternion to euler
        pose_quat = np.array(state["pose"], dtype=np.float64)
        xyz = pose_quat[:3]
        quat = pose_quat[3:7]  # xyzw format
        rpy = quat_2_euler(quat)  # roll, pitch, yaw

        # Combine into 14D state
        obs_state = np.concatenate([q, [gripper], xyz, rpy])
        return obs_state.astype(np.float64)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions (7D)."""
        state = self.get_state()
        return np.array(state["q"], dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities (7D).

        Returns:
            7D array of joint velocities (dq)
        """
        state = self.get_state()
        return np.array(state["dq"], dtype=np.float64)

    def get_eef_velocity(self) -> np.ndarray:
        """Get current end-effector velocity (6D).

        Returns:
            6D array [vx, vy, vz, wx, wy, wz] (linear + angular velocity)
        """
        state = self.get_state()
        return np.array(state["vel"], dtype=np.float64)

    def get_gripper_position(self) -> float:
        """Get current gripper position (0-1, 1=open)."""
        state = self.get_state()
        return float(np.clip(state["gripper_pos"], 0.0, 1.0))

    def get_eef_pose_quat(self) -> np.ndarray:
        """Get current EEF pose as xyz + quaternion (7D)."""
        state = self.get_state()
        return np.array(state["pose"], dtype=np.float64)

    def get_eef_pose_euler(self) -> np.ndarray:
        """Get current EEF pose as xyz + RPY (6D)."""
        pose_quat = self.get_eef_pose_quat()
        xyz = pose_quat[:3]
        rpy = quat_2_euler(pose_quat[3:7])
        return np.concatenate([xyz, rpy])

    def goto_pose(self, pose: Sequence[float]) -> None:
        """Send EEF pose command.

        Args:
            pose: 7D pose [x, y, z, qx, qy, qz, qw]
        """
        url = f"{self.server_url}/pose"
        message = {"arr": list(pose)}
        try:
            requests.post(url, json=message, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] pose request failed: {e}")

    def goto_pose_euler(self, pose_euler: Sequence[float]) -> None:
        """Send EEF pose command with euler angles.

        Args:
            pose_euler: 6D pose [x, y, z, roll, pitch, yaw]
        """
        xyz = np.array(pose_euler[:3])
        rpy = np.array(pose_euler[3:6])
        quat = euler_2_quat(rpy)
        pose_quat = np.concatenate([xyz, quat])
        self.goto_pose(pose_quat.tolist())

    def open_gripper(self) -> None:
        """Open the gripper."""
        url = f"{self.server_url}/open_gripper"
        try:
            requests.post(url, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] open gripper request failed: {e}")

    def close_gripper(self) -> None:
        """Close the gripper."""
        url = f"{self.server_url}/close_gripper"
        try:
            requests.post(url, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] close gripper request failed: {e}")

    def set_gripper(self, position: float, threshold: float = 0.5) -> None:
        """Set gripper based on position value.

        Args:
            position: Gripper position 0-1 (1=open, 0=closed)
            threshold: Threshold for open/close decision
        """
        if position >= threshold:
            self.open_gripper()
        else:
            self.close_gripper()

    def move_gripper(self, position: int) -> None:
        """Move gripper to specific position (for Robotiq gripper).

        Args:
            position: Gripper position 0-255
        """
        url = f"{self.server_url}/move_gripper"
        message = {"gripper_pos": position}
        try:
            requests.post(url, json=message, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] move gripper request failed: {e}")

    def reset(
        self,
        max_s: float = 10.0,
        hz: float = 30.0,
        pos_tol: float = 0.01,
        rot_tol_rad: float = 0.2,
    ) -> None:
        """Reset robot to default pose and open gripper."""
        url = f"{self.server_url}/reset_all"
        payload = {
            "max_s": float(max_s),
            "hz": float(hz),
            "pos_tol": float(pos_tol),
            "rot_tol_rad": float(rot_tol_rad),
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"[WARN] reset_all failed: {resp.status_code} {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] reset request failed: {e}")

    def clear_error(self) -> None:
        """Clear robot error state."""
        url = f"{self.server_url}/clearerr"
        try:
            requests.post(url, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] clear error request failed: {e}")
