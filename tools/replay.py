import h5py
import time
import requests
import numpy as np

from franka_env.utils.rotations import euler_2_quat

def euler_to_quat(euler_angles: np.ndarray) -> np.ndarray:
    """Euler -> quat using the same convention as the robot stack (xyzw)."""
    euler_angles = np.asarray(euler_angles, dtype=np.float64).reshape((3,))
    return euler_2_quat(euler_angles)


def goto_pose(pose):
    """发送机械臂位姿控制指令"""
    url = "http://127.0.0.2:5000/pose"
    message = {"arr": pose}
    try:
        requests.post(url, json=message, timeout=1)
    except requests.exceptions.RequestException as e:
        print(f"[WARN] pose request failed: {e}")

def send_gripper(position):
    """发送夹爪控制指令"""

    if position >= 0.7:
        url = "http://127.0.0.2:5000/open_gripper"
    else:
        url = "http://127.0.0.2:5000/close_gripper"

    try:
        requests.post(url, timeout=1)
    except requests.exceptions.RequestException as e:
        print(f"[WARN] gripper request failed: {e}")
        


def replay_hdf5(hdf5_path, delay=0.05):
    """
    回放 HDF5 数据文件
    :param hdf5_path: 文件路径
    :param delay: 每帧间隔秒数（根据需要调整）
    """
    # 初始化夹爪
    # gripper = RobotiqGripper()
    # gripper.activate()

    



    # 打开HDF5文件
    with h5py.File(hdf5_path, "r") as f:
        # 读取机械臂 pose 与 gripper 数据
        pose_data = f["puppet/pose"][:]      # shape = (N, 6)
        gripper_data = f["puppet/gripper"][:]  # shape = (N, 2)
        joint_data = f["puppet/joint"][:]  # shape = (N, 7)

        num_frames = len(pose_data)
        print(f"🎬 开始回放，共 {num_frames} 帧")

        last_grip_cmd = None
        for i in range(num_frames):
            pose = pose_data[i].tolist()
            grip = gripper_data[i].tolist()
            joint = joint_data[i].tolist()

            moved = pose[0:3] + euler_to_quat(np.array(pose[3:6])).tolist()

            # 控制机械臂
            goto_pose(moved)

            # 控制夹爪开合：加一点滞回/去抖，避免每帧都重复发送开/关导致夹爪“抖”或看起来没响应。
            cmd = 1 if grip[0] >= 0.7 else 0
            if cmd != last_grip_cmd:
                send_gripper(grip[0])
                last_grip_cmd = cmd

            print(f"Frame {i+1}/{num_frames} | Pose={pose} | Gripper={grip} | Joint={joint}")
            time.sleep(delay)

        print("✅ 回放完成")


if __name__ == "__main__":
    # 示例用法
    hdf5_path = "/home/franka/test/plug_charger_20251111/59.hdf5"
    replay_hdf5(hdf5_path, delay=0.05)
