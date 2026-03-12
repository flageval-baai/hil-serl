"""
Replay actions from actions.json file using EEF pose or joint position control.

Action format (14 dims):
  action[0:7]  - joint positions (q1-q7)
  action[7]    - gripper: 0-1, where 1=fully open, 0=fully closed
  action[8:14] - EEF pose: (x, y, z, roll, pitch, yaw)
"""

import argparse
import json
import time
from typing import Optional

import numpy as np

from inference.robot_client import RobotClient, DEFAULT_SERVER_URL
from franka_env.utils.rotations import euler_2_quat


def load_actions_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_eef_and_gripper(action):
    """Extract (eef_pose_quat[7], gripper) from action."""
    if len(action) == 14:
        gripper, xyz, rpy = action[7], action[8:11], action[11:14]
    else:
        gripper, xyz, rpy = action[6], action[0:3], action[3:6]
    quat = euler_2_quat(np.array(rpy))
    return list(xyz) + quat.tolist(), gripper


def extract_joints_and_gripper(action):
    """Extract (joints[7], gripper) from action."""
    return action[:7], action[7] if len(action) > 7 else None


def wait_for_joints(robot, target, timeout=5.0, tol=0.01):
    """Wait until robot joints converge to target."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.1)
        try:
            q = robot.get_joint_positions()
            if max(abs(c - t) for c, t in zip(q, target)) < tol:
                return True
        except Exception:
            pass
    return False


def replay_actions_json(
    json_path: str,
    control_mode: str = "eef",
    hz: float = 30.0,
    server_url: str = DEFAULT_SERVER_URL,
    gripper_threshold: float = 0.9,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
):
    robot = RobotClient(server_url=server_url)
    data = load_actions_json(json_path)
    metadata = data.get("metadata", {})
    frames = data.get("frames", [])

    if not frames:
        print("No frames to replay!")
        return

    # Use --hz if given, otherwise fall back to file metadata fps
    file_fps = metadata.get("fps", 30)
    delay = 1.0 / hz

    if end_frame is None:
        end_frame = len(frames)
    frames_to_play = frames[start_frame:end_frame]
    num_frames = len(frames_to_play)

    print(f"Loaded {len(frames)} frames (fps={file_fps}) from {json_path}")
    print(f"Replaying frames {start_frame}-{end_frame} ({num_frames} frames) "
          f"at {hz:.1f} Hz, control_mode={control_mode}")

    # Switch control mode if needed
    current_mode = robot.get_control_mode()
    if current_mode != control_mode:
        if control_mode == "joint":
            robot.start_joint_control()
        else:
            robot.start_eef_control()
    else:
        print(f"Already in {control_mode} mode, skipping controller switch.")
    time.sleep(2)

    # Move to start position
    first_action = frames_to_play[0].get("action", [])
    if control_mode == "joint" and len(first_action) >= 7:
        target = first_action[:7]
        print(f"Moving to start joints: [{', '.join(f'{j:.3f}' for j in target)}]")
        robot.goto_joints(target)
        wait_for_joints(robot, target)
    else:
        pose, _ = extract_eef_and_gripper(first_action)
        print("Moving to start pose...")
        robot.goto_pose(pose)
        time.sleep(2)
    print("Ready.")

    last_gripper_cmd = None
    try:
        for i, frame in enumerate(frames_to_play):
            step_start = time.time()
            action = frame.get("action", [])
            frame_idx = frame.get("frame_index", start_frame + i)

            if control_mode == "joint":
                joints, gripper_value = extract_joints_and_gripper(action)
                robot.goto_joints(joints)
                print(f"Frame {i+1}/{num_frames} (idx={frame_idx}) | "
                      f"Joints=[{', '.join(f'{j:.3f}' for j in joints)}] | "
                      f"Gripper={gripper_value:.2f}" if gripper_value is not None
                      else f"Frame {i+1}/{num_frames} (idx={frame_idx}) | "
                           f"Joints=[{', '.join(f'{j:.3f}' for j in joints)}]")
            else:
                pose, gripper_value = extract_eef_and_gripper(action)
                robot.goto_pose(pose)
                xyz = action[8:11] if len(action) == 14 else action[0:3]
                rpy = action[11:14] if len(action) == 14 else action[3:6]
                print(f"Frame {i+1}/{num_frames} (idx={frame_idx}) | "
                      f"XYZ=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] | "
                      f"RPY=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}] | "
                      f"Gripper={gripper_value:.2f}")

            if gripper_value is not None:
                cmd = 1 if gripper_value >= gripper_threshold else 0
                if cmd != last_gripper_cmd:
                    robot.open_gripper() if cmd == 1 else robot.close_gripper()
                    last_gripper_cmd = cmd

            elapsed = time.time() - step_start
            if elapsed < delay:
                time.sleep(delay - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user")

    print("Replay complete!")


def main():
    parser = argparse.ArgumentParser(description="Replay robot actions from actions.json.")
    parser.add_argument("json_path", nargs="?", default="actions.json",
                        help="Path to actions.json file")
    parser.add_argument("--control-mode", default="eef", choices=["eef", "joint"],
                        help="Control mode (default: eef)")
    parser.add_argument("--hz", type=float, default=None,
                        help="Replay frequency in Hz (default: use file fps)")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                        help=f"Robot server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--gripper-threshold", type=float, default=0.9,
                        help="Gripper threshold (default: 0.9)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    args = parser.parse_args()

    # Determine replay rate: --hz flag > file metadata fps > 30
    data = load_actions_json(args.json_path)
    file_fps = data.get("metadata", {}).get("fps", 30)
    hz = args.hz if args.hz is not None else file_fps

    replay_actions_json(
        json_path=args.json_path,
        control_mode=args.control_mode,
        hz=hz,
        server_url=args.server_url,
        gripper_threshold=args.gripper_threshold,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
