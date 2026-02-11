"""
Replay actions from actions.json file using EEF pose control.

Action format (14 dims):
  action[0:7]  - joint pose (not used for EEF control)
  action[7]    - gripper: 0-1, where 1=fully open, 0=fully closed
  action[8:14] - EEF pose: (x, y, z, roll, pitch, yaw)
"""

import argparse
import json
import time
from typing import List, Optional

import numpy as np

from inference.robot_client import RobotClient, DEFAULT_SERVER_URL
from franka_env.utils.rotations import euler_2_quat


def load_actions_json(json_path: str) -> dict:
    """Load actions.json file."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_eef_pose_and_gripper(action: List[float]) -> tuple:
    """Extract EEF pose and gripper from action vector.

    Args:
        action: 14D action vector
            [0:7] = joint pose
            [7] = gripper (0-1, 1=open)
            [8:14] = EEF pose (x, y, z, roll, pitch, yaw)

    Returns:
        (eef_pose_quat, gripper_value): 7D pose with quaternion and gripper value
    """
    gripper = action[7]
    xyz = action[8:11]
    rpy = action[11:14]

    # Convert RPY to quaternion
    quat = euler_2_quat(np.array(rpy))

    # Combine xyz + quaternion
    eef_pose_quat = list(xyz) + quat.tolist()

    return eef_pose_quat, gripper


def replay_actions_json(
    json_path: str,
    delay: float = 0.033,  # ~30Hz by default
    server_url: str = DEFAULT_SERVER_URL,
    reset_wait: float = 5.0,
    gripper_threshold: float = 0.9,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Replay actions from actions.json using EEF pose control.

    Args:
        json_path: Path to actions.json file
        delay: Delay between frames in seconds
        server_url: Robot server URL
        reset_wait: Time to wait after reset
        gripper_threshold: Threshold for gripper open/close
        start_frame: Start from this frame index
        end_frame: End at this frame index (exclusive), None for all
    """
    # Initialize robot client
    robot = RobotClient(server_url=server_url)

    # Load actions
    data = load_actions_json(json_path)

    metadata = data.get("metadata", {})
    frames = data.get("frames", [])

    num_total_frames = len(frames)
    fps = metadata.get("fps", 30)

    print(f"Loaded {num_total_frames} frames from {json_path}")
    print(f"Metadata: {metadata}")

    if not frames:
        print("No frames to replay!")
        return

    # Apply frame range
    if end_frame is None:
        end_frame = num_total_frames
    frames_to_play = frames[start_frame:end_frame]
    num_frames = len(frames_to_play)

    print(f"Replaying frames {start_frame} to {end_frame} ({num_frames} frames)")

    # Reset robot first
    print("Resetting robot...")
    robot.reset()
    print(f"Waiting {reset_wait}s for reset to complete...")
    time.sleep(reset_wait)
    print(f"Starting replay at {1/delay:.1f} Hz...")

    last_gripper_cmd = None

    for i, frame in enumerate(frames_to_play):
        action = frame.get("action", [])

        if len(action) < 14:
            print(f"[WARN] Frame {i}: action has {len(action)} dims, expected 14. Skipping.")
            continue

        # Extract EEF pose and gripper
        eef_pose_quat, gripper_value = extract_eef_pose_and_gripper(action)

        # Send EEF pose command
        robot.goto_pose(eef_pose_quat)

        # Send gripper command with hysteresis to avoid jitter
        gripper_cmd = 1 if gripper_value >= gripper_threshold else 0

        if gripper_cmd != last_gripper_cmd:
            if gripper_cmd == 1:
                robot.open_gripper()
            else:
                robot.close_gripper()
            last_gripper_cmd = gripper_cmd

        # Print progress
        frame_idx = frame.get("frame_index", start_frame + i)
        xyz = action[8:11]
        rpy = action[11:14]
        print(
            f"Frame {i+1}/{num_frames} (idx={frame_idx}) | "
            f"XYZ=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] | "
            f"RPY=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}] | "
            f"Gripper={gripper_value:.2f}"
        )

        time.sleep(delay)

    print("Replay complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Replay robot actions from actions.json using EEF pose control."
    )
    parser.add_argument(
        "json_path",
        type=str,
        nargs="?",
        default="actions.json",
        help="Path to actions.json file (default: actions.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.033,
        help="Delay between frames in seconds (default: 0.033 for ~30Hz)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"Robot server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--reset-wait",
        type=float,
        default=3.0,
        help="Time to wait after reset in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.9,
        help="Gripper threshold: >= threshold = open (default: 0.9)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start from this frame index (default: 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End at this frame index (exclusive, default: all frames)",
    )

    args = parser.parse_args()

    replay_actions_json(
        json_path=args.json_path,
        delay=args.delay,
        server_url=args.server_url,
        reset_wait=args.reset_wait,
        gripper_threshold=args.gripper_threshold,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
