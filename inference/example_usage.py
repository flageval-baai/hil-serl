#!/usr/bin/env python3
"""
Example showing how to use the inference module programmatically.

This demonstrates:
1. Getting robot observations (state + images)
2. Sending observations to a policy server
3. Executing returned actions on the robot
"""

import time
import numpy as np

from inference.robot_client import RobotClient
from inference.camera_client import MultiCameraClient, CAMERA_SERIALS
from inference.policy_client import PolicyClient, extract_action_components
from franka_env.utils.rotations import euler_2_quat


def main():
    # Configuration
    ROBOT_URL = "http://127.0.0.2:5000"
    POLICY_HOST = "localhost"
    POLICY_PORT = 8000
    PROMPT = "B1a_Put_the_building_blocks_into_the_basket_3camera"

    # Initialize clients
    print("Initializing robot client...")
    robot = RobotClient(server_url=ROBOT_URL)

    print("Initializing camera client...")
    cameras = MultiCameraClient(
        camera_configs=CAMERA_SERIALS,
        width=640,
        height=480,
        fps=30,
    )
    print(f"Active cameras: {cameras.camera_names}")

    print("Initializing policy client...")
    policy = PolicyClient(host=POLICY_HOST, port=POLICY_PORT)

    # Print server metadata
    try:
        metadata = policy.get_server_metadata()
        print(f"Policy server metadata: {metadata}")
    except Exception as e:
        print(f"[WARN] Could not get server metadata: {e}")

    # Reset robot
    print("Resetting robot...")
    robot.reset()
    time.sleep(5.0)

    # Main inference loop
    print("Starting inference loop...")
    hz = 10.0
    delay = 1.0 / hz
    max_steps = 100
    gripper_threshold = 0.5
    last_gripper_cmd = None

    try:
        for step in range(max_steps):
            # 1. Get robot state (14D)
            # Format: [joint(7), gripper(1), eef_xyz(3), eef_rpy(3)]
            state = robot.get_observation_state()
            print(f"State shape: {state.shape}")
            print(f"  Joints: {state[0:7]}")
            print(f"  Gripper: {state[7]:.3f}")
            print(f"  EEF XYZ: {state[8:11]}")
            print(f"  EEF RPY: {state[11:14]}")

            # 2. Get camera images (RGB, 480x640x3)
            images = cameras.get_observation_images()
            for name, img in images.items():
                print(f"  {name}: shape={img.shape}, dtype={img.dtype}")

            # 3. Run policy inference
            # Returns dict with 'actions' shape (N, 7) where each row is [x, y, z, r, p, y, gripper]
            action = policy.infer(
                state=state,
                images=images,
                prompt=PROMPT,
            )
            print(f"Action keys: {action.keys()}")

            # 4. Parse action components to get the full chunk
            components = extract_action_components(action, action_index=0)
            all_actions = components.get("all_actions", np.array([]))
            print(f"Received action chunk with {len(all_actions)} actions")

            # 5. Execute ALL actions in the chunk
            for i, action_vec in enumerate(all_actions):
                action_start = time.time()

                # Extract pose and gripper from action vector
                xyz = action_vec[:3]
                rpy = action_vec[3:6]
                gripper_value = action_vec[6] if len(action_vec) > 6 else None

                # Convert euler to quaternion and send pose
                quat = euler_2_quat(rpy)
                pose_quat = np.concatenate([xyz, quat])
                robot.goto_pose(pose_quat.tolist())

                # Handle gripper with hysteresis
                if gripper_value is not None:
                    gripper_cmd = 1 if gripper_value >= gripper_threshold else 0
                    if gripper_cmd != last_gripper_cmd:
                        if gripper_cmd == 1:
                            robot.open_gripper()
                        else:
                            robot.close_gripper()
                        last_gripper_cmd = gripper_cmd

                print(
                    f"  Action {i + 1}/{len(all_actions)} | "
                    f"XYZ=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] | "
                    f"Gripper={gripper_value:.2f if gripper_value else 0:.2f}"
                )

                # Maintain control rate for each action
                elapsed = time.time() - action_start
                if elapsed < delay:
                    time.sleep(delay - elapsed)

            print(f"Step {step + 1}/{max_steps} complete - executed {len(all_actions)} actions")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cameras.close()
        print("Done!")


if __name__ == "__main__":
    main()
