"""Main inference loop for policy execution."""

import argparse
import time
from typing import Dict, Optional

import numpy as np

from inference.robot_client import RobotClient
from inference.camera_client import MultiCameraClient, CAMERA_SERIALS
from inference.policy_client import PolicyClient, extract_action_components
from franka_env.utils.rotations import euler_2_quat


class InferenceLoop:
    """Main inference loop for executing a policy on the robot."""

    def __init__(
        self,
        robot_url: str = "http://127.0.0.2:5000",
        policy_host: str = "localhost",
        policy_port: int = 8000,
        camera_configs: Optional[Dict[str, str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        control_mode: str = "eef",
    ):
        """Initialize inference loop.

        Args:
            robot_url: Robot server URL
            policy_host: Policy server hostname
            policy_port: Policy server port
            camera_configs: Camera name to serial number mapping
            camera_width: Camera image width
            camera_height: Camera image height
            camera_fps: Camera frame rate
            control_mode: "eef" for cartesian pose control, "joint" for joint position control
        """
        assert control_mode in ("eef", "joint"), f"Unknown control_mode: {control_mode}"
        self.control_mode = control_mode
        self.robot = RobotClient(server_url=robot_url)
        self.policy = PolicyClient(host=policy_host, port=policy_port)

        # Initialize cameras
        if camera_configs is None:
            camera_configs = CAMERA_SERIALS
        self.cameras = MultiCameraClient(
            camera_configs=camera_configs,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
        )

        self.last_gripper_cmd = None

    def get_observation(self) -> Dict:
        """Get current observation from robot and cameras.

        Returns:
            Observation dict with state and images
        """
        # Get robot state (14D)
        state = self.robot.get_observation_state()

        # Get camera images (RGB, 480x640x3)
        images = self.cameras.get_observation_images()

        return {
            "state": state,
            "images": images,
        }

    def execute_single_action(
        self,
        action_vec: np.ndarray,
        gripper_threshold: float = 0.5,
    ) -> None:
        """Execute a single action on the robot.

        Args:
            action_vec: Action vector.
                EEF mode:   7D [x, y, z, roll, pitch, yaw, gripper]
                Joint mode: 8D [q1, q2, q3, q4, q5, q6, q7, gripper]
            gripper_threshold: Threshold for gripper open/close
        """
        if self.control_mode == "joint":
            # Joint mode: first 7 values are joint positions
            joint_positions = action_vec[:7]
            gripper_value = action_vec[7] if len(action_vec) > 7 else None
            self.robot.goto_joints(joint_positions.tolist())
        else:
            # EEF mode: [x, y, z, roll, pitch, yaw, gripper]
            xyz = action_vec[:3]
            rpy = action_vec[3:6]
            gripper_value = action_vec[6] if len(action_vec) > 6 else None

            quat = euler_2_quat(rpy)
            pose_quat = np.concatenate([xyz, quat])
            self.robot.goto_pose(pose_quat.tolist())

        # Handle gripper with hysteresis
        if gripper_value is not None:
            gripper_cmd = 1 if gripper_value >= gripper_threshold else 0
            if gripper_cmd != self.last_gripper_cmd:
                if gripper_cmd == 1:
                    self.robot.open_gripper()
                else:
                    self.robot.close_gripper()
                self.last_gripper_cmd = gripper_cmd

    def execute_action_chunk(
        self,
        action: Dict,
        gripper_threshold: float = 0.5,
        delay: float = 0.033,
        verbose: bool = False,
    ) -> int:
        """Execute all actions in the action chunk.

        Args:
            action: Action dict from policy (contains 'actions' with shape (N, 7))
            gripper_threshold: Threshold for gripper open/close
            delay: Delay between actions in seconds
            verbose: Print progress for each action

        Returns:
            Number of actions executed
        """
        components = extract_action_components(action, action_index=0)
        all_actions = components.get("all_actions", np.array([]))

        if len(all_actions) == 0:
            return 0

        for i, action_vec in enumerate(all_actions[15:16]):
        # for i, action_vec in enumerate(all_actions):
            step_start = time.time()

            self.execute_single_action(
                action_vec=action_vec,
                gripper_threshold=gripper_threshold,
            )

            if verbose:
                if self.control_mode == "joint":
                    joints = action_vec[:7]
                    gripper = action_vec[7] if len(action_vec) > 7 else 0
                    joints_str = ", ".join(f"{j:.3f}" for j in joints)
                    print(
                        f"  Action {i + 1}/{len(all_actions)} | "
                        f"Joints=[{joints_str}] | "
                        f"Gripper={gripper:.2f}"
                    )
                else:
                    xyz = action_vec[:3]
                    rpy = action_vec[3:6]
                    gripper = action_vec[6] if len(action_vec) > 6 else 0
                    print(
                        f"  Action {i + 1}/{len(all_actions)} | "
                        f"XYZ=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] | "
                        f"RPY=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}] | "
                        f"Gripper={gripper:.2f}"
                    )

            # Maintain control rate
            elapsed = time.time() - step_start
            if elapsed < delay:
                time.sleep(delay - elapsed)

        return len(all_actions)

    def _ensure_control_mode(self, verbose: bool = True):
        """Switch controller mode if needed."""
        current_mode = self.robot.get_control_mode()
        if current_mode != self.control_mode:
            if verbose:
                print(f"Switching from {current_mode} to {self.control_mode} control...")
            if self.control_mode == "joint":
                self.robot.start_joint_control()
            else:
                self.robot.start_eef_control()
        elif verbose:
            print(f"Already in {self.control_mode} mode.")

    def run(
        self,
        prompt: str,
        max_steps: int = 1000,
        hz: float = 10.0,
        gripper_threshold: float = 0.5,
        reset_first: bool = True,
        verbose: bool = True,
    ) -> None:
        """Run the inference loop.

        Args:
            prompt: Task prompt for the policy
            max_steps: Maximum number of steps
            hz: Control frequency in Hz
            gripper_threshold: Threshold for gripper open/close
            reset_first: Reset robot before starting
            verbose: Print step information
        """
        delay = 1.0 / hz

        self._ensure_control_mode(verbose)

        if reset_first:
            if verbose:
                print("Resetting robot...")
            result = self.robot.reset()
            if verbose and result:
                print(f"Reset result: {result}")

        if verbose:
            print(f"Starting inference loop at {hz} Hz (control_mode={self.control_mode})")
            print(f"Prompt: {prompt}")
            print(f"Max steps: {max_steps}")
            try:
                metadata = self.policy.get_server_metadata()
                print(f"Policy server metadata: {metadata}")
            except Exception as e:
                print(f"[WARN] Failed to get server metadata: {e}")

        total_actions = 0
        try:
            for step in range(max_steps):
                # Get observation
                obs = self.get_observation()

                # Run inference
                action = self.policy.infer(
                    state=obs["state"],
                    images=obs["images"],
                    prompt=prompt,
                )
                # save images for debugging
                from PIL import Image
                print(list(obs["state"]))
                for name, img in obs["images"].items():
                    print(img.dtype, img.shape)
                    Image.fromarray(img).save(f"output/images/frame_{step}_{name.replace('observation.','')}.png")
                
                # Get chunk size for logging
                components = extract_action_components(action, action_index=0)
                chunk_size = len(components.get("all_actions", []))

                if verbose:
                    print(f"Step {step + 1}/{max_steps} | Executing {chunk_size} actions...")

                # Execute all actions in the chunk
                num_executed = self.execute_action_chunk(
                    action=action,
                    gripper_threshold=gripper_threshold,
                    delay=delay,
                    verbose=verbose,
                )

                total_actions += num_executed

                if verbose:
                    print(f"Step {step + 1}/{max_steps} complete | Total actions: {total_actions}")

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            if verbose:
                print("Inference loop finished")

    def close(self) -> None:
        """Clean up resources."""
        self.cameras.close()


def main():
    parser = argparse.ArgumentParser(description="Run policy inference on Franka robot")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Task prompt for the policy",
    )
    parser.add_argument(
        "--robot-url",
        type=str,
        default="http://127.0.0.2:5000",
        help="Robot server URL",
    )
    parser.add_argument(
        "--policy-host",
        type=str,
        default="localhost",
        help="Policy server hostname",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=8000,
        help="Policy server port",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=20.0,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.9,
        help="Gripper threshold (>= threshold = open)",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip initial robot reset",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="eef",
        choices=["eef", "joint"],
        help="Control mode: 'eef' for cartesian pose, 'joint' for joint positions",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    loop = InferenceLoop(
        robot_url=args.robot_url,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        control_mode=args.control_mode,
    )

    try:
        loop.run(
            prompt=args.prompt,
            max_steps=args.max_steps,
            hz=args.hz,
            gripper_threshold=args.gripper_threshold,
            reset_first=not args.skip_reset,
            verbose=not args.quiet,
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
