"""Policy client for connecting to inference server."""

from typing import Any, Dict, Optional

import numpy as np

try:
    from openpi_client import websocket_client_policy
except ImportError:
    websocket_client_policy = None


class PolicyClient:
    """Client for connecting to a policy inference server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
    ):
        """Initialize policy client.

        Args:
            host: Server hostname or IP
            port: Server port
        """
        if websocket_client_policy is None:
            raise ImportError(
                "openpi_client is required. Install with: pip install openpi-client"
            )

        self.host = host
        self.port = port
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
        )

    def get_server_metadata(self) -> Dict[str, Any]:
        """Get server metadata."""
        return self.policy.get_server_metadata()

    def infer(
        self,
        state: np.ndarray,
        images: Dict[str, np.ndarray],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference to get action from observation.

        Args:
            state: 14D robot state vector
                state[0:7]  - joint positions
                state[7]    - gripper (0-1, 1=open)
                state[8:14] - EEF pose (x, y, z, roll, pitch, yaw)
            images: Dict of camera images in RGB format (480, 640, 3)
                Expected keys: observation.image_front, observation.image_wrist, etc.
            prompt: Optional task prompt string

        Returns:
            Action dict from the policy server
        """
        # Build observation dict
        obs = {
            "observation.state": state.astype(np.float32),
        }

        # Add images
        for key, image in images.items():
            if not key.startswith("observation."):
                key = f"observation.{key}"
            obs[key] = image.astype(np.uint8)

        # Add prompt if provided
        if prompt is not None:
            obs["prompt"] = prompt

        # Run inference
        return self.policy.infer(obs)

    def infer_raw(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with raw observation dict.

        Args:
            obs: Full observation dictionary

        Returns:
            Action dict from the policy server
        """
        return self.policy.infer(obs)


def extract_action_components(
    action: Dict[str, Any],
    action_index: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract action components from policy output.

    The policy returns an action chunk with shape (N, 7) in the 'actions' field.
    Each action row is: [x, y, z, roll, pitch, yaw, gripper]

    Args:
        action: Raw action dict from policy with 'actions' field
        action_index: Which action in the chunk to use (default: 0, the first)

    Returns:
        Dict with parsed action components:
            - gripper: scalar gripper value (0-1)
            - eef_pose_euler: 6D EEF pose (x, y, z, roll, pitch, yaw)
            - raw: raw action array (single action, 7D)
            - all_actions: full action chunk (N, 7)
    """
    result = {}

    # Extract actions array from dict
    if "actions" in action:
        all_actions = np.array(action["actions"], dtype=np.float64)
    elif "action" in action:
        all_actions = np.array(action["action"], dtype=np.float64)
    elif isinstance(action, np.ndarray):
        all_actions = action.astype(np.float64)
    else:
        # Try to find action array in dict
        for key in action:
            if isinstance(action[key], (list, np.ndarray)):
                all_actions = np.array(action[key], dtype=np.float64)
                break
        else:
            raise ValueError(f"Cannot extract action from: {action}")

    # Ensure 2D shape
    if all_actions.ndim == 1:
        all_actions = all_actions.reshape(1, -1)

    result["all_actions"] = all_actions

    # Get the specific action to execute
    if action_index >= len(all_actions):
        action_index = len(all_actions) - 1
    single_action = all_actions[action_index]

    result["raw"] = single_action

    # Parse action: [x, y, z, roll, pitch, yaw, gripper]
    if len(single_action) >= 7:
        result["eef_pose_euler"] = single_action[0:6]
        result["gripper"] = float(single_action[6])
    elif len(single_action) == 6:
        # No gripper in action
        result["eef_pose_euler"] = single_action[0:6]
        result["gripper"] = None
    else:
        raise ValueError(f"Unexpected action dimension: {len(single_action)}")

    return result
