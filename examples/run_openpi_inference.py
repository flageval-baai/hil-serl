#!/usr/bin/env python3
"""
Run a Franka real-robot inference loop using an OpenPI websocket policy server.

This script acts as a bridge:
  1) Read robot state via the existing Franka Flask server.
  2) Read two RealSense RGB streams (front + wrist).
  3) Send an observation dict to the remote OpenPI server via websocket (msgpack+numpy).
  4) Receive an action dict and execute the (absolute) target pose + gripper command.

Observation keys match the provided OpenPI RoboInputs transform:
  - "observation.state": float32[14] = [q(7), gripper(1), pose_xyzrpy(6)] (default)
  - "observation.state": float32[15] = [q(7), gripper(1), pose_xyzquat_xyzw(7)] (optional, see --state_format)
  - "observation.image_front": uint8[H,W,3]
  - "observation.image_wrist": uint8[H,W,3]
  - "prompt": str (optional)

Action is expected to be:
  - action_dict[action_key] = float32[T,D] or float32[D], where D is one of:
      - 7:  [x,y,z,rx,ry,rz,gripper]                 (rpy, legacy)
      - 8:  [x,y,z,qx,qy,qz,qw,gripper]               (quat xyzw)
      - 15: [j1..j7,gripper,x,y,z,qx,qy,qz,qw]        (dataset-style; joints currently ignored)
    where gripper >= threshold triggers open, else close.

Example:
  cd examples
  python run_openpi_inference.py \
    --franka_url http://127.0.0.1:5000/ \
    --openpi_ws ws://<OPENPI_HOST>:<PORT> \
    --front_serial <SERIAL> \
    --wrist_serial <SERIAL> \
    --prompt "Place the object into the basket"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as e:  # pragma: no cover
    rs = None
    _RS_IMPORT_ERROR = e

try:
    from openpi_client import msgpack_numpy as _mpn  # type: ignore
except Exception:
    _mpn = None

try:
    import msgpack_numpy as _mpn_fallback  # type: ignore
except Exception:
    _mpn_fallback = None

try:
    from websockets.asyncio.client import connect as ws_connect  # type: ignore
except Exception:  # pragma: no cover
    try:
        from websockets import connect as ws_connect  # type: ignore
    except Exception as e:  # pragma: no cover
        ws_connect = None
        _WS_IMPORT_ERROR = e


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("franka_url is empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    if not url.endswith("/"):
        url += "/"
    return url


class FrankaHttpClient:
    """Client responsible for Franka control"""
    def __init__(self, base_url: str, *, timeout_s: float = 3.0) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout_s = float(timeout_s)

    def _post_json(self, endpoint: str, payload: Any | None = None, *, timeout_s: Optional[float] = None) -> dict:
        """
        Post JSON payload to Franka Flask server and parse JSON response.
        """
        url = self.base_url + endpoint.lstrip("/")
        # Request the robot
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s or self.timeout_s)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to reach Franka server at {url}: {e}") from e
        # Validate the HTTP status code
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            body = (resp.text or "").strip()
            snippet = body[:500] if body else "<empty body>"
            raise RuntimeError(f"Franka server error calling {url}: HTTP {resp.status_code}. Response: {snippet}") from e
        # Parse the JSON response
        try:
            data = resp.json()
        except Exception as e:
            body = (resp.text or "").strip()
            snippet = body[:500] if body else "<empty body>"
            raise RuntimeError(f"Expected JSON from {url} but got invalid JSON. Response: {snippet}") from e
        # Check for server busy status
        if isinstance(data, dict) and data.get("busy"):
            raise RuntimeError(f"Franka server busy for {endpoint}: {data}")
        return data

    def clear_errors(self) -> None:
        self._post_json("clearerr", payload=None)

    def get_status(self) -> dict:
        return self._post_json("status", payload=None)

    def reset_all(self) -> dict:
        return self._post_json("reset_all", payload={}, timeout_s=5.0)

    def joint_reset(self) -> dict:
        return self._post_json("jointreset", payload={}, timeout_s=5.0)

    def start_joint_control(self) -> None:
        self._post_json("start_joint_control", payload=None, timeout_s=8.0)

    def stop_joint_control(self) -> None:
        self._post_json("stop_joint_control", payload=None, timeout_s=8.0)

    def get_joint_positions(self) -> np.ndarray:
        data = self._post_json("getq", payload=None)
        q = np.asarray(data["q"], dtype=np.float32).reshape((7,))
        return q

    def send_joint_q(self, q_target: np.ndarray) -> None:
        q = np.asarray(q_target, dtype=np.float32).reshape((7,))
        self._post_json("joint_q", payload={"q": q.tolist()})

    def get_pose_euler(self) -> np.ndarray:
        data = self._post_json("getpos_euler", payload=None)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape((6,))
        return pose

    def get_gripper_distance(self) -> float:
        data = self._post_json("get_gripper", payload=None)
        return float(data["gripper"])

    def send_pose_xyz_quat(self, pose_xyz_quat: np.ndarray) -> None:
        arr = np.asarray(pose_xyz_quat, dtype=np.float32).reshape((7,))
        self._post_json("pose", payload={"arr": arr.tolist()})

    def send_pose_xyzrpy(self, pose_xyzrpy: np.ndarray) -> None:
        pose_xyzrpy = np.asarray(pose_xyzrpy, dtype=np.float32).reshape((6,))
        xyz = pose_xyzrpy[:3]
        rpy = pose_xyzrpy[3:]
        quat_xyzw = R.from_euler("xyz", rpy).as_quat().astype(np.float32)
        self.send_pose_xyz_quat(np.concatenate([xyz, quat_xyzw], axis=0))

    def open_gripper(self) -> None:
        self._post_json("open_gripper", payload=None, timeout_s=5.0)

    def close_gripper(self) -> None:
        self._post_json("close_gripper", payload=None, timeout_s=5.0)


@dataclass
class RealSenseColorCamera:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30

    def __post_init__(self) -> None:
        if rs is None:  # pragma: no cover
            raise RuntimeError(
                f"pyrealsense2 is required for camera capture but could not be imported: {_RS_IMPORT_ERROR}"
            )
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._pipeline.start(cfg)

    def read_bgr(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"RealSense {self.serial}: no color frame available")
        return np.asanyarray(color_frame.get_data(), dtype=np.uint8)

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


def _detect_realsense_serials() -> list[str]:
    if rs is None:  # pragma: no cover
        raise RuntimeError(
            f"pyrealsense2 is required for camera capture but could not be imported: {_RS_IMPORT_ERROR}"
        )
    serials: list[str] = []
    for d in rs.context().devices:
        try:
            serials.append(d.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return serials


def _mpn_module():
    """
    Get a msgpack-numpy module for packing/unpacking.
    """
    if _mpn is not None:
        return _mpn
    if _mpn_fallback is not None:
        return _mpn_fallback
    raise RuntimeError(
        "msgpack numpy support is missing. Install `openpi_client` or `msgpack_numpy` to use this script."
    )


def _extract_action(action_dict: dict, *, action_key: str) -> np.ndarray:
    """Extract action from a dictionary, regardless of shape"""
    if action_key not in action_dict:
        raise KeyError(f"Missing action key '{action_key}' in response keys={list(action_dict.keys())}")
    actions = np.asarray(action_dict[action_key], dtype=np.float32)
    if actions.ndim == 1:
        if actions.shape[0] not in (7, 8, 15):
            raise ValueError(f"Expected action shape (7,), (8,), or (15,), got {actions.shape}")
        return actions[None, :]
    if actions.ndim == 2:
        if actions.shape[1] not in (7, 8, 15):
            raise ValueError(f"Expected action shape (T,7), (T,8), or (T,15), got {actions.shape}")
        return actions
    raise ValueError(f"Unsupported action array rank {actions.ndim} with shape {actions.shape}")


def _normalize_quat_xyzw(quat_xyzw: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape((4,))
    n = float(np.linalg.norm(quat))
    if not np.isfinite(n) or n < eps:
        raise ValueError(f"Invalid quaternion (norm={n}): {quat.tolist()}")
    return (quat / n).astype(np.float32)


def _maybe_flip_quat_sign(quat_xyzw: np.ndarray, prev_quat_xyzw: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape((4,))
    if prev_quat_xyzw is None:
        return quat, quat
    prev = np.asarray(prev_quat_xyzw, dtype=np.float32).reshape((4,))
    if float(np.dot(prev, quat)) < 0.0:
        quat = -quat
    return quat, quat


def _parse_action_step(
    a: np.ndarray,
    *,
    action_format: str,
    quat_normalize: bool,
    quat_flip_sign: bool,
    prev_quat_xyzw: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse one action step into (xyz, rpy, quat_xyzw, gripper, joints7, next_prev_quat_xyzw).
    Exactly one of rpy/quat_xyzw is returned as not-None.
    """
    a = np.asarray(a, dtype=np.float32).reshape((-1,))
    dim = int(a.shape[0])

    if action_format != "auto":
        expected = {"rpy7": 7, "quat8": 8, "quat15": 15, "joint8": 8}[action_format]
        if dim != expected:
            raise ValueError(f"--action_format={action_format} requires action dim {expected}, got {dim}")
    else:
        if dim == 7:
            action_format = "rpy7"
        elif dim == 8:
            action_format = "quat8"
        elif dim == 15:
            action_format = "quat15"
        else:
            raise ValueError(f"Unsupported action dim {dim}; expected 7, 8, or 15")

    joints7: Optional[np.ndarray] = None
    rpy: Optional[np.ndarray] = None
    quat_xyzw: Optional[np.ndarray] = None

    if action_format == "joint8":
        joints7 = a[:7].copy()
        gripper = float(a[7])
        xyz = np.zeros((3,), dtype=np.float32)
        return xyz, None, None, gripper, joints7, prev_quat_xyzw

    if action_format == "rpy7":
        xyz = a[:3]
        rpy = a[3:6]
        gripper = float(a[6])
        return xyz, rpy, None, gripper, None, prev_quat_xyzw

    if action_format == "quat8":
        xyz = a[:3]
        quat_xyzw = a[3:7]
        gripper = float(a[7])
    elif action_format == "quat15":
        joints7 = a[:7].copy()
        gripper = float(a[7])
        xyz = a[8:11]
        quat_xyzw = a[11:15]
    else:
        raise ValueError(f"Unsupported action_format={action_format!r}")

    if quat_xyzw is None:
        raise AssertionError("quat_xyzw must be set for quaternion action formats")
    if quat_normalize:
        quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
    if quat_flip_sign:
        quat_xyzw, prev_quat_xyzw = _maybe_flip_quat_sign(quat_xyzw, prev_quat_xyzw)

    return xyz, None, quat_xyzw, float(gripper), joints7, prev_quat_xyzw


def _maybe_resize_bgr(img: np.ndarray, *, resize_hw: Optional[tuple[int, int]]) -> np.ndarray:
    """Resize BGR image if requested."""
    if resize_hw is None:
        return img
    h, w = resize_hw
    import cv2  # local import: optional dependency in some robot setups

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _maybe_bgr_to_rgb(img: np.ndarray, *, bgr_to_rgb: bool) -> np.ndarray:
    """Convert BGR image to RGB if requested."""
    if not bgr_to_rgb:
        return img
    return img[..., ::-1]


def _maybe_save_obs(
    *,
    enabled: bool,
    every: int,
    run: Optional["_ObsSaver"],
    out_dir: str,
    step_idx: int,
    state: np.ndarray,
    state_spec: str,
    front_img: np.ndarray,
    wrist_img: np.ndarray,
    front_serial: str,
    wrist_serial: str,
    bgr_to_rgb: bool,
    resize_hw: Optional[tuple[int, int]],
    prompt: Optional[str],
    video_fps: float,
) -> Optional["_ObsSaver"]:
    """Save the latest model inputs (overwrite) and record videos (finalize on exit)."""
    if not enabled:
        return run
    if every <= 0:
        return run
    if (step_idx % every) != 0:
        return run

    if run is None:
        base = Path(out_dir).expanduser()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"obs_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "provided_to_model": {
                "observation.state": state_spec,
                "observation.image_front": "front camera image",
                "observation.image_wrist": "wrist camera image",
                "prompt": "optional text prompt",
            },
            "camera_serials": {"front": front_serial, "wrist": wrist_serial},
            "preprocess": {
                "bgr_to_rgb": bool(bgr_to_rgb),
                "resize_hw": list(resize_hw) if resize_hw is not None else None,
                "note": (
                    "This folder is updated in-place during inference: latest .png frames overwrite "
                    "previous ones for easy live observation; videos are finalized on exit."
                ),
            },
        }
        (run_dir / "obs_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[obs] save latest frames + record videos to: {run_dir} (every={every})")
        run = _ObsSaver(
            run_dir=run_dir,
            bgr_to_rgb=bgr_to_rgb,
            video_fps=float(video_fps),
        )

    run.update(step_idx=step_idx, state=state, front_img=front_img, wrist_img=wrist_img, prompt=prompt)
    return run


def _try_create_video_writer(path: Path, *, fps: float, frame_hw: tuple[int, int]):
    try:
        import cv2  # local import

        # cv2.VideoWriter expects (W, H)
        h, w = frame_hw
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)))
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            return None
        return writer
    except Exception:
        return None


@dataclass
class _ObsSaver:
    run_dir: Path
    bgr_to_rgb: bool
    video_fps: float
    _front_writer: Any | None = None
    _wrist_writer: Any | None = None
    _video_ready: bool = False

    def _ensure_video_writers(self, *, front_bgr: np.ndarray, wrist_bgr: np.ndarray) -> None:
        if self._video_ready:
            return

        front_path = self.run_dir / "front.mp4"
        wrist_path = self.run_dir / "wrist.mp4"

        self._front_writer = _try_create_video_writer(front_path, fps=self.video_fps, frame_hw=front_bgr.shape[:2])
        self._wrist_writer = _try_create_video_writer(wrist_path, fps=self.video_fps, frame_hw=wrist_bgr.shape[:2])

        if self._front_writer is None or self._wrist_writer is None:
            self._front_writer = None
            self._wrist_writer = None
            print("[obs] warning: failed to initialize video writers (cv2 unavailable or codec not supported); skipping videos")
            self._video_ready = True
            return

        self._video_ready = True

    def update(
        self,
        *,
        step_idx: int,
        state: np.ndarray,
        front_img: np.ndarray,
        wrist_img: np.ndarray,
        prompt: Optional[str],
    ) -> None:
        # cv2.imwrite expects BGR; if we converted to RGB for the model, swap back for correct colors.
        front_bgr = front_img[..., ::-1] if self.bgr_to_rgb else front_img
        wrist_bgr = wrist_img[..., ::-1] if self.bgr_to_rgb else wrist_img

        try:
            import cv2  # local import

            cv2.imwrite(str(self.run_dir / "observation.image_front.png"), front_bgr)
            cv2.imwrite(str(self.run_dir / "observation.image_wrist.png"), wrist_bgr)
        except Exception as e:
            print(f"[obs] warning: failed to save latest .png images (cv2 unavailable or write failed): {e}")

        step_info = {
            "step_idx": int(step_idx),
            "prompt": prompt,
            "state_shape": list(np.asarray(state).shape),
            "front_shape": list(np.asarray(front_img).shape),
            "wrist_shape": list(np.asarray(wrist_img).shape),
        }
        (self.run_dir / "step_latest.json").write_text(
            json.dumps(step_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._ensure_video_writers(front_bgr=front_bgr, wrist_bgr=wrist_bgr)
        if self._front_writer is not None and self._wrist_writer is not None:
            try:
                self._front_writer.write(front_bgr)
                self._wrist_writer.write(wrist_bgr)
            except Exception as e:
                print(f"[obs] warning: failed to append to video writers: {e}")

    def finalize(self) -> None:
        for w in (self._front_writer, self._wrist_writer):
            if w is None:
                continue
            try:
                w.release()
            except Exception:
                pass


def _start_ssh_tunnels(cmds: list[str], *, startup_wait_s: float = 1.0) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for cmd in cmds:
        argv = shlex.split(cmd)
        if not argv:
            continue
        procs.append(
            subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    if not procs:
        return []
    time.sleep(max(0.0, float(startup_wait_s)))
    for proc, cmd in zip(procs, cmds, strict=False):
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"SSH tunnel exited early (rc={rc}). cmd={cmd!r}")
    return procs


def _stop_processes(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=2.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def _wait_for_franka_not_resetting(
    franka: FrankaHttpClient,
    *,
    timeout_s: float,
    poll_s: float = 0.2,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last = None
    while time.monotonic() < deadline:
        last = franka.get_status()
        if not bool(last.get("resetting", False)):
            return
        time.sleep(float(poll_s))
    raise TimeoutError(f"Timed out waiting for Franka to become not-resetting. Last status: {last}")


def _run_franka_reset_and_wait(
    franka: FrankaHttpClient,
    *,
    reset_method: str,
    start_timeout_s: float,
    finish_timeout_s: float,
    poll_s: float,
    strict: bool,
) -> tuple[dict, bool]:
    """Trigger a reset endpoint and wait until it is safe to proceed.

    We avoid sending anything to the model side until this returns.

    Some resets may finish very quickly and we might never observe `resetting=True`
    via polling. In non-strict mode we treat that as "already reset" and proceed.
    """
    if reset_method == "jointreset":
        resp = franka.joint_reset()
    elif reset_method == "reset_all":
        resp = franka.reset_all()
    else:
        raise ValueError(f"Unsupported reset_method={reset_method!r}")

    saw_resetting = False
    start_deadline = time.monotonic() + float(start_timeout_s)
    last = None
    while time.monotonic() < start_deadline:
        last = franka.get_status()
        if bool(last.get("resetting", False)):
            saw_resetting = True
            break
        time.sleep(float(poll_s))

    if saw_resetting:
        _wait_for_franka_not_resetting(franka, timeout_s=finish_timeout_s, poll_s=poll_s)
        return resp, True

    if strict:
        raise TimeoutError(
            f"Franka reset flag never became true within {start_timeout_s}s. "
            f"Last status: {last}, reset response: {resp}"
        )

    return resp, False


async def run_loop(args: argparse.Namespace) -> None:
    if ws_connect is None:  # pragma: no cover
        raise RuntimeError(f"websockets is required but could not be imported: {_WS_IMPORT_ERROR}")

    mpn = _mpn_module()
    packer = mpn.Packer()

    franka = FrankaHttpClient(args.franka_url, timeout_s=args.franka_timeout_s)
    tunnel_procs: list[subprocess.Popen] = []

    stop_event = asyncio.Event()
    try:
        loop = asyncio.get_running_loop()

        def _request_stop() -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass

    serials = _detect_realsense_serials()
    if args.front_serial is None or args.wrist_serial is None:
        if len(serials) != 2:
            raise RuntimeError(
                f"Expected exactly 2 RealSense devices when serials are not provided; found {len(serials)}: {serials}"
            )
        if args.camera_order == "front_first":
            # Mapping: front=first, wrist=second (override via flags).
            front_serial = args.front_serial or serials[0]
            wrist_serial = args.wrist_serial or serials[1]
        elif args.camera_order == "wrist_first":
            # Mapping: wrist=first, front=second (override via flags).
            wrist_serial = args.wrist_serial or serials[0]
            front_serial = args.front_serial or serials[1]
        else:
            raise ValueError(f"Unsupported camera_order={args.camera_order!r}")
    else:
        wrist_serial = args.wrist_serial
        front_serial = args.front_serial

    if wrist_serial == front_serial:
        raise ValueError("front_serial and wrist_serial must be different")

    wrist_cam = RealSenseColorCamera(wrist_serial, width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    front_cam = RealSenseColorCamera(front_serial, width=args.cam_width, height=args.cam_height, fps=args.cam_fps)

    print(f"[cams] wrist_serial={wrist_serial} front_serial={front_serial}")
    save_obs_run: Optional[_ObsSaver] = None

    ping_interval = None if args.ws_ping_interval_s <= 0 else float(args.ws_ping_interval_s)
    ping_timeout = None if args.ws_ping_timeout_s <= 0 else float(args.ws_ping_timeout_s)
    close_timeout = float(args.ws_close_timeout_s)

    control_mode: str = "cartesian"
    try:
        if args.ssh_tunnel:
            tunnel_procs = _start_ssh_tunnels(args.ssh_tunnel, startup_wait_s=args.ssh_tunnel_wait_s)

        # Reset the robot BEFORE we connect/send anything to the OpenPI server.
        # This guarantees no messages are sent to the model side during reset.
        if args.reset_before_start:
            before_pose = None
            try:
                before_pose = franka.get_pose_euler()
            except Exception:
                pass

            print(f"[robot] {args.reset_method} starting...")
            resp, saw_flag = _run_franka_reset_and_wait(
                franka,
                reset_method=args.reset_method,
                start_timeout_s=args.reset_start_timeout_s,
                finish_timeout_s=args.reset_timeout_s,
                poll_s=args.reset_poll_s,
                strict=args.reset_strict,
            )
            if args.print_reset_status:
                print(f"[robot] {args.reset_method} response: {resp} (observed_resetting_flag={saw_flag})")
            # Extra short settle to make motion/audible behavior more apparent and avoid immediate follow-up commands.
            time.sleep(max(0.0, float(args.reset_settle_s)))

            after_pose = None
            try:
                after_pose = franka.get_pose_euler()
            except Exception:
                pass

            if before_pose is not None and after_pose is not None:
                pos_err = float(np.linalg.norm(after_pose[:3] - before_pose[:3]))
                rot_err = float(
                    (
                        R.from_euler("xyz", before_pose[3:]).inv()
                        * R.from_euler("xyz", after_pose[3:])
                    ).magnitude()
                )
                print(f"[robot] reset_all done. delta_xyz_m={pos_err:.4f} delta_rot_rad={rot_err:.3f}")
            else:
                print("[robot] reset done.")

        if args.control_mode == "joint":
            franka.start_joint_control()
            control_mode = "joint"
            print("[robot] control_mode=joint (streaming)")

        # IMPORTANT: OpenPI server-side inference may block its event loop (sync infer),
        # which can cause websocket keepalive ping timeouts. Default is to disable keepalive
        # (ws_ping_interval_s=0) unless you explicitly enable it.
        try:
            ws_ctx = ws_connect(
                args.openpi_ws,
                compression=None,
                max_size=None,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                close_timeout=close_timeout,
            )
        except TypeError:
            # Older websockets versions may not support these kwargs.
            ws_ctx = ws_connect(args.openpi_ws, compression=None, max_size=None)

        async with ws_ctx as ws:
            # Server sends metadata first.
            meta_msg = await ws.recv()
            if isinstance(meta_msg, str):
                raise RuntimeError(f"Expected metadata bytes, got text: {meta_msg[:200]}")
            metadata = mpn.unpackb(meta_msg)
            print(f"[openpi] metadata: {metadata}")

            period_s = 1.0 / max(args.hz, 1e-6)
            next_tick = time.monotonic()
            last_gripper_open: Optional[bool] = None
            prev_quat_xyzw: Optional[np.ndarray] = None
            warned_safe_rpy_ignored = False

            for step_idx in range(args.max_steps if args.max_steps > 0 else 1_000_000_000):
                if stop_event.is_set():
                    break
                t0 = time.monotonic()

                q = franka.get_joint_positions()
                pose_xyzrpy = franka.get_pose_euler()
                gripper_pos = franka.get_gripper_distance()

                if args.state_format == "rpy14":
                    state = np.concatenate(
                        [q, np.asarray([gripper_pos], dtype=np.float32), pose_xyzrpy.astype(np.float32)],
                        axis=0,
                    ).astype(np.float32)
                    state_spec = "float32[14] = [q(7), gripper(1), pose_xyzrpy(6)]"
                elif args.state_format == "quat15":
                    xyz = pose_xyzrpy[:3].astype(np.float32)
                    rpy = pose_xyzrpy[3:].astype(np.float32)
                    quat_xyzw = R.from_euler("xyz", rpy).as_quat().astype(np.float32)
                    state = np.concatenate(
                        [q, np.asarray([gripper_pos], dtype=np.float32), xyz, quat_xyzw],
                        axis=0,
                    ).astype(np.float32)
                    state_spec = "float32[15] = [q(7), gripper(1), pose_xyzquat_xyzw(7)]"
                else:
                    raise ValueError(f"Unsupported --state_format={args.state_format!r}")

                front_bgr = front_cam.read_bgr()
                wrist_bgr = wrist_cam.read_bgr()
                front_bgr = _maybe_resize_bgr(front_bgr, resize_hw=args.resize_hw)
                wrist_bgr = _maybe_resize_bgr(wrist_bgr, resize_hw=args.resize_hw)
                front_img = _maybe_bgr_to_rgb(front_bgr, bgr_to_rgb=args.bgr_to_rgb)
                wrist_img = _maybe_bgr_to_rgb(wrist_bgr, bgr_to_rgb=args.bgr_to_rgb)

                obs: dict[str, Any] = {
                    "observation.state": state,
                    "observation.image_front": front_img,
                    "observation.image_wrist": wrist_img,
                }
                if args.prompt is not None:
                    obs["prompt"] = args.prompt

                if args.print_obs_keys and step_idx == 0:
                    print(f"[obs] keys={list(obs.keys())} state_shape={state.shape} front={front_img.shape} wrist={wrist_img.shape}")

                save_obs_run = _maybe_save_obs(
                    enabled=args.save_obs,
                    every=int(args.save_obs_every),
                    run=save_obs_run,
                    out_dir=args.save_obs_dir,
                    step_idx=step_idx,
                    state=state,
                    state_spec=state_spec,
                    front_img=front_img,
                    wrist_img=wrist_img,
                    front_serial=front_serial,
                    wrist_serial=wrist_serial,
                    bgr_to_rgb=args.bgr_to_rgb,
                    resize_hw=args.resize_hw,
                    prompt=args.prompt,
                    video_fps=max(1.0, float(args.hz) / max(1, int(args.exec_horizon)) / max(1, int(args.save_obs_every))),
                )

                await ws.send(packer.pack(obs))

                try:
                    resp_msg = await asyncio.wait_for(ws.recv(), timeout=args.ws_recv_timeout_s)
                except asyncio.TimeoutError as e:
                    raise RuntimeError(
                        f"Timed out waiting for OpenPI inference response after {args.ws_recv_timeout_s}s. "
                        "Model may be compiling or stuck; try waiting longer (increase --ws_recv_timeout_s) "
                        "or check server logs."
                    ) from e
                if isinstance(resp_msg, str):
                    raise RuntimeError(f"OpenPI server returned error text:\n{resp_msg}")
                action_dict = mpn.unpackb(resp_msg)

                action_seq = _extract_action(action_dict, action_key=args.action_key)  # (T, D), D in {7,8,15}

                
                waited_ms = (time.monotonic() - t0) * 1000.0
                print(f"[step {step_idx}] action0={action_seq[0].tolist()} action_T={int(action_seq.shape[0])} step_ms={waited_ms:.1f}")

                # Execute multiple steps for each inference result (open-loop).
                exec_horizon = int(args.exec_horizon)
                if exec_horizon <= 0:
                    raise ValueError(f"--exec_horizon must be >= 1, got {exec_horizon}")

                for k in range(exec_horizon):
                    if stop_event.is_set():
                        break
                    a7 = action_seq[min(k, action_seq.shape[0] - 1)]
                    (
                        target_xyz,
                        target_rpy,
                        target_quat_xyzw,
                        target_gripper,
                        _ignored_joints7,
                        prev_quat_xyzw,
                    ) = _parse_action_step(
                        a7,
                        action_format=args.action_format,
                        quat_normalize=args.quat_normalize,
                        quat_flip_sign=args.quat_flip_sign,
                        prev_quat_xyzw=prev_quat_xyzw,
                    )

                    joints7 = _ignored_joints7
                    if args.control_mode == "auto":
                        want_mode = "joint" if joints7 is not None else "cartesian"
                    else:
                        want_mode = args.control_mode

                    if want_mode == "joint" and joints7 is None:
                        raise RuntimeError(
                            "Joint control requested but action does not include joints. "
                            "Use --control_mode cartesian or ensure model returns joint actions "
                            "(either 8D joint8=[joints(7),gripper] or 15D quat15=[joints,gripper,xyz,quat])."
                        )

                    if want_mode != control_mode:
                        if want_mode == "joint":
                            franka.start_joint_control()
                            control_mode = "joint"
                            print("[robot] switched to joint control")
                        elif want_mode == "cartesian":
                            franka.stop_joint_control()
                            control_mode = "cartesian"
                            print("[robot] switched to cartesian control")
                        else:
                            raise ValueError(f"Unsupported --control_mode={args.control_mode!r}")

                    if control_mode == "cartesian" and args.safe_xyz_low is not None and args.safe_xyz_high is not None:
                        xyz = target_xyz[:3].copy()
                        clipped = np.clip(xyz, args.safe_xyz_low, args.safe_xyz_high)
                        if args.stop_on_safety_violation and not np.allclose(xyz, clipped):
                            raise RuntimeError(f"Target xyz out of safety bounds: xyz={xyz.tolist()} clipped={clipped.tolist()}")
                        target_xyz = target_xyz.copy()
                        target_xyz[:3] = clipped

                    if control_mode == "cartesian" and args.safe_rpy_low is not None and args.safe_rpy_high is not None:
                        if target_rpy is None:
                            if not warned_safe_rpy_ignored:
                                print("[warn] safe_rpy_* bounds are set but quaternion actions are in use; rpy bounds are ignored.")
                                warned_safe_rpy_ignored = True
                        else:
                            rpy = target_rpy.copy()
                            clipped = np.clip(rpy, args.safe_rpy_low, args.safe_rpy_high)
                            if args.stop_on_safety_violation and not np.allclose(rpy, clipped):
                                raise RuntimeError(f"Target rpy out of safety bounds: rpy={rpy.tolist()} clipped={clipped.tolist()}")
                            target_rpy = clipped.astype(np.float32)

                    if args.dry_run:
                        next_tick += period_s
                        sleep_s = next_tick - time.monotonic()
                        if sleep_s > 0:
                            await asyncio.sleep(sleep_s)
                        else:
                            next_tick = time.monotonic()
                        continue

                    if args.recover_each_step:
                        franka.clear_errors()
                    if control_mode == "joint":
                        franka.send_joint_q(joints7)
                    elif target_quat_xyzw is not None:
                        franka.send_pose_xyz_quat(
                            np.concatenate([target_xyz.astype(np.float32), target_quat_xyzw], axis=0)
                        )
                    else:
                        if target_rpy is None:
                            raise AssertionError("Expected target_rpy for non-quaternion action")
                        franka.send_pose_xyzrpy(np.concatenate([target_xyz.astype(np.float32), target_rpy.astype(np.float32)], axis=0))

                    want_open = target_gripper >= args.gripper_open_threshold
                    if last_gripper_open is None or (want_open != last_gripper_open):
                        if want_open:
                            franka.open_gripper()
                        else:
                            franka.close_gripper()
                        last_gripper_open = want_open

                    next_tick += period_s
                    sleep_s = next_tick - time.monotonic()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    else:
                        next_tick = time.monotonic()

                if args.print_timing:
                    timing = action_dict.get("server_timing", None)
                    total_ms = (time.monotonic() - t0) * 1000.0
                    print(f"[timing] step_ms={total_ms:.1f} server_timing={timing}")

    finally:
        if save_obs_run is not None:
            try:
                save_obs_run.finalize()
            except Exception:
                pass
        if args.on_stop == "hold" and control_mode == "joint":
            try:
                franka.stop_joint_control()
                control_mode = "cartesian"
            except Exception:
                pass
        if args.on_stop == "hold":
            try:
                pose_xyzrpy = franka.get_pose_euler()
                franka.send_pose_xyzrpy(pose_xyzrpy)
            except Exception:
                pass
        elif args.on_stop == "stopimp":
            try:
                franka._post_json("stopimp", payload=None, timeout_s=5.0)
            except Exception:
                pass
        _stop_processes(tunnel_procs)
        front_cam.close()
        wrist_cam.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--franka_url", required=True, help="Base URL of Franka Flask server (e.g., http://127.0.0.1:5000/)")
    p.add_argument("--openpi_ws", required=True, help="OpenPI websocket URL (e.g., ws://host:port)")

    p.add_argument("--front_serial", default=None, help="RealSense serial for front camera (optional if exactly 2 devices connected)")
    p.add_argument("--wrist_serial", default=None, help="RealSense serial for wrist camera (optional if exactly 2 devices connected)")
    p.add_argument(
        "--camera_order",
        choices=["front_first", "wrist_first"],
        default="front_first",
        help="When serials are omitted and exactly 2 RealSense devices are connected, map devices by enumeration order.",
    )
    p.add_argument("--cam_width", type=int, default=640)
    p.add_argument("--cam_height", type=int, default=480)
    p.add_argument("--cam_fps", type=int, default=30)

    p.add_argument("--resize_hw", type=int, nargs=2, default=None, metavar=("H", "W"), help="Optionally resize images before sending")
    p.add_argument("--bgr_to_rgb", action="store_true", help="Convert camera BGR to RGB before sending")

    p.add_argument("--prompt", default=None, help="Optional text prompt for the policy")
    p.add_argument("--action_key", default="actions", help="Key in OpenPI response dict containing actions (default: actions)")
    p.add_argument(
        "--state_format",
        choices=["rpy14", "quat15"],
        default="rpy14",
        help="Observation state encoding: rpy14=[q,gripper,xyzrpy], quat15=[q,gripper,xyz,quat_xyzw].",
    )
    p.add_argument(
        "--action_format",
        choices=["auto", "rpy7", "quat8", "quat15", "joint8"],
        default="auto",
        help=(
            "Action encoding: auto infers from dim; rpy7=[xyzrpy,gripper]; quat8=[xyz,quat_xyzw,gripper]; "
            "quat15=[joints,gripper,xyz,quat_xyzw]; joint8=[joints(7),gripper]."
        ),
    )
    p.add_argument(
        "--quat_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize quaternion actions to unit length (recommended).",
    )
    p.add_argument(
        "--quat_flip_sign",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip quaternion sign for continuity (if dot(prev, curr)<0).",
    )

    p.add_argument("--hz", type=float, default=10.0, help="Control loop frequency")
    p.add_argument("--exec_horizon", type=int, default=50, help="Execute N action steps per inference result")
    p.add_argument("--max_steps", type=int, default=0, help="Max steps before exit (0 = run forever)")
    p.add_argument("--dry_run", action="store_true", help="Do not command robot; just print actions")
    p.add_argument("--recover_each_step", action="store_true", help="Call /clearerr before each /pose command")
    p.add_argument("--franka_timeout_s", type=float, default=3.0)
    p.add_argument(
        "--control_mode",
        choices=["cartesian", "joint", "auto"],
        default="cartesian",
        help=(
            "Robot command mode: cartesian sends /pose; joint sends /joint_q (requires 15D actions with joints). "
            "auto uses joint when joints are present, else cartesian."
        ),
    )

    p.add_argument("--gripper_open_threshold", type=float, default=0.7, help="Open if action gripper >= threshold, else close")

    p.add_argument(
        "--save_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the observation snapshots (as sent to the model).",
    )
    p.add_argument(
        "--save_obs_dir",
        type=str,
        default="saved_obs",
        help="Directory to save the observation snapshots.",
    )
    p.add_argument(
        "--save_obs_every",
        type=int,
        default=1,
        help="Save observations every N steps (1 = every step; 0 disables saving even if --save_obs is enabled).",
    )

    p.add_argument(
        "--reset_before_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset robot once before starting inference (default: enabled).",
    )
    p.add_argument(
        "--reset_method",
        choices=["jointreset", "reset_all"],
        default="jointreset",
        help="Reset endpoint to call before inference.",
    )
    p.add_argument("--reset_strict", action="store_true", help="Fail if /status.resetting is never observed true during reset")
    p.add_argument("--reset_start_timeout_s", type=float, default=2.0, help="Seconds to wait for /status.resetting to become true after reset")
    p.add_argument("--reset_timeout_s", type=float, default=60.0, help="Max seconds to wait for reset to finish")
    p.add_argument("--reset_poll_s", type=float, default=0.2, help="Polling interval while waiting for reset")
    p.add_argument("--reset_settle_s", type=float, default=0.5, help="Extra seconds to wait after reset completes")
    p.add_argument("--print_reset_status", action="store_true", help="Print reset endpoint response and observed resetting flag")

    p.add_argument("--ws_ping_interval_s", type=float, default=0.0, help="Websocket keepalive ping interval in seconds (0 disables)")
    p.add_argument("--ws_ping_timeout_s", type=float, default=0.0, help="Websocket keepalive ping timeout in seconds (0 disables)")
    p.add_argument("--ws_close_timeout_s", type=float, default=1.0, help="Websocket close timeout in seconds")
    p.add_argument("--ws_recv_timeout_s", type=float, default=300.0, help="Max seconds to wait for each inference response")
    p.add_argument("--print_actions_every", type=int, default=0, help="Print action every N steps (0 disables; step 0 always prints)")

    p.add_argument(
        "--ssh_tunnel",
        action="append",
        default=[],
        help="Optional: start an SSH port-forward command inside this process (repeatable). Example: \"ssh -N -L 9000:127.0.0.1:8000 -J user@jump user@model -o ExitOnForwardFailure=yes\"",
    )
    p.add_argument("--ssh_tunnel_wait_s", type=float, default=1.0, help="Seconds to wait after starting SSH tunnels before connecting")

    p.add_argument("--safe_xyz_low", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional safety lower bound for target xyz")
    p.add_argument("--safe_xyz_high", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional safety upper bound for target xyz")
    p.add_argument("--safe_rpy_low", type=float, nargs=3, default=None, metavar=("R", "P", "Y"), help="Optional safety lower bound for target rpy (rad)")
    p.add_argument("--safe_rpy_high", type=float, nargs=3, default=None, metavar=("R", "P", "Y"), help="Optional safety upper bound for target rpy (rad)")
    p.add_argument("--stop_on_safety_violation", action="store_true", help="If set, stop when model outputs out-of-bounds pose (instead of clipping)")
    p.add_argument("--on_stop", choices=["hold", "stopimp", "none"], default="hold", help="What to do on exit (Ctrl-C): hold current pose, stop impedance, or nothing")

    p.add_argument("--print_timing", action="store_true")
    p.add_argument("--print_obs_keys", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
