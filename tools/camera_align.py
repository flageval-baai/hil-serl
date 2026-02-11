#!/usr/bin/env python3
"""
Multi-camera alignment tool - overlay target images on live camera feeds to align camera positions.

Usage:
    # Single camera
    python tools/camera_align.py --front target_front.png

    # Two cameras
    python tools/camera_align.py --front target_front.png --wrist target_wrist.png

    # All three cameras
    python tools/camera_align.py --front target_front.png --wrist target_wrist.png --side target_side.png

    # Using serial numbers
    python tools/camera_align.py --camera 247122073147:target_front.png --camera 347622076012:target_wrist.png

Controls:
    - 'q' or ESC: Quit
    - '+' / '=': Increase overlay opacity
    - '-': Decrease overlay opacity
    - 'r': Reset opacity to default (0.5)
    - 'o': Overlay mode (default)
    - 'd': Difference mode (shows pixel difference)
    - 'e': Edge mode (shows edge alignment)
    - 's': Save current frames
    - '1'/'2'/'3': Toggle individual camera display
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 is required. Install with: pip install pyrealsense2")
    sys.exit(1)


# Camera serial numbers
CAMERA_SERIALS = {
    "wrist": "347622076012",
    "front": "247122073147",
    "side": "243322074118",
}


class CameraStream:
    """Single camera stream handler."""

    def __init__(
        self,
        name: str,
        serial: str,
        target_image_path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.enabled = True

        # Load target image
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")

        self.target_image = cv2.imread(target_image_path)
        if self.target_image is None:
            raise ValueError(f"Failed to load image: {target_image_path}")

        # Resize target to match camera resolution
        self.target_image = cv2.resize(self.target_image, (width, height))
        self.target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        self.target_edges = cv2.Canny(self.target_gray, 50, 150)

        # Initialize camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)

        # Warm up
        for _ in range(30):
            self.pipeline.wait_for_frames()

        print(f"Camera '{name}' ({serial}) initialized")

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a frame."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        except Exception:
            pass
        return None

    def close(self):
        """Stop the camera."""
        self.pipeline.stop()


class MultiCameraAligner:
    """Tool for aligning multiple cameras with target images."""

    def __init__(
        self,
        camera_targets: Dict[str, str],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """Initialize multi-camera aligner.

        Args:
            camera_targets: Dict mapping camera name/serial to target image path
            width: Camera width
            height: Camera height
            fps: Camera FPS
        """
        self.width = width
        self.height = height
        self.cameras: Dict[str, CameraStream] = {}

        # Check available cameras
        ctx = rs.context()
        available = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
        print(f"Available cameras: {available}")

        # Initialize each camera
        for cam_id, target_path in camera_targets.items():
            # Resolve serial number
            if cam_id in CAMERA_SERIALS:
                name = cam_id
                serial = CAMERA_SERIALS[cam_id]
            else:
                # Assume it's a serial number
                serial = cam_id
                # Find name by serial
                name = next((k for k, v in CAMERA_SERIALS.items() if v == serial), serial)

            if serial not in available:
                print(f"Warning: Camera '{name}' ({serial}) not connected, skipping")
                continue

            try:
                self.cameras[name] = CameraStream(
                    name=name,
                    serial=serial,
                    target_image_path=target_path,
                    width=width,
                    height=height,
                    fps=fps,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize camera '{name}': {e}")

        if not self.cameras:
            raise RuntimeError("No cameras initialized")

        print(f"Initialized {len(self.cameras)} cameras: {list(self.cameras.keys())}")

        # Display settings
        self.opacity = 0.5
        self.mode = "overlay"  # overlay, difference, edge

    def create_overlay(self, frame: np.ndarray, camera: CameraStream) -> np.ndarray:
        """Create overlay visualization for a single camera."""
        if self.mode == "overlay":
            overlay = cv2.addWeighted(
                frame, 1 - self.opacity,
                camera.target_image, self.opacity,
                0
            )
            return overlay

        elif self.mode == "difference":
            diff = cv2.absdiff(frame, camera.target_image)
            diff = cv2.convertScaleAbs(diff, alpha=2.0)
            mean_diff = np.mean(diff)
            cv2.putText(
                diff, f"Diff: {mean_diff:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            return diff

        elif self.mode == "edge":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_edges = cv2.Canny(frame_gray, 50, 150)

            overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            overlay[:, :, 1] = camera.target_edges  # Green = target
            overlay[:, :, 2] = frame_edges  # Red = current

            result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            return result

        return frame

    def create_combined_view(self) -> np.ndarray:
        """Create combined view of all cameras."""
        displays = []
        camera_list = list(self.cameras.items())

        for name, camera in camera_list:
            if not camera.enabled:
                # Show disabled placeholder
                placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, f"{name} (disabled)",
                    (self.width // 4, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2
                )
                displays.append(placeholder)
                continue

            frame = camera.get_frame()
            if frame is None:
                # Show error placeholder
                placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, f"{name} (no frame)",
                    (self.width // 4, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                displays.append(placeholder)
                continue

            # Create overlay
            display = self.create_overlay(frame, camera)

            # Add camera label
            cv2.putText(
                display, name.upper(),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            displays.append(display)

        # Combine views based on number of cameras
        if len(displays) == 1:
            combined = displays[0]
        elif len(displays) == 2:
            # Side by side
            combined = np.hstack(displays)
        elif len(displays) == 3:
            # Top: 2 cameras, Bottom: 1 camera centered
            top = np.hstack(displays[:2])
            # Pad bottom to match width
            bottom = np.zeros((self.height, self.width * 2, 3), dtype=np.uint8)
            x_offset = self.width // 2
            bottom[:, x_offset:x_offset + self.width] = displays[2]
            combined = np.vstack([top, bottom])
        else:
            # Grid layout for 4+
            rows = []
            for i in range(0, len(displays), 2):
                if i + 1 < len(displays):
                    rows.append(np.hstack([displays[i], displays[i + 1]]))
                else:
                    padded = np.hstack([displays[i], np.zeros_like(displays[i])])
                    rows.append(padded)
            combined = np.vstack(rows)

        return combined

    def run(self):
        """Run the alignment tool."""
        window_name = "Multi-Camera Align"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Calculate window size
        n_cameras = len(self.cameras)
        if n_cameras <= 2:
            win_width = self.width * n_cameras
            win_height = self.height
        else:
            win_width = self.width * 2
            win_height = self.height * 2

        cv2.resizeWindow(window_name, win_width, win_height)

        camera_names = list(self.cameras.keys())
        print("\nControls:")
        print("  q/ESC : Quit")
        print("  +/=   : Increase opacity")
        print("  -     : Decrease opacity")
        print("  r     : Reset opacity to 0.5")
        print("  o     : Overlay mode (default)")
        print("  d     : Difference mode")
        print("  e     : Edge mode")
        print("  s     : Save current frames")
        for i, name in enumerate(camera_names):
            print(f"  {i + 1}     : Toggle {name} camera")
        print(f"\nCameras: {camera_names}")
        print(f"Mode: {self.mode}, Opacity: {self.opacity:.2f}")

        try:
            while True:
                # Create combined view
                display = self.create_combined_view()

                # Add status bar at bottom
                status = f"Mode: {self.mode} | Opacity: {self.opacity:.2f} | Press 'q' to quit"
                cv2.putText(
                    display, status,
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

                cv2.imshow(window_name, display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('+') or key == ord('='):
                    self.opacity = min(1.0, self.opacity + 0.05)
                    print(f"Opacity: {self.opacity:.2f}")
                elif key == ord('-'):
                    self.opacity = max(0.0, self.opacity - 0.05)
                    print(f"Opacity: {self.opacity:.2f}")
                elif key == ord('r'):
                    self.opacity = 0.5
                    print(f"Opacity reset to: {self.opacity:.2f}")
                elif key == ord('d'):
                    self.mode = "difference"
                    print(f"Mode: {self.mode}")
                elif key == ord('e'):
                    self.mode = "edge"
                    print(f"Mode: {self.mode}")
                elif key == ord('o'):
                    self.mode = "overlay"
                    print(f"Mode: {self.mode}")
                elif key == ord('s'):
                    # Save current frames
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    for name, camera in self.cameras.items():
                        if camera.enabled:
                            frame = camera.get_frame()
                            if frame is not None:
                                filename = f"aligned_{name}_{timestamp}.png"
                                cv2.imwrite(filename, frame)
                                print(f"Saved: {filename}")
                elif ord('1') <= key <= ord('9'):
                    # Toggle camera
                    idx = key - ord('1')
                    if idx < len(camera_names):
                        cam_name = camera_names[idx]
                        self.cameras[cam_name].enabled = not self.cameras[cam_name].enabled
                        status = "enabled" if self.cameras[cam_name].enabled else "disabled"
                        print(f"Camera '{cam_name}' {status}")

        finally:
            self.close()

    def close(self):
        """Clean up resources."""
        for camera in self.cameras.values():
            camera.close()
        cv2.destroyAllWindows()
        print("All cameras closed")


def main():
    parser = argparse.ArgumentParser(
        description="Align multiple cameras with target images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Named camera arguments
    parser.add_argument(
        "--front",
        type=str,
        metavar="IMAGE",
        help="Target image for front camera",
    )
    parser.add_argument(
        "--wrist",
        type=str,
        metavar="IMAGE",
        help="Target image for wrist camera",
    )
    parser.add_argument(
        "--side",
        type=str,
        metavar="IMAGE",
        help="Target image for side camera",
    )

    # Generic camera argument (serial:image format)
    parser.add_argument(
        "--camera",
        type=str,
        action="append",
        metavar="SERIAL:IMAGE",
        help="Camera serial and target image (format: serial:image_path)",
    )

    # Display settings
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )

    args = parser.parse_args()

    # Build camera targets dict
    camera_targets = {}

    if args.front:
        camera_targets["front"] = args.front
    if args.wrist:
        camera_targets["wrist"] = args.wrist
    if args.side:
        camera_targets["side"] = args.side

    if args.camera:
        for cam_spec in args.camera:
            if ":" not in cam_spec:
                print(f"Error: Invalid camera spec '{cam_spec}'. Use format: serial:image_path")
                sys.exit(1)
            serial, image_path = cam_spec.split(":", 1)
            camera_targets[serial] = image_path

    if not camera_targets:
        print("Error: No cameras specified. Use --front, --wrist, --side, or --camera")
        print("Example: python tools/camera_align.py --front target_front.png --side target_side.png")
        sys.exit(1)

    print(f"Aligning {len(camera_targets)} camera(s):")
    for cam, target in camera_targets.items():
        print(f"  {cam}: {target}")

    aligner = MultiCameraAligner(
        camera_targets=camera_targets,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    aligner.run()


if __name__ == "__main__":
    main()
