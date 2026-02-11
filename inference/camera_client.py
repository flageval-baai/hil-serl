"""Camera client for RealSense cameras."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

import cv2


# Camera serial numbers from dataflow.yml
CAMERA_SERIALS = {
    "wrist": "347622076012",
    "front": "247122073147",
    "side": "243322074118",
}


class CameraClient:
    """Single RealSense camera client."""

    def __init__(
        self,
        name: str,
        serial_number: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth: bool = False,
        exposure: int = 40000,
    ):
        """Initialize camera client.

        Args:
            name: Camera name identifier
            serial_number: RealSense camera serial number
            width: Image width
            height: Image height
            fps: Frames per second
            depth: Enable depth stream
            exposure: Camera exposure setting
        """
        if rs is None:
            raise ImportError("pyrealsense2 is required for camera capture")

        self.name = name
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.depth = depth

        # Verify device is connected
        available = self._get_device_serial_numbers()
        if serial_number not in available:
            raise RuntimeError(
                f"Camera {serial_number} not found. Available: {available}"
            )

        # Configure pipeline
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(serial_number)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if depth:
            self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start pipeline
        self.profile = self.pipe.start(self.cfg)

        # Set exposure
        sensor = self.profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.exposure, exposure)

        # Alignment for depth
        self.align = rs.align(rs.stream.color)

        # Get depth scale if depth enabled
        if depth:
            self.depth_scale = (
                self.profile.get_device().first_depth_sensor().get_depth_scale()
            )
        else:
            self.depth_scale = None

    @staticmethod
    def _get_device_serial_numbers() -> List[str]:
        """Get list of connected RealSense device serial numbers."""
        if rs is None:
            return []
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    @staticmethod
    def list_devices() -> List[Dict[str, str]]:
        """List all connected RealSense devices."""
        if rs is None:
            return []
        devices = rs.context().devices
        return [
            {
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
            }
            for d in devices
        ]

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            (success, image): success flag and BGR image (or BGRD if depth enabled)
        """
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame.is_video_frame():
            return False, None

        image = np.asarray(color_frame.get_data())

        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)

        return True, image

    def read_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame in RGB format.

        Returns:
            (success, image): success flag and RGB image
        """
        success, image = self.read()
        if success and image is not None:
            # Convert BGR to RGB
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return success, image

    def close(self) -> None:
        """Stop the camera pipeline."""
        self.pipe.stop()
        self.cfg.disable_all_streams()


class MultiCameraClient:
    """Multi-camera client for capturing from multiple RealSense cameras."""

    def __init__(
        self,
        camera_configs: Optional[Dict[str, str]] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth: bool = False,
        exposure: int = 40000,
    ):
        """Initialize multi-camera client.

        Args:
            camera_configs: Dict mapping camera names to serial numbers.
                           If None, uses default CAMERA_SERIALS.
            width: Image width
            height: Image height
            fps: Frames per second
            depth: Enable depth stream
            exposure: Camera exposure setting
        """
        if camera_configs is None:
            camera_configs = CAMERA_SERIALS

        self.cameras: Dict[str, CameraClient] = {}
        self.width = width
        self.height = height

        # Get available devices
        available = CameraClient._get_device_serial_numbers()
        print(f"Available cameras: {available}")

        # Initialize only cameras that are connected
        for name, serial in camera_configs.items():
            if serial in available:
                try:
                    self.cameras[name] = CameraClient(
                        name=name,
                        serial_number=serial,
                        width=width,
                        height=height,
                        fps=fps,
                        depth=depth,
                        exposure=exposure,
                    )
                    print(f"Initialized camera '{name}' (serial: {serial})")
                except Exception as e:
                    print(f"[WARN] Failed to initialize camera '{name}': {e}")
            else:
                print(f"[WARN] Camera '{name}' (serial: {serial}) not connected")

        if not self.cameras:
            raise RuntimeError("No cameras initialized")

    def read_all(self) -> Dict[str, np.ndarray]:
        """Read frames from all cameras.

        Returns:
            Dict mapping camera names to BGR images
        """
        images = {}
        for name, camera in self.cameras.items():
            success, image = camera.read()
            if success and image is not None:
                images[name] = image
        return images

    def read_all_rgb(self) -> Dict[str, np.ndarray]:
        """Read frames from all cameras in RGB format.

        Returns:
            Dict mapping camera names to RGB images
        """
        images = {}
        for name, camera in self.cameras.items():
            success, image = camera.read_rgb()
            if success and image is not None:
                images[name] = image
        return images

    def get_observation_images(self) -> Dict[str, np.ndarray]:
        """Get images in the observation format for policy inference.

        Returns:
            Dict with keys like 'observation.image_front', 'observation.image_wrist', etc.
        """
        images = self.read_all_rgb()
        obs_images = {}
        for name, image in images.items():
            obs_images[f"observation.image_{name}"] = image
        return obs_images

    def close(self) -> None:
        """Close all camera pipelines."""
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()

    @property
    def camera_names(self) -> List[str]:
        """Get list of active camera names."""
        return list(self.cameras.keys())
