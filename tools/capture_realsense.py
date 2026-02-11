#!/usr/bin/env python3
"""
Capture photos from RealSense cameras.
Usage: python capture_realsense.py [--camera wrist|front|side|all] [--count 3]
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import argparse

# Camera configurations from dataflow.yml
CAMERAS = {
    "wrist": "347622076012",
    "front": "247122073147",
    "side": "243322074118",
}

def capture_from_camera(serial: str, name: str, output_dir: str, count: int = 1):
    """Capture photos from a specific RealSense camera."""
    print(f"\n[{name}] Connecting to camera (serial: {serial})...")

    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        # Start streaming
        pipeline.start(config)
        print(f"[{name}] Camera connected. Warming up...")

        # Wait for auto-exposure to stabilize
        for _ in range(30):
            pipeline.wait_for_frames()

        print(f"[{name}] Capturing {count} photo(s)...")

        for i in range(count):
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                print(f"[{name}] Warning: No color frame received")
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            color_path = os.path.join(output_dir, f"{name}_{timestamp}_{i+1}.jpg")

            # Save color image
            cv2.imwrite(color_path, color_image)
            print(f"[{name}] Saved: {color_path}")

            # Save depth image if available
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                # Normalize depth for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                depth_path = os.path.join(output_dir, f"{name}_{timestamp}_{i+1}_depth.jpg")
                cv2.imwrite(depth_path, depth_colormap)
                print(f"[{name}] Saved: {depth_path}")

        print(f"[{name}] Done!")

    except Exception as e:
        print(f"[{name}] Error: {e}")
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="Capture photos from RealSense cameras")
    parser.add_argument("--camera", "-c", choices=["wrist", "front", "side", "all"],
                        default="all", help="Which camera to use (default: all)")
    parser.add_argument("--count", "-n", type=int, default=1,
                        help="Number of photos to capture per camera (default: 1)")
    parser.add_argument("--output", "-o", default="realsense_photos",
                        help="Output directory (default: realsense_photos)")
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which cameras to use
    if args.camera == "all":
        cameras_to_use = CAMERAS
    else:
        cameras_to_use = {args.camera: CAMERAS[args.camera]}

    # Capture from each camera
    for name, serial in cameras_to_use.items():
        capture_from_camera(serial, name, output_dir, args.count)

    print(f"\nAll photos saved to: {output_dir}")


if __name__ == "__main__":
    main()
