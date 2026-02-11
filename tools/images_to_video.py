#!/usr/bin/env python3
"""
Combine images from multiple cameras into a video.

Usage:
    python tools/images_to_video.py output/images_success -o output/video.mp4
    python tools/images_to_video.py output/images_success --layout horizontal --fps 30
    python tools/images_to_video.py output/images_success --layout grid --fps 15
"""

import argparse
import os
import re
from typing import List, Tuple

import cv2
import numpy as np


def find_frames(image_dir: str) -> Tuple[List[int], List[str]]:
    """Find all frames and camera names in the directory.

    Args:
        image_dir: Directory containing images

    Returns:
        (frame_indices, camera_names): Sorted list of frame indices and camera names
    """
    files = os.listdir(image_dir)

    # Parse filenames: frame_{index}_image_{camera}.png
    pattern = re.compile(r"frame_(\d+)_image_(\w+)\.(png|jpg|jpeg)")

    frames = set()
    cameras = set()

    for f in files:
        match = pattern.match(f)
        if match:
            frame_idx = int(match.group(1))
            camera_name = match.group(2)
            frames.add(frame_idx)
            cameras.add(camera_name)

    return sorted(frames), sorted(cameras)


def load_frame_images(
    image_dir: str,
    frame_idx: int,
    cameras: List[str],
) -> dict:
    """Load images for a single frame from all cameras.

    Args:
        image_dir: Directory containing images
        frame_idx: Frame index
        cameras: List of camera names

    Returns:
        Dict mapping camera name to image (BGR)
    """
    images = {}
    for camera in cameras:
        # Try different extensions
        for ext in ["png", "jpg", "jpeg"]:
            filename = f"frame_{frame_idx}_image_{camera}.{ext}"
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    images[camera] = img
                break
    return images


def combine_images(
    images: dict,
    layout: str = "horizontal",
    target_height: int = 480,
    padding: int = 5,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Combine multiple camera images into a single frame.

    Args:
        images: Dict mapping camera name to image
        layout: 'horizontal', 'vertical', or 'grid'
        target_height: Target height for each image (maintains aspect ratio)
        padding: Padding between images
        bg_color: Background color (BGR)

    Returns:
        Combined image
    """
    if not images:
        return np.zeros((target_height, 640, 3), dtype=np.uint8)

    # Resize images to same height
    resized = {}
    for name, img in images.items():
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized[name] = cv2.resize(img, (new_w, target_height))

    # Sort by camera name for consistent ordering
    camera_order = sorted(resized.keys())
    imgs = [resized[name] for name in camera_order]

    if layout == "horizontal":
        # Side by side
        total_width = sum(img.shape[1] for img in imgs) + padding * (len(imgs) - 1)
        combined = np.full((target_height, total_width, 3), bg_color, dtype=np.uint8)

        x = 0
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            combined[:h, x:x+w] = img
            # Add label
            cv2.putText(
                combined, camera_order[i].upper(),
                (x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            x += w + padding

    elif layout == "vertical":
        # Stacked vertically
        max_width = max(img.shape[1] for img in imgs)
        total_height = sum(img.shape[0] for img in imgs) + padding * (len(imgs) - 1)
        combined = np.full((total_height, max_width, 3), bg_color, dtype=np.uint8)

        y = 0
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            combined[y:y+h, :w] = img
            cv2.putText(
                combined, camera_order[i].upper(),
                (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            y += h + padding

    elif layout == "grid":
        # 2x2 grid (or 2xN for more cameras)
        n = len(imgs)
        cols = 2 if n > 1 else 1
        rows = (n + cols - 1) // cols

        max_width = max(img.shape[1] for img in imgs)
        total_width = max_width * cols + padding * (cols - 1)
        total_height = target_height * rows + padding * (rows - 1)
        combined = np.full((total_height, total_width, 3), bg_color, dtype=np.uint8)

        for i, img in enumerate(imgs):
            row = i // cols
            col = i % cols
            y = row * (target_height + padding)
            x = col * (max_width + padding)
            h, w = img.shape[:2]
            combined[y:y+h, x:x+w] = img
            cv2.putText(
                combined, camera_order[i].upper(),
                (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    else:
        raise ValueError(f"Unknown layout: {layout}")

    return combined


def create_video(
    image_dir: str,
    output_path: str,
    fps: float = 30.0,
    layout: str = "horizontal",
    target_height: int = 480,
    codec: str = "mp4v",
):
    """Create video from multi-camera images.

    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        layout: 'horizontal', 'vertical', or 'grid'
        target_height: Target height for each camera view
        codec: Video codec (mp4v, XVID, etc.)
    """
    # Find frames and cameras
    frames, cameras = find_frames(image_dir)

    if not frames:
        print(f"No frames found in {image_dir}")
        return

    print(f"Found {len(frames)} frames with cameras: {cameras}")

    # Load first frame to get dimensions
    first_images = load_frame_images(image_dir, frames[0], cameras)
    first_combined = combine_images(first_images, layout, target_height)
    height, width = first_combined.shape[:2]

    print(f"Output video size: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Failed to create video writer for {output_path}")
        return

    # Process each frame
    for i, frame_idx in enumerate(frames):
        images = load_frame_images(image_dir, frame_idx, cameras)
        combined = combine_images(images, layout, target_height)

        # Ensure correct size
        if combined.shape[:2] != (height, width):
            combined = cv2.resize(combined, (width, height))

        # Add frame number
        cv2.putText(
            combined, f"Frame: {frame_idx}",
            (width - 150, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        out.write(combined)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")

    out.release()
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multi-camera images into a video"
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images (frame_N_image_CAMERA.png)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path (default: image_dir/video.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical", "grid"],
        help="Layout for combining cameras (default: horizontal)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Target height for each camera view (default: 480)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec (default: mp4v)",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = os.path.join(args.image_dir, "video.mp4")

    create_video(
        image_dir=args.image_dir,
        output_path=args.output,
        fps=args.fps,
        layout=args.layout,
        target_height=args.height,
        codec=args.codec,
    )


if __name__ == "__main__":
    main()
