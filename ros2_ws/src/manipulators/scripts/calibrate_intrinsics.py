#!/usr/bin/env python3
"""
Camera intrinsic calibration using chessboard images.

Usage:
  python3 calibrate_intrinsics.py \
    --image-dir ./calibration_images \
    --output ./config/camera.yaml \
    --board-size 8 6 \
    --square-size 25.0

Finds chessboard corners in all images, runs cv2.calibrateCamera,
saves camera matrix and distortion coefficients to YAML.
"""

import argparse
import glob
import sys

import cv2
import numpy as np
import yaml


def calibrate(image_dir: str, board_size: tuple, square_size_mm: float):
    """Run chessboard calibration. Returns camera_matrix, dist_coeffs, error."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []
    img_points = []
    image_size = None

    images = sorted(
        glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpg")
    )
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            print(f"  Skipping {path} (no corners found)")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  OK: {path}")

    if len(obj_points) < 3:
        print("Need at least 3 images with detected corners")
        sys.exit(1)

    print(f"\nCalibrating with {len(obj_points)} images ...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Compute reprojection error
    total_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(obj_points)

    print(f"Calibration done. RMS error: {ret:.4f}, Mean reprojection error: {mean_error:.4f}")
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion:\n{dist_coeffs}")

    return camera_matrix, dist_coeffs, image_size, mean_error


def save_to_yaml(filepath: str, camera_matrix, dist_coeffs, image_size, error):
    """Save or merge intrinsics into a YAML file."""
    # Load existing data if file exists (to preserve extrinsics)
    data = {}
    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass

    data["intrinsics"] = {
        "camera_matrix": camera_matrix.flatten().tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_size": list(image_size),
        "reprojection_error": float(error),
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved intrinsics to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument("--image-dir", required=True, help="Directory with calibration images")
    parser.add_argument("--output", default="camera.yaml", help="Output YAML file")
    parser.add_argument("--board-size", nargs=2, type=int, default=[8, 6],
                        help="Chessboard inner corners (cols rows)")
    parser.add_argument("--square-size", type=float, default=25.0,
                        help="Chessboard square size in mm")
    args = parser.parse_args()

    board_size = tuple(args.board_size)
    camera_matrix, dist_coeffs, image_size, error = calibrate(
        args.image_dir, board_size, args.square_size
    )
    save_to_yaml(args.output, camera_matrix, dist_coeffs, image_size, error)


if __name__ == "__main__":
    main()
