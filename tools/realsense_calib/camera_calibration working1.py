#!/usr/bin/env python3
import argparse
import glob
import os

import cv2
import numpy as np


# ==============================
# 1. Corner Detection
# ==============================
def find_corners_in_images(images_dir, nx, ny, square_size, show=False):
    """
    Find chessboard corners in all images in images_dir.

    nx, ny = number of inner corners along x and y (columns, rows).
    square_size in meters.
    """

    # Prepare object points for the chessboard pattern
    # (0,0,0), (1,0,0), ..., (nx-1, ny-1, 0) scaled by square_size
    objp = np.zeros((nx * ny, 3), np.float32)
    # OpenCV expects pattern size as (nx, ny) -> columns, rows
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= float(square_size)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Collect images
    image_files = []
    image_files.extend(sorted(glob.glob(os.path.join(images_dir, "*.png"))))
    image_files.extend(sorted(glob.glob(os.path.join(images_dir, "*.jpg"))))

    print(f"Searching images in: {images_dir}")
    print(f"Found {len(image_files)} images for calibration.")

    if not image_files:
        raise RuntimeError("No images found. Are you sure the run has been captured?")

    image_size = None
    valid_count = 0

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ Could not read image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
            print(f"Image resolution: {image_size[0]} x {image_size[1]}")

        # Find the chessboard corners
        pattern_size = (nx, ny)
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner positions
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners_subpix = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )

            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            valid_count += 1

            print(f"✔️ Chessboard found in: {os.path.basename(fname)}")

            if show:
                cv2.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
                cv2.imshow("Corners", img)
                cv2.waitKey(200)
        else:
            print(f"❌ Chessboard NOT found in: {os.path.basename(fname)}")

    if show:
        cv2.destroyAllWindows()

    print(f"✔️ Valid detections: {valid_count}")

    if valid_count < 5:
        raise RuntimeError("Not enough valid detections for a reliable calibration.")

    return objpoints, imgpoints, image_size


# ==============================
# 2. Calibration Routine
# ==============================
def calibrate_camera(objpoints, imgpoints, image_size):
    """
    Run OpenCV's calibrateCamera and return RMS, camera matrix and distortion.
    """
    print("\n=== Performing calibration... ===")

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error: {ret}")
    print("\nCamera Matrix (K):\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    return ret, camera_matrix, dist_coeffs


# ==============================
# 3. Save Calibration
# ==============================
def save_calibration(output_path, camera_matrix, dist_coeffs):
    np.savez(output_path, K=camera_matrix, dist=dist_coeffs)
    print(f"\n📁 Saved calibration to: {output_path}")


# ==============================
# 4. Main
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Camera calibration from chessboard images")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base directory where run_* folders live",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID, e.g. run_001",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        required=True,
        help="Camera name, e.g. realsense, zed_left, etc.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=7,
        help="Number of inner corners along X (columns)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=10,
        help="Number of inner corners along Y (rows)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.024,
        help="Chessboard square size in meters (e.g. 0.024 for 24mm)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="CameraParams.npz",
        help="Output .npz file name",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show corner detection results",
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    run_id = args.run_id
    camera_name = args.camera_name
    nx = args.nx
    ny = args.ny
    square_size = args.square_size
    output = args.output

    # Print configuration summary
    print("=== Camera Calibration ===")
    print(f"📂 Base dir   : {base_dir}")
    print(f"🏃 Run ID     : {run_id}")
    print(f"📷 Camera     : {camera_name}")

    # IMPORTANT: match your actual directory layout:
    # /root/ur_ws_sim/data/run_001/images/realsense
    images_dir = os.path.join(base_dir, run_id, "images", camera_name)
    print(f"📂 Image dir  : {images_dir}")
    print(f"Chessboard nx x ny: {nx} x {ny}")
    print(f"Square size       : {square_size} m\n")

    # 1) Detect corners on all images
    objpoints, imgpoints, image_size = find_corners_in_images(
        images_dir, nx, ny, square_size, show=args.show
    )

    # 2) Run calibration
    _, K, dist = calibrate_camera(objpoints, imgpoints, image_size)

    # 3) Save calibration
    output_path = os.path.join(os.getcwd(), output)
    save_calibration(output_path, K, dist)


if __name__ == "__main__":
    main()

