#!/usr/bin/env python3
import os
import glob
import argparse

import cv2
import numpy as np


# =========================
# === 1. Argument parsing ===
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera calibration using chessboard images from a run folder"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base data directory (where run folders live). "
             "Default: /root/ur_ws_sim/data",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID, e.g. run_001, run_010",
    )

    parser.add_argument(
        "--camera-name",
        type=str,
        required=True,
        help="Camera name, e.g. realsense, zed_left, zed_right",
    )

    # Chessboard spec: you told me 8x11
    # -> 8 corners in vertical (rows), 11 in horizontal (cols)
    parser.add_argument(
        "--nx",
        type=int,
        default=10,
        help="Number of inner corners along the board width (columns). Default: 10",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=7,
        help="Number of inner corners along the board height (rows). Default: 7",
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=0.03,
        help="Chessboard square size in meters. Default: 0.03 m",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional explicit path to save calibration .npz. "
            "If not given, saves to <base-dir>/calib/CameraParams_<camera>.npz"
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detected corners during calibration (slow).",
    )

    return parser.parse_args()


# ==============================================
# === 2. Prepare object points (chessboard 3D) ===
# ==============================================

def create_chessboard_object_points(nx, ny, square_size):
    """
    Create an array of 3D points for the chessboard corners in the board frame.

    nx: number of inner corners along width  (columns)
    ny: number of inner corners along height (rows)
    """
    objp = np.zeros((ny * nx, 3), np.float32)
    # (0,0), (1,0), ... (nx-1, ny-1) scaled by square_size
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size
    return objp


# ========================================
# === 3. Load images & find chessboard ===
# ========================================

def find_corners_in_images(image_dir, nx, ny, show=False):
    """
    Scan all img_*.png in image_dir, detect chessboard corners.

    Returns:
      objpoints: list of 3D points (same board pattern each time)
      imgpoints: list of 2D points in images
      image_size: (w, h) from the first valid image
    """
    pattern_size = (nx, ny)

    # Collect images: we only rely on the img_XXXX_pose_YYYY.png naming
    image_paths = sorted(glob.glob(os.path.join(image_dir, "img_*.png")))

    print(f"Searching images in: {image_dir}")
    print(f"Found {len(image_paths)} images for calibration.")

    if not image_paths:
        raise RuntimeError("No images found. Are you sure the run has been captured?")

    objpoints = []
    imgpoints = []

    objp_template = create_chessboard_object_points(nx, ny, square_size=1.0)
    # We'll scale later for real square size; for corner detection pattern_size is enough.

    image_size = None
    valid_count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Failed to read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # Try find corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if not ret:
            print(f"❌ Chessboard NOT found in: {os.path.basename(path)}")
            continue

        # Refine corner positions
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp_template.copy())
        imgpoints.append(corners_refined)
        valid_count += 1

        if show:
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
            cv2.imshow("Corners", img)
            key = cv2.waitKey(50)
            if key == 27:  # ESC to quit early
                break

    if show:
        cv2.destroyAllWindows()

    print(f"✔️ Valid detections: {valid_count}")

    if valid_count == 0:
        raise RuntimeError("No valid chessboard detections. Cannot calibrate.")

    return objpoints, imgpoints, image_size


# ===================================
# === 4. Run cv2.calibrateCamera  ===
# ===================================

def calibrate_camera(objpoints, imgpoints, image_size, square_size):
    """
    Run OpenCV calibrateCamera with scaled object points.
    """
    # Scale object points by square size in meters
    objpoints_scaled = [op * square_size for op in objpoints]

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_scaled,
        imgpoints,
        image_size,
        None,
        None,
    )

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error: {ret}\n")
    print("Camera Matrix (K):\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


# ====================================
# === 5. Save calibration to .npz  ===
# ====================================

def save_calibration(base_dir, camera_name, output_path, camera_matrix, dist_coeffs):
    """
    Save calibration as npz with keys 'K' and 'dist':
      - K: 3x3 camera matrix
      - dist: distortion coefficients
    """
    if output_path is None:
        calib_dir = os.path.join(base_dir, "calib")
        os.makedirs(calib_dir, exist_ok=True)
        output_path = os.path.join(calib_dir, f"CameraParams_{camera_name}.npz")

    np.savez(output_path, K=camera_matrix, dist=dist_coeffs)
    print(f"\n📁 Saved calibration to: {output_path}")


# ======================
# === 6. Main entry  ===
# ======================

def main():
    args = parse_args()

    # Real image directory from run + camera
    image_dir = os.path.join(args.base_dir, args.run_id, args.camera_name)

    print("=== Camera Calibration ===")
    print(f"📂 Base dir   : {args.base_dir}")
    print(f"🏃 Run ID     : {args.run_id}")
    print(f"📷 Camera     : {args.camera_name}")
    print(f"📂 Image dir  : {image_dir}")
    print(f"Chessboard nx x ny: {args.nx} x {args.ny}")
    print(f"Square size       : {args.square_size} m\n")

    # 1) Find corners
    objpoints, imgpoints, image_size = find_corners_in_images(
        image_dir, args.nx, args.ny, show=args.show
    )

    print(f"\nImage resolution: {image_size[0]} x {image_size[1]}")

    # 2) Calibrate
    print("\n=== Performing calibration... ===")
    _, K, dist, _, _ = calibrate_camera(
        objpoints, imgpoints, image_size, args.square_size
    )

    # 3) Save
    save_calibration(args.base_dir, args.camera_name, args.output, K, dist)


if __name__ == "__main__":
    main()

