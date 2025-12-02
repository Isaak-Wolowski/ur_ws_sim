#!/usr/bin/env python3
import os
import glob
import argparse

import cv2
import numpy as np


# =========================
# 1. Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from chessboard images in a run."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base data directory (default: /root/ur_ws_sim/data)",
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
        help="Camera name, e.g. realsense, zed_left, zed_right",
    )

    parser.add_argument(
        "--nx",
        type=int,
        required=True,
        help="Number of inner corners along board width (columns).",
    )

    parser.add_argument(
        "--ny",
        type=int,
        required=True,
        help="Number of inner corners along board height (rows).",
    )

    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square size in meters.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output file name or path for calibration npz. "
            "If not given, uses "
            "<base-dir>/<run-id>/calib/CameraParams_<camera>.npz. "
            "If a bare filename is given (no '/'), it is placed "
            "inside <base-dir>/<run-id>/calib/."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show chessboard detections while processing.",
    )

    return parser.parse_args()


# =========================
# 2. Chessboard helpers
# =========================

def create_chessboard_object_points(nx, ny, square_size):
    """
    3D points for the chessboard corners in the board frame.
    x axis: along width (nx)
    y axis: along height (ny)
    """
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_corners_in_images(image_dir, nx, ny, show=False):
    """
    Scan image_dir for img_*.png, detect chessboard corners.
    Returns:
      - objpoints: list of (N, 3) arrays in board frame
      - imgpoints: list of (N, 2) arrays in image pixels
      - image_size: (width, height)
      - used_filenames: list[str] of image file basenames used
      - per_image_indices: indices in original sorted file list
    """
    pattern_size = (nx, ny)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "img_*.png")))

    print(f"Searching images in: {image_dir}")
    print(f"Found {len(image_paths)} images for calibration.")

    if not image_paths:
        raise RuntimeError("No images found. Are you sure the run has been captured?")

    objpoints = []
    imgpoints = []
    used_filenames = []
    per_image_indices = []

    objp_template = create_chessboard_object_points(nx, ny, square_size=1.0)  # scale later

    img_size = None
    valid_count = 0

    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Failed to read image: {os.path.basename(path)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if not ret:
            print(f"❌ Chessboard NOT found in: {os.path.basename(path)}")
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        valid_count += 1
        print(f"✔️ Chessboard found in: {os.path.basename(path)}")

        # we store object points scaled later by square_size
        objpoints.append(objp_template.copy())
        imgpoints.append(corners_refined.reshape(-1, 2))
        used_filenames.append(os.path.basename(path))
        per_image_indices.append(idx)

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
            cv2.imshow("Chessboard detection", vis)
            key = cv2.waitKey(50)
            if key == 27:  # ESC
                break

    if show:
        cv2.destroyAllWindows()

    print(f"✔️ Valid detections: {valid_count}")
    if valid_count == 0:
        raise RuntimeError("No valid chessboard detections, cannot calibrate.")

    return objpoints, imgpoints, img_size, used_filenames


# =========================
# 3. Calibration
# =========================

def calibrate_camera(objpoints, imgpoints, image_size, nx, ny, square_size):
    """
    Run cv2.calibrateCamera and compute per-view reprojection errors.
    objpoints come in as unit-square board; we scale by square_size here.
    """
    # scale object points
    objpoints_scaled = [op * square_size for op in objpoints]

    print("\n=== Performing calibration... ===")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_scaled,
        [ip.reshape(-1, 1, 2) for ip in imgpoints],  # OpenCV expects (N,1,2)
        image_size,
        None,
        None,
    )

    # compute per-view reprojection error
    per_view_errors = []
    for op, ip, rv, tv in zip(objpoints_scaled, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(op, rv, tv, K, dist)
        projected = projected.reshape(-1, 2)
        ip2 = ip.reshape(-1, 2)
        err = np.linalg.norm(ip2 - projected, axis=1)
        per_view_errors.append(err.mean())

    per_view_errors = np.array(per_view_errors, dtype=np.float64)

    print("\n=== Calibration Results ===")
    print(f"RMS reprojection error (overall): {ret}")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients:")
    print(dist)
    print("\nPer-view mean reprojection errors (px):")
    print(per_view_errors)

    return ret, K, dist, rvecs, tvecs, per_view_errors


# =========================
# 4. Saving
# =========================

def resolve_output_path(base_dir, run_id, camera_name, output_arg):
    """
    Decide where to save CameraParams_<camera>.npz

    Rules:
      - If output_arg is None:
        <base>/<run>/calib/CameraParams_<camera>.npz
      - If output_arg contains '/', use it as given.
      - Else:
        <base>/<run>/calib/<output_arg>
    """
    run_dir = os.path.join(base_dir, run_id)
    calib_dir = os.path.join(run_dir, "calib")
    os.makedirs(calib_dir, exist_ok=True)

    if output_arg is None:
        filename = f"CameraParams_{camera_name}.npz"
        return os.path.join(calib_dir, filename)

    if "/" in output_arg:
        # user gave path
        out_dir = os.path.dirname(output_arg)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return output_arg

    # bare filename -> put into calib_dir
    return os.path.join(calib_dir, output_arg)


def save_calibration(
    output_path,
    rms,
    K,
    dist,
    rvecs,
    tvecs,
    objpoints,
    imgpoints,
    image_size,
    nx,
    ny,
    square_size,
    filenames,
    per_view_errors,
):
    np.savez(
        output_path,
        rms=rms,
        K=K,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        nx=nx,
        ny=ny,
        square_size=square_size,
        image_size=np.array(image_size, dtype=np.int32),
        filenames=np.array(filenames),
        per_view_errors=per_view_errors,
    )
    print(f"\n📁 Saved calibration to: {output_path}")


# =========================
# 5. Main
# =========================

def main():
    args = parse_args()

    run_dir = os.path.join(args.base_dir, args.run_id)
    image_dir = os.path.join(run_dir, "images", args.camera_name)

    print("=== Camera Calibration ===")
    print(f"📂 Base dir   : {args.base_dir}")
    print(f"🏃 Run ID     : {args.run_id}")
    print(f"📷 Camera     : {args.camera_name}")
    print(f"📂 Image dir  : {image_dir}")
    print(f"Chessboard nx x ny: {args.nx} x {args.ny}")
    print(f"Square size       : {args.square_size} m")
    print("")

    objpoints, imgpoints, image_size, filenames = find_corners_in_images(
        image_dir, args.nx, args.ny, show=args.show
    )

    rms, K, dist, rvecs, tvecs, per_view_errors = calibrate_camera(
        objpoints, imgpoints, image_size, args.nx, args.ny, args.square_size
    )

    out_path = resolve_output_path(
        args.base_dir, args.run_id, args.camera_name, args.output
    )

    save_calibration(
        out_path,
        rms,
        K,
        dist,
        rvecs,
        tvecs,
        objpoints,
        imgpoints,
        image_size,
        args.nx,
        args.ny,
        args.square_size,
        filenames,
        per_view_errors,
    )


if __name__ == "__main__":
    main()

