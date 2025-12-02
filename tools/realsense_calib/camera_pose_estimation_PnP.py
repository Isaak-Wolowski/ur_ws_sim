#!/usr/bin/env python3
import os
import glob
import argparse
import csv

import cv2
import numpy as np


# =========================
# === 1. Argument parsing ==
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate board pose for each image in a run using PnP."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base directory where run folders live. "
             "Default: /root/ur_ws_sim/data",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID, e.g. run_001, run_100",
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
        default=10,
        help="Number of inner corners along board width (columns). Default: 10",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=7,
        help="Number of inner corners along board height (rows). Default: 7",
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=0.024,
        help="Chessboard square size in meters. Default: 0.024 m",
    )

    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Optional explicit calibration file path (.npz). "
             "If not given, uses <base-dir>/<run-id>/calib/CameraParams_<camera>.npz",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for poses (.npz). "
             "If not given, saves to <base-dir>/<run-id>/calib/poses_<camera>.npz",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show detected corners and coordinate axes (for debugging).",
    )

    return parser.parse_args()


# ==============================================
# === 2. Chessboard object points in 3D (board) =
# ==============================================

def create_chessboard_object_points(nx, ny, square_size):
    """
    Create 3D points for the chessboard corners in the board frame.

    Coordinate system:
      - x along board width (columns, 0..nx-1)
      - y along board height (rows, 0..ny-1)
      - z = 0 on board plane
    """
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= square_size
    return objp


# ==================================
# === 3. Load camera calibration  ===
# ==================================

def load_calibration(base_dir, run_id, camera_name, calib_file=None):
    """
    Load camera intrinsics from npz.

    Expected keys:
      - K   : 3x3 camera matrix
      - dist: distortion coefficients

    Also accepts old names:
      - cameraMatrix, distCoeffs
    """
    if calib_file is None:
        calib_dir = os.path.join(base_dir, run_id, "calib")
        calib_file = os.path.join(calib_dir, f"CameraParams_{camera_name}.npz")

    if not os.path.exists(calib_file):
        raise FileNotFoundError(
            f"Calibration file not found: {calib_file}"
        )

    print(f"🔧 Camera: {camera_name}")
    print(f"📂 Calibration file: {calib_file}")

    data = np.load(calib_file)

    # Try new keys
    if "K" in data:
        K = data["K"]
    elif "cameraMatrix" in data:
        K = data["cameraMatrix"]
    else:
        raise KeyError("No 'K' or 'cameraMatrix' in calibration file.")

    if "dist" in data:
        dist = data["dist"]
    elif "distCoeffs" in data:
        dist = data["distCoeffs"]
    else:
        raise KeyError("No 'dist' or 'distCoeffs' in calibration file.")

    return K, dist


# ======================================
# === 4. Detect corners in all images ===
# ======================================

def find_chessboard_corners(image_dir, nx, ny, show=False):
    pattern_size = (nx, ny)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "img_*.png")))

    print(f"📂 Image directory : {image_dir}")
    print(f"Found {len(image_paths)} images.")

    if not image_paths:
        raise RuntimeError("No images found for pose estimation.")

    all_results = []  # list of (image_path, bgr_img, gray_img, corners_refined)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Failed to read image: {os.path.basename(path)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        print(f"✔️ Chessboard found in: {os.path.basename(path)}")
        all_results.append((path, img, gray, corners_refined))

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, True)
            cv2.imshow("Corners", vis)
            key = cv2.waitKey(50)
            if key == 27:  # ESC
                break

    if show:
        cv2.destroyAllWindows()

    if not all_results:
        raise RuntimeError("No valid chessboard detections; cannot estimate poses.")

    return all_results


# ===============================
# === 5. Pose estimation (PnP) ==
# ===============================

def estimate_poses_pnp(objp, detections, K, dist, show=False):
    """
    detections: list of (path, bgr_img, gray_img, corners_refined)
    returns:
      filenames: list[str]
      rvecs:     list[(3,1)]
      tvecs:     list[(3,1)]
    """
    filenames = []
    rvecs = []
    tvecs = []

    for path, img, gray, corners in detections:
        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        name = os.path.basename(path)

        if not ok:
            print(f"❌ solvePnP failed for image: {name}")
            continue

        filenames.append(name)
        rvecs.append(rvec)
        tvecs.append(tvec)

        t = tvec.reshape(-1)
        print(f"\n📸 {name}")
        print(f"  t (board in camera) [m]: [{t[0]: .4f}, {t[1]: .4f}, {t[2]: .4f}]")
        print(f"  rvec (Rodrigues)   [rad]: [{rvec[0,0]: .4f}, {rvec[1,0]: .4f}, {rvec[2,0]: .4f}]")

        if show:
            axis_len = 0.05  # 5 cm
            axis_3d = np.float32([
                [0, 0, 0],
                [axis_len, 0, 0],
                [0, axis_len, 0],
                [0, 0, axis_len],
            ])
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
            img_vis = img.copy()
            p0 = tuple(imgpts[0].ravel().astype(int))
            pX = tuple(imgpts[1].ravel().astype(int))
            pY = tuple(imgpts[2].ravel().astype(int))
            pZ = tuple(imgpts[3].ravel().astype(int))

            cv2.line(img_vis, p0, pX, (0, 0, 255), 2)  # X - red
            cv2.line(img_vis, p0, pY, (0, 255, 0), 2)  # Y - green
            cv2.line(img_vis, p0, pZ, (255, 0, 0), 2)  # Z - blue

            cv2.imshow("Pose (axes)", img_vis)
            key = cv2.waitKey(50)
            if key == 27:  # ESC
                break

    if show:
        cv2.destroyAllWindows()

    print(f"\nTotal successful PnP solutions: {len(filenames)}")
    return filenames, rvecs, tvecs


# ====================================
# === 6. Save poses for this run   ===
# ====================================

def save_poses(base_dir, run_id, camera_name, output_path,
               filenames, rvecs, tvecs, K, dist):
    """
    Save results to npz (machine) and CSV (human).

    npz:
      - filenames: (N,)
      - rvecs:     (N,3)
      - tvecs:     (N,3)
      - K, dist

    csv:
      filename,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z
    """
    if output_path is None:
        run_dir = os.path.join(base_dir, run_id)
        calib_dir = os.path.join(run_dir, "calib")
        os.makedirs(calib_dir, exist_ok=True)
        output_path = os.path.join(calib_dir, f"poses_{camera_name}.npz")

    # Pack into arrays
    rvecs_arr = np.array([rv.reshape(3) for rv in rvecs], dtype=np.float64)
    tvecs_arr = np.array([tv.reshape(3) for tv in tvecs], dtype=np.float64)
    filenames_arr = np.array(filenames)

    np.savez(
        output_path,
        filenames=filenames_arr,
        rvecs=rvecs_arr,
        tvecs=tvecs_arr,
        K=K,
        dist=dist,
    )

    print(f"\n📁 Saved poses (binary npz) to: {output_path}")

    # ---- Human-readable CSV export ----
    # Derive a CSV path next to the npz
    base, _ = os.path.splitext(output_path)
    csv_path = base + ".csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "rvec_x", "rvec_y", "rvec_z",
            "tvec_x", "tvec_y", "tvec_z",
        ])
        for fname, rv, tv in zip(filenames_arr, rvecs_arr, tvecs_arr):
            writer.writerow([
                fname,
                float(rv[0]), float(rv[1]), float(rv[2]),
                float(tv[0]), float(tv[1]), float(tv[2]),
            ])

    print(f"📝 Saved human-readable CSV to: {csv_path}")
    print("Each row stores board-in-camera pose per image.")


# ======================
# === 7. Main entry  ===
# ======================

def main():
    args = parse_args()

    # NOTE: images are in: <base-dir>/<run-id>/images/<camera-name>
    image_dir = os.path.join(args.base_dir, args.run_id, "images", args.camera_name)

    print("=== Camera Pose Estimation (PnP) ===")
    print(f"📂 Base dir    : {args.base_dir}")
    print(f"🏃 Run ID      : {args.run_id}")
    print(f"📷 Camera      : {args.camera_name}")
    print(f"📂 Image dir   : {image_dir}")
    print(f"Chessboard nx x ny: {args.nx} x {args.ny}")
    print(f"Square size       : {args.square_size} m\n")

    # 1) Load intrinsics
    K, dist = load_calibration(args.base_dir, args.run_id, args.camera_name, args.calib_file)

    # 2) Prepare 3D points
    objp = create_chessboard_object_points(args.nx, args.ny, args.square_size)

    # 3) Detect corners in all images
    detections = find_chessboard_corners(image_dir, args.nx, args.ny, show=args.show)

    # 4) Run PnP for each image
    filenames, rvecs, tvecs = estimate_poses_pnp(objp, detections, K, dist, show=args.show)

    if not filenames:
        raise RuntimeError("No poses estimated (no successful PnP).")

    # 5) Save everything
    save_poses(
        args.base_dir,
        args.run_id,
        args.camera_name,
        args.output,
        filenames,
        rvecs,
        tvecs,
        K,
        dist,
    )


if __name__ == "__main__":
    main()

