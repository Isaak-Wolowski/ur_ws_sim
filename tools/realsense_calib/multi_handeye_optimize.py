#!/usr/bin/env python3
import os
import argparse

import numpy as np
import cv2
from scipy.optimize import minimize


# ======================================
# 1. Argument parsing
# ======================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Phase 2: joint multi-camera hand–eye refinement.\n"
            "Optimizes EE->camera extrinsics for several cameras so that\n"
            "all of them agree on a single rigid chessboard pose in base."
        )
    )

    p.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base directory for runs and calibration data "
             "(default: /root/ur_ws_sim/data)",
    )

    p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID, e.g. run_001",
    )

    p.add_argument(
        "--camera-names",
        type=str,
        required=True,
        help=(
            "Comma-separated list of camera names, "
            "e.g. 'realsense,webcam' or 'realsense,zed_left,zed_right'."
        ),
    )

    p.add_argument(
        "--board-poses-dir",
        type=str,
        default=None,
        help=(
            "Optional dir where poses_<camera>.npz live. "
            "If not given, uses <base-dir>/<run-id>/calib"
        ),
    )

    p.add_argument(
        "--robot-poses",
        type=str,
        default=None,
        help="Optional explicit robot poses file. If not given, "
             "tries <base-dir>/<run-id>/calib/robot_poses.npz, "
             "then <base-dir>/<run-id>/calib/robot_poses.csv.",
    )

    p.add_argument(
        "--lambda-trans",
        type=float,
        default=1.0,
        help="Weight for translation consistency term (default: 1.0)",
    )

    p.add_argument(
        "--lambda-rot",
        type=float,
        default=1.0,
        help="Weight for rotation consistency term (default: 1.0)",
    )

    p.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations for SLSQP (default: 200)",
    )

    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output npz file for refined multi-camera extrinsics. "
            "Default: <base-dir>/<run-id>/calib/multi_handeye_optimized.npz"
        ),
    )

    return p.parse_args()


# ======================================
# 2. Quaternion & transform helpers
# ======================================

def quat_to_rot(q):
    """
    Quaternion -> 3x3 rotation matrix.
    q = [qx, qy, qz, qw] (same order as geometry_msgs)
    """
    qx, qy, qz, qw = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = q / n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ])
    return R


def rt_to_T(R, t):
    """R (3x3), t (3,) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T):
    """Invert 4x4 transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def format_matrix_rows(M):
    """Helper: format a 4x4 (or 3x3) into YAML-like rows."""
    lines = []
    for row in M:
        formatted = ", ".join(f"{v:.6f}" for v in row)
        lines.append(f"  - [{formatted}]")
    return "\n".join(lines)


# ======================================
# 3. Load board poses (PnP)
# ======================================

def load_board_poses_raw(base_dir, run_id, camera_name, board_poses_dir=None):
    """
    Load raw board-in-camera poses from PnP npz:
      - filenames: (N,)
      - rvecs:     (N,3)
      - tvecs:     (N,3)

    Default path:
      <base-dir>/<run-id>/calib/poses_<camera>.npz
    """
    if board_poses_dir is None:
        board_poses_dir = os.path.join(base_dir, run_id, "calib")

    board_poses_path = os.path.join(
        board_poses_dir, f"poses_{camera_name}.npz"
    )

    if not os.path.exists(board_poses_path):
        raise FileNotFoundError(f"Board poses file not found: {board_poses_path}")

    data = np.load(board_poses_path, allow_pickle=True)
    filenames = data["filenames"]       # (N,) of strings
    rvecs = data["rvecs"]               # (N,3)
    tvecs = data["tvecs"]               # (N,3)

    return filenames, rvecs, tvecs


# ======================================
# 4. Load robot poses (ee in base)
# ======================================

def load_robot_poses(base_dir, run_id, robot_poses_path=None):
    """
    Try npz first:
      - filenames: (N,)
      - pos:       (N,3)   3D position in base
      - quat:      (N,4)   [qx,qy,qz,qw] orientation of ee in base

    If that doesn't exist, try CSV.

    For your CSV from hemi_motion_rs_node:
      - filenames: column 'image_name'
      - pos:       px, py, pz
      - quat:      qx, qy, qz, qw

    Default locations:
      NPZ: <base-dir>/<run-id>/calib/robot_poses.npz
      CSV: <base-dir>/<run-id>/calib/robot_poses.csv
    """
    if robot_poses_path is None:
        robot_poses_path = os.path.join(base_dir, run_id, "calib", "robot_poses.npz")

    # --- NPZ path ---
    if os.path.exists(robot_poses_path) and robot_poses_path.endswith(".npz"):
        data = np.load(robot_poses_path, allow_pickle=True)
        filenames = data["filenames"]
        pos = data["pos"]      # (N,3)
        quat = data["quat"]    # (N,4) [qx,qy,qz,qw]
        return filenames, pos, quat

    # --- CSV fallback ---
    if robot_poses_path is None or not os.path.exists(robot_poses_path):
        csv_path = os.path.join(base_dir, run_id, "calib", "robot_poses.csv")
    else:
        csv_path = robot_poses_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Robot poses file not found. Tried:\n"
            f"  {robot_poses_path}\n"
            f"  {csv_path}"
        )

    print(f"Loading robot poses from CSV: {csv_path}")
    raw = np.genfromtxt(
        csv_path, delimiter=",", dtype=None, names=True, encoding=None
    )

    colnames = raw.dtype.names
    print(f"Detected CSV columns: {colnames}")

    if colnames is None:
        raise RuntimeError(
            "CSV has no header / columns could not be inferred. "
            "Make sure the first line is a header."
        )

    # --- pick filename column ---
    filename_col = None
    for name in ("image_name", "filename", "file", "img", "image"):
        if name in colnames:
            filename_col = name
            break

    if filename_col is None:
        for name in colnames:
            if np.issubdtype(raw[name].dtype, np.str_):
                filename_col = name
                break

    if filename_col is None:
        filename_col = colnames[0]

    filenames = raw[filename_col]
    print(f"Using column '{filename_col}' as filenames.")

    # --- choose pos/quat columns ---
    expected_pos = ["px", "py", "pz"]
    expected_quat = ["qx", "qy", "qz", "qw"]

    if all(name in colnames for name in expected_pos + expected_quat):
        pos_cols = expected_pos
        quat_cols = expected_quat
        print(f"Using position columns: {pos_cols}")
        print(f"Using quaternion columns: {quat_cols}")
    else:
        numeric_cols = [
            n for n in colnames
            if n != filename_col and not np.issubdtype(raw[n].dtype, np.str_)
        ]

        if len(numeric_cols) < 7:
            raise RuntimeError(
                f"Expected at least 7 numeric columns for x,y,z,qx,qy,qz,qw, "
                f"but found {len(numeric_cols)}: {numeric_cols}"
            )

        pos_cols = numeric_cols[0:3]
        quat_cols = numeric_cols[3:7]

        print("Explicit px/py/pz + qx/qy/qz/qw not found, "
              "falling back to numeric columns.")
        print(f"Using position columns: {pos_cols}")
        print(f"Using quaternion columns: {quat_cols}")

    pos = np.vstack([raw[c] for c in pos_cols]).T.astype(float)
    quat = np.vstack([raw[c] for c in quat_cols]).T.astype(float)

    return filenames, pos, quat


# ======================================
# 5. Match board & robot by filename
# ======================================

def match_by_filenames_raw(board_files, robot_files, rvecs, tvecs, pos, quat):
    """
    board_files: list/array of filenames (from PnP)
    robot_files: list/array of filenames (from robot logging)
    returns matched arrays for a single camera.
    """
    board_map = {f: i for i, f in enumerate(board_files)}
    robot_map = {f: i for i, f in enumerate(robot_files)}

    common = sorted(set(board_map.keys()) & set(robot_map.keys()))
    if len(common) == 0:
        raise RuntimeError("No overlapping filenames between board and robot poses.")

    print(f"  Found {len(common)} common images for this camera.")

    pos_used = []
    quat_used = []
    rvecs_used = []
    tvecs_used = []
    used_filenames = []

    for fname in common:
        i_b = board_map[fname]
        i_r = robot_map[fname]

        used_filenames.append(fname)
        pos_used.append(pos[i_r])
        quat_used.append(quat[i_r])
        rvecs_used.append(rvecs[i_b])
        tvecs_used.append(tvecs[i_b])

    return (np.array(used_filenames),
            np.array(pos_used),
            np.array(quat_used),
            np.array(rvecs_used),
            np.array(tvecs_used))


# ======================================
# 6. Parameter packing / unpacking
# ======================================

def pack_params(cam_names, cam2_Ree_cam, cam2_tee_cam):
    """
    Pack per-camera extrinsics into a single parameter vector.
    Each camera k has:
      - 3 params: Rodrigues vector for R_ee_cam
      - 3 params: translation t_ee_cam
    """
    theta_list = []
    for cam in cam_names:
        R = cam2_Ree_cam[cam]
        t = cam2_tee_cam[cam]
        rvec, _ = cv2.Rodrigues(R)  # (3,1)
        theta_list.append(rvec.reshape(3))
        theta_list.append(t.reshape(3))
    return np.concatenate(theta_list)


def unpack_params(theta, cam_names):
    """
    Inverse of pack_params.
    Returns:
      cam2_Ree_cam, cam2_tee_cam
    where each is a dict: cam_name -> R or t
    """
    cam2_R = {}
    cam2_t = {}
    idx = 0
    for cam in cam_names:
        rvec = theta[idx:idx+3]
        t = theta[idx+3:idx+6]
        idx += 6
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam2_R[cam] = R
        cam2_t[cam] = t
    return cam2_R, cam2_t


# ======================================
# 7. Objective: board-in-base consistency across all cameras
# ======================================

def objective(theta, cam_names, cam_data, lambda_trans, lambda_rot):
    """
    theta: packed params for all cameras (EE->camera extrinsics).
           For camera k: [rvec(3), t(3)] describing ^eR_cam, ^e t_cam.
    cam_data: dict cam_name -> dict with:
       - pos:   (N_k, 3) EE position in base
       - quat:  (N_k, 4) EE orientation in base
       - rvecs: (N_k, 3) board in camera (Rodrigues)
       - tvecs: (N_k, 3) board in camera (translation)

    We build, for every sample and every camera:
       ^bT_t = ^bT_e  ^eT_cam  ^cT_t
    and penalize the variance (spread) of ^bT_t across all samples and cameras:
      - translation variance
      - rotation variance (angle to mean)
    """
    cam2_Ree_cam, cam2_tee_cam = unpack_params(theta, cam_names)

    board_positions = []
    board_rotations = []

    # Build all board poses in base for all cameras & samples
    for cam in cam_names:
        data = cam_data[cam]
        pos = data["pos"]      # (N_k,3)
        quat = data["quat"]    # (N_k,4)
        rvecs = data["rvecs"]  # (N_k,3)
        tvecs = data["tvecs"]  # (N_k,3)

        R_ee_cam = cam2_Ree_cam[cam]
        t_ee_cam = cam2_tee_cam[cam]
        T_ee_cam = rt_to_T(R_ee_cam, t_ee_cam)

        for p, q, rv, tv in zip(pos, quat, rvecs, tvecs):
            # base -> ee
            R_b_e = quat_to_rot(q)
            T_b_e = rt_to_T(R_b_e, p)

            # camera -> board: we have board in camera, so ^cT_t
            R_c_t, _ = cv2.Rodrigues(rv.reshape(3, 1))
            T_c_t = rt_to_T(R_c_t, tv.reshape(3))

            # base -> camera: ^bT_cam = ^bT_e ^eT_cam
            T_b_cam = T_b_e @ T_ee_cam

            # base -> board: ^bT_t = ^bT_cam ^cT_t
            T_b_t = T_b_cam @ T_c_t

            board_positions.append(T_b_t[:3, 3])
            board_rotations.append(T_b_t[:3, :3])

    board_positions = np.array(board_positions)          # (M,3)
    board_rotations = np.stack(board_rotations, axis=0)  # (M,3,3)
    M = board_positions.shape[0]

    if M == 0:
        # something is badly wrong; just return large cost
        return 1e9

    # --- Translation consistency: penalize spread around mean ---
    t_mean = board_positions.mean(axis=0)
    trans_res = board_positions - t_mean[None, :]
    # total squared norm
    trans_cost = np.sum(np.sum(trans_res ** 2, axis=1)) / M  # average

    # --- Rotation consistency: mean rotation via SVD + angle deviations ---
    R_sum = board_rotations.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt

    # rotation deviation: angle between R_mean and each R
    angles = []
    for R in board_rotations:
        R_delta = R_mean.T @ R
        tr = np.trace(R_delta)
        val = (tr - 1.0) / 2.0
        val = np.clip(val, -1.0, 1.0)
        angle = np.arccos(val)
        angles.append(angle)
    angles = np.array(angles)
    rot_cost = np.mean(angles ** 2)  # average squared angle [rad^2]

    # Combine with weights
    total_cost = lambda_trans * trans_cost + lambda_rot * rot_cost

    # Debugging: you can uncomment next line for verbose prints
    # print(f"cost: trans={trans_cost:.6e}, rot={rot_cost:.6e}, total={total_cost:.6e}")

    return float(total_cost)


# ======================================
# 8. Saving YAML summaries (per camera)
# ======================================

def save_per_camera_yaml(
    base_dir,
    run_id,
    cam_names,
    cam2_Ree_cam,
    cam2_tee_cam,
    cam2_Tbase_cam,
    suffix="_opt",
):
    calib_dir = os.path.join(base_dir, run_id, "calib")
    os.makedirs(calib_dir, exist_ok=True)

    for cam in cam_names:
        R_ee_c = cam2_Ree_cam[cam]
        t_ee_c = cam2_tee_cam[cam]
        T_ee_c = rt_to_T(R_ee_c, t_ee_c)
        T_b_c = cam2_Tbase_cam[cam]
        T_c_b = invert_T(T_b_c)

        yaml_path = os.path.join(
            calib_dir,
            f"handeye_{cam}{suffix}.yaml",
        )

        with open(yaml_path, "w") as f:
            f.write(f"method: joint_board_consistency\n")
            f.write(f"camera_name: {cam}\n")
            f.write(f"run_id: {run_id}\n")
            f.write("\n")

            f.write("T_ee2cam:\n")
            f.write(format_matrix_rows(T_ee_c) + "\n\n")

            f.write("T_base2cam:\n")
            f.write(format_matrix_rows(T_b_c) + "\n\n")

            f.write("T_cam2base:\n")
            f.write(format_matrix_rows(T_c_b) + "\n")

        print(f"  📝 Saved optimized YAML for '{cam}' to: {yaml_path}")


# ======================================
# 9. Main
# ======================================

def main():
    args = parse_args()

    run_dir = os.path.join(args.base_dir, args.run_id)
    calib_dir = os.path.join(run_dir, "calib")
    os.makedirs(calib_dir, exist_ok=True)

    cam_names = [
        c.strip() for c in args.camera_names.split(",") if c.strip()
    ]
    if not cam_names:
        raise RuntimeError("No valid camera names parsed from --camera-names")

    print("=== Phase 2: Joint Multi-Camera Hand–Eye Optimization ===")
    print(f"📂 Base dir     : {args.base_dir}")
    print(f"🏃 Run ID       : {args.run_id}")
    print(f"📷 Cameras      : {cam_names}")
    print(f"📂 Run dir      : {run_dir}")
    print(f"λ_trans (w_t)   : {args.lambda_trans}")
    print(f"λ_rot   (w_R)   : {args.lambda_rot}")
    print(f"max_iter        : {args.max_iter}\n")

    # 1) Load robot poses
    robot_files, pos_all, quat_all = load_robot_poses(
        args.base_dir, args.run_id, args.robot_poses
    )

    # 2) For each camera: load board poses, match by filename
    cam_data = {}
    for cam in cam_names:
        print(f"\n=== Camera '{cam}' data preparation ===")
        board_files, rvecs, tvecs = load_board_poses_raw(
            args.base_dir, args.run_id, cam, args.board_poses_dir
        )

        used_filenames, pos_used, quat_used, rvecs_used, tvecs_used = \
            match_by_filenames_raw(
                board_files,
                robot_files,
                rvecs,
                tvecs,
                pos_all,
                quat_all,
            )

        print(f"  Using {len(used_filenames)} samples for joint optimization.")

        cam_data[cam] = {
            "filenames": used_filenames,
            "pos": pos_used,
            "quat": quat_used,
            "rvecs": rvecs_used,
            "tvecs": tvecs_used,
        }

    # 3) Build initial guess from phase-1 handeye_<cam>.npz if available
    cam2_Ree_cam_init = {}
    cam2_tee_cam_init = {}

    for cam in cam_names:
        he_path = os.path.join(calib_dir, f"handeye_{cam}.npz")
        if os.path.exists(he_path):
            data = np.load(he_path, allow_pickle=True)
            if "R_cam2ee" in data and "t_cam2ee" in data:
                R_cam2ee = data["R_cam2ee"]
                t_cam2ee = data["t_cam2ee"].reshape(3)

                # In phase 1 we treated this as T_ee^cam (EE->cam).
                # Keep the same convention: ^eR_cam, ^e t_cam
                cam2_Ree_cam_init[cam] = R_cam2ee
                cam2_tee_cam_init[cam] = t_cam2ee
                print(f"  Init for '{cam}' from {he_path}")
            else:
                print(f"  [WARN] {he_path} has no R_cam2ee/t_cam2ee; using identity init.")
                cam2_Ree_cam_init[cam] = np.eye(3)
                cam2_tee_cam_init[cam] = np.zeros(3)
        else:
            print(f"  [WARN] {he_path} not found; using identity init for '{cam}'.")
            cam2_Ree_cam_init[cam] = np.eye(3)
            cam2_tee_cam_init[cam] = np.zeros(3)

    theta0 = pack_params(cam_names, cam2_Ree_cam_init, cam2_tee_cam_init)

    print("\n=== Starting SLSQP optimization ===")

    res = minimize(
        objective,
        theta0,
        args=(cam_names, cam_data, args.lambda_trans, args.lambda_rot),
        method="SLSQP",
        options={
            "maxiter": args.max_iter,
            "ftol": 1e-10,
            "disp": True,
        },
    )

    print("\n=== Optimization result ===")
    print("success:", res.success)
    print("status :", res.status)
    print("message:", res.message)
    print("final cost:", res.fun)

    theta_opt = res.x

    # 4) Decode optimized params into transforms
    cam2_Ree_cam_opt, cam2_tee_cam_opt = unpack_params(theta_opt, cam_names)

    # Compute ^bT_cam per camera using *mean* EE pose from that camera's samples
    cam2_Tbase_cam_opt = {}
    for cam in cam_names:
        data = cam_data[cam]
        pos = data["pos"]
        quat = data["quat"]

        # Mean EE pose in base (very rough, but fine just to define base->cam)
        p_mean = pos.mean(axis=0)
        # For orientation, just take the first one (we only need one ^bT_e
        # to define a "typical" base->cam)
        q0 = quat[0]
        R_b_e = quat_to_rot(q0)
        T_b_e = rt_to_T(R_b_e, p_mean)

        R_ee_c = cam2_Ree_cam_opt[cam]
        t_ee_c = cam2_tee_cam_opt[cam]
        T_e_c = rt_to_T(R_ee_c, t_ee_c)

        T_b_c = T_b_e @ T_e_c
        cam2_Tbase_cam_opt[cam] = T_b_c

        print(f"\n--- Optimized EE->cam for '{cam}' ---")
        print("R_ee2cam:\n", R_ee_c)
        print("t_ee2cam:", t_ee_c)
        print("T_base2cam (approx, using mean EE pose):\n", T_b_c)

    # 5) Save optimized multi-camera summary
    if args.output is None:
        out_path = os.path.join(calib_dir, "multi_handeye_optimized.npz")
    else:
        out_path = args.output

    # Build a rough average board pose in base (from all samples & cameras)
    # using optimized extrinsics (for curiosity + debugging)
    board_positions = []
    board_rotations = []
    for cam in cam_names:
        data = cam_data[cam]
        pos = data["pos"]
        quat = data["quat"]
        rvecs = data["rvecs"]
        tvecs = data["tvecs"]

        R_ee_c = cam2_Ree_cam_opt[cam]
        t_ee_c = cam2_tee_cam_opt[cam]
        T_e_c = rt_to_T(R_ee_c, t_ee_c)

        for p, q, rv, tv in zip(pos, quat, rvecs, tvecs):
            R_b_e = quat_to_rot(q)
            T_b_e = rt_to_T(R_b_e, p)

            R_c_t, _ = cv2.Rodrigues(rv.reshape(3, 1))
            T_c_t = rt_to_T(R_c_t, tv.reshape(3))

            T_b_c = T_b_e @ T_e_c
            T_b_t = T_b_c @ T_c_t

            board_positions.append(T_b_t[:3, 3])
            board_rotations.append(T_b_t[:3, :3])

    board_positions = np.array(board_positions)
    board_rotations = np.stack(board_rotations, axis=0)
    t_board_mean = board_positions.mean(axis=0)
    R_sum = board_rotations.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_board_mean = U @ Vt
    T_base_board_mean = rt_to_T(R_board_mean, t_board_mean)

    print("\n=== Optimized mean board pose in base ===")
    print("T_base2board (mean):\n", T_base_board_mean)

    np.savez(
        out_path,
        run_id=args.run_id,
        camera_names=np.array(cam_names),
        lambda_trans=args.lambda_trans,
        lambda_rot=args.lambda_rot,
        success=res.success,
        final_cost=res.fun,
        Ree_cam=cam2_Ree_cam_opt,
        tee_cam=cam2_tee_cam_opt,
        Tbase_cam=cam2_Tbase_cam_opt,
        Tbase_board_mean=T_base_board_mean,
    )

    print(f"\n📁 Saved optimized multi-camera summary to: {out_path}")

    # 6) Also write per-camera YAML with *_opt suffix
    save_per_camera_yaml(
        args.base_dir,
        args.run_id,
        cam_names,
        cam2_Ree_cam_opt,
        cam2_tee_cam_opt,
        cam2_Tbase_cam_opt,
        suffix="_opt",
    )


if __name__ == "__main__":
    main()

