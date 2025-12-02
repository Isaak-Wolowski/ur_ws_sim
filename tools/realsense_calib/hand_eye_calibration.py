#!/usr/bin/env python3
import os
import argparse

import numpy as np
import cv2


# ======================================
# 1. Argument parsing
# ======================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hand–eye calibration: solve base–camera transform "
                    "from robot TCP poses and PnP board poses."
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
        "--camera-name",
        type=str,
        required=True,
        help="Camera name, e.g. realsense",
    )

    p.add_argument(
        "--board-poses",
        type=str,
        default=None,
        help="Optional explicit path to poses_<camera>.npz "
             "(board in camera). If not given, uses "
             "<base-dir>/<run-id>/calib/poses_<camera>.npz",
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
        "--output",
        type=str,
        default=None,
        help="Optional output npz file path. If not given, saves to "
             "<base-dir>/<run-id>/calib/handeye_<camera>.npz",
    )

    p.add_argument(
        "--method",
        type=str,
        default="Tsai",
        choices=["Tsai", "Park", "Horaud", "Andreff", "Daniilidis"],
        help="Hand–eye method (OpenCV): Tsai, Park, Horaud, Andreff, Daniilidis. "
             "Default: Tsai",
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
    # normalize
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
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)]
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


# ======================================
# 3. Load board poses (PnP)
# ======================================

def load_board_poses(base_dir, run_id, camera_name, board_poses_path=None):
    """
    Load board-in-camera poses from PnP:
      - filenames: (N,)
      - rvecs:     (N,3)
      - tvecs:     (N,3)

    Default path:
      <base-dir>/<run-id>/calib/poses_<camera>.npz
    """
    if board_poses_path is None:
        board_poses_path = os.path.join(
            base_dir, run_id, "calib", f"poses_{camera_name}.npz"
        )

    if not os.path.exists(board_poses_path):
        raise FileNotFoundError(f"Board poses file not found: {board_poses_path}")

    data = np.load(board_poses_path, allow_pickle=True)
    filenames = data["filenames"]       # (N,) of strings
    rvecs = data["rvecs"]               # (N,3)
    tvecs = data["tvecs"]               # (N,3)

    # Convert rvecs to rotation matrices
    R_target2cam = []
    t_target2cam = []

    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        R_target2cam.append(R)
        t_target2cam.append(tv.reshape(3, 1))

    return filenames, R_target2cam, t_target2cam


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
    # Prefer known names if available
    for name in ("image_name", "filename", "file", "img", "image"):
        if name in colnames:
            filename_col = name
            break

    # fallback: first string-like column
    if filename_col is None:
        for name in colnames:
            if np.issubdtype(raw[name].dtype, np.str_):
                filename_col = name
                break

    # last resort: first column
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
        # Generic fallback: take numeric columns only
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
# 5. Match poses by filename
# ======================================

def match_by_filenames(board_files, robot_files,
                       R_target2cam, t_target2cam, pos, quat):
    """
    board_files: list/array of filenames (from PnP)
    robot_files: list/array of filenames (from robot logging)
    returns matched lists for hand–eye.
    """
    board_map = {f: i for i, f in enumerate(board_files)}
    robot_map = {f: i for i, f in enumerate(robot_files)}

    common = sorted(set(board_map.keys()) & set(robot_map.keys()))
    if len(common) == 0:
        raise RuntimeError("No overlapping filenames between board and robot poses.")

    print(f"Found {len(common)} common images for hand–eye calibration.")

    Rg2b = []  # gripper (ee) in base
    tg2b = []
    Rt2c = []  # target (board) in camera
    tt2c = []

    for fname in common:
        i_b = board_map[fname]
        i_r = robot_map[fname]

        # board in camera
        Rt2c.append(R_target2cam[i_b])
        tt2c.append(t_target2cam[i_b])

        # ee in base
        p = pos[i_r]
        q = quat[i_r]
        R = quat_to_rot(q)
        t = p.reshape(3, 1)
        Rg2b.append(R)
        tg2b.append(t)

    return common, Rg2b, tg2b, Rt2c, tt2c


# ======================================
# 6. Hand–eye solve & saving
# ======================================

def method_to_flag(name):
    mapping = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    return mapping[name]


def format_matrix_rows(M):
    """Helper: format a 4x4 (or 3x3) into YAML-like rows."""
    lines = []
    for row in M:
        formatted = ", ".join(f"{v:.6f}" for v in row)
        lines.append(f"  - [{formatted}]")
    return "\n".join(lines)


def save_handeye(base_dir, run_id, camera_name, output_path,
                 R_cam2ee, t_cam2ee, T_cam2ee, T_base2cam, T_cam2base,
                 used_filenames, method_name):
    # Everything lives in <base-dir>/<run-id>/calib
    calib_dir = os.path.join(base_dir, run_id, "calib")
    os.makedirs(calib_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(calib_dir, f"handeye_{camera_name}.npz")

    np.savez(
        output_path,
        method=method_name,
        filenames=np.array(used_filenames),
        R_cam2ee=R_cam2ee,
        t_cam2ee=t_cam2ee,
        T_cam2ee=T_cam2ee,
        T_base2cam=T_base2cam,
        T_cam2base=T_cam2base,
    )

    print(f"\n📁 Saved hand–eye results (binary npz) to: {output_path}")
    print("  - T_cam2ee   : camera in ee frame")
    print("  - T_base2cam : camera in base frame")
    print("  - T_cam2base : inverse of T_base2cam")

    # ===== Human-readable YAML summary (same calib dir) =====
    base, _ = os.path.splitext(output_path)
    yaml_path = base + ".yaml"

    with open(yaml_path, "w") as f:
        f.write(f"method: {method_name}\n")
        f.write(f"camera_name: {camera_name}\n")
        f.write(f"run_id: {run_id}\n")
        f.write("\n")
        f.write("used_images:\n")
        for fname in used_filenames:
            f.write(f"  - {fname}\n")
        f.write("\n")

        f.write("T_cam2ee:\n")
        f.write(format_matrix_rows(T_cam2ee) + "\n\n")

        f.write("T_base2cam:\n")
        f.write(format_matrix_rows(T_base2cam) + "\n\n")

        f.write("T_cam2base:\n")
        f.write(format_matrix_rows(T_cam2base) + "\n")

    print(f"📝 Saved human-readable YAML to: {yaml_path}")
    print("Open this in any text editor to inspect the transforms.")


# ======================================
# 7. Main
# ======================================

def main():
    args = parse_args()

    run_dir = os.path.join(args.base_dir, args.run_id)

    print("=== Hand–Eye Calibration ===")
    print(f"📂 Base dir : {args.base_dir}")
    print(f"🏃 Run ID   : {args.run_id}")
    print(f"📷 Camera   : {args.camera_name}")
    print(f"📂 Run dir  : {run_dir}")
    print(f"⚙️  Method   : {args.method}\n")

    # 1) Load board poses (board in camera)
    board_files, R_target2cam, t_target2cam = load_board_poses(
        args.base_dir, args.run_id, args.camera_name, args.board_poses
    )

    # 2) Load robot poses (ee in base)
    robot_files, pos, quat = load_robot_poses(
        args.base_dir, args.run_id, args.robot_poses
    )

    # 3) Match by filename
    used_filenames, Rg2b, tg2b, Rt2c, tt2c = match_by_filenames(
        board_files, robot_files,
        R_target2cam, t_target2cam,
        pos, quat
    )

    # 4) Solve hand–eye: camera in ee frame
    flag = method_to_flag(args.method)
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        Rg2b, tg2b, Rt2c, tt2c, method=flag
    )

    t_cam2ee = t_cam2ee.reshape(3)
    T_cam2ee = rt_to_T(R_cam2ee, t_cam2ee)

    print("\n=== Result: Camera in EE frame (T_cam^ee) ===")
    print("R_cam2ee:")
    print(R_cam2ee)
    print("t_cam2ee [m]:", t_cam2ee)

    # 5) Compute camera in base frame for each pose & average.
    # OpenCV returns camera IN ee frame: T_cam2ee = ^ee T_c
    # So for each sample:  T_base^cam = T_base^ee * T_cam2ee  (no inversion!)

    
    # === Consistency check: board pose in base for each sample ===
    T_boards = []
    for R_t2c, t_t2c, p, q in zip(Rt2c, tt2c, pos, quat):
        # base -> ee
        T_b_e = rt_to_T(quat_to_rot(q), p)

        # camera -> board (we currently have target/board in camera)
        T_c_t = rt_to_T(R_t2c, t_t2c.reshape(3))

        # base -> board
     ##########################################   T_b_t = T_b_e @ T_ee2cam @ T_c_t
        T_b_t = T_b_e @ T_cam2ee @ T_c_t
        T_boards.append(T_b_t)

    T_boards = np.stack(T_boards, axis=0)

    # Translation stats
    board_positions = T_boards[:, :3, 3]
    mean_board_pos = board_positions.mean(axis=0)
    pos_err = np.linalg.norm(board_positions - mean_board_pos, axis=1)

    print("\n=== Board-in-base consistency check ===")
    print("Mean board position [m]:", mean_board_pos)
    print("Board position error [m]:")
    print("  mean =", pos_err.mean())
    print("  std  =", pos_err.std())
    print("  max  =", pos_err.max())


    Ts_base2cam = []
    for p, q in zip(pos, quat):
        R_b_ee = quat_to_rot(q)
        t_b_ee = p
        T_b_ee = rt_to_T(R_b_ee, t_b_ee)

        ##T_b_cam = T_b_ee @ T_ee2cam
        T_b_cam = T_b_ee @ T_cam2ee        
        Ts_base2cam.append(T_b_cam)

    Ts_base2cam = np.stack(Ts_base2cam, axis=0)

    # simple average of translation; rotation average via SVD
    t_mean = Ts_base2cam[:, :3, 3].mean(axis=0)

    R_stack = Ts_base2cam[:, :3, :3]
    R_sum = R_stack.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt

    T_base2cam = rt_to_T(R_mean, t_mean)
    T_cam2base = invert_T(T_base2cam)

    print("\n=== Result: Camera in BASE frame (T_base^cam, averaged) ===")
    print("T_base2cam:")
    print(T_base2cam)
    print("\nT_cam2base (inverse):")
    print(T_cam2base)

    # 6) Save everything (npz + yaml in calib/)
    save_handeye(
        args.base_dir,
        args.run_id,
        args.camera_name,
        args.output,
        R_cam2ee,
        t_cam2ee,
        T_cam2ee,
        T_base2cam,
        T_cam2base,
        used_filenames,
        args.method,
    )


if __name__ == "__main__":
    main()

