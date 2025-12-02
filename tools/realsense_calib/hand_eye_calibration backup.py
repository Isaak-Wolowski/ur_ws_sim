#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2


# =================================
# 1. Argument parsing
# =================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hand–eye calibration using robot poses and board poses."
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
        help="Run ID, e.g. run_100",
    )

    parser.add_argument(
        "--camera-name",
        type=str,
        required=True,
        help="Camera name, e.g. realsense, zed_left, zed_right",
    )

    parser.add_argument(
        "--robot-poses-file",
        type=str,
        default=None,
        help="Optional path to robot poses npz. "
             "If not given, uses <base-dir>/<run-id>/robot_poses.npz",
    )

    parser.add_argument(
        "--pnp-file",
        type=str,
        default=None,
        help="Optional path to PnP poses npz. "
             "If not given, uses <base-dir>/<run-id>/poses_<camera>.npz",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for hand–eye result npz. "
             "If not given, uses <base-dir>/calib/HandEye_<camera>.npz",
    )

    return parser.parse_args()


# =================================
# 2. Helpers
# =================================

def rodrigues_to_R(rvecs):
    """
    rvecs: (N, 3)
    returns: (N, 3, 3)
    """
    R_list = []
    for rv in rvecs:
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        R_list.append(R)
    return np.stack(R_list, axis=0)


def invert_RT(R, t):
    """
    Invert a rigid transform T = [R | t].
    R: (3,3)
    t: (3,)
    returns R_inv, t_inv such that T_inv = [R_inv | t_inv]
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


# =================================
# 3. Load data
# =================================

def load_robot_poses(base_dir, run_id, robot_poses_file=None):
    if robot_poses_file is None:
        robot_poses_file = os.path.join(base_dir, run_id, "robot_poses.npz")

    if not os.path.exists(robot_poses_file):
        raise FileNotFoundError(
            f"Robot poses file not found: {robot_poses_file}"
        )

    data = np.load(robot_poses_file, allow_pickle=True)

    if "filenames" not in data:
        raise KeyError("Robot poses file has no 'filenames' key.")
    if "rvecs_base_ee" not in data or "tvecs_base_ee" not in data:
        raise KeyError("Robot poses file must contain 'rvecs_base_ee' and 'tvecs_base_ee'.")

    filenames = data["filenames"]
    rvecs_base_ee = data["rvecs_base_ee"]
    tvecs_base_ee = data["tvecs_base_ee"]

    if rvecs_base_ee.shape[0] != filenames.shape[0]:
        raise RuntimeError("Robot poses: rvecs length != filenames length.")
    if tvecs_base_ee.shape[0] != filenames.shape[0]:
        raise RuntimeError("Robot poses: tvecs length != filenames length.")

    return filenames, rvecs_base_ee, tvecs_base_ee


def load_board_poses(base_dir, run_id, camera_name, pnp_file=None):
    if pnp_file is None:
        pnp_file = os.path.join(base_dir, run_id, f"poses_{camera_name}.npz")

    if not os.path.exists(pnp_file):
        raise FileNotFoundError(
            f"PnP poses file not found: {pnp_file}"
        )

    data = np.load(pnp_file, allow_pickle=True)

    if "filenames" not in data or "rvecs" not in data or "tvecs" not in data:
        raise KeyError("PnP file must contain 'filenames', 'rvecs', 'tvecs'.")

    filenames = data["filenames"]
    rvecs_board_cam = data["rvecs"]
    tvecs_board_cam = data["tvecs"]

    if rvecs_board_cam.shape[0] != filenames.shape[0]:
        raise RuntimeError("PnP poses: rvecs length != filenames length.")
    if tvecs_board_cam.shape[0] != filenames.shape[0]:
        raise RuntimeError("PnP poses: tvecs length != filenames length.")

    return filenames, rvecs_board_cam, tvecs_board_cam


# =================================
# 4. Match robot and board samples
# =================================

def match_by_filename(
    fn_robot, rvecs_base_ee, tvecs_base_ee,
    fn_board, rvecs_board_cam, tvecs_board_cam
):
    """
    Join the two sets on filename.
    returns matched:
      r_base_ee, t_base_ee, r_board_cam, t_board_cam
    """
    # Build dict for fast lookup by filename
    idx_robot = {name: i for i, name in enumerate(fn_robot)}
    idx_board = {name: i for i, name in enumerate(fn_board)}

    common_names = sorted(set(idx_robot.keys()) & set(idx_board.keys()))

    if not common_names:
        raise RuntimeError("No overlapping filenames between robot poses and board poses.")

    r_base_ee_list = []
    t_base_ee_list = []
    r_board_cam_list = []
    t_board_cam_list = []

    for name in common_names:
        i_r = idx_robot[name]
        i_b = idx_board[name]

        r_base_ee_list.append(rvecs_base_ee[i_r])
        t_base_ee_list.append(tvecs_base_ee[i_r])
        r_board_cam_list.append(rvecs_board_cam[i_b])
        t_board_cam_list.append(tvecs_board_cam[i_b])

    print(f"Matched {len(common_names)} samples by filename.")
    return (
        np.array(r_base_ee_list),
        np.array(t_base_ee_list),
        np.array(r_board_cam_list),
        np.array(t_board_cam_list),
        common_names,
    )


# =================================
# 5. Hand–eye via OpenCV
# =================================

def run_hand_eye(
    rvecs_base_ee, tvecs_base_ee,
    rvecs_board_cam, tvecs_board_cam
):
    """
    Input:
      - rvecs_base_ee, tvecs_base_ee: ee (gripper) in base frame
      - rvecs_board_cam, tvecs_board_cam: board (target) in camera frame

    We convert to:
      - R_gripper2base, t_gripper2base   (ee->base)
      - R_target2cam,   t_target2cam     (board->camera)

    Then call cv2.calibrateHandEye, which returns:
      - R_cam2gripper, t_cam2gripper
    """

    # Convert to rotation matrices
    R_base_ee_all = rodrigues_to_R(rvecs_base_ee)       # (N,3,3)
    R_board_cam_all = rodrigues_to_R(rvecs_board_cam)   # (N,3,3)

    t_base_ee_all = tvecs_base_ee.reshape(-1, 3)
    t_board_cam_all = tvecs_board_cam.reshape(-1, 3)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for R_be, t_be, R_bc, t_bc in zip(
        R_base_ee_all, t_base_ee_all, R_board_cam_all, t_board_cam_all
    ):
        # Invert base->ee to get ee->base
        R_eb, t_eb = invert_RT(R_be, t_be)

        R_gripper2base.append(R_eb)
        t_gripper2base.append(t_eb)

        # board->camera already in correct direction
        R_target2cam.append(R_bc)
        t_target2cam.append(t_bc)

    # cv2.calibrateHandEye expects Python lists of (3x3) and (3x1)
    R_gripper2base = [R for R in R_gripper2base]
    t_gripper2base = [t.reshape(3, 1) for t in t_gripper2base]
    R_target2cam = [R for R in R_target2cam]
    t_target2cam = [t.reshape(3, 1) for t in t_target2cam]

    # Solve for camera in gripper frame
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    return R_cam2gripper, t_cam2gripper


# =================================
# 6. Save result
# =================================

def save_hand_eye(base_dir, camera_name, output, R_cam2gripper, t_cam2gripper):
    if output is None:
        calib_dir = os.path.join(base_dir, "calib")
        os.makedirs(calib_dir, exist_ok=True)
        output = os.path.join(calib_dir, f"HandEye_{camera_name}.npz")

    np.savez(
        output,
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper.reshape(3),
    )

    print(f"\n📁 Saved hand–eye result to: {output}")


# =================================
# 7. Main
# =================================

def main():
    args = parse_args()

    print("=== Hand–Eye Calibration ===")
    print(f"📂 Base dir   : {args.base_dir}")
    print(f"🏃 Run ID     : {args.run_id}")
    print(f"📷 Camera     : {args.camera_name}\n")

    # 1) Load robot poses
    fn_robot, rvecs_base_ee, tvecs_base_ee = load_robot_poses(
        args.base_dir, args.run_id, args.robot_poses_file
    )
    print(f"Loaded {len(fn_robot)} robot poses.")

    # 2) Load board-in-camera poses from PnP
    fn_board, rvecs_board_cam, tvecs_board_cam = load_board_poses(
        args.base_dir, args.run_id, args.camera_name, args.pnp_file
    )
    print(f"Loaded {len(fn_board)} board poses.\n")

    # 3) Match by filename
    (
        r_base_ee,
        t_base_ee,
        r_board_cam,
        t_board_cam,
        common_names,
    ) = match_by_filename(
        fn_robot, rvecs_base_ee, tvecs_base_ee,
        fn_board, rvecs_board_cam, tvecs_board_cam
    )

    print(f"Using {len(common_names)} matched samples for hand–eye.\n")

    # 4) Run hand–eye
    R_cam2gripper, t_cam2gripper = run_hand_eye(
        r_base_ee, t_base_ee,
        r_board_cam, t_board_cam
    )

    # 5) Print nicely
    print("=== Hand–Eye Result ===")
    print("Camera in gripper frame: T_cam^gripper")
    print("Rotation (R_cam2gripper):")
    print(R_cam2gripper)
    print("\nTranslation (t_cam2gripper) [m]:")
    print(t_cam2gripper.reshape(3))

    # 6) Save
    save_hand_eye(args.base_dir, args.camera_name, args.output, R_cam2gripper, t_cam2gripper)


if __name__ == "__main__":
    main()

