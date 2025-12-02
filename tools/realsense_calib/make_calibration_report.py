#!/usr/bin/env python3
import os
import argparse
import textwrap

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from glob import glob


# =========================
# 1. Small helpers
# =========================

def quat_to_rot(q):
    """
    Quaternion -> 3x3 rotation matrix.
    q = [qx, qy, qz, qw]
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
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def rot_to_rpy(R):
    """
    Rotation matrix -> roll, pitch, yaw (ZYX), radians.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


# =========================
# 2. Parsing
# =========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Make a PDF calibration report with text + TF plots."
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="/root/ur_ws_sim/data",
        help="Base data directory (default: /root/ur_ws_sim/data)",
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
        "--md-path",
        type=str,
        default=None,
        help="Optional path to the markdown report. "
             "If not given, uses "
             "<base-dir>/<run-id>/results/calib_report_<camera>.md",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PDF path. "
             "If not given, uses "
             "<base-dir>/<run-id>/results/calib_report_<camera>.pdf",
    )
    # board parameters for reprojection analysis
    p.add_argument(
        "--nx",
        type=int,
        default=10,
        help="Number of inner corners along board width (columns).",
    )
    p.add_argument(
        "--ny",
        type=int,
        default=7,
        help="Number of inner corners along board height (rows).",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=0.024,
        help="Chessboard square size in meters.",
    )
    return p.parse_args()


# =========================
# 3. Load stuff
# =========================

def load_handeye(base_dir, run_id, camera_name):
    calib_dir = os.path.join(base_dir, run_id, "calib")
    path = os.path.join(calib_dir, f"handeye_{camera_name}.npz")
    if not os.path.exists(path):
        print(f"[WARN] hand–eye file not found: {path}")
        return None, None
    data = np.load(path, allow_pickle=True)
    return path, data


def load_robot_poses(base_dir, run_id):
    """
    Expect your current format:

      image_index,image_name,frame_id,
      px,py,pz,
      qx,qy,qz,qw
    """
    calib_dir = os.path.join(base_dir, run_id, "calib")
    csv_path = os.path.join(calib_dir, "robot_poses.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] robot_poses.csv not found: {csv_path}")
        return None, None, None

    raw = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding=None
    )

    px = raw["px"]
    py = raw["py"]
    pz = raw["pz"]
    qx = raw["qx"]
    qy = raw["qy"]
    qz = raw["qz"]
    qw = raw["qw"]

    pos = np.vstack([px, py, pz]).T.astype(float)
    quat = np.vstack([qx, qy, qz, qw]).T.astype(float)
    return csv_path, pos, quat


def load_markdown(md_path):
    if md_path is None or not os.path.exists(md_path):
        print(f"[WARN] markdown report not found: {md_path}")
        return None
    with open(md_path, "r") as f:
        return f.read()


def load_camera_calib(base_dir, run_id, camera_name):
    """
    Try run-local calib first, then global calib.
    """
    run_calib_dir = os.path.join(base_dir, run_id, "calib")
    path_run = os.path.join(run_calib_dir, f"CameraParams_{camera_name}.npz")
    global_calib_dir = os.path.join(base_dir, "calib")
    path_global = os.path.join(global_calib_dir, f"CameraParams_{camera_name}.npz")

    path = None
    if os.path.exists(path_run):
        path = path_run
    elif os.path.exists(path_global):
        path = path_global

    if path is None:
        print(f"[WARN] CameraParams not found in:\n  {path_run}\n  {path_global}")
        return None, None, None

    data = np.load(path)
    if "K" in data:
        K = data["K"]
    elif "cameraMatrix" in data:
        K = data["cameraMatrix"]
    else:
        print("[WARN] No 'K' or 'cameraMatrix' in camera params.")
        return path, None, None

    if "dist" in data:
        dist = data["dist"]
    elif "distCoeffs" in data:
        dist = data["distCoeffs"]
    else:
        print("[WARN] No 'dist' or 'distCoeffs' in camera params.")
        return path, None, None

    return path, K, dist


def load_pnp_poses(base_dir, run_id, camera_name):
    calib_dir = os.path.join(base_dir, run_id, "calib")
    path = os.path.join(calib_dir, f"poses_{camera_name}.npz")
    if not os.path.exists(path):
        print(f"[WARN] PnP poses file not found: {path}")
        return None, None, None, None
    data = np.load(path, allow_pickle=True)
    filenames = data["filenames"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    return path, filenames, rvecs, tvecs


# =========================
# 4. Compute per-image camera / board / EE TFs
# =========================

def compute_camera_poses_in_base(handeye_data, pos, quat):
    """
    From:
      - handeye_data: contains R_cam2ee, t_cam2ee  (camera in EE)
      - pos, quat: ee in base for each image

    Compute:
      - T_base^cam for each image
    """
    if handeye_data is None or pos is None or quat is None:
        return None

    if "R_cam2ee" not in handeye_data or "t_cam2ee" not in handeye_data:
        print("[WARN] hand–eye file missing R_cam2ee/t_cam2ee.")
        return None

    R_cam2ee = handeye_data["R_cam2ee"]
    t_cam2ee = handeye_data["t_cam2ee"].reshape(3)

    # OpenCV returns camera in EE frame: ^ee T_c
    T_cam2ee = rt_to_T(R_cam2ee, t_cam2ee)

    Ts_base2cam = []
    for p, q in zip(pos, quat):
        R_b_ee = quat_to_rot(q)
        T_b_ee = rt_to_T(R_b_ee, p)     # ^b T_ee
        T_b_cam = T_b_ee @ T_cam2ee     # ^b T_c = ^b T_ee ^ee T_c
        Ts_base2cam.append(T_b_cam)

    return np.stack(Ts_base2cam, axis=0)


def compute_board_poses_in_base(handeye_data, pos, quat, rvecs, tvecs):
    """
    Compute board poses in base for each sample:

      ^bT_t = ^bT_ee  ^eeT_c  ^cT_t

    Using:
      - handeye_data: R_cam2ee, t_cam2ee (camera in ee frame)
      - pos, quat   : ee in base
      - rvecs, tvecs: board in camera (PnP)
    """
    if (
        handeye_data is None or pos is None or quat is None or
        rvecs is None or tvecs is None
    ):
        return None

    if "R_cam2ee" not in handeye_data or "t_cam2ee" not in handeye_data:
        print("[WARN] hand–eye file missing R_cam2ee/t_cam2ee.")
        return None

    R_cam2ee = handeye_data["R_cam2ee"]
    t_cam2ee = handeye_data["t_cam2ee"].reshape(3)
    T_cam2ee = rt_to_T(R_cam2ee, t_cam2ee)  # ^ee T_c

    Ts_base2board = []
    for p, q, rv, tv in zip(pos, quat, rvecs, tvecs):
        # base -> ee
        T_b_ee = rt_to_T(quat_to_rot(q), p)

        # camera -> board: we have board in camera (rvec,tvec)
        R_t2c, _ = cv2.Rodrigues(rv.reshape(3, 1))
        T_c_t = rt_to_T(R_t2c, tv.reshape(3))  # ^c T_t

        # base -> board
        T_b_t = T_b_ee @ T_cam2ee @ T_c_t
        Ts_base2board.append(T_b_t)

    return np.stack(Ts_base2board, axis=0)


def build_ee_poses_in_base(pos, quat):
    """
    Build T_base^ee for each robot pose from position + quaternion.
    """
    if pos is None or quat is None:
        return None
    Ts = []
    for p, q in zip(pos, quat):
        R = quat_to_rot(q)
        T = rt_to_T(R, p)
        Ts.append(T)
    return np.stack(Ts, axis=0)


# =========================
# 5. Reprojection + pose stats
# =========================

def create_chessboard_object_points(nx, ny, square_size):
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= square_size
    return objp


def compute_reprojection_stats(image_dir, K, dist, nx, ny, square_size):
    """
    Re-detect corners and compute reprojection errors using fixed K, dist.

    Returns:
      reproj_errors_all : list[float] per corner
      per_image_rms     : list[(filename, rms)]
      heatmap           : (ny, nx) RMS error per board corner
      detect_flags      : list[bool] detection success per image
      image_paths       : list[str]
    """
    if K is None or dist is None:
        return [], [], None, [], []

    objp = create_chessboard_object_points(nx, ny, square_size)
    pattern_size = (nx, ny)

    image_paths = sorted(glob(os.path.join(image_dir, "img_*.png")))

    reproj_errors_all = []
    per_image_rms = []
    detect_flags = []

    n_points = nx * ny
    sum_err_per_corner = np.zeros(n_points, dtype=np.float64)
    count_per_corner = np.zeros(n_points, dtype=np.int32)

    for path in image_paths:
        img = cv2.imread(path)
        fname = os.path.basename(path)
        if img is None:
            print(f"[WARN] Failed to read image {fname}")
            detect_flags.append(False)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"[WARN] Chessboard NOT found in: {fname}")
            detect_flags.append(False)
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners_refined,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            print(f"[WARN] solvePnP failed for {fname}")
            detect_flags.append(False)
            continue

        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        corners2 = corners_refined.reshape(-1, 2)

        err = np.linalg.norm(corners2 - proj, axis=1)
        reproj_errors_all.extend(err.tolist())

        rms = float(np.sqrt(np.mean(err ** 2)))
        per_image_rms.append((fname, rms))
        detect_flags.append(True)

        sum_err_per_corner += err
        count_per_corner += 1

    heatmap = None
    if n_points > 0:
        heatmap = np.zeros((ny, nx), dtype=np.float64)
        for idx in range(n_points):
            if count_per_corner[idx] > 0:
                mean_e = sum_err_per_corner[idx] / count_per_corner[idx]
            else:
                mean_e = np.nan
            ix = idx % nx
            iy = idx // nx
            heatmap[iy, ix] = mean_e

    return reproj_errors_all, per_image_rms, heatmap, detect_flags, image_paths


def compute_camera_pose_deviation(Ts_base2cam):
    if Ts_base2cam is None:
        return None, None, None, None

    R_all = Ts_base2cam[:, :3, :3]
    t_all = Ts_base2cam[:, :3, 3]

    t_mean = t_all.mean(axis=0)

    R_sum = R_all.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt

    t_dev = np.linalg.norm(t_all - t_mean[None, :], axis=1)

    rot_dev_deg = []
    for R in R_all:
        R_delta = R_mean.T @ R
        angle = np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
        rot_dev_deg.append(np.degrees(angle))

    return np.array(t_dev), np.array(rot_dev_deg), R_mean, t_mean


# =========================
# 6. Plot helpers
# =========================

def set_equal_3d(ax):
    xlims = ax.get_xlim3d()
    ylims = ax.get_ylim3d()
    zlims = ax.get_zlim3d()

    x_range = xlims[1] - xlims[0]
    y_range = ylims[1] - ylims[0]
    z_range = zlims[1] - zlims[0]
    max_range = max([x_range, y_range, z_range])

    x_mid = (xlims[0] + xlims[1]) / 2.0
    y_mid = (ylims[0] + ylims[1]) / 2.0
    z_mid = (zlims[0] + zlims[1]) / 2.0

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])


def apply_workspace_limits(ax):
    """
    Clamp axes to a nice workspace so things aren't tiny.
    Adjust these numbers if your setup changes.
    """
    ax.set_xlim(0.7, 1.5)
    ax.set_ylim(-0.3, 0.5)
    ax.set_zlim(0.8, 1.5)


def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4-ish
    ax = fig.add_subplot(111)
    ax.axis("off")

    y = 0.95
    ax.text(0.02, y, title, fontsize=16, weight="bold", va="top")
    y -= 0.05

    for line in lines:
        ax.text(0.02, y, line, fontsize=9, va="top", family="monospace")
        y -= 0.02
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.95

    pdf.savefig(fig)
    plt.close(fig)


def add_markdown_page(pdf, md_text):
    if md_text is None:
        return
    wrapped_lines = []
    for line in md_text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=100) or [""])
    add_text_page(pdf, "Calibration Markdown Report", wrapped_lines)


def draw_frame(ax, T, length=0.05, lw=1.2, alpha=0.9):
    """
    Draw a small 3D coordinate frame for a homogeneous transform T (4x4).

    - X axis: red
    - Y axis: green
    - Z axis: blue
    """
    o = T[:3, 3]
    R = T[:3, :3]

    x_axis = o + R[:, 0] * length
    y_axis = o + R[:, 1] * length
    z_axis = o + R[:, 2] * length

    # X (red)
    ax.plot(
        [o[0], x_axis[0]],
        [o[1], x_axis[1]],
        [o[2], x_axis[2]],
        "r-",
        linewidth=lw,
        alpha=alpha,
    )
    # Y (green)
    ax.plot(
        [o[0], y_axis[0]],
        [o[1], y_axis[1]],
        [o[2], y_axis[2]],
        "g-",
        linewidth=lw,
        alpha=alpha,
    )
    # Z (blue)
    ax.plot(
        [o[0], z_axis[0]],
        [o[1], z_axis[1]],
        [o[2], z_axis[2]],
        "b-",
        linewidth=lw,
        alpha=alpha,
    )


def draw_board_plane(ax, T_b_t, nx, ny, square_size):
    """
    Draw a rectangle approximating the chessboard in the base frame.
    """
    if T_b_t is None:
        return

    # Board corners in board frame (same convention as PnP)
    w = (nx - 1) * square_size
    h = (ny - 1) * square_size
    corners_board = np.array([
        [0.0, 0.0, 0.0],
        [w,   0.0, 0.0],
        [w,   h,   0.0],
        [0.0, h,   0.0],
        [0.0, 0.0, 0.0],
    ])

    R = T_b_t[:3, :3]
    t = T_b_t[:3, 3]
    corners_world = (R @ corners_board.T).T + t

    ax.plot(
        corners_world[:, 0],
        corners_world[:, 1],
        corners_world[:, 2],
        linewidth=2.0,
    )


def add_3d_frames_page(pdf, Ts_base2ee, Ts_base2cam):
    """
    Show EE frames (thin) and camera frames (thicker) in the base frame.
    """
    if Ts_base2ee is None and Ts_base2cam is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ---- World / base frame at origin ----
    T_base = np.eye(4)
    draw_frame(ax, T_base, length=0.25, lw=3.0, alpha=1.0)
    ax.text(
        0.0, 0.0, 0.0,
        "BASE / WORLD",
        fontsize=10,
        color="k",
        ha="left",
        va="bottom",
    )

    # Draw EE frames: short, thin, semi-transparent
    if Ts_base2ee is not None:
        for T in Ts_base2ee:
            draw_frame(ax, T, length=0.05, lw=0.7, alpha=0.4)

    # Draw camera frames: longer, thicker, opaque
    if Ts_base2cam is not None:
        for T in Ts_base2cam:
            draw_frame(ax, T, length=0.10, lw=2.0, alpha=1.0)

        # Also scatter camera positions for clarity
        cam_positions = Ts_base2cam[:, :3, 3]
        ax.scatter(
            cam_positions[:, 0],
            cam_positions[:, 1],
            cam_positions[:, 2],
            marker="^",
            s=10,
            label="Camera origin",
        )

    # Optional: scatter EE positions (as dots)
    if Ts_base2ee is not None:
        ee_positions = Ts_base2ee[:, :3, 3]
        ax.scatter(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            marker="o",
            s=5,
            alpha=0.5,
            label="EE origin",
        )

    apply_workspace_limits(ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Robot EE (thin) vs Camera (thick) Frames in Base")
    ax.legend(loc="best")

    pdf.savefig(fig)
    plt.close(fig)


def add_scene_page(pdf, Ts_base2ee, Ts_base2cam, Ts_base2board, nx, ny, square_size):
    """
    Show base frame, EE poses, camera poses, and mean board pose.
    """
    if Ts_base2ee is None or Ts_base2cam is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Base/world frame
    T_base = np.eye(4)
    draw_frame(ax, T_base, length=0.25, lw=3.0, alpha=1.0)
    ax.text(0.0, 0.0, 0.0, "BASE/WORLD", fontsize=10, color="k")

    # EE positions
    ee_pos = Ts_base2ee[:, :3, 3]
    ax.scatter(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2],
               s=10, alpha=0.6, label="EE origin")

    # Camera positions
    cam_pos = Ts_base2cam[:, :3, 3]
    ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
               marker="^", s=20, alpha=0.8, label="Camera origin")

    # Optionally draw a subset of frames to avoid clutter
    step = max(1, len(Ts_base2ee) // 40)
    for T in Ts_base2ee[::step]:
        draw_frame(ax, T, length=0.05, lw=0.7, alpha=0.3)
    for T in Ts_base2cam[::step]:
        draw_frame(ax, T, length=0.08, lw=1.4, alpha=0.9)

    # Mean board pose & plane
    if Ts_base2board is not None:
        R_all = Ts_base2board[:, :3, :3]
        t_all = Ts_base2board[:, :3, 3]
        t_mean = t_all.mean(axis=0)

        R_sum = R_all.sum(axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_mean = U @ Vt

        T_b_t_mean = rt_to_T(R_mean, t_mean)

        draw_frame(ax, T_b_t_mean, length=0.15, lw=2.0, alpha=1.0)
        draw_board_plane(ax, T_b_t_mean, nx, ny, square_size)
        ax.text(t_mean[0], t_mean[1], t_mean[2],
                "BOARD", fontsize=9, color="k")

    apply_workspace_limits(ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Base, EE, Camera, and Board in Base Frame")
    ax.legend(loc="best")

    pdf.savefig(fig)
    plt.close(fig)


def add_3d_plot_page(pdf, pos_robot, Ts_base2cam):
    if pos_robot is None or Ts_base2cam is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    cams = Ts_base2cam[:, :3, 3]

    ax.scatter(
        pos_robot[:, 0],
        pos_robot[:, 1],
        pos_robot[:, 2],
        marker="o",
        s=20,
        label="EE in base",
        alpha=0.6,
    )

    ax.scatter(
        cams[:, 0],
        cams[:, 1],
        cams[:, 2],
        marker="^",
        s=25,
        label="Camera in base (per image)",
        alpha=0.8,
    )

    apply_workspace_limits(ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Robot EE & Camera Trajectory in Base Frame")
    ax.legend(loc="best")

    pdf.savefig(fig)
    plt.close(fig)


def add_mean_camera_frame_page(pdf, R_mean, t_mean, Ts_base2board, nx, ny, square_size):
    """
    Show a big camera frame at the mean camera pose, plus the board.
    """
    if R_mean is None or t_mean is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Base frame
    T_base = np.eye(4)
    draw_frame(ax, T_base, length=0.25, lw=3.0, alpha=1.0)
    ax.text(0.0, 0.0, 0.0, "BASE/WORLD", fontsize=10, color="k")

    # Mean camera frame
    T_cam_mean = rt_to_T(R_mean, t_mean)
    draw_frame(ax, T_cam_mean, length=0.25, lw=3.0, alpha=1.0)
    ax.text(t_mean[0], t_mean[1], t_mean[2], "CAM (mean)", fontsize=10, color="k")

    # Board mean pose & plane (if available)
    if Ts_base2board is not None:
        R_all = Ts_base2board[:, :3, :3]
        t_all = Ts_base2board[:, :3, 3]
        t_b_mean = t_all.mean(axis=0)

        R_sum = R_all.sum(axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_b_mean = U @ Vt

        T_b_t_mean = rt_to_T(R_b_mean, t_b_mean)
        draw_frame(ax, T_b_t_mean, length=0.20, lw=2.0, alpha=1.0)
        draw_board_plane(ax, T_b_t_mean, nx, ny, square_size)
        ax.text(t_b_mean[0], t_b_mean[1], t_b_mean[2], "BOARD", fontsize=9, color="k")

    apply_workspace_limits(ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Mean Camera Pose and Board in Base Frame")

    pdf.savefig(fig)
    plt.close(fig)


def add_camera_look_ray_page(pdf, R_mean, t_mean, Ts_base2board, nx, ny, square_size):
    """
    Show the camera optical axis as a ray in the base frame.
    """
    if R_mean is None or t_mean is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Base frame
    T_base = np.eye(4)
    draw_frame(ax, T_base, length=0.25, lw=3.0, alpha=1.0)
    ax.text(0.0, 0.0, 0.0, "BASE/WORLD", fontsize=10, color="k")

    # Camera center
    cam_o = t_mean
    ax.scatter(cam_o[0], cam_o[1], cam_o[2], s=40, c="k", label="Camera center")

    # Camera optical axis (camera z-axis in base frame)
    z_cam = R_mean[:, 2]  # third column
    ray_len = 1.0
    p1 = cam_o + ray_len * z_cam

    ax.plot(
        [cam_o[0], p1[0]],
        [cam_o[1], p1[1]],
        [cam_o[2], p1[2]],
        linestyle="-",
        linewidth=3.0,
    )

    # Board mean pose / plane if available
    if Ts_base2board is not None:
        R_all = Ts_base2board[:, :3, :3]
        t_all = Ts_base2board[:, :3, 3]
        t_b_mean = t_all.mean(axis=0)

        R_sum = R_all.sum(axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_b_mean = U @ Vt

        T_b_t_mean = rt_to_T(R_b_mean, t_b_mean)
        draw_board_plane(ax, T_b_t_mean, nx, ny, square_size)
        ax.text(t_b_mean[0], t_b_mean[1], t_b_mean[2], "BOARD", fontsize=9, color="k")

    apply_workspace_limits(ax)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Camera Look Ray in Base Frame")
    ax.legend(loc="best")

    pdf.savefig(fig)
    plt.close(fig)


def add_reproj_hist_page(pdf, reproj_errors_all):
    if not reproj_errors_all:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(reproj_errors_all, bins=40)
    ax.set_xlabel("Reprojection error [pixels]")
    ax.set_ylabel("Count")
    ax.set_title("Reprojection Error Histogram (all corners, all images)")
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)


def add_reproj_rms_page(pdf, per_image_rms):
    if not per_image_rms:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    rms_vals = [e for _, e in per_image_rms]
    ax.plot(range(len(rms_vals)), rms_vals, marker="o")
    ax.set_xlabel("Image index (successful detections)")
    ax.set_ylabel("RMS reprojection error [px]")
    ax.set_title("Per-Image RMS Reprojection Error")
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)


def add_reproj_heatmap_page(pdf, heatmap, nx, ny):
    if heatmap is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, origin="lower", aspect="auto")
    ax.set_xlabel("Board X index (0..nx-1)")
    ax.set_ylabel("Board Y index (0..ny-1)")
    ax.set_title("Mean Reprojection Error per Chessboard Corner [px]")
    fig.colorbar(im, ax=ax, label="Error [px]")
    pdf.savefig(fig)
    plt.close(fig)


def add_detection_diag_page(pdf, detect_flags, image_paths):
    if not image_paths:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(image_paths))
    y = [1 if f else 0 for f in detect_flags]
    ax.bar(x, y)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Detection (1=success, 0=failure)")
    ax.set_title("Chessboard Detection per Image")
    pdf.savefig(fig)
    plt.close(fig)


def add_distortion_coeffs_page(pdf, dist):
    if dist is None:
        return
    dist = dist.ravel()
    n = len(dist)
    if n == 4:
        labels = ["k1", "k2", "p1", "p2"]
    elif n == 5:
        labels = ["k1", "k2", "p1", "p2", "k3"]
    elif n == 8:
        labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    else:
        labels = [f"d{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n), dist)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Distortion Coefficients")
    ax.grid(True, axis="y")
    pdf.savefig(fig)
    plt.close(fig)


def add_board_pose_vs_index_page(pdf, rvecs, tvecs):
    if rvecs is None or tvecs is None:
        return
    t_norm = np.linalg.norm(tvecs, axis=1)
    r_norm = np.linalg.norm(rvecs, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(range(len(t_norm)), t_norm, marker="o")
    ax[0].set_ylabel("||t|| [m]")
    ax[0].set_title("Board Distance from Camera")

    ax[1].plot(range(len(r_norm)), r_norm, marker="o")
    ax[1].set_ylabel("||rvec|| [rad]")
    ax[1].set_xlabel("Sample index")
    ax[1].set_title("Board Orientation Magnitude")

    for a in ax:
        a.grid(True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_camera_deviation_page(pdf, t_dev, rot_dev_deg):
    if t_dev is None or rot_dev_deg is None:
        return
    idx = range(len(t_dev))
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(idx, t_dev, marker="o")
    ax[0].set_ylabel("Δ translation [m]")
    ax[0].set_title("Camera Pose Deviation from Mean (Translation)")

    ax[1].plot(idx, rot_dev_deg, marker="o")
    ax[1].set_ylabel("Δ rotation [deg]")
    ax[1].set_xlabel("Sample index")
    ax[1].set_title("Camera Pose Deviation from Mean (Rotation)")

    for a in ax:
        a.grid(True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_xy_xz_distributions_page(pdf, Ts_base2cam):
    if Ts_base2cam is None:
        return
    cam_positions = Ts_base2cam[:, :3, 3]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].scatter(cam_positions[:, 0], cam_positions[:, 1], s=10)
    ax[0].set_xlabel("X [m]")
    ax[0].set_ylabel("Y [m]")
    ax[0].set_title("Camera XY distribution (top view)")
    ax[0].grid(True)

    ax[1].scatter(cam_positions[:, 0], cam_positions[:, 2], s=10)
    ax[1].set_xlabel("X [m]")
    ax[1].set_ylabel("Z [m]")
    ax[1].set_title("Camera XZ distribution (side view)")
    ax[1].grid(True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_camera_orientation_vs_index_page(pdf, Ts_base2cam):
    if Ts_base2cam is None:
        return
    R_list = Ts_base2cam[:, :3, :3]
    rolls, pitches, yaws = [], [], []
    for R in R_list:
        r, p, y = rot_to_rpy(R)
        rolls.append(np.degrees(r))
        pitches.append(np.degrees(p))
        yaws.append(np.degrees(y))

    idx = range(len(rolls))
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(idx, rolls, marker="o")
    ax[0].set_ylabel("Roll [deg]")
    ax[1].plot(idx, pitches, marker="o")
    ax[1].set_ylabel("Pitch [deg]")
    ax[2].plot(idx, yaws, marker="o")
    ax[2].set_ylabel("Yaw [deg]")
    ax[2].set_xlabel("Sample index")

    ax[0].set_title("Camera Orientation (Base Frame, RPY)")
    for a in ax:
        a.grid(True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =========================
# 7. Main
# =========================

def main():
    args = parse_args()

    run_dir = os.path.join(args.base_dir, args.run_id)
    images_dir = os.path.join(run_dir, "images", args.camera_name)
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.md_path is None:
        md_path = os.path.join(
            results_dir, f"calib_report_{args.camera_name}.md"
        )
    else:
        md_path = args.md_path

    if args.output is None:
        pdf_path = os.path.join(
            results_dir, f"calib_report_{args.camera_name}.pdf"
        )
    else:
        pdf_path = args.output

    print("=== PDF Calibration Report ===")
    print(f"Base dir   : {args.base_dir}")
    print(f"Run ID     : {args.run_id}")
    print(f"Camera     : {args.camera_name}")
    print(f"Images dir : {images_dir}")
    print(f"Markdown   : {md_path}")
    print(f"PDF output : {pdf_path}\n")

    # Load data
    handeye_path, handeye_data = load_handeye(
        args.base_dir, args.run_id, args.camera_name
    )
    robot_csv_path, pos_robot, quat_robot = load_robot_poses(
        args.base_dir, args.run_id
    )
    md_text = load_markdown(md_path)

    cam_params_path, K, dist = load_camera_calib(
        args.base_dir, args.run_id, args.camera_name
    )
    pnp_path, pnp_filenames, pnp_rvecs, pnp_tvecs = load_pnp_poses(
        args.base_dir, args.run_id, args.camera_name
    )

    Ts_base2cam = compute_camera_poses_in_base(
        handeye_data, pos_robot, quat_robot
    )
    Ts_base2ee = build_ee_poses_in_base(pos_robot, quat_robot)
    Ts_base2board = compute_board_poses_in_base(
        handeye_data, pos_robot, quat_robot, pnp_rvecs, pnp_tvecs
    )

    print("Computing reprojection statistics ...")
    reproj_errors_all, per_image_rms, heatmap, detect_flags, image_paths = \
        compute_reprojection_stats(
            images_dir, K, dist, args.nx, args.ny, args.square_size
        )

    t_dev, rot_dev_deg, R_mean, t_mean = compute_camera_pose_deviation(
        Ts_base2cam
    )
    if t_mean is not None:
        roll_m, pitch_m, yaw_m = rot_to_rpy(R_mean)
        print("Mean camera position in base [m]:", t_mean)
        print("Mean camera RPY [deg]:",
              np.degrees(roll_m),
              np.degrees(pitch_m),
              np.degrees(yaw_m))

    # Build PDF
    with PdfPages(pdf_path) as pdf:
        # Page 1: summary
        summary_lines = [
            f"Base directory : {args.base_dir}",
            f"Run ID        : {args.run_id}",
            f"Camera        : {args.camera_name}",
            "",
            f"Hand–eye file : {handeye_path or 'NOT FOUND'}",
            f"Robot poses   : {robot_csv_path or 'NOT FOUND'}",
            f"Camera params : {cam_params_path or 'NOT FOUND'}",
            f"PnP poses     : {pnp_path or 'NOT FOUND'}",
            f"Markdown      : {md_path if os.path.exists(md_path) else 'NOT FOUND'}",
            "",
        ]

        if handeye_data is not None and "T_base2cam" in handeye_data:
            T_b_c = handeye_data["T_base2cam"]
            summary_lines.append("T_base2cam (average):")
            for row in T_b_c:
                summary_lines.append("  " + " ".join(f"{v: .4f}" for v in row))
        else:
            summary_lines.append("T_base2cam not stored in handeye file.")

        add_text_page(pdf, "Calibration Summary", summary_lines)

        # Page 2: markdown content (raw)
        add_markdown_page(pdf, md_text)

        # Page 3: 3D trajectory plot (EE vs camera)
        add_3d_plot_page(pdf, pos_robot, Ts_base2cam)

        # Page 4: EE vs camera frames
        add_3d_frames_page(pdf, Ts_base2ee, Ts_base2cam)

        # Page 5: full scene (base, EE, camera frames, board)
        add_scene_page(
            pdf,
            Ts_base2ee,
            Ts_base2cam,
            Ts_base2board,
            args.nx,
            args.ny,
            args.square_size,
        )

        # Page 6: mean camera frame + board
        add_mean_camera_frame_page(
            pdf,
            R_mean,
            t_mean,
            Ts_base2board,
            args.nx,
            args.ny,
            args.square_size,
        )

        # Page 7: camera look ray
        add_camera_look_ray_page(
            pdf,
            R_mean,
            t_mean,
            Ts_base2board,
            args.nx,
            args.ny,
            args.square_size,
        )

        # --- Extra plots: reprojection + intrinsics ---
        add_reproj_hist_page(pdf, reproj_errors_all)
        add_reproj_rms_page(pdf, per_image_rms)
        add_reproj_heatmap_page(pdf, heatmap, args.nx, args.ny)
        add_detection_diag_page(pdf, detect_flags, image_paths)
        add_distortion_coeffs_page(pdf, dist)

        # Board pose vs index
        add_board_pose_vs_index_page(pdf, pnp_rvecs, pnp_tvecs)

        # Camera pose diagnostics
        add_xy_xz_distributions_page(pdf, Ts_base2cam)
        add_camera_orientation_vs_index_page(pdf, Ts_base2cam)
        add_camera_deviation_page(pdf, t_dev, rot_dev_deg)

    print(f"[OK] PDF report written to: {pdf_path}")


if __name__ == "__main__":
    main()

