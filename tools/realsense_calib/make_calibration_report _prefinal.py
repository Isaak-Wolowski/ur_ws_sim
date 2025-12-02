#!/usr/bin/env python3
import os
import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

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


# =========================
# 4. Compute per-image camera TFs
# =========================

def compute_camera_poses_in_base(handeye_data, pos, quat):
    """
    From:
      - handeye_data: contains R_cam2ee, t_cam2ee
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

    T_cam2ee = rt_to_T(R_cam2ee, t_cam2ee)
    T_ee2cam = invert_T(T_cam2ee)

    Ts_base2cam = []
    for p, q in zip(pos, quat):
        R_b_ee = quat_to_rot(q)
        T_b_ee = rt_to_T(R_b_ee, p)
        T_b_cam = T_b_ee @ T_ee2cam
        Ts_base2cam.append(T_b_cam)

    return np.stack(Ts_base2cam, axis=0)


# =========================
# 5. Plot helpers
# =========================

def set_equal_3d(ax):
    """
    Make 3D axes have equal scale (so spheres look like spheres).
    """
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
    # We don't try to render markdown properly; just show raw text.
    wrapped_lines = []
    for line in md_text.splitlines():
        # wrap long lines
        wrapped_lines.extend(textwrap.wrap(line, width=100) or [""])

    add_text_page(pdf, "Calibration Markdown Report", wrapped_lines)


def add_3d_plot_page(pdf, pos_robot, Ts_base2cam):
    if pos_robot is None or Ts_base2cam is None:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Robot EE positions
    ax.scatter(
        pos_robot[:, 0],
        pos_robot[:, 1],
        pos_robot[:, 2],
        marker="o",
        label="EE in base",
        alpha=0.6,
    )

    # Camera positions (per image)
    cams = Ts_base2cam[:, :3, 3]
    ax.scatter(
        cams[:, 0],
        cams[:, 1],
        cams[:, 2],
        marker="^",
        label="Camera in base (per image)",
        alpha=0.8,
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Robot EE & Camera Trajectory in Base Frame")
    ax.legend(loc="best")

    set_equal_3d(ax)

    pdf.savefig(fig)
    plt.close(fig)


# =========================
# 6. Main
# =========================

def main():
    args = parse_args()

    run_dir = os.path.join(args.base_dir, args.run_id)
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
    Ts_base2cam = compute_camera_poses_in_base(
        handeye_data, pos_robot, quat_robot
    )

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
            f"Markdown      : {md_path if os.path.exists(md_path) else 'NOT FOUND'}",
            "",
        ]

        if handeye_data is not None:
            if "T_base2cam" in handeye_data:
                T_b_c = handeye_data["T_base2cam"]
                summary_lines.append("T_base2cam (average):")
                for row in T_b_c:
                    summary_lines.append("  " + " ".join(f"{v: .4f}" for v in row))
            else:
                summary_lines.append("T_base2cam not stored in handeye file.")
        else:
            summary_lines.append("No hand–eye data loaded.")

        add_text_page(pdf, "Calibration Summary", summary_lines)

        # Page 2: markdown content (raw)
        add_markdown_page(pdf, md_text)

        # Page 3: 3D trajectory plot
        add_3d_plot_page(pdf, pos_robot, Ts_base2cam)

    print(f"[OK] PDF report written to: {pdf_path}")


if __name__ == "__main__":
    main()

