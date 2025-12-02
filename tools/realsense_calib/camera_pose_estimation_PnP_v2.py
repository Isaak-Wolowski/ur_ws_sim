import numpy as np
import os
import glob
import cv2
from enum import Enum

class DrawOption(Enum):
    NONE = 0
    AXES = 1
    CUBE = 2

def tuple_of_ints(arr):
    return tuple(int(x) for x in arr)

def draw_axes(img, corners, imgpts):
    """Draw 3D axes (X=red, Y=green, Z=blue) at the first chessboard corner."""
    corner = tuple_of_ints(corners[0].ravel())
    img = cv2.line(img, corner, tuple_of_ints(imgpts[0].ravel()), (0, 0, 255), 2)  # X (red)
    img = cv2.line(img, corner, tuple_of_ints(imgpts[1].ravel()), (0, 255, 0), 2)  # Y (green)
    img = cv2.line(img, corner, tuple_of_ints(imgpts[2].ravel()), (255, 0, 0), 2)  # Z (blue)
    return img

def draw_cube(img, imgpts):
    """Draw a 3D cube on the chessboard."""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # bottom face (green)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
    # top face (red)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)
    # vertical edges (white)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 255), 2)
    return img

def compute_homogeneous_poses(rvecs, tvecs):
    """Convert lists of (rvec, tvec) from solvePnP into 4x4 homogeneous matrices."""
    poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        poses.append(T)
    return np.array(poses, dtype=np.float64)

def pose_estimation_for_camera(
    root_dir,
    camera_name,
    run_id,
    chess_rows=8,
    chess_cols=11,
    square_size_m=0.024,
    draw_option=DrawOption.NONE,
):
    """
    Generic pose estimation for a single camera.

    Directory layout it expects:
        root_dir/
            CameraParams_<camera_name>.npz
            run_<run_id>/
                images/
                    <camera_name>/
                        *.png

    This will produce:
        root_dir/run_<run_id>/images/<camera_name>/poses.npy
            -> array of shape (N, 4, 4), board pose in camera frame.
    """

    # ------------------------------------------------------------------
    # 1. Paths
    # ------------------------------------------------------------------
    calib_file = os.path.join(root_dir, f"CameraParams_{camera_name}.npz")
    img_dir    = os.path.join(root_dir, f"run_{run_id}", "images", camera_name)
    out_dir    = img_dir  # save poses next to images

    os.makedirs(out_dir, exist_ok=True)

    print(f"🔧 Camera: {camera_name}")
    print(f"📂 Calibration file: {calib_file}")
    print(f"📂 Image directory : {img_dir}")

    # ------------------------------------------------------------------
    # 2. Load intrinsics
    # ------------------------------------------------------------------
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")


    # camera matrix (common name)
    if "cameraMatrix" in data:
        cameraMatrix = data["cameraMatrix"]
    elif "K" in data:
        cameraMatrix = data["K"]
    else:
        raise KeyError("Could not find camera matrix in npz (tried 'cameraMatrix', 'K').")

    # distortion coeffs: support both 'dist' and 'distCoeffs'
    if "dist" in data:
        distCoeffs = data["dist"]
    elif "distCoeffs" in data:
        distCoeffs = data["distCoeffs"]
    else:
        raise KeyError("Could not find distortion coeffs in npz (tried 'dist', 'distCoeffs').")
        

    print("\nLoaded intrinsics:")
    print("K =\n", cameraMatrix)
    print("dist =", distCoeffs.ravel())

    # ------------------------------------------------------------------
    # 3. Chessboard model (same for all cameras)
    # ------------------------------------------------------------------
    pattern_size = (chess_cols, chess_rows)  # (cols, rows)
    objp = np.zeros((chess_cols * chess_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1, 2)
    objp *= square_size_m

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]) * square_size_m
    cube_corners = np.float32(
        [
            [0, 0, 0],
            [0, 3, 0],
            [3, 3, 0],
            [3, 0, 0],
            [0, 0, -3],
            [0, 3, -3],
            [3, 3, -3],
            [3, 0, -3],
        ]
    ) * square_size_m

    # ------------------------------------------------------------------
    # 4. Load images
    # ------------------------------------------------------------------
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_paths:
        raise RuntimeError(f"No PNG images found in {img_dir}")

    print(f"\nFound {len(img_paths)} images for camera '{camera_name}'.")

    # ------------------------------------------------------------------
    # 5. Run PnP on each image
    # ------------------------------------------------------------------
    term_crit = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    rvecs, tvecs = [], []
    used_indices = []
    reproj_errors = []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            print(f"❌ Chessboard NOT found in: {os.path.basename(img_path)}")
            continue

        # refine corners
        corners_ref = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), term_crit
        )

        # solve PnP (board in camera frame)
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners_ref, cameraMatrix, distCoeffs
        )
        if not ok:
            print(f"❌ solvePnP failed for: {os.path.basename(img_path)}")
            continue

        rvecs.append(rvec)
        tvecs.append(tvec)
        used_indices.append(idx)

        # reprojection error
        projected_pts, _ = cv2.projectPoints(
            objp, rvec, tvec, cameraMatrix, distCoeffs
        )
        err = np.mean(
            np.linalg.norm(
                projected_pts.squeeze() - corners_ref.squeeze(), axis=1
            )
        )
        reproj_errors.append(err)

        # optional visualization
        if draw_option != DrawOption.NONE:
            if draw_option == DrawOption.AXES:
                imgpts, _ = cv2.projectPoints(
                    axis, rvec, tvec, cameraMatrix, distCoeffs
                )
                img = draw_axes(img, corners_ref, imgpts)
            elif draw_option == DrawOption.CUBE:
                imgpts, _ = cv2.projectPoints(
                    cube_corners, rvec, tvec, cameraMatrix, distCoeffs
                )
                img = draw_cube(img, imgpts)

            cv2.imshow(f"{camera_name} pose", img)
            cv2.waitKey(50)

    cv2.destroyAllWindows()

    if not rvecs:
        raise RuntimeError("No valid PnP results – no chessboards were detected.")

    print(f"\n✔️ Valid PnP poses: {len(rvecs)} / {len(img_paths)}")
    print(f"   Mean reprojection error: {np.mean(reproj_errors):.3f} px")

    # ------------------------------------------------------------------
    # 6. Save poses as 4x4 matrices
    # ------------------------------------------------------------------
    poses = compute_homogeneous_poses(rvecs, tvecs)
    poses_path = os.path.join(out_dir, "poses.npy")
    np.save(poses_path, poses)

    print(f"\n💾 Saved {poses.shape[0]} poses to:\n   {poses_path}")
    print("Each pose is a 4x4 [R|t] with board in camera frame.")

    # Optionally also save raw rvecs/tvecs
    np.save(os.path.join(out_dir, "rvecs.npy"), np.array(rvecs))
    np.save(os.path.join(out_dir, "tvecs.npy"), np.array(tvecs))

    return poses, np.array(reproj_errors), used_indices


if __name__ == "__main__":
    # Example usage for Realsense:
    root_dir = os.path.dirname(os.path.abspath(__file__))  # tools/realsense_calib or similar
    pose_estimation_for_camera(
        root_dir=root_dir,
        camera_name="realsense",   # e.g. folder name under images/
        run_id="100",              # e.g. run_100
        chess_rows=8,
        chess_cols=11,
        square_size_m=0.024,
        draw_option=DrawOption.AXES,
    )

