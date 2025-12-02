# THWS Multi-Camera Robot Calibration Workspace

This repository contains the complete robot–camera calibration workflow used in the THWS Robotics Project. It includes:

- UR10 robot motion (simulation + real robot)
- RealSense image capture
- Chessboard detection and PnP pose estimation
- Intrinsic camera calibration
- Hand–eye calibration
- Rich visualization and PDF report generation
- Basis for future multi-camera calibration (RealSense + webcam + ZED)
- Backend for a future ROS 2 GUI

The full software environment (ROS 2, UR drivers, RealSense SDK, Python stack, etc.) is provided as a Docker image:

    thws_multi_calibration.tar

This tar file is **not in the GitHub repo** (too large).  
It will be distributed via USB or shared storage.

--------------------------------------------------------------------------------
## 1. Repository Structure

Top-level layout:

    ur_ws_sim/
      ├── src/        ROS 2 packages (UR driver, calibration nodes, utilities)
      ├── tools/      Python calibration scripts (PnP, intrinsics, hand–eye, PDF)
      ├── data/       Calibration datasets (images, poses, results)
      └── README.md   This file

All paths in this README assume that, inside the container, you work from:

    /root/ur_ws_sim

--------------------------------------------------------------------------------
## 2. Load the THWS Docker Environment

On your host (Linux) machine, after you have:

    thws_multi_calibration.tar

load the image:

    docker load -i thws_multi_calibration.tar

You should see something like:

    thws_multi_calibration   latest

This image contains:

- ROS 2 Humble
- UR robot driver
- RealSense SDK
- OpenCV, NumPy, SciPy, Matplotlib
- All Python tools under tools/realsense_calib
- Other dependencies needed for this workspace

--------------------------------------------------------------------------------
## 3. Clone the Official Repository

On the host (outside Docker):

    git clone https://github.com/thws-project/ur_ws_sim
    cd ur_ws_sim

You now have the workspace on your host.

--------------------------------------------------------------------------------
## 4. Start the Development Container (Mounting the Workspace)

From inside the cloned repo folder on the host (ur_ws_sim/):

    docker run -it --net=host \
        -v $PWD:/root/ur_ws_sim \
        --name ur_dev \
        thws_multi_calibration:latest \
        bash

Inside the container, your prompt should look something like:

    root@<container-id>:/root/ur_ws_sim#

The source tree you see there is the same as on your host (because of the -v bind mount).

If you stop the container and want to re-enter it later:

    docker start ur_dev
    docker exec -it ur_dev bash

--------------------------------------------------------------------------------
## 5. Build the ROS 2 Workspace (inside the container)

Inside the container:

    cd /root/ur_ws_sim
    colcon build
    source install/setup.bash

You should do this once after cloning (and again after changing ROS 2 packages).

--------------------------------------------------------------------------------
## 6. Simulation Workflow (No Real Robot Needed)

Simulation lets you test the hemisphere motion and calibration scripts without a physical UR10.

### 6.1 Run simulated hemisphere motion + capture

From inside the container:

    cd /root/ur_ws_sim
    source install/setup.bash

Run the simulated capture node (name may be hemi_capture_sim or similar depending on the package; adjust if needed):

    ros2 run realsense_calib hemi_capture_sim

This will:

- Move a simulated UR robot through a hemi-spherical trajectory
- Simulate a RealSense camera capturing chessboard images
- Save data under something like:

      /root/ur_ws_sim/data/run_001/images/realsense/
      /root/ur_ws_sim/data/run_001/calib/robot_poses.csv

You can then use that run_001 for the calibration pipeline described below.

--------------------------------------------------------------------------------
## 7. Real Robot Workflow

This is the real UR10 + RealSense pipeline.

### 7.1 Start UR10 driver (on the same machine or a networked PC)

Inside the container (with network access to the robot):

    source /opt/ros/humble/setup.bash
    ros2 launch ur_robot_driver ur10_bringup.launch.py \
        ur_type:=ur10e \
        robot_ip:=<ROBOT_IP> \
        launch_rviz:=false

Replace <ROBOT_IP> with the actual UR controller IP.

Activate the scaled trajectory controller:

    ros2 control switch_controllers \
        --activate scaled_joint_trajectory_controller

Check controllers:

    ros2 control list_controllers

The scaled_joint_trajectory_controller should be active.

### 7.2 Start the RealSense camera node

In another terminal (inside the same Docker container):

    cd /root/ur_ws_sim
    source install/setup.bash

Then:

    ros2 launch realsense2_camera rs_launch.py \
        align_depth:=true \
        pointcloud.enable:=true

(You can adjust options as needed; this is a typical setup.)

### 7.3 Run hemisphere capture with the real robot

In another terminal (inside the same container):

    cd /root/ur_ws_sim
    source install/setup.bash

Then:

    ros2 run realsense_calib hemi_capture

This node will:

- Command the UR10 to move in a hemi-spherical motion around the chessboard
- Capture RealSense RGB frames
- Save synchronized robot poses and images

The resulting structure will look like:

    /root/ur_ws_sim/data/run_001/images/realsense/img_000.png, ...
    /root/ur_ws_sim/data/run_001/calib/robot_poses.csv

Different runs will use different run IDs (run_002, run_003, ...).

--------------------------------------------------------------------------------
## 8. Calibration Pipeline (Order Matters)

We assume:

- Base directory: /root/ur_ws_sim/data
- Run ID: run_001
- Camera name: realsense

You can adjust these if your run or camera name is different.

### 8.1 Step 1 — Intrinsic Camera Calibration

This step estimates the camera intrinsics K and distortion coefficients from chessboard images.

Run (inside the container):

    cd /root/ur_ws_sim
    python3 tools/realsense_calib/camera_calibration.py \
        --base-dir /root/ur_ws_sim/data \
        --run-id run_001 \
        --camera-name realsense

This produces (either in the run-specific calib folder or a global calib folder):

    /root/ur_ws_sim/data/run_001/calib/CameraParams_realsense.npz

or

    /root/ur_ws_sim/data/calib/CameraParams_realsense.npz

The file contains:

- K  : 3x3 intrinsic matrix
- dist : distortion vector

### 8.2 Step 2 — PnP (Board Pose in Camera Frame)

Using the intrinsics from the previous step, we estimate the chessboard pose for each image using solvePnP.

Run:

    cd /root/ur_ws_sim
    python3 tools/realsense_calib/pnp_solve.py \
        --base-dir /root/ur_ws_sim/data \
        --run-id run_001 \
        --camera-name realsense

This creates:

    /root/ur_ws_sim/data/run_001/calib/poses_realsense.npz

It typically contains:

- filenames : N image names
- rvecs     : N x 3 Rodrigues rotation vectors
- tvecs     : N x 3 translation vectors
- K, dist   : copied intrinsics

These represent "board in camera" (target in camera frame).

### 8.3 Step 3 — Hand–Eye Calibration

Here we solve for the camera-with-respect-to-robot transform using OpenCV’s calibrateHandEye.

Run:

    cd /root/ur_ws_sim
    python3 tools/realsense_calib/hand_eye_calibration.py \
        --base-dir /root/ur_ws_sim/data \
        --run-id run_001 \
        --camera-name realsense

This outputs:

    /root/ur_ws_sim/data/run_001/calib/handeye_realsense.npz

Containing, among others:

- R_cam2ee, t_cam2ee : camera in end-effector frame
- T_cam2ee           : 4x4 homogeneous transform
- T_base2cam         : average camera pose in base frame
- T_cam2base         : inverse of T_base2cam

A YAML summary (handeye_realsense.yaml) is also written for human inspection.

### 8.4 Step 4 — PDF Calibration Report

Finally, we generate a detailed PDF combining all data and visualizations.

Run:

    cd /root/ur_ws_sim
    python3 tools/realsense_calib/make_calibration_report.py \
        --base-dir /root/ur_ws_sim/data \
        --run-id run_001 \
        --camera-name realsense

This creates:

    /root/ur_ws_sim/data/run_001/results/calib_report_realsense.pdf

The PDF typically includes:

- 3D visualization of:
    - Base frame
    - EE trajectory
    - Camera poses
    - Board plane (mean pose)
- Histograms of reprojection error
- Per-image RMS reprojection plot
- Corner-wise reprojection heatmap
- Detection success per image
- Distortion coefficients bar plot
- Pose stability and deviations
- World/EE/camera frames drawn as colored axes

This is the main artifact to judge calibration quality.

--------------------------------------------------------------------------------
## 9. Future Multi-Camera Calibration (Webcam, ZED, etc.)

The long-term goal is to support multiple cameras simultaneously, for example:

- RealSense
- USB webcam
- ZED stereo (left/right)

A typical data layout might be:

    data/run_001/images/realsense/
    data/run_001/images/webcam/
    data/run_001/images/zed_left/
    data/run_001/images/zed_right/
    data/run_001/calib/...

For each camera, the same pattern will apply:

- CameraParams_<camera>.npz
- poses_<camera>.npz
- handeye_<camera>.npz

A future script (e.g. tools/realsense_calib/multi_handeye.py) will then:

- Solve all camera–EE transforms jointly
- Estimate camera–camera transforms
- Perform graph-based optimization
- Output a multi-camera report and TF tree

At the moment, the infrastructure is primarily built and tested for:

- single camera = RealSense RGB

--------------------------------------------------------------------------------
## 10. GUI Integration Notes

This repository is intended to be the computational backend for a future GUI.

The planned GUI will:

- Detect connected cameras
- Connect to the UR robot
- Start the appropriate hemisphere trajectory
- Capture images while logging robot poses
- Run:
    - Intrinsic calibration
    - PnP
    - Hand–eye
- Visualize the 3D scene (EE, camera, board)
- Generate the PDF report
- Eventually handle multiple cameras

Most of the numerical heavy lifting is already implemented in:

    tools/realsense_calib/

GUI developers should treat these scripts as reference implementations and/or callable modules.

--------------------------------------------------------------------------------
## 11. Troubleshooting

**Problem: Docker image not found after docker load**

- Ensure you used the correct file name:
  
      docker load -i thws_multi_calibration.tar

- Check with:

      docker images

**Problem: Robot does not move**

- Verify the UR driver node is running.
- Check that scaled_joint_trajectory_controller is active.
- Ensure the robot is out of protective stop and in remote mode.

**Problem: RealSense node fails**

- Check USB connection.
- Sometimes helps to unplug / replug and relaunch.
- Check permissions on /dev/bus/usb devices if running on bare metal.

**Problem: Chessboard not detected**

- Ensure the printed board matches the nx, ny, and square-size used by the scripts.
- Improve lighting and avoid motion blur.
- Ensure the board is fully visible in the image.

**Problem: Hand–eye transform looks wrong**

- Common causes:
    - Intrinsic calibration is poor (reprojection errors too high).
    - Chessboard motion is too small or only in a narrow region.
    - Board is always at nearly the same distance / orientation.
- Check:
    - Number of valid images in pnp_solve output.
    - RMS reprojection errors in the PDF.
    - Board-in-base consistency section in the PDF.

--------------------------------------------------------------------------------
## 12. Credits

This workspace is developed within the THWS Robotics Project by students and staff.

- Base ROS 2 and UR driver setup
- RealSense integration
- Calibration pipeline design
- Visualization and reporting tools

Contributions and improvements are welcome via pull requests.

