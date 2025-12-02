# THWS Multi-Camera Robot Calibration Workspace

This repository contains:

- ROS 2 workspace (src/)
- Calibration tools (tools/)
- Calibration data (data/)
- Scripts for RealSense, multi-camera PnP, hand–eye calibration, visualization, and PDF reporting

The full software environment (ROS 2, Python dependencies, RealSense SDK, etc.)
is distributed separately as a Docker image archive:

    ur_env_snapshot.tar

This file is NOT part of the GitHub repo.
You will receive it via USB, shared drive, or download link.

--------------------------------------------------------------------

## 1. Load the Docker Environment

After receiving the file `ur_env_snapshot.tar`, load it:

    docker load -i ur_env_snapshot.tar

Expected output:

    Loaded image: thws_multi_calibration:latest

--------------------------------------------------------------------

## 2. Run the Environment

Start the container:

    docker run -it --net=host --privileged thws_multi_calibration:latest

Inside the container:

    cd /root/ur_ws_sim

--------------------------------------------------------------------

## 3. Recommended Workflow (Git + Docker Together)

Clone the repository on your **host** machine:

    git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
    cd <YOUR-REPO>

Run the container while mounting the repo:

    docker run -it --net=host --privileged \
        -v $(pwd):/root/ur_ws_sim \
        thws_multi_calibration:latest

Then inside the container:

    cd /root/ur_ws_sim

This gives you:
- Environment from Docker
- Code from GitHub  
The correct professional robotics workflow.

--------------------------------------------------------------------

## 4. Project Structure

    ur_ws_sim/
    ├── src/            # ROS 2 packages
    ├── tools/          # Calibration scripts
    ├── data/           # Calibration runs and results
    ├── README.md
    └── .gitignore

--------------------------------------------------------------------

## 5. Running Calibration Tools (Inside Container)

Run PnP:

    python3 tools/realsense_calib/pnp_rs.py --run-id run_001 --camera-name realsense

Run hand–eye:

    python3 tools/realsense_calib/hand_eye.py --run-id run_001 --camera-name realsense

Generate calibration report PDF:

    python3 tools/realsense_calib/make_calibration_report.py --run-id run_001 --camera-name realsense

--------------------------------------------------------------------

## 6. Notes

- Do NOT commit `ur_env_snapshot.tar` to GitHub.
- When environment dependencies change, regenerate the tar with:
  
      docker commit <container_id> thws_multi_calibration:latest
      docker save thws_multi_calibration:latest -o ur_env_snapshot.tar

- Anyone who:
  1) loads the tar file and  
  2) clones this repo  
can reproduce the full environment.

