# Calibration report – run `run_001`, camera `realsense`

- Base directory: `/root/ur_ws_sim/data`
- Run directory : `/root/ur_ws_sim/data/run_001`

---

## 1. Camera intrinsics

- Calibration file: `/root/ur_ws_sim/data/run_001/calib/CameraParams_realsense.npz`
- Number of images used: **98**
- Image resolution: **1280 x 720**
- Chessboard inner corners (nx × ny): **7 × 10**
- Chessboard square size: **0.024000 m**
- RMS reprojection error: **0.192799 px**
- Per-image mean reprojection error: min=0.1090, mean=0.1575, max=0.2373 px

Camera matrix K:

```
[[919.18559339   0.         649.55459405]
 [  0.         920.90454306 365.54467847]
 [  0.           0.           1.        ]]
```

Distortion coefficients:

```
[[ 1.11659574e-01  8.12637857e-03  7.44873124e-04 -1.59891864e-03
  -7.89971697e-01]]
```

---

## 2. PnP board poses (board in camera frame)

- PnP file: `/root/ur_ws_sim/data/run_001/calib/poses_realsense.npz`
- Number of images with successful PnP: **98**
- Board distance from camera (min/mean/max): **0.390 / 0.559 / 0.625 m**

Example poses (first up to 3 images):

**Image:** `img_0000_pose_0000.png`

- t (board in camera) [m] = [-0.0016, 0.1433, 0.5908]
- rvec (Rodrigues) [rad] = [0.2766, 0.0480, -3.0640]

**Image:** `img_0001_pose_0001.png`

- t (board in camera) [m] = [-0.0012, 0.1430, 0.5887]
- rvec (Rodrigues) [rad] = [0.2717, 0.0156, -3.0665]

**Image:** `img_0002_pose_0002.png`

- t (board in camera) [m] = [-0.0008, 0.1428, 0.5853]
- rvec (Rodrigues) [rad] = [0.2569, -0.0160, -3.0691]

---

## 3. Hand–eye calibration

- Hand–eye file: `/root/ur_ws_sim/data/run_001/calib/handeye_realsense.npz`
- Method: **Tsai**
- Number of samples used: **98**

### 3.1 Camera in EE frame (T_cam^ee)

Rotation matrix R_cam2ee:

```
[[ 0.9941614   0.03865706  0.10074097]
 [-0.04107452  0.99891312  0.02203331]
 [-0.09977974 -0.02604255  0.99466868]]
```

Translation t_cam2ee [m]: [0.1024, -0.0784, 0.0439]

RPY (deg, ZYX convention) ≈ roll=-1.50, pitch=5.73, yaw=-2.37

Homogeneous transform T_cam2ee:

```
array([[ 0.9941614 ,  0.03865706,  0.10074097,  0.10243068],
       [-0.04107452,  0.99891312,  0.02203331, -0.07842157],
       [-0.09977974, -0.02604255,  0.99466868,  0.04386567],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
```

### 3.2 Camera in BASE frame (T_base^cam, averaged)

Homogeneous transform T_base2cam:

```
array([[-0.03868316, -0.99891242,  0.02603064,  1.09752458],
       [-0.99415935,  0.04109951,  0.09978987,  0.14687683],
       [-0.10075118, -0.02201842, -0.99466798,  1.45043942],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
```

Translation base→camera [m]: [1.0975, 0.1469, 1.4504]

RPY (deg, ZYX convention) ≈ roll=-178.73, pitch=5.78, yaw=-92.23

Homogeneous transform T_cam2base (inverse of T_base2cam):

```
array([[-0.03868316, -0.99415935, -0.10075118,  0.33460817],
       [-0.99891242,  0.04109951, -0.02201842,  1.12223076],
       [ 0.02603064,  0.09978987, -0.99466798,  1.39947956],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
```


_This report was auto-generated. Graphs and plots are available in the PDF: `calib_report_realsense.pdf`_
