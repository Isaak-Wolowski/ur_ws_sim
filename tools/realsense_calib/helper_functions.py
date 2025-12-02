import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R
import cv2 as cv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

import inspect
import csv





def rod_to_rmat(rod_vec):
    # Initialize a Rotation object with the Rodrigues vector
    rotation = R.from_rotvec(rod_vec)
    
    # Convert to rotation matrix representation
    rmat = rotation.as_matrix()
    
    # Return the rotation matrix
    return rmat


def rmat_to_rod(rmat):
    """
    Converts a rotation matrix to a Rodrigues vector.
    
    Parameters:
    - rmat: The rotation matrix.
    
    Returns:
    - Rodrigues vector representing the rotation.
    """
    # Initialize a Rotation object with the rotation matrix
    rotation = R.from_matrix(rmat)
    
    # Convert to Rodrigues vector representation
    rod_vec = rotation.as_rotvec()
    
    # Return the Rodrigues vector
    return rod_vec

def rot_mat_to_axis_angle(R):
    """
    Converts a rotation matrix to axis-angle representation.
    
    Parameters:
    - R: 3x3 rotation matrix
    
    Returns:
    - waxis: 3x1 rotation axis vector
    - wangle: Rotation angle in radians
    """
    # Compute the angle of rotation using the trace of the matrix
    angle = np.arccos((np.trace(R) - 1) / 2)
    
    if np.isclose(angle, 0.0):
        # If angle is near zero, any vector is an axis (default to [1, 0, 0])
        return np.array([1.0, 0.0, 0.0]), 0.0
    
    # Compute the rotation axis
    axis = np.array([R[2, 1] - R[1, 2], 
                      R[0, 2] - R[2, 0], 
                      R[1, 0] - R[0, 1]])
    
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    return axis, angle

def rod_tvec_to_pose(params):
    """
    Convert a 6-element vector [rx, ry, rz, tx, ty, tz] to a 4x4 homogeneous transformation matrix.
    - [rx, ry, rz]: Rodrigues vector for rotation
    - [tx, ty, tz]: Translation vector
    """
    rx, ry, rz = params[:3]
    tx, ty, tz = params[3:]

    # Rotation matrix from Rodrigues vector
    r = np.array([rx, ry, rz])
    theta = np.linalg.norm(r)
    if theta < 1e-10:
        R = np.eye(3)
    else:
        r_hat = r / theta
        K = np.array([[0, -r_hat[2], r_hat[1]],
                      [r_hat[2], 0, -r_hat[0]],
                      [-r_hat[1], r_hat[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Translation vector
    t = np.array([tx, ty, tz])

    # Construct 4x4 homogeneous matrix
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t
    return X


# def rot_mat_to_axis_angle(R):
#     """
#     Converts a rotation matrix to axis-angle representation.

#     Parameters:
#         R (numpy.ndarray): A 3x3 rotation matrix.

#     Returns:
#         numpy.ndarray: The rotation axis (3D vector).
#         float: The rotation angle (in radians).
#     """
#     # Ensure the input is a valid 3x3 matrix
#     if R.shape != (3, 3):
#         raise ValueError("Input must be a 3x3 rotation matrix.")

#     # Compute the angle
#     trace = np.trace(R)
#     angle = np.arccos((trace - 1) / 2)

#     # Handle numerical precision issues for angle close to 0 or π
#     if np.isclose(angle, 0.0):  # No rotation
#         return np.array([1.0, 0.0, 0.0]), 0.0
#     elif np.isclose(angle, np.pi):  # 180-degree rotation
#         # Special case: extract the axis from the diagonal
#         axis = np.sqrt((np.diagonal(R) + 1) / 2.0)
#         axis[np.isnan(axis)] = 0.0  # Handle numerical issues
#         return axis / np.linalg.norm(axis), angle

#     # General case: compute the axis
#     axis = np.array([
#         R[2, 1] - R[1, 2],
#         R[0, 2] - R[2, 0],
#         R[1, 0] - R[0, 1]
#     ]) / (2 * np.sin(angle))

#     return axis / np.linalg.norm(axis), angle







def axis_angle_to_rot_mat(axis, angle):
    """ Converts axis-angle representation to a rotation matrix """
    #K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])  # Skew-symmetric matrix
    #I = np.eye(3)
    #R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    R = (np.eye(3) * np.cos(angle) + np.sin(angle) * skew(axis) + (1 - np.cos(angle)) * np.outer(axis, axis)).T
    return R




def rodrigues_rotation_matrix(v, theta):
    """
    Computes the rotation matrix using Rodrigues' rotation formula.
    
    v: numpy array, rotation axis (should be a unit vector)
    theta: float, rotation angle in radians
    """
    v = np.asarray(v).flatten()  # Ensures v is a flattened 1D array
    if v.shape[0] != 3:
        raise ValueError("The rotation axis v must be a 3-dimensional vector")
    v = v / np.linalg.norm(v)  # Normalize v to make it a unit vector
    
    # Identity matrix
    I = np.eye(3)
    
    # Skew-symmetric matrix of v
    Vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    # Rodrigues' formula components
    Vx2 = np.dot(Vx, Vx)
    R = I + np.sin(theta) * Vx + (1 - np.cos(theta)) * Vx2
    
    return R


def rmat_to_quat(R):
    """
    Converts a 3x3 rotation matrix to a quaternion [q_w, q_x, q_y, q_z].
    
    Parameters:
    R (ndarray): 3x3 rotation matrix.
    
    Returns:
    ndarray: Quaternion [q_w, q_x, q_y, q_z].
    """
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * q_w
        q_w = 0.25 * S
        q_x = (R[2, 1] - R[1, 2]) / S
        q_y = (R[0, 2] - R[2, 0]) / S
        q_z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * q_x
        q_w = (R[2, 1] - R[1, 2]) / S
        q_x = 0.25 * S
        q_y = (R[0, 1] + R[1, 0]) / S
        q_z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * q_y
        q_w = (R[0, 2] - R[2, 0]) / S
        q_x = (R[0, 1] + R[1, 0]) / S
        q_y = 0.25 * S
        q_z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * q_z
        q_w = (R[1, 0] - R[0, 1]) / S
        q_x = (R[0, 2] + R[2, 0]) / S
        q_y = (R[1, 2] + R[2, 1]) / S
        q_z = 0.25 * S

    return np.array([q_w, q_x, q_y, q_z])



def quat_to_rmat(quaternion):
    """
    Converts a quaternion [q_w, q_x, q_y, q_z] to a 3x3 rotation matrix.
    
    Parameters:
    quaternion (ndarray): Quaternion [q_w, q_x, q_y, q_z].
    
    Returns:
    ndarray: 3x3 rotation matrix.
    """
    q_w, q_x, q_y, q_z = quaternion

    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
    ])
    return R




def load_and_slice_poses(file_path, start_index, end_index):
    """
    Loads pose data from a file, validates indices, and returns a sliced batch of poses.

    Parameters:
    - file_path (str): Path to the .npy file containing pose data.
    - start_index (int): Starting index for slicing the poses.
    - end_index (int): Ending index for slicing the poses (exclusive).

    Returns:
    - list: A list of sliced pose arrays.

    Raises:
    - ValueError: If the start or end index is invalid.
    """
    # Load the pose data
    poses = np.load(file_path)
    poses = [np.array(row) for row in poses]

    # Validate indices
    if start_index < 0 or end_index > len(poses) or start_index >= end_index:
        raise ValueError("Invalid start or end index. Please ensure that the indices are within the range of the list.")

    # Slice and return the poses
    return poses[start_index:end_index]




def nearest_rotation_matrix_svd(noisy_rmat):
    U, _, Vt = np.linalg.svd(noisy_rmat)
    R = np.dot(U, Vt)
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    distance = np.linalg.norm(noisy_rmat - R, 'fro')
    #print(f"The Frobenius norm distance is: {distance}")def nearest_rotation_matrix_svd(noisy_rmat):
    U, _, Vt = np.linalg.svd(noisy_rmat)
    R = np.dot(U, Vt)
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    distance = np.linalg.norm(noisy_rmat - R, 'fro')
    #print(f"The Frobenius norm distance is: {distance}")
    
    
def generate_random_rotation_matrix(seed=None):
    """
    Generate a random 3x3 rotation matrix.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
   
    # Set the random seed
    np.random.seed(seed) 
    # Generate a random 3x3 matrix
    random_matrix = np.random.rand(3, 3)
    
    # Perform QR decomposition to ensure orthonormal columns
    q, r = np.linalg.qr(random_matrix)
    
    # Ensure the determinant is 1 for a proper rotation matrix
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1  # Flip the sign of the last column
    
    return q

def generate_random_translation_vector(scale=1.0):
    """
    Generates a random 3D translation vector.

    Parameters:
        scale (float): The scale factor for the translation values. Default is 1.0.

    Returns:
        numpy.ndarray: A 3D translation vector as a (3,) numpy array.
    """
    return scale * np.random.uniform(-1, 1, size=3)




def noisify_pose(pose, deg_noise, axis_noise, trans_noise):
    Pnoise = np.eye(4)  # Initialize Pnoise as a 4x4 identity matrix
    
    # Extract the rotation matrix and convert it to axis-angle representation
    waxis, wnorm = rot_mat_to_axis_angle(pose[:3, :3])
    
    # Add noise to the axis and normalize it
    waxis_noise = waxis + axis_noise * np.random.randn(3)
    waxis_noise = waxis_noise / np.linalg.norm(waxis_noise)
    
    # Add noise to the angle
    wnorm_noise = wnorm + deg_noise * np.random.randn(1)
    
    # Convert back from axis-angle to rotation matrix with noise
    Rnoise = axis_angle_to_rot_mat(waxis_noise, wnorm_noise)
    
    # Update the rotation part of Pnoise
    Pnoise[:3, :3] = Rnoise
    
    # Add noise to the translation part
    Pnoise[:3, 3] = pose[:3, 3] + trans_noise * np.random.randn(3)
    tnoise = Pnoise[:3, 3]
    tnoise = tnoise.reshape(3,1)
    
    return Pnoise, Rnoise, tnoise


def noisify_pose2(pose, rvec_stdDev, tvec_stdDev):
    pose_noisy = np.eye(4)  # Initialize Pnoise as a 4x4 identity matrix
    
    # Extract the rotation matrix and convert it to rodrigues vector representation
    rvec = rmat_to_rod(pose[:3, :3])
    
    # Add noise to the rotation vector
    rotation_noise = np.random.normal(0, rvec_stdDev, rvec.shape)
    rvec_noisy = rvec + rotation_noise
    
    # Convert rot_vector back to matrix
    rmat_noisy = rod_to_rmat(rvec_noisy)
    
    # Update the rotation part of pose_noisy
    pose_noisy[:3, :3] = rmat_noisy
    
    # Add normally distributed noise to the translation vector
    tvec = pose[:3, 3]
    translation_noise = np.random.normal(0, tvec_stdDev, tvec.shape)
    tvec_noisy = tvec + translation_noise
    
    # Update the translation part of pose_noisy
    pose_noisy[:3, 3] = tvec_noisy
    
    return pose_noisy, rmat_noisy, tvec_noisy
  
# This function uses the noisify_pose() or nosify_poses2() function.
def noisify_poses(poses, rvec_stdDev, tvec_stdDev):
    poses_noisy = []
    rmat_noisy_list = []
    tvec_noisy_list = []

    for item_pose in poses:
        pose_noisy, rmat_noisy, tvec_noisy = noisify_pose2(item_pose, rvec_stdDev, tvec_stdDev)
        poses_noisy.append(pose_noisy)
        rmat_noisy_list.append(rmat_noisy)
        tvec_noisy_list.append(tvec_noisy)

    return poses_noisy, rmat_noisy_list, tvec_noisy_list 

    

def save_poses(poses, file_path):
    """
    Saves poses to a text file and a numpy file.

    Parameters:
    - poses (list): List of poses to save.
    - file_path (str): Base path (excluding extension) for saving the files.

    Returns:
    - None
    """
    # Save poses in a text file
    text_file_path = f"{file_path}.txt"
    with open(text_file_path, 'w') as file:
        for item in poses:
            file.write(str(item) + '\n')

    # Save poses as a numpy file
    numpy_file_path = f"{file_path}.npy"
    np.save(numpy_file_path, poses)

    #print(f"Poses saved to:\n  Text file: {text_file_path}\n  Numpy file: {numpy_file_path}")


    



def skew(v):
    """Generate skew-symmetric matrix for a vector v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def tsai(A_list, B_list):
    """Calculates the least squares solution of AX = XB with list of 4x4 matrices."""
    
    # Get the number of matrices
    n = len(A_list)
    
    # Initialize S and v matrices
    S = np.zeros((3 * n, 3))
    v = np.zeros((3 * n, 1))
    
    # Calculate the best rotation R
    for i in range(n):
        # Extract the 3x3 rotation matrix from the 4x4 transformation matrices A and B
        A1 = logm(A_list[i][0:3, 0:3])
        B1 = logm(B_list[i][0:3, 0:3])
        
        # Extract axis-angle representations
        a = np.array([A1[2, 1], A1[0, 2], A1[1, 0]])
        a = a / np.linalg.norm(a)
        
        b = np.array([B1[2, 1], B1[0, 2], B1[1, 0]])
        b = b / np.linalg.norm(b)
        
        # Build S matrix and v vector
        S[3 * i:3 * i + 3, :] = skew(a + b)
        v[3 * i:3 * i + 3, :] = (a - b).reshape(3, 1)
    
    # Solve for x
    x = np.linalg.lstsq(S, v, rcond=None)[0].flatten()
    
    # Calculate rotation angle theta
    theta = 2 * np.arctan(np.linalg.norm(x))
    
    # Normalize x
    x = x / np.linalg.norm(x)
    
    # Calculate rotation matrix R
    R = (np.eye(3) * np.cos(theta) +
         np.sin(theta) * skew(x) +
         (1 - np.cos(theta)) * np.outer(x, x)).T
    
    # Calculate the best translation t
    C = np.zeros((3 * n, 3))
    d = np.zeros((3 * n, 1))
    I = np.eye(3)
    
    for i in range(n):
        # Build C matrix and d vector
        C[3 * i:3 * i + 3, :] = I - A_list[i][0:3, 0:3]  # I - rotation matrix from A
        d[3 * i:3 * i + 3, :] = A_list[i][0:3, 3].reshape(3, 1) - np.dot(R, B_list[i][0:3, 3].reshape(3, 1))
    
    # Solve for translation vector t
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    
    # Form the final transformation matrix X
    X = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
    
    return X




def pose_decomposition_list(hom_matrix_list):
    rmats = []
    tvecs = []
    for item in hom_matrix_list:
        rmat = item[:3, :3]
        tvec = item[:3, 3].reshape(3,1)
        rmats.append(rmat)
        tvecs.append(tvec)

    return rmats, tvecs


def pose_decomposition(hom_matrix):
    rmat = hom_matrix[:3, :3]
    tvec = hom_matrix[:3, 3]

    return rmat, tvec


def create_homogeneous_pose(rmat, tvec):
    #assert rmat.shape == (3, 3) "Rotation matrix should be a 3x3 matrix"
    #assert len(tvec) == 3 "Translation vector should have 3 elements"
    
    #hom_pose = np.eye(4)
    #hom_pose[:3, :3] = rmat
    #hom_pose[:3, 3] = tvec
    
    hom_pose = np.vstack((np.hstack((rmat, tvec)), (0,0,0,1)))
    
    return hom_pose


def compute_homogeneous_poses(rvecs, tvecs):
    if len(rvecs) != len(tvecs):
        raise ValueError("rvecs and tvecs must have the same length.")
    
    homogeneous_poses = []
    
    for rvec, tvec in zip(rvecs, tvecs):
        # Convert rotation vector to rotation matrix
        rmat, _ = cv.Rodrigues(rvec)
        #rmat = rod_to_rmat(rvec)
        
        # Create a 4x4 homogeneous transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = rmat  # Top-left 3x3 is the rotation matrix
        pose[:3, 3] = tvec.ravel()  # Top-right 3x1 is the translation vector
        
        homogeneous_poses.append(pose)
    
    return homogeneous_poses




def create_homogeneous_pose(rmat, tvec):
    #assert rmat.shape == (3, 3) "Rotation matrix should be a 3x3 matrix"
    #assert len(tvec) == 3 "Translation vector should have 3 elements"
    
    #hom_pose = np.eye(4)
    #hom_pose[:3, :3] = rmat
    #hom_pose[:3, 3] = tvec
    
    hom_pose = np.vstack((np.hstack((rmat, tvec)), (0,0,0,1)))
    
    return hom_pose


def pose_inverse(pose):    
    # Extract the rotation matrix (3x3) from the homogeneous pose matrix
    rot_mat = pose[:3, :3]

    # Extract the translation vector (3x1) from the homogeneous pose matrix
    tvec = pose[:3, 3]

    # Compute the inverse of the rotation matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    #rot_mat_inv = rot_mat.T

    # Compute the negative of the rotation matrix multiplied by the translation vector
    tvec_inv = -np.dot(rot_mat_inv, tvec)

    # Construct the inverse homogeneous pose matrix
    inv_pose = np.identity(4)
    inv_pose[:3, :3] = rot_mat_inv
    inv_pose[:3, 3] = tvec_inv
    
    return inv_pose
    


    
    

def plot_translation_vectors(vectors1, vectors2, origin=None):
    """
    Plots the endpoints of two sets of 3D translation vectors in 3D space for comparison.
    
    Parameters:
    vectors1 (list of arrays): The first set of 3D vectors (translation vectors) to plot.
    vectors2 (list of arrays): The second set of 3D vectors (translation vectors) to plot.
    origin (array, optional): The starting point for the vectors, defaults to the origin [0, 0, 0].
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # If no origin is provided, default to the origin (0,0,0)
    if origin is None:
        origin = np.array([0, 0, 0])

    origin = np.array(origin)
    
    # Calculate the endpoints of the first set of vectors
    endpoints1 = np.array([origin + np.array(vector) for vector in vectors1])
    # Calculate the endpoints of the second set of vectors
    endpoints2 = np.array([origin + np.array(vector) for vector in vectors2])

    # Plot the endpoints of the first set as blue points
    ax.scatter(endpoints1[:, 0], endpoints1[:, 1], endpoints1[:, 2], color='b', label='noisified position')
    # Plot the endpoints of the second set as red points
    ax.scatter(endpoints2[:, 0], endpoints2[:, 1], endpoints2[:, 2], color='r', label='true position')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set limits to give a clear plot
    max_val = np.max([np.abs(endpoints1).max(), np.abs(endpoints2).max()]) * 1.2  # Adding some margin for clarity
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Add a legend
    ax.legend()

    plt.show()






# def plot_poses(poses, ax, color):
#     """
#     Plots the 3D positions (translation vectors) and orientations (rotation matrices) from 4x4 transformation matrices in 3D space.
    
#     Parameters:
#     poses (list of arrays): A list of 4x4 transformation matrices (each representing a pose).
#                             - The top-left 3x3 submatrix represents the rotation matrix (orientation).
#                             - The top-right 3x1 submatrix represents the translation vector (position).
#     ax (Axes3D object): The 3D axis to plot on.
#     color (str): The color to use for the pose plot.
#     """
#     positions = []  # To store positions for connecting with lines

#     for pose in poses:
#         # Decompose the 4x4 transformation matrix
#         position = pose[:3, 3]         # Translation vector (x, y, z)
#         orientation = pose[:3, :3]     # Rotation matrix (3x3)

#         # Store the position for line connection
#         positions.append(position)

#         # Plot the position as a point
#         ax.scatter(position[0], position[1], position[2], color=color)

#         # Define unit vectors to represent the orientation axes
#         unit_vectors = np.identity(3)  # x-axis: [1, 0, 0], y-axis: [0, 1, 0], z-axis: [0, 0, 1]
        
#         # Transform the unit vectors by the rotation matrix
#         transformed_vectors = orientation @ unit_vectors
        
#         # Plot the orientation as arrows (quivers)
#         ax.quiver(position[0], position[1], position[2], 
#                   transformed_vectors[0, 0], transformed_vectors[1, 0], transformed_vectors[2, 0], color='r', length=10)
#         ax.quiver(position[0], position[1], position[2], 
#                   transformed_vectors[0, 1], transformed_vectors[1, 1], transformed_vectors[2, 1], color='g', length=10)
#         ax.quiver(position[0], position[1], position[2], 
#                   transformed_vectors[0, 2], transformed_vectors[1, 2], transformed_vectors[2, 2], color='b', length=10)

#     # Connect positions with a line
#     positions = np.array(positions)  # Convert to NumPy array for easier plotting
#     ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, linestyle='-', linewidth=0.5)

def plot_poses(poses, ax, color, axis_scale=0.1):
    """
    Plots 3D positions and orientations from 4x4 transformation matrices.
    SCALED TO FIT THE SCENE
    """
    positions = []

    # Collect all positions first
    for pose in poses:
        positions.append(pose[:3, 3])
    positions = np.array(positions)

    # Estimate a reasonable axis length (e.g., 10% of the trajectory size)
    max_range = np.max(np.ptp(positions, axis=0))  # peak-to-peak range
    axis_length = axis_scale * max_range if max_range > 0 else 1.0

    for pose in poses:
        position = pose[:3, 3]
        orientation = pose[:3, :3]

        # Plot the position
        ax.scatter(*position, color=color)

        # Unit vectors
        unit_vectors = np.identity(3)
        transformed_vectors = orientation @ unit_vectors

        # Plot axes as quivers
        ax.quiver(*position, *transformed_vectors[:, 0], color='r', length=axis_length)
        ax.quiver(*position, *transformed_vectors[:, 1], color='g', length=axis_length)
        ax.quiver(*position, *transformed_vectors[:, 2], color='b', length=axis_length)

    # Connect positions with a line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color=color, linestyle='-', linewidth=0.5)



def plot_two_sets_of_poses(poses1, poses2):
    """
    Plots two sets of 3D poses (positions and orientations) in 3D space for comparison.
    
    Parameters:
    poses1 (list of arrays): The first set of 4x4 transformation matrices (poses).
    poses2 (list of arrays): The second set of 4x4 transformation matrices (poses).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first set of poses in blue
    plot_poses(poses1, ax, 'b')

    # Plot the second set of poses in red
    plot_poses(poses2, ax, 'r')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set limits to give a clear plot
    all_poses = poses1 + poses2  # Combine both sets to determine axis limits
    max_val = np.max(np.abs([pose[:3, 3] for pose in all_poses])) * 1.2  # Adding some margin for clarity
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Add legend for clarity
    ax.legend(['Set 1', 'Set 2'])

    plt.show()

    



    
# Note: This function uses the plot_poses() function
# def plot_one_set_of_poses(poses):
#     """
#     Plots two sets of 3D poses (positions and orientations) in 3D space for comparison.
    
#     Parameters:
#     poses1 (list of arrays): The first set of 4x4 transformation matrices (poses).
#     poses2 (list of arrays): The second set of 4x4 transformation matrices (poses).
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the first set of poses in blue
#     plot_poses(poses, ax, 'b')


#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Set limits to give a clear plot
#     all_poses = poses  # Combine both sets to determine axis limits
#     max_val = np.max(np.abs([pose[:3, 3] for pose in all_poses])) * 1.2  # Adding some margin for clarity
#     ax.set_xlim([-max_val, max_val])
#     ax.set_ylim([-max_val, max_val])
#     ax.set_zlim([-max_val, max_val])

#     # Add legend for clarity
#     ax.legend(['Set 1', 'Set 2'])

#     plt.show()
    
    


def plot_one_set_of_poses(poses, save_path=None):
    """
    Plots a set of 3D poses (positions and orientations) in 3D space.
    
    Parameters:
    poses (list of arrays): The set of 4x4 transformation matrices (poses).
    save_path (str): Optional path to save the figure as an image file.
    """
    # Enable LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the set of poses
    plot_poses(poses, ax, 'black')

    # Set axis labels with LaTeX formatting
    ax.set_xlabel(r'\textbf{X-axis (m)}', fontsize=12)
    ax.set_ylabel(r'\textbf{Y-axis (m)}', fontsize=12)
    ax.set_zlabel(r'\textbf{Z-axis (m)}', fontsize=12)

    # Remove grid for a cleaner look
    ax.grid(False)
    
    # Remove background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove spines
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Set limits to give a clear plot
    all_positions = np.array([pose[:3, 3] for pose in poses])
    max_val = np.max(np.abs(all_positions)) * 1.2  # Adding some margin for clarity
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Set aspect ratio to equal
    ax.set_box_aspect([1, 1, 1])
    
    # Set the perspective view
    ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth as needed

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')

    # Show the plot
    plt.show()





def calculate_L_Ri(R_Ai, R_Bi, RX):
    """
    Calculate the L_RX value for given matrices R_Ai, R_Bi, and RX in AX = XB.

    Parameters:
    - R_Ai: list of numpy arrays
    - R_Bi: list of numpy arrays
    - RX: numpy array

    Returns:
    - L_RX: float
    """
    L_RX_list = []
    for item_A, item_B in zip(R_Ai, R_Bi):
        L_RX_norm = np.linalg.norm((item_A @ RX) - (RX @ item_B), 'fro')
        L_RX_norm_sqr = L_RX_norm ** 2
        L_RX_list.append(L_RX_norm_sqr)
    
    L_RX = sum(L_RX_list) / (2 * len(L_RX_list))
    return L_RX




def calculate_L_ti(R_Ai, t_Ai, t_Bi, tX, RX):
    """
    Calculate the L_tX value for given matrices R_Ai, vectors t_Ai, t_Bi, and matrices tX and RX.

    Parameters:
    - R_Ai: list of numpy arrays (3x3 rotation matrices)
    - t_Ai: list of numpy arrays (3x1 translation vectors)
    - t_Bi: list of numpy arrays (3x1 translation vectors)
    - tX: numpy array (3x1 translation vector)
    - RX: numpy array (3x3 rotation matrix)

    Returns:
    - L_tX: float
    """
    I = np.identity(3, dtype=int)
    L_tX_list = []
    
    for item_RA, item_tA, item_tB in zip(R_Ai, t_Ai, t_Bi):        
        expression = (item_RA - I) @ tX - (RX @ item_tB) + item_tA
        expression_norm = np.linalg.norm(expression, 'fro')
        norm_sq = expression_norm ** 2
        L_tX_list.append(norm_sq)
    
    L_tX = sum(L_tX_list) / (2 * len(L_tX_list))
    return L_tX





def dR_ds_i(R, i):
    """
    Based on: A Compact Formula for the Derivative of a 3-D Rotation in Exponential 
    Coordinates Guillermo Gallego, Anthony Yezzi J Math Imaging Vis 2015
    
    Computes the derivative of the rotation matrix R with respect to
    the i-th component of the scaled rotation vector s.
    
    Parameters:
    - R: 3x3 rotation matrix
    - i: integer index (0, 1, or 2) representing which component of s the derivative is with respect to
    
    Returns:
    - dR_dsi: 3x3 matrix representing the partial derivative of R with respect to s_i
    """
    # Compute the scaled rotation vector s from R
    axis, angle = rot_mat_to_axis_angle(R)
    s = angle * axis
    norm_s = np.linalg.norm(s)
    
    if norm_s == 0:
        raise ValueError("The rotation matrix R should not correspond to zero rotation.")

    # Standard basis vector e_i
    e_i = np.zeros(3)
    e_i[i] = 1

    # Compute components of the derivative formula
    s_i_skew = s[i] * skew(s)
    s_cross_term = skew(s) @ (np.eye(3) - R) @ e_i

    # Calculate the derivative
    dR_dsi = (s_i_skew + skew(s_cross_term)) @ R / (norm_s**2)

    return dR_dsi, s[i]


def dR_ds(R):
    """
    Based on: A Compact Formula for the Derivative of a 3-D Rotation in Exponential 
    Coordinates Guillermo Gallego, Anthony Yezzi J Math Imaging Vis 2015
    
    Computes the partial derivatives of a rotation matrix R with respect to
    the rotation parameters in exponential coordinates.
    
    Parameters:
    - R: 3x3 rotation matrix
    
    Returns:
    - dR1, dR2, dR3: 3x3 matrices representing the partial derivatives of R
    """
    # Get the axis and angle from the rotation matrix
    axis, angle = rot_mat_to_axis_angle(R)
    s = angle * axis
    
    if angle == 0:
        return [np.zeros((3, 3)) for _ in range(3)], s
    
    s_hat = skew(s)
    I = np.eye(3)

    # Compute dR1
    temp0 = s[0] * s_hat + skew(s_hat @ (I - R) @ np.array([1, 0, 0]))
    dR_ds0 = (1 / angle**2) * temp0 @ R
    
    # Compute dR2
    temp1 = s[1] * s_hat + skew(s_hat @ (I - R) @ np.array([0, 1, 0]))
    dR_ds1 = (1 / angle**2) * temp1 @ R
    
    # Compute dR3
    temp2 = s[2] * s_hat + skew(s_hat @ (I - R) @ np.array([0, 0, 1]))
    dR_ds2 = (1 / angle**2) * temp2 @ R
    

    return dR_ds0, dR_ds1, dR_ds2, s[0], s[1], s[2]


def dLR_ds(R_Ai, R_Bi, RX, dRX_dsk): 

    dLR_ds0_list = []
    for R_A, R_B in zip(R_Ai, R_Bi):
        dL_dRX = 2*RX - (R_A.T)@RX@R_B - R_A@RX@(R_B.T)
        expression = np.trace((dL_dRX.T) @ dRX_dsk) 
        dLR_ds0_list.append(expression)          
    dLR_dsk = sum(dLR_ds0_list)/(len(dLR_ds0_list))
              
    return dLR_dsk


def dL_dt(t, R_A, t_A, t_B, R_X):
    t_X = t.reshape(3,1)
    
    # Initialize gradient as a 3-element vector
    gradient = []
    
    for RA, tA, tB in zip(R_A, t_A, t_B):
        # Compute RA - I
        RA_minus_I = RA - np.eye(3)
        
        # Compute residual
        residual = (RA_minus_I @ t_X) - (R_X @ tB) + tA
        
        # Accumulate the gradient
        gradient.append(RA_minus_I.T @ residual)
        
    grad_t = sum(gradient)/len(gradient)
    
    return grad_t[0], grad_t[1], grad_t[2]




def angular_error(R_est, R_true):
    # Compute the geodesic distance between two rotation matrices
    R_diff = np.dot(R_est.T, R_true)
    trace = np.trace(R_diff)
    return np.arccos((trace - 1) / 2)



def trans_error(t_est, t_true):
    return np.linalg.norm(t_est - t_true)


def obj_fun_R(R_X, S1_i, S2_i):
    loss = sum([np.linalg.norm(R_A @ R_X - R_X @ R_B)**2 for R_A, R_B in zip(S1_i, S2_i)])
    return (1 / (2 * len(S1_i))) * loss


def gradient_2_sensors(R_X, S1_i, S2_i):
    
    # Compute partial derivatives of RX with respect to s (dR_ds)
    dR_ds0, dR_ds1, dR_ds2, _, _, _ = dR_ds(R_X)
        
    # Compute gradients for each parameter using dLR_ds
    grad0 = dLR_ds(S1_i, S2_i, R_X, dR_ds0)
    grad1 = dLR_ds(S1_i, S2_i, R_X, dR_ds1)
    grad2 = dLR_ds(S1_i, S2_i, R_X, dR_ds2)
    
    # Combine gradients into a single array
    grad = np.array([grad0, grad1, grad2])
    return grad


def constraint_gradX(RX, RY, RZ, lambdah):  
    dR0 = lambdah*np.trace((RX - RZ@(RY.T)).T @ dR_ds(RX)[0])
    dR1 = lambdah*np.trace((RX - RZ@(RY.T)).T @ dR_ds(RX)[1])
    dR2 = lambdah*np.trace((RX - RZ@(RY.T)).T @ dR_ds(RX)[2])
    
    # Combine gradients into a single array
    gradX_C = np.array([dR0, dR1, dR2])
    return gradX_C


def constraint_gradY(RX, RY, RZ, lambdah):
    dR0 = lambdah*np.trace((RY - (RX.T) @ RZ).T @ dR_ds(RY)[0])    
    dR1 = lambdah*np.trace((RY - (RX.T) @ RZ).T @ dR_ds(RY)[1]) 
    dR2 = lambdah*np.trace((RY - (RX.T) @ RZ).T @ dR_ds(RY)[2]) 
    
    # Combine gradients into a single array
    gradY_C = np.array([dR0, dR1, dR2])
    return gradY_C
    
    

def constraint_gradZ(RX, RY, RZ, lambdah):    
    dR0 = lambdah*np.trace((RZ - RX@RY).T @ dR_ds(RZ)[0])
    dR1 = lambdah*np.trace((RZ - RX@RY).T @ dR_ds(RZ)[1])
    dR2 = lambdah*np.trace((RZ - RX@RY).T @ dR_ds(RZ)[2])
    
    # Combine gradients into a single array
    gradZ_C = np.array([dR0, dR1, dR2])
    return gradZ_C



def constraint_grad_tX(tX, tY, tZ, RX, lambdah_t):
    con_tX = lambdah_t * (tX + (RX@tY)- tZ)
    return con_tX

def constraint_grad_tY(tX, tY, tZ, RX, lambdah_t):
    con_tY = lambdah_t * (tY + RX.T@(tX- tZ))
    return con_tY
    
def constraint_grad_tZ(tX, tY, tZ, RX, lambdah_t):
    con_tZ = lambdah_t * (tZ - tX - (RX@tY))
    return con_tZ



##### ----------------  Semi-implicit New Variant -----------------------------


def dLR_dsX_semi_implicit(R_Ai, R_Bi, R_Ci, RX, RZ, dRX_dsk):
    if not (len(R_Ai) == len(R_Bi) == len(R_Ci)):
        raise ValueError("R_A, R_B, and R_C must have the same length.")
        
    dL_dsXk_list = []
    for RA, RB, RC in zip(R_Ai, R_Bi, R_Ci):
        dL_dRX = (
            4 * RX
            - RA.T @ (RX @ RB)
            - RA @ (RX @ RB.T)
            - RZ @ (RC @ (RZ.T @ (RX @ RB.T)))
            + RZ @ (RC.T @ (RZ.T @ (RX @ RB)))
        )
        expression = np.trace(dL_dRX.T @ dRX_dsk)
        dL_dsXk_list.append(expression)
        
    dL_dsX = sum(dL_dsXk_list) / len(dL_dsXk_list)
    return dL_dsX




def dLR_dsZ_semi_implicit(R_Ai, R_Bi, R_Ci, RX, RZ, dRZ_dsk):
    if not (len(R_Ai) == len(R_Bi) == len(R_Ci)):
        raise ValueError("R_A, R_B, and R_C must have the same length.")
        
    dL_dsZk_list = []
    for RA, RB, RC in zip(R_Ai, R_Bi, R_Ci):
        dL_dRZ = (
            4 * RZ
            - RA.T @ (RZ @ RC)
            - RA @ (RZ @ RC.T)
            - RX @ (RB.T @ (RX @ (RZ @ RC)))
            + RX @ (RB @ (RX.T @ (RZ @ RC.T)))
        )
        expression = np.trace(dL_dRZ.T @ dRZ_dsk)
        dL_dsZk_list.append(expression)
        
    dL_dsZ = sum(dL_dsZk_list) / len(dL_dsZk_list)
    return dL_dsZ





##### ------------------    Implicit Constraint  -----------------------------


def dLR_dsX_implicit(R_Ai, R_Bi, R_Ci, RX, RY, RZ, dRX_dsk):
    dL_dsXk_list = []
    for R_A, R_B, R_C in zip(R_Ai, R_Bi, R_Ci):
        dL_dRX = 4 * RX - (
            RZ @ R_C @ (RZ.T) @ RX @ R_B.T
            + RZ @ (R_C.T) @ (RZ.T) @ RX @ R_B
            + (R_A.T) @ RX @ RY @ R_C @ RY.T
            + R_A @ RX @ RY @ (R_C.T) @ RY.T
        )
        expression = np.trace((dL_dRX.T) @ dRX_dsk)
        dL_dsXk_list.append(expression)

    # Corrected normalization
    dL_dsX = sum(dL_dsXk_list) / len(dL_dsXk_list)
    return dL_dsX




def dLR_dsY_implicit(R_Ai, R_Bi, R_Ci, RX, RY, RZ, dRY_dsk):
    dL_dsYk_list = []
    for R_A, R_B, R_C in zip(R_Ai, R_Bi, R_Ci):
        dL_dRY = 4*RY - (R_B @ RY @ (RZ.T) @ (R_A.T) @ RZ +
                         (R_B.T) @ RY @ (RZ.T) @ R_A @ RZ +
                         (RX.T) @ (R_A.T) @ RX @ RY @ R_C +
                         (RX.T) @ R_A @ RX @ RY @ (R_C.T)) 
        expression = np.trace((dL_dRY.T) @ dRY_dsk)
        dL_dsYk_list.append(expression)
        
    dL_dsY = sum(dL_dsYk_list) / (len(dL_dsYk_list))
    return dL_dsY




def dLR_dsZ_implicit(R_Ai, R_Bi, R_Ci, RX, RY, RZ, dRZ_dsk):
    dL_dsZk_list = []
    for R_A, R_B, R_C in zip(R_Ai, R_Bi, R_Ci):
        dL_dRZ = 4*RZ - ((R_A.T) @ RZ @ (RY.T) @ R_B @ RY +
                         R_A @ RZ @ (RY.T) @ (R_B.T) @ RY +
                         RX @ (R_B.T) @ (RX.T) @ RZ @ R_C +
                         RX @ R_B @ (RX.T) @ RZ @ (R_C.T)) 
        expression = np.trace((dL_dRZ.T) @ dRZ_dsk)
        dL_dsZk_list.append(expression)
        
    dL_dsZ = sum(dL_dsZk_list) / (len(dL_dsZk_list))
    return dL_dsZ

##########################################################################
#######              Error metrics for real dataset                 ######
##########################################################################


def rotation_angle(R):
    """Compute rotation angle (radians) from a rotation matrix R."""
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # numerical stability
    return angle

# Compute relative rotation error
def compute_eR1(RX, RA_list, RB_list):
    """
    Compute e_R1 as defined in the formula.
    
    Parameters:
        RX : (3,3) numpy array
            Fixed rotation matrix.
        RA_list : list of (3,3) numpy arrays
            List of RA_i rotation matrices.
        RB_list : list of (3,3) numpy arrays
            List of RB_i rotation matrices (same length as RA_list).
    
    Returns:
        float : e_R1 error value (in radians).
    """
    assert len(RA_list) == len(RB_list), "RA_list and RB_list must have the same length"
    N = len(RA_list)
    
    total_error = 0.0
    for i in range(N):
        RBi = RB_list[i]
        RAi = RA_list[i]
        
        R = (RX @ RBi).T @ (RAi @ RX)  # equivalent to (RX RBi)^T (RAi RX)
        total_error += rotation_angle(R)
    
    return total_error / N

# Compute relative translation error
def compute_et1(RA_list, tA_list, tB_list, tX, RX):
    """
    Compute e_t1.

    """
    N = len(RA_list)
    total_error = 0.0
    
    for i in range(N):
        term1 = RA_list[i] @ tX + tA_list[i]
        term2 = RX @ tB_list[i] + tX
        diff = term1 - term2
        total_error += np.linalg.norm(diff, 2)
    
    return total_error / N

# Compute relative transformation error
def compute_eT(A_list, B_list, X):
    """
    Compute e_T from the given formula:
        e_T = (1/N) * sum || A_i X - X B_i ||_F

    Parameters
    ----------
    A_list : list of np.ndarray
        List of matrices A_i, each (m, m).
    B_list : list of np.ndarray
        List of matrices B_i, each (n, n).
    X : np.ndarray
        Matrix X, shape (m, n).

    Returns
    -------
    float
        The computed e_T value.
    """
    N = len(A_list)
    total_error = 0.0

    for i in range(N):
        diff = A_list[i] @ X - X @ B_list[i]
        total_error += np.linalg.norm(diff, 'fro')  # Frobenius norm
    
    return total_error / N




##########################################################################
###                          STATISTICS                                ###
##########################################################################



def plot_histogram_with_stats(data, filename='histogram.png'):
    """
    Plots a histogram of the given data with mean and standard deviation lines.
    Saves the figure as a high-resolution PNG file.
    
    Parameters:
    data (array-like): The dataset to visualize.
    filename (str): Name of the output PNG file.
    """
    # Compute mean and standard deviation
    mean_data = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    median = np.median(data)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_vals, bins, patches = ax.hist(data, bins=100, color=(214/255, 110/255, 19/255), alpha=0.75)
    
    # Add vertical lines for mean and standard deviation
    ax.axvline(mean_data, color='red', linewidth=2, label='Mean')
    ax.axvline(mean_data + std_dev, color='green', linestyle='--', linewidth=2, label='+1 std dev')
    ax.axvline(mean_data - std_dev, color='green', linestyle='--', linewidth=2, label='-1 std dev')
    
    # Formatting labels
    ax.set_xlabel(r'$\mathbf{mm}$', fontsize=14)
    ax.set_ylabel(r'$\mathcal{E}_\mathbf{t}$', fontsize=14)
    
    # Remove top and right axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure with high DPI
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Histogram saved as {filename}")
    return mean_data, median, variance


def save_stats_to_csv(data, csv_filename='statistics.csv'):
    """
    Calculates the mean, median, and standard deviation of data and appends it to a CSV file in row format.
    
    Parameters:
    data (array-like): The dataset to analyze.
    csv_filename (str): Name of the output CSV file.
    """
    mean_data = np.mean(data)
    median_data = np.median(data)
    std_dev = np.std(data)
    
    stats_df = pd.DataFrame([[mean_data, median_data, std_dev]],
                            columns=['Mean', 'Median', 'Std_Dev'])
    
    # Append to the CSV file if it exists, otherwise create a new one
    if os.path.exists(csv_filename):
        stats_df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        stats_df.to_csv(csv_filename, index=False)
    
    print(f"Statistics appended to {csv_filename}")
    



def outlier_removal_ModifiedZ_and_stats(data, csv_filename='statistics_modified_z.csv', threshold=3.5):
    """Removes outliers using the Modified Z-Score method and computes mean, median, and standard deviation. Results are saved to a CSV file."""

    # Ensure data is a NumPy array
    data = np.array(data)  

    # Compute the median
    median = np.median(data)

    # Compute the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(data - median))

    # Handle case where MAD is zero (all values are identical)
    if mad == 0:
        print("Warning: MAD is zero, meaning all values are the same. No outliers removed.")
        return data, np.mean(data), np.median(data), np.std(data)

    # Compute Modified Z-Scores
    mod_z_scores = 0.6745 * (data - median) / mad

    # Remove outliers (threshold |Modified Z| > 3.5)
    filtered_data = data[np.abs(mod_z_scores) <= threshold]  

    # Handle case where all data points are outliers
    if len(filtered_data) == 0:
        print("Warning: All data points were removed as outliers. No statistics computed.")
        return None

    # Compute final statistics
    mean_final = np.mean(filtered_data)
    median_final = np.median(filtered_data)
    std_dev_final = np.std(filtered_data)

    # Save results to CSV
    stats_df = pd.DataFrame([[mean_final, median_final, std_dev_final]],
                            columns=['Mean', 'Median', 'Std_Dev'])

    # Append if file exists, otherwise create a new one
    file_exists = os.path.exists(csv_filename)

    stats_df.to_csv(csv_filename, mode='a' if file_exists else 'w', header=not file_exists, index=False)

    print(f"Statistics appended to {csv_filename}")

    return filtered_data, mean_final, median_final, std_dev_final  # Return values for further use


# def outlier_removal_ModifiedZ_and_stats_EXTRA(data, csv_filename='statistics_modified_z.csv', filtered_filename='filtered_data.csv', threshold=3.5):
#     """Removes outliers using the Modified Z-Score method and computes mean, median, and standard deviation. Results are saved to CSV files."""
    
#     # Ensure data is a NumPy array
#     data = np.array(data)  

#     # Compute the median
#     median = np.median(data)

#     # Compute the Median Absolute Deviation (MAD)
#     mad = np.median(np.abs(data - median))

#     # Handle case where MAD is zero (all values are identical)
#     if mad == 0:
#         print("Warning: MAD is zero, meaning all values are the same. No outliers removed.")
#         return data, np.mean(data), np.median(data), np.std(data)

#     # Compute Modified Z-Scores
#     mod_z_scores = 0.6745 * (data - median) / mad

#     # Remove outliers (threshold |Modified Z| > 3.5)
#     filtered_data = data[np.abs(mod_z_scores) <= threshold]  

#     # Handle case where all data points are outliers
#     if len(filtered_data) == 0:
#         print("Warning: All data points were removed as outliers. No statistics computed.")
#         return None

#     # Compute final statistics
#     mean_final = np.mean(filtered_data)
#     median_final = np.median(filtered_data)
#     std_dev_final = np.std(filtered_data)

#     # Save statistics to CSV
#     stats_df = pd.DataFrame([[mean_final, median_final, std_dev_final]],
#                             columns=['Mean', 'Median', 'Std_Dev'])
#     file_exists = os.path.exists(csv_filename)
#     stats_df.to_csv(csv_filename, mode='a' if file_exists else 'w', header=not file_exists, index=False)
#     print(f"Statistics appended to {csv_filename}")

#     # Save filtered data to CSV
#     filtered_df = pd.DataFrame(filtered_data, columns=['Filtered Data'])
#     filtered_df.to_csv(filtered_filename, index=False)
#     print(f"Filtered data saved to {filtered_filename}")

#     return filtered_data, mean_final, median_final, std_dev_final





def outlier_removal_IQR_and_stats(data, csv_filename='5statistics.csv'):
    """Removes outliers using the IQR method and computes mean, median, and standard deviation. Results are saved to a CSV file."""
    
    # Ensure data is a NumPy array
    data = np.array(data)
    
    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Compute IQR
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # Handle case where all data points are outliers
    if len(filtered_data) == 0:
        print("Warning: All data points were removed as outliers. No statistics computed.")
        return None

    # Compute final statistics
    mean_final = np.mean(filtered_data)
    median_final = np.median(filtered_data)
    std_dev_final = np.std(filtered_data)

    # Save results to CSV
    stats_df = pd.DataFrame([[mean_final, median_final, std_dev_final]],
                            columns=['Mean', 'Median', 'Std_Dev'])
    
    # Append if file exists, otherwise create a new one
    file_exists = os.path.exists(csv_filename)
    
    stats_df.to_csv(csv_filename, mode='a' if file_exists else 'w', header=not file_exists, index=False)

    print(f"Statistics appended to {csv_filename}")

    return filtered_data, mean_final, median_final, std_dev_final  # Return values for further use





def get_variable_name(var):
    """Return the variable name(s) from caller's frame."""
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [name for name, val in callers_local_vars if val is var]

def save_list_to_csv(filename, data_list, column_name=None):
    """
    Saves a list to a CSV file. Each call appends the list as a new column.
    
    Args:
        filename (str): Path to CSV file.
        data_list (list): The list of values to save.
        column_name (int, optional): Used to customize column name.
    """
    # Try to get variable name
    var_names = get_variable_name(data_list)
    var_name = var_names[0] if var_names else "List"

    # Build column name
    if column_name is not None:
        column_name = f"{var_name}_{column_name}"
    else:
        column_name = var_name

    # Convert values to strings for CSV
    data_list = [str(x) for x in data_list]

    # Check if file exists
    file_exists = os.path.isfile(filename)

    if not file_exists:
        # Create new CSV
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([column_name])
            for val in data_list:
                writer.writerow([val])
    else:
        # Read existing
        with open(filename, mode="r", newline="") as f:
            reader = list(csv.reader(f))

        max_len = max(len(reader) - 1, len(data_list))
        while len(reader) - 1 < max_len:
            reader.append([])

        reader[0].append(column_name)

        for i in range(max_len):
            val = data_list[i] if i < len(data_list) else ""
            if len(reader[i + 1]) < len(reader[0]) - 1:
                reader[i + 1].extend([""] * (len(reader[0]) - 1 - len(reader[i + 1])))
            reader[i + 1].append(val)

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(reader)



######################################################################
######   Solving Translations with KKT lagrangian Multipliers  #######
######################################################################

def solve_constrained_lsq_LM(RA_list, RB_list, RX, RY, RZ, tAi_list, tBi_list, tCi_list, reg=1e-12):
    """
    Solve the equality-constrained least squares problem:
        min ||r1||^2 + ||r2||^2 + ||r3||^2
        s.t. RX tY + tX = tZ

    Parameters
    ----------
    RA_list : list of (3x3) arrays, rotations RA_i
    RB_list : list of (3x3) arrays, rotations RB_i
    RX, RY, RZ : (3x3) arrays
    tAi_list, tBi_list, tCi_list : lists of (3,) arrays
    reg : float, regularization for numerical stability

    Returns
    -------
    tX, tY, tZ : (3,) arrays
    lambda_vec : (3,) array, Lagrange multiplier
    """

    N = len(RA_list)

    # Initialize accumulators
    Sx = np.zeros((3, 3))
    Sy = np.zeros((3, 3))
    Sz = np.zeros((3, 3))
    bx = np.zeros(3)
    by = np.zeros(3)
    bz = np.zeros(3)

    # Loop over dataset
    for i in range(N):
        RA, RB = RA_list[i], RB_list[i]
        tAi, tBi, tCi = tAi_list[i], tBi_list[i], tCi_list[i]

        # Flatten vectors
        tAi = np.asarray(tAi).reshape(3,)
        tBi = np.asarray(tBi).reshape(3,)
        tCi = np.asarray(tCi).reshape(3,)

        # r1: (RA - I) tX = RX tB - tA
        Mx = RA - np.eye(3)
        Sx += Mx.T @ Mx
        bx += Mx.T @ (RX @ tBi - tAi)

        # r2: (RB - I) tY = RY tC - tB
        My = RB - np.eye(3)
        Sy += My.T @ My
        by += My.T @ (RY @ tCi - tBi)

        # r3: (RA - I) tZ = RZ tC - tA
        Mz = RA - np.eye(3)
        Sz += Mz.T @ Mz
        bz += Mz.T @ (RZ @ tCi - tAi)

    # Add regularization
    Sx += reg * np.eye(3)
    Sy += reg * np.eye(3)
    Sz += reg * np.eye(3)

    # Build block system
    AtA = np.block([
        [Sx, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), Sy, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), Sz]
    ])
    Atb = np.concatenate([bx, by, bz])

    # Constraint: RX tY + tX = tZ  ->  tX + RX tY - tZ = 0
    C = np.hstack([np.eye(3), RX, -np.eye(3)])   # (3x9)

    # Schur complement solve
    AtA_inv = np.linalg.inv(AtA)
    S = C @ AtA_inv @ C.T
    rhs = C @ AtA_inv @ Atb

    # Solve for Lagrange multipliers
    lambda_vec = np.linalg.solve(S, rhs)

    # Recover translations
    t = AtA_inv @ (Atb - C.T @ lambda_vec)
    tX, tY, tZ = t[:3], t[3:6], t[6:9]

    return tX.reshape(3,1), tY.reshape(3,1), tZ.reshape(3,1), lambda_vec.reshape(3,1)







def build_A_and_d_SU(RA_list, RB_list, tA_list, tB_list, tC_list, RX, RY, RZ):
    """
    Standard Unconstrained Method (SU)
    Build the big block matrix A and vector d for the system At = d
    based on the updated equations:

        (RA - I) tX = RX tB - tA
        (RB - I) tY = RY tC - tB
        (RA - I) tZ = RZ tC - tA

    Parameters
    ----------
    RA_list, RB_list : list of (3x3) numpy arrays
        Rotation matrices RA_i and RB_i.
    tA_list, tB_list, tC_list : list of (3,) numpy arrays
        Translation vectors (can also be (3,1); will be flattened).
    RX, RY, RZ : (3x3) numpy arrays
        Rotation matrices.

    Returns
    -------
    A : (9N x 9) numpy array
    d : (9N,) numpy array
    """

    N = len(RA_list)
    A = np.zeros((9 * N, 9))
    d = np.zeros(9 * N)
    I = np.eye(3)

    for i in range(N):
        RA = RA_list[i]
        RB = RB_list[i]
        tA = np.asarray(tA_list[i]).ravel()
        tB = np.asarray(tB_list[i]).ravel()
        tC = np.asarray(tC_list[i]).ravel()

        # Build block A^(i)
        Ai = np.zeros((9, 9))
        Ai[0:3, 0:3] = RA - I       # tX
        Ai[3:6, 3:6] = RB - I       # tY
        Ai[6:9, 6:9] = RA - I       # tZ

        # Build block d^(i)
        di = np.zeros(9)
        di[0:3] = (RX @ tB - tA).ravel()
        di[3:6] = (RY @ tC - tB).ravel()
        di[6:9] = (RZ @ tC - tA).ravel()

        # Place into big A and d
        A[9*i:9*(i+1), :] = Ai
        d[9*i:9*(i+1)] = di

    return A, d



def build_Q_and_d_VE(RA_list, RB_list, tA_list, tB_list, tC_list, RX, RY, RZ):
    """
    Variable Elimination Method (VE) - Unconstrained
    Build the stacked linear system Q t ≈ d for the residuals:

        r_t1 = (RA - I) tX - RX tB + tA
        r_t2 = (RA - I) tZ - RZ tC + tA
        r_t3 = (RB - I) RX.T (tZ - tX) - RX.T RZ tC + tB

    Returns
    -------
    Q : (9N x 9) numpy array
        Block matrix of coefficients.
    d : (9N,) numpy array
        Stacked right-hand side vector.
    """
    N = len(RA_list)
    I = np.eye(3)

    Q = np.zeros((9 * N, 9))
    d = np.zeros(9 * N)

    for i in range(N):
        RA = RA_list[i]
        RB = RB_list[i]
        tA = np.asarray(tA_list[i]).ravel()
        tB = np.asarray(tB_list[i]).ravel()
        tC = np.asarray(tC_list[i]).ravel()

        # --- Build block Q^(i) ---
        Qi = np.zeros((9, 9))
        # (1) (RA - I) tX
        Qi[0:3, 0:3] = RA - I
        # (2) (RA - I) tZ
        Qi[3:6, 6:9] = RA - I
        # (3) (RB - I) RX.T (tZ - tX)
        Qi[6:9, 0:3] = - (RB - I) @ RX.T
        Qi[6:9, 6:9] =   (RB - I) @ RX.T

        # --- Build block d^(i) ---
        di = np.zeros(9)
        di[0:3] = (RX @ tB - tA)
        di[3:6] = (RZ @ tC - tA)
        di[6:9] = (RX.T @ RZ @ tC - tB)

        # Place into big Q and d
        Q[9*i:9*(i+1), :] = Qi
        d[9*i:9*(i+1)]   = di

    return Q, d





def solve_least_squares(A, d):
    """
    Solve the unconstrained least-squares problem At ≈ d.
    """
    t, residuals, rank, s = np.linalg.lstsq(A, d, rcond=None)
    return t, residuals, rank, s




def solve_t_paiwise(RA_list, tA_list, tB_list, RX):
    """
    Solve (RA - I) tX = RX tB - tA for tX using least squares.

    Parameters
    ----------
    RA_list : list of (3x3) numpy arrays
        Rotation matrices RA_i
    tA_list : list of (3,) or (3,1) numpy arrays
        Translation vectors tA_i
    tB_list : list of (3,) or (3,1) numpy arrays
        Translation vectors tB_i
    RX : (3x3) numpy array
        Rotation matrix

    Returns
    -------
    tX : (3,) numpy array
        Solution vector
    """

    N = len(RA_list)

    A = []
    d = []

    I = np.eye(3)

    for i in range(N):
        RA = RA_list[i]
        tA = np.asarray(tA_list[i]).ravel()
        tB = np.asarray(tB_list[i]).ravel()

        Ai = RA - I
        di = (RX @ tB - tA).ravel()

        A.append(Ai)
        d.append(di)

    # Stack into big system
    A = np.vstack(A)    # shape (3N, 3)
    d = np.hstack(d)    # shape (3N,)

    # Solve least squares
    tX, *_ = np.linalg.lstsq(A, d, rcond=None)

    return tX





