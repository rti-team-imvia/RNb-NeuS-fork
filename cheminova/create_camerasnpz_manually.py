import numpy as np
import os
from understanding_camerasnpz import preprocess_cameras

# Define a rotation matrix around the Y-axis (clockwise in steps of 15 degrees)
def rotation_matrix_y(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ], dtype=float)

def get_cameras_npz_v1(PATH_TO_SAVE_CAMERAS_NPZ):

    n_cameras = 24

    # Create a homogeneous row [0, 0, 0, 1] to be appended to matrices later to convert to 4x4 format
    bottom = np.array([0, 0, 0, 1], dtype=float).reshape((1, 4))

    # Create the intrinsic matrix 'K' by defining the 3x3 camera matrix
    # Append a column of zeros to the right of the 3x3 'K' to make it a 3x4 matrix
    # Then, append the 'bottom' row to make it a 4x4 matrix
    k = np.array([
        [1.00729775e+03, 0,              3.21808408e+02],
        [0,              9.99578354e+02, 2.20436686e+02],
        [0,              0,              1]
    ])

    K = np.concatenate([np.concatenate([k, np.zeros((3, 1), dtype=float)], axis=1), bottom], axis=0)

    # Rotation matrix (3x3) for view 0 (initial position)
    R_w2c_mat_0 = np.array([
        [0.1020467,   -0.98730305, -0.12173397],
        [0.90492353,   0.0413067,   0.42356483],
        [-0.41315842, -0.15338332,  0.89764897]
    ])

    # Translation vector (3x1) for the camera at the starting position
    T_w2c_mat_0 = np.array([[0.07423388], [-0.03511917], [0.59900515]])

    # Extract the initial Z-distance from the camera to the object (this is the radius)
    initial_distance = T_w2c_mat_0[2, 0]  # 0.59900515

    # Initialize the list of extrinsic matrices, starting with RT_w2c_mat_0
    RT_w2c_mats = []

    # Generate rotation matrices and translations for views 0 to 23 (15-degree increments around Y-axis)
    for i in range(n_cameras):
        # Rotate the initial rotation matrix (3x3 part of R_w2c_mat_0) by 15 degrees around the Y-axis
        R_y = rotation_matrix_y(i * 15)  # Y-axis rotation matrix
        R_w2c_mat_i = R_y @ R_w2c_mat_0  # Apply the Y-axis rotation to the initial 3x3 rotation matrix

        # Calculate the new translation vector (camera moving on a circular path)
        x_new = initial_distance * np.sin(np.radians(i * 15))  # X-coordinate
        z_new = initial_distance * np.cos(np.radians(i * 15))  # Z-coordinate
        T_w2c_mat_i = np.array([[x_new], [T_w2c_mat_0[1, 0]], [z_new]])  # Camera's new position (Y remains unchanged)

        # Combine the new 3x3 rotation matrix with the new translation vector (3x1)
        RT_w2c_mat_i = np.concatenate([R_w2c_mat_i, T_w2c_mat_i], axis=1)
        RT_w2c_mat_i = np.concatenate([RT_w2c_mat_i, bottom], axis=0)

        # Append the new extrinsic matrix to the list
        RT_w2c_mats.append(RT_w2c_mat_i)

    # Compute the projection matrices for all views by multiplying the intrinsic matrix 'K'
    # with the extrinsic matrix 'RT_w2c_mats' for each view (i.e., K * [R | T])
    proj_mats = [K @ RT_w2c_mats[idx] for idx in range(n_cameras)]  # Dimensions matrix (24, 4, 4)

    # Create an empty dictionary to store the projection matrices with specific keys
    proj_dict = {}

    # Loop through each view index and store the corresponding projection matrix in the dictionary
    # The key for each projection matrix will be in the format 'world_mat_X' where X is the view index
    for i in range(n_cameras):  # Use 'range(24)' to correctly iterate over all view indices
        proj_dict[f"world_mat_{i}"] = proj_mats[i]

    # Save the projection matrices to a .npz file named 'cameras_v1.npz'
    # The **proj_dict unpacks the dictionary, saving each projection matrix as a separate array in the .npz file
    np.savez(os.path.join(PATH_TO_SAVE_CAMERAS_NPZ, "cameras_v1.npz"), **proj_dict)

if __name__ == "__main__":
    PATH_TO_SAVE_CAMERAS_NPZ = r"C:/Users/Deivid/OneDrive - Université de Bourgogne/3D/in/head_cs_3D_in_011"
    get_cameras_npz_v1(PATH_TO_SAVE_CAMERAS_NPZ)

    source_dir = PATH_TO_SAVE_CAMERAS_NPZ
    preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ, source_dir)
