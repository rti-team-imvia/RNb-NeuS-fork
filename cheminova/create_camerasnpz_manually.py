import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from understanding_camerasnpz import preprocess_cameras
import matplotlib
matplotlib.use('TkAgg')

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

def draw_camera(ax, R, T, size=0.1, label=None):
    """
    Draw a camera as a cube with an X on the front face to represent the lens.
    R: Rotation matrix (3x3)
    T: Translation vector (3x1)
    size: Size of the camera cube
    label: Label for the camera (optional)
    """
    # Define a cube centered at the origin
    cube = np.array([[-1, -1, -1],
                     [1, -1, -1],
                     [1, 1, -1],
                     [-1, 1, -1],
                     [-1, -1, 1],
                     [1, -1, 1],
                     [1, 1, 1],
                     [-1, 1, 1]]) * size / 2

    # Define faces of the cube
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    
    # Rotate and translate the cube according to R and T
    cube_transformed = cube @ R.T + T.T
    
    # Draw the cube
    ax.add_collection3d(Poly3DCollection([cube_transformed[face] for face in faces], facecolors='cyan', edgecolors='r', linewidths=1, alpha=0.25))

    # Draw the "X" on the front face to represent the lens
    front_face = cube_transformed[:4]  # First four vertices are the front face
    ax.plot([front_face[0, 0], front_face[2, 0]], [front_face[0, 1], front_face[2, 1]], [front_face[0, 2], front_face[2, 2]], 'r-')
    ax.plot([front_face[1, 0], front_face[3, 0]], [front_face[1, 1], front_face[3, 1]], [front_face[1, 2], front_face[3, 2]], 'r-')

    # Optional label for the camera
    if label:
        ax.text(T[0], T[1], T[2], label, color='black')

def visualize_camera_positions(RT_w2c_mats, object_position=[0, 0, 0]):
    """
    Visualize the position and orientation of the cameras.
    RT_w2c_mats: List of extrinsic matrices for each camera
    object_position: The fixed position of the object (default is at the origin)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Plot the object at the center (fixed object)
    ax.scatter(object_position[0], object_position[1], object_position[2], c='r', marker='o', s=100, label="Object")

    # Plot each camera in the list
    for idx, RT in enumerate(RT_w2c_mats):
        R = RT[:3, :3]  # Extract the rotation matrix (3x3)
        T = RT[:3, 3]   # Extract the translation vector (3x1)
        draw_camera(ax, R, T, label=f"Cam {idx}")
    
    # Add labels and show
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show(block=True)

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

    # Visualize the camera positions
    visualize_camera_positions(RT_w2c_mats)

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
