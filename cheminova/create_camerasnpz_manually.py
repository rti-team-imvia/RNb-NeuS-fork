import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from understanding_camerasnpz import preprocess_cameras
import matplotlib
matplotlib.use('TkAgg')

# Define a rotation matrix around the specified axis (X, Y, or Z)
def rotation_matrix(axis, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)

    if axis == 'X':
        return np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ], dtype=float)
    
    elif axis == 'Y':
        return np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=float)
    
    elif axis == 'Z':
        return np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ], dtype=float)



def visualize_camera_positions(RT_w2c_mats, scale_factor=1000, object_position=[0, 1, 0]):
    """
    Visualize the position and orientation of the cameras as blue dots.
    RT_w2c_mats: List of extrinsic matrices for each camera
    scale_factor: Scaling factor to convert units (e.g., mm to meters)
    object_position: The fixed position of the object (default is at the origin)
    """
    # Convert object position to meters (divide by scale_factor)
    object_position = np.array(object_position) / scale_factor

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the range of the camera positions for dynamic axis scaling
    all_positions = np.array([RT[:3, 3] for RT in RT_w2c_mats]) / scale_factor  # Convert camera positions to meters
    max_range = np.ptp(all_positions, axis=0).max() * 1.1  # Get peak-to-peak range

    # Set up plot limits dynamically based on the camera positions
    mid_x = (all_positions[:, 0].min() + all_positions[:, 0].max()) * 0.5
    mid_y = (all_positions[:, 1].min() + all_positions[:, 1].max()) * 0.5
    mid_z = (all_positions[:, 2].min() + all_positions[:, 2].max()) * 0.5

    # ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    # ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    # ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # Plot the object at the center (fixed object)
    ax.scatter(object_position[0], object_position[1], object_position[2], c='r', marker='o', s=100, label="Object")

    # Plot each camera as a blue dot in the list
    for idx, RT in enumerate(RT_w2c_mats):
        T = RT[:3, 3] / scale_factor   # Extract the translation vector (3x1) and convert to meters
        ax.scatter(T[0], T[1], T[2], c='b', marker='o', s=50, label=f"Cam {idx}" if idx == 0 else "")  # Label only the first camera for the legend
    
    # Add labels and show
    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Z axis (m)')

    # Set the viewing angle
    ax.view_init(elev=20, azim=-60)

    ax.legend()
    plt.show(block=True)

def get_cameras_npz_v1(PATH_TO_SAVE_CAMERAS_NPZ, rotation_axis='Y', unit='meters'):

    # Unit scaling factor
    if unit == 'centimeters':
        scale_factor = 100  # 1 meter = 100 cm
    elif unit == 'millimeters':
        scale_factor = 1000  # 1 meter = 1000 mm
    else:
        scale_factor = 1  # Default is meters

    n_cameras = 24

    # Create a homogeneous row [0, 0, 0, 1] to be appended to matrices later to convert to 4x4 format
    bottom = np.array([0, 0, 0, 1], dtype=float).reshape((1, 4))

    # Create the intrinsic matrix 'K' by defining the 3x3 camera matrix
    # Append a column of zeros to the right of the 3x3 'K' to make it a 3x4 matrix
    # Then, append the 'bottom' row to make it a 4x4 matrix

    # Matlab K in pixels
    k = np.array([
        [1006.2,  0,        323.6],
        [0,       998.3,    217.0],
        [0,       0,        1]
    ]) 
        
    # # OpenCV K in pixels
    # k = np.array([
    #     [1007.29775, 0,            321.808408],
    #     [0,          999.578354,   220.436686],
    #     [0,          0,            1]
    # ]) 

    K = np.concatenate([np.concatenate([k, np.zeros((3, 1), dtype=float)], axis=1), bottom], axis=0)

    # Matlab Rotation matrix (3x3) for view 0 (initial position)
    R_w2c_mat_0 = np.array([
        [0.944322942090362, -0.232042096098093, 0.233260898309608],
        [0.25468969870751, 0.964356881113204, -0.0717562765330722],
        [-0.208296275564589, 0.127170246079878, 0.969763058740589]
    ])

    # # OpenCV Rotation matrix (3x3) for view 0 (initial position)
    # R_w2c_mat_0 = np.array([
    #     [0.1020467,   -0.98730305, -0.12173397],
    #     [0.90492353,   0.0413067,   0.42356483],
    #     [-0.41315842, -0.15338332,  0.89764897]
    # ])

    # Translation vector (3x1) for the camera at the starting position, scaled by the unit factor
    T_w2c_mat_0 = np.array([[-0.110498313543743], [-0.0049519362840422], [0.551621156856599]]) * scale_factor # Matlab changed from millimeters to meters
    # T_w2c_mat_0 = np.array([[0.07423388], [-0.03511917], [0.59900515]]) * scale_factor # OpenCV in meters

    # Extract the initial Z-distance from the camera to the object (this is the radius), scaled by the unit factor
    initial_distance = T_w2c_mat_0[2, 0]  # 0.59900515 (already scaled by unit factor)

    # Initialize the list of extrinsic matrices, starting with RT_w2c_mat_0
    RT_w2c_mats = []

    # Generate rotation matrices and translations for views 0 to 23 (15-degree increments around the specified axis)
    for i in range(n_cameras):
        # Rotate the initial rotation matrix (3x3 part of R_w2c_mat_0) by 15 degrees around the specified axis
        R_axis = rotation_matrix(rotation_axis, i * 15)  # Rotation matrix based on chosen axis
        R_w2c_mat_i = R_axis @ R_w2c_mat_0  # Apply the rotation to the initial 3x3 rotation matrix

        # Calculate the new translation vector (camera moving on a circular path along the chosen axis)
        if rotation_axis == 'X':
            y_new = initial_distance * np.sin(np.radians(i * 15))  # Y-coordinate
            z_new = initial_distance * np.cos(np.radians(i * 15))  # Z-coordinate
            T_w2c_mat_i = np.array([[T_w2c_mat_0[0, 0]], [y_new], [z_new]])  # X remains unchanged

        elif rotation_axis == 'Y':
            x_new = initial_distance * np.sin(np.radians(i * 15))  # X-coordinate
            z_new = initial_distance * np.cos(np.radians(i * 15))  # Z-coordinate
            T_w2c_mat_i = np.array([[x_new], [T_w2c_mat_0[1, 0]], [z_new]])  # Y remains unchanged

        elif rotation_axis == 'Z':
            x_new = initial_distance * np.sin(np.radians(i * 15))  # X-coordinate
            y_new = initial_distance * np.cos(np.radians(i * 15))  # Y-coordinate
            T_w2c_mat_i = np.array([[x_new], [y_new], [T_w2c_mat_0[2, 0]]])  # Z remains unchanged

        # Combine the new 3x3 rotation matrix with the new translation vector (3x1)
        RT_w2c_mat_i = np.concatenate([R_w2c_mat_i, T_w2c_mat_i], axis=1)
        RT_w2c_mat_i = np.concatenate([RT_w2c_mat_i, bottom], axis=0)

        # Append the new extrinsic matrix to the list
        RT_w2c_mats.append(RT_w2c_mat_i)

    # Visualize the camera positions, passing the scale_factor for proper unit scaling
    visualize_camera_positions(RT_w2c_mats, scale_factor)

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
    np.savez(os.path.join(PATH_TO_SAVE_CAMERAS_NPZ, "cameras_v1.npz"), **proj_dict)

if __name__ == "__main__":
    PATH_TO_SAVE_CAMERAS_NPZ = r"C:/Users/Deivid/OneDrive - Université de Bourgogne/3D/in/head_cs_3D_in_011"
    
    # Pass 'X', 'Y', or 'Z' to rotate around a different axis
    get_cameras_npz_v1(PATH_TO_SAVE_CAMERAS_NPZ, rotation_axis='Y', unit='millimeters')  # Change 'Y' to 'X' or 'Z' and 'meters' to 'centimeters' or 'millimeters'

    source_dir = PATH_TO_SAVE_CAMERAS_NPZ
    preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ, source_dir, number_of_normalization_points = 100)
