import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('TkAgg')

# Helper function to retrieve all image file paths (mask images) from a given directory.
def glob_imgs(path):
    imgs = []
    # Searching for files with common image extensions.
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))  # Add files found with the given extension.
    return imgs

# GPUT: Computes the fundamental matrix F that transforms points from image of camera 2 to lines in the image of camera 1.
# Authors: #Gets the fundamental matrix that transforms points from the image of camera 2, to a line in the image of camera 1
def get_fundamental_matrix(P_1, P_2):
    # Calculate the camera center of P_2 by taking the last row of the SVD (singular value decomposition) of P_2.
    P_2_center = np.linalg.svd(P_2)[-1][-1, :]
    # Compute the epipole in camera 1 by projecting the camera center of P_2 onto P_1.
    epipole = P_1 @ P_2_center
    # Create a skew-symmetric matrix for the epipole, used to compute the cross-product.
    epipole_cross = np.zeros((3, 3))
    epipole_cross[0, 1] = -epipole[2]
    epipole_cross[1, 0] = epipole[2]

    epipole_cross[0, 2] = epipole[1]
    epipole_cross[2, 0] = -epipole[1]
    
    epipole_cross[1, 2] = -epipole[0]
    epipole_cross[2, 1] = epipole[0]
    
    # Compute the fundamental matrix F = [e]_x P1 pinv(P2)
    F = epipole_cross @ P_1 @ np.linalg.pinv(P_2)
    return F

# GPT: Computes the fundamental matrices for all pairs of cameras, mapping points from camera 0 to epipolar lines in other cameras.
# get all fundamental matrices that trasform points from camera 0 to lines in Ps
def get_fundamental_matrices(P_0, Ps):
    Fs = []
    for i in range(0, Ps.shape[0]):
        # Compute the fundamental matrix between camera 0 and camera i.
        F_i0 = get_fundamental_matrix(Ps[i], P_0)
        Fs.append(F_i0)
    return np.array(Fs)

def get_all_mask_points(masks_dir):
    # Use the helper function 'glob_imgs' to retrieve all image files (masks) from the specified directory.
    mask_paths = sorted(glob_imgs(masks_dir))  
    # Initialize an empty list to store the coordinates of points for each mask.
    mask_points_all = []
    # Initialize an empty list to store the binary mask images (True/False for mask regions).
    mask_ims = []
    
    # Iterate over each mask path (image file).
    for path in mask_paths:
        # Read the image from the current path. 'mpimg.imread' reads the image as an array.
        img = mpimg.imread(path); # plt.figure();plt.imshow(img);plt.show(block=True)
        # Add an extra axis to the image array, ensuring it's in the correct format for further operations.
        img = img[:, :, np.newaxis] # plt.figure();plt.imshow(img[:,:,0]);plt.show(block=True)
        # Create a binary mask where pixels with values greater than 0.5 are considered "foreground" (True), others are "background" (False).
        cur_mask = img.max(axis=2) > 0.5 # plt.figure();plt.imshow(cur_mask);plt.show(block=True)
        # Identify the (x, y) coordinates of all the points that belong to the foreground (True) in the mask.
        mask_points = np.where(img.max(axis=2) > 0.5)
        # Store the x (column) and y (row) coordinates of the mask points.
        xs = mask_points[1]
        ys = mask_points[0]
        # Stack the x and y coordinates, adding a third row of ones (homogeneous coordinates).
        mask_points_all.append(np.stack((xs, ys, np.ones_like(xs))).astype(np.float32))
        # Append the current binary mask (True/False values) to the 'mask_ims' list.
        mask_ims.append(cur_mask)
    
    # Return the list of mask points for all masks, and an array of the binary mask images.
    return mask_points_all, np.array(mask_ims)

# Function to extract camera projection matrices (P matrices) from a dictionary of camera data.
def get_Ps(cameras, number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        # Extract the 3x4 projection matrix for each camera and store it.
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64)
        Ps.append(P)
    return np.array(Ps)

# GPT:Computes the minimum and maximum possible depth of a point (curx, cury) in the reference image,
# by considering its projection in a second image (camera j) and the mask (silhouette) for camera j.
# Authors: Given a point (curx,cury) in image 0, get the  maximum and minimum
# possible depth of the point, considering the second image silhouette (index j)
def get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j):
    # Use the fundamental matrix to map the point (curx, cury) to a line in the second image (epipolar line).
    cur_l_1 = Fj0 @ np.array([curx, cury, 1.0]).astype(np.float32)
    cur_l_1 = cur_l_1 / np.linalg.norm(cur_l_1[:2])  # Normalize the line equation.

    # Calculate distances from all points in the silhouette of camera j to the epipolar line.
    dists = np.abs(silhouette_j.T @ cur_l_1)
    # Select the points in the silhouette that are close to the epipolar line (within a threshold of 0.7).
    relevant_matching_points_1 = silhouette_j[:, dists < 0.7]
    
    # If no relevant points found, return (0.0, 0.0) as min/max depth.
    if relevant_matching_points_1.shape[1] == 0:
        return (0.0, 0.0)
    # Perform triangulation to compute the 3D coordinates of the relevant matching points.
    X = cv2.triangulatePoints(P_0, P_j, np.tile(np.array([curx, cury]).astype(np.float32),
                                                (relevant_matching_points_1.shape[1], 1)).T,
                              relevant_matching_points_1[:2, :])
    
    depths = P_0[2] @ (X / X[3]) # Project the triangulated points back into camera 0 to compute their depth.
    reldepth = depths >= 0 # Filter out points with negative depth (invalid points).
    depths = depths[reldepth]
    
    # If no valid depth values are found, return (0.0, 0.0).
    if depths.shape[0] == 0:
        return (0.0, 0.0)

    # Compute the minimum and maximum depth among the valid points.
    min_depth = depths.min()
    max_depth = depths.max()

    return min_depth, max_depth

# the normaliztion script needs a set of 2D object masks and camera projection matrices (P_i=K_i[R_i |t_i] where [R_i |t_i] is world to camera transformation)
def get_normalization_function(Ps,mask_points_all,number_of_normalization_points,number_of_cameras,masks_all):
    P_0 = Ps[0] # Use the first camera as the reference camera.
    Fs = get_fundamental_matrices(P_0, Ps) # [20,3,3] # Compute the fundamental matrices between camera 0 and all other cameras.
    P_0_center = np.linalg.svd(P_0)[-1][-1, :] # Compute the camera center for camera 0.
    P_0_center = P_0_center / P_0_center[3] # Normalize the camera center.

    # Get the x and y coordinates of mask points in camera 0. Use image 0 as a references
    xs = mask_points_all[0][0, :]
    ys = mask_points_all[0][1, :]

    counter = 0 # Initialize a counter for the number of valid 3D points.
    all_Xs = [] # List to store the computed 3D points.

    # Randomly select a subset of points from camera 0 to use for normalization. Sample a subset of 2D points from camera 0
    indss = np.random.permutation(xs.shape[0])[:number_of_normalization_points]

    # Iterate over the selected subset of points. 
    for i in indss:
        curx = xs[i] # Current x coordinate.
        cury = ys[i] # Current y coordinate. 

        # Initialize variables to track the min and max depth across all views.
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10

        # For each camera (except the reference camera 0), compute the min and max depth of the point.
        for j in range(1, number_of_cameras, 1):
            min_d, max_d = get_min_max_d(curx, cury, Ps[j], mask_points_all[j], P_0, Fs[j], j)

            # If no valid depth range is found, skip the point.
            if abs(min_d) < 0.00001:
                observerved_in_all = False
                break
            max_d_all = np.min(np.array([max_d_all, max_d]))
            min_d_all = np.max(np.array([min_d_all, min_d]))

            # If the maximum depth is smaller than the minimum depth, skip the point.
            if max_d_all < min_d_all + 1e-2:
                observerved_in_all = False
                break

        # If the point is observed in all cameras, compute its 3D position using triangulation.
        if observerved_in_all:
            # Compute the direction of the point in camera 0's coordinate system.
            direction = np.linalg.inv(P_0[:3, :3]) @ np.array([curx, cury, 1.0])

            # Compute the 3D position at both the minimum and maximum depth.
            all_Xs.append(P_0_center[:3] + direction * min_d_all)
            all_Xs.append(P_0_center[:3] + direction * max_d_all)
            counter = counter + 1

    print("Number of points:%d" % counter)
    centroid = np.array(all_Xs).mean(axis=0)
    # mean_norm=np.linalg.norm(np.array(allXs)-centroid,axis=1).mean()
    scale = np.array(all_Xs).std()

    # OPTIONAL: refine the visual hull
    # centroid,scale,all_Xs = refine_visual_hull(masks_all, Ps, scale, centroid)

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    normalization[0, 0] = scale
    normalization[1, 1] = scale
    normalization[2, 2] = scale
    return normalization,all_Xs

def preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ, source_dir, number_of_normalization_points):
    cameras_filename = "cameras_v1"
    cameras_out_filename = "cameras_v2"
    # Define the directory containing the mask images.
    masks_dir = '{0}/mask'.format(source_dir) # C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/mask
    # Load the camera data from the cameras file.
    cameras = np.load('{0}/{1}.npz'.format(source_dir, cameras_filename)) # 'C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/cameras.npz'

    # Get the mask points and binary mask images from the masks directory.
    mask_points_all, masks_all = get_all_mask_points(masks_dir) #plt.figure();plt.imshow(masks_all[0]);plt.show(block=True)
    number_of_cameras = len(masks_all) # Get the number of cameras. plt.figure();plt.show(masks_all[0]);plt.show(block=True)
    Ps = get_Ps(cameras, number_of_cameras) # [20,3,4] # Extract the projection matrices (P matrices) for each camera. 

    # Compute the normalization matrix and 3D points for the object.
    normalization, all_Xs = get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras, masks_all)
    
    # Create a new dictionary to store the normalized camera data.
    cameras_new = {}
    for i in range(number_of_cameras):
        # Save the normalization matrix (scale and translation) for each camera.
        cameras_new['scale_mat_%d' % i] = normalization
        # Save the world-to-camera transformation matrix for each camera, adding an extra row for homogeneity.
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i], np.array([[0, 0, 0, 1.0]])), axis=0).astype(np.float32)
    
    # Save the updated camera data to a new file.
    np.savez('{0}/{1}.npz'.format(source_dir, cameras_out_filename), **cameras_new)
    
    print(normalization)
    print('--------------------------------------------------------')

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

def update_camerasnpz_v1_with_bear_scale_matrix(cameras_v1_path):

    cameras_filename = "cameras_v1"
    cameras_out_filename = "cameras_v2"
    # Load the camera data from the cameras file.
    cameras_file_path = os.path.join(cameras_v1_path, f"{cameras_filename}.npz")
    cameras = np.load(cameras_file_path)  # Load the npz file
    
    # Iterate over the keys to get the number of world matrices
    num_cameras = len([key for key in cameras.keys() if key.startswith('world_mat_')])
    
    # Define the scale matrix to be applied to each camera
    scale_mat = np.array([
        [84.4945297241211, 0, 0, 87.13922119140625],
        [0, 84.4945297241211, 0, 83.05925750732422],
        [0, 0, 84.4945297241211, 36.135494232177734],
        [0, 0, 0, 1]
    ])
    
    # Create a dictionary to store the new data (both world and scale matrices)
    updated_mats = {}
    
    # Copy the existing world matrices into the new dictionary
    for i in range(num_cameras):
        # Add the corresponding scale matrix
        scale_mat_key = f'scale_mat_{i}'
        updated_mats[scale_mat_key] = scale_mat
        # Add the corresponding world matrix
        world_mat_key = f'world_mat_{i}'
        updated_mats[world_mat_key] = cameras[world_mat_key]

    # Save the updated matrices (both world and scale) to a new .npz file
    output_file = os.path.join(os.path.abspath(cameras_v1_path), f"{cameras_out_filename}.npz")
    np.savez(output_file, **updated_mats)

    print(f"Updated .npz file saved to {output_file}")    

if __name__ == "__main__":
    PATH_TO_SAVE_CAMERAS_NPZ = r"C:/Users/Deivid/OneDrive - Université de Bourgogne/3D/in/head_cs_3D_in_011"
    
    # Pass 'X', 'Y', or 'Z' to rotate around a different axis
    get_cameras_npz_v1(PATH_TO_SAVE_CAMERAS_NPZ, rotation_axis='Y', unit='millimeters')  # Change 'Y' to 'X' or 'Z' and 'meters' to 'centimeters' or 'millimeters'

    source_dir = PATH_TO_SAVE_CAMERAS_NPZ
    update_camerasnpz_v1_with_bear_scale_matrix(PATH_TO_SAVE_CAMERAS_NPZ)
    # preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ, source_dir, number_of_normalization_points = 100)
