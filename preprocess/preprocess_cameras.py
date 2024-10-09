import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import os

# Helper function to retrieve all image file paths (mask images) from a given directory.
def glob_imgs(path):
    imgs = []
    # Searching for files with common image extensions.
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))  # Add files found with the given extension.
    return imgs

# Function to extract camera projection matrices (P matrices) from a dictionary of camera data.
def get_Ps(cameras, number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        # Extract the 3x4 projection matrix for each camera and store it.
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64)
        Ps.append(P)
    return np.array(Ps)

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
        img = mpimg.imread(path)
        # Add an extra axis to the image array, ensuring it's in the correct format for further operations.
        img = img[:, :, np.newaxis]
        # Create a binary mask where pixels with values greater than 0.5 are considered "foreground" (True), others are "background" (False).
        cur_mask = img.max(axis=2) > 0.5
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

def refine_visual_hull(masks, Ps, scale, center):
    num_cam=masks.shape[0]
    GRID_SIZE=100
    MINIMAL_VIEWS=20 # Fitted for DTU, might need to change for different data.
    im_height=masks.shape[1]
    im_width = masks.shape[2]
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE*GRID_SIZE*GRID_SIZE, 1))
    for i in range(num_cam):
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE*GRID_SIZE*GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.longlong)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]
        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid=final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid,np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3,final_points.T

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

# Main function that handles the normalization process.
def get_normalization(source_dir, use_linear_init=False):
    print('Preprocessing', source_dir)
    # Depending on the initialization option, set the number of normalization points and the camera filename.
    if use_linear_init:
        number_of_normalization_points = 1000  # More points are used in the linear initialization case.
        cameras_filename = "cameras_linear_init"
    else:
        number_of_normalization_points = 100  # Fewer points are used for the default case.
        cameras_filename = "cameras"
    
    # Define the directory containing the mask images.
    masks_dir = '{0}/mask'.format(source_dir) # C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/mask
    # Load the camera data from the cameras file.
    cameras = np.load('{0}/{1}.npz'.format(source_dir, cameras_filename)) # 'C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/cameras.npz'
    
    mask_points_all, masks_all = get_all_mask_points(masks_dir) # Get the mask points and binary mask images from the masks directory.
    number_of_cameras = len(masks_all) # Get the number of cameras.
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
    np.savez('{0}/{1}.npz'.format(source_dir, cameras_filename), **cameras_new)
    
    print(normalization)
    print('--------------------------------------------------------')
    
    # Optional: for debugging purposes, visualize the 3D points in the mask images.
    if False:  # This block will not execute unless changed to True.
        
        for i in range(number_of_cameras):
            plt.figure()
            plt.imshow(mpimg.imread('%s/%03d.png' % (masks_dir, i)))
            
            # Project the 3D points into the 2D image plane of camera i and plot them on the image.
            xy = (Ps[i, :2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                        Ps[i, 2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))
            plt.plot(xy[0, :], xy[1, :], '*')
            plt.show()
            plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='', help='data source folder for preprocess')
    parser.add_argument('--dtu', default=False, action="store_true", help='If set, apply preprocess to all DTU scenes.')
    parser.add_argument('--use_linear_init', default=False, action="store_true", help='If set, preprocess for linear init cameras.')

    opt = parser.parse_args()

    if opt.dtu:
        source_dir = '../data/DTU'
        scene_dirs = sorted(glob(os.path.join(source_dir, "scan*")))
        for scene_dir in scene_dirs:
            get_normalization(scene_dir,opt.use_linear_init)
    else:
        get_normalization(opt.source_dir, opt.use_linear_init)

    print('Done!')
