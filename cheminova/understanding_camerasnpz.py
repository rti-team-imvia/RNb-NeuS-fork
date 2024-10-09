import scipy
import numpy as np
import os

def get_cameras_npz(PATH_TO_CALIB_RESULTS_MAT, PATH_TO_SAVE_CAMERAS_NPZ):

    # Define the number of camera views (20 different views for the bear object)
    n_views = 20

    # Load the calibration results from a .mat file containing intrinsic and extrinsic parameters
    camera_dict = scipy.io.loadmat(PATH_TO_CALIB_RESULTS_MAT)

    # Create a homogeneous row [0, 0, 0, 1] to be appended to matrices later to convert to 4x4 format
    bottom = np.array([0, 0, 0, 1], dtype=float).reshape((1, 4))

    # Create the intrinsic matrix 'K' by extracting 'KK' (3x3 camera matrix) from the loaded data
    # Append a column of zeros to the right of the 3x3 'KK' to make it a 3x4 matrix
    # Then, append the 'bottom' row to make it a 4x4 matrix
    K = np.concatenate([np.concatenate([camera_dict['KK'], np.zeros((3, 1), dtype=float)], axis=1), bottom], axis=0)

    # Extract rotation matrices ('Rc_X' fields) for all views, converting them to float32 for computation
    R_w2c_mats = [camera_dict[f"Rc_{idx+1}"].astype(np.float32) for idx in range(n_views)]
    
    # Extract translation vectors ('Tc_X' fields) for all views, also converting to float32
    T_w2c_mats = [camera_dict[f"Tc_{idx+1}"].astype(np.float32) for idx in range(n_views)]

    # Combine the rotation and translation matrices into full extrinsic matrices (4x4)
    # Concatenate each rotation matrix (3x3) with its corresponding translation vector (3x1) horizontally
    # Then, append the homogeneous bottom row to create a 4x4 extrinsic matrix for each view
    RT_w2c_mats = [np.concatenate([np.concatenate([R_w2c_mats[idx], T_w2c_mats[idx]], axis=1), bottom], axis=0) for idx in range(n_views)]

    # Compute the projection matrices for all views by multiplying the intrinsic matrix 'K'
    # with the extrinsic matrix 'RT_w2c_mats' for each view (i.e., K * [R | T])
    proj_mats = [K @ RT_w2c_mats[idx] for idx in range(n_views)] # Dimensions matrix (20, 4, 4)

    # Create an empty dictionary to store the projection matrices with specific keys
    proj_dict = {}

    # Loop through each view index and store the corresponding projection matrix in the dictionary
    # The key for each projection matrix will be in the format 'world_mat_X' where X is the view index
    for i in range(n_views):  # Use 'range(n_views)' to correctly iterate over all view indices
        proj_dict[f"world_mat_{i}"] = proj_mats[i]

    # Save the projection matrices to a .npz file named 'cameras.npz'
    # The **proj_dict unpacks the dictionary, saving each projection matrix as a separate array in the .npz file
    np.savez(os.path.join(PATH_TO_SAVE_CAMERAS_NPZ, "cameras.npz"), **proj_dict)

def load_cameras_npz(PATH_TO_SAVE_CAMERAS_NPZ):
    # Load the .npz file using numpy
    data = np.load(os.path.join(PATH_TO_SAVE_CAMERAS_NPZ, "cameras.npz"))
    # data_orig = np.load(r"C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/cameras_orig.npz")

    # Loop through each key-value pair in the .npz file
    for key in data:
        matrix = data[key]
        
        # Print the key (like 'world_mat_X') and its corresponding matrix
        print(f"Key: {key}")
        
        # Explanation of the matrix
        if "world_mat" in key:
            view_number = key.split('_')[-1]  # Extract the view number from the key
            print(f"This is the projection matrix for view {view_number}.")
            print("It combines both intrinsic and extrinsic camera parameters.")
        
        # Show the actual matrix values
        print("Matrix values:")
        print(matrix)
        print("\n---\n")

def compare_cameras_npz(PATH_TO_SAVE_CAMERAS_NPZ, PATH_TO_ORIG_CAMERAS_NPZ):
    # Load the original cameras_orig.npz file
    data_orig = np.load(os.path.join(PATH_TO_ORIG_CAMERAS_NPZ, "cameras_orig.npz"))
    
    # Load the newly created cameras.npz file
    data = np.load(os.path.join(PATH_TO_SAVE_CAMERAS_NPZ, "cameras.npz"))

    # Iterate through the 20 matrices (your file only has 20 views)
    for i in range(20):
        key_save = f"world_mat_{i}"      # Key for the saved file's matrix
        key_orig_world = f"world_mat_{i}" # Key for the original file's world matrix
        key_orig_scale = f"scale_mat_{i}" # Key for the original file's scale matrix

        print(f"### Comparison for view {i} ###\n")
        
        # Ensure all required keys exist in both datasets
        if key_save in data and key_orig_world in data_orig and key_orig_scale in data_orig:
            # Extract the matrices
            matrix_save = data[key_save]
            matrix_orig_world = data_orig[key_orig_world]
            matrix_orig_scale = data_orig[key_orig_scale]

            # Display the saved matrix
            print(f"Matrix from saved file (key: {key_save}):")
            print(matrix_save)
            print("\n---\n")

            # Display the world matrix from the original file
            print(f"Matrix from original file (key: {key_orig_world} - world_mat):")
            print(matrix_orig_world)
            print("\n---\n")

            # Display the scale matrix from the original file
            print(f"Matrix from original file (key: {key_orig_scale} - scale_mat):")
            print(matrix_orig_scale)
            print("\n---\n")

        else:
            print(f"Missing data for view {i}.")
            print("\n---\n")


def preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ):
    pass

if __name__ == "__main__":

    # From the original Calib_Results.mat file create cameras.npz file
    PATH_TO_CALIB_RESULTS_MAT = r'C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/Calib_Results.mat'
    PATH_TO_SAVE_CAMERAS_NPZ = r'C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/'
    get_cameras_npz(PATH_TO_CALIB_RESULTS_MAT, PATH_TO_SAVE_CAMERAS_NPZ)

    # # See the content of the cameras.npz file
    # load_cameras_npz(PATH_TO_SAVE_CAMERAS_NPZ)

    # # Compare the cameras.npz file with the original cameras.npz file
    # PAT_TO_ORIG_CAMERAS_NPZ = r'C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG'
    # compare_cameras_npz(PATH_TO_SAVE_CAMERAS_NPZ, PAT_TO_ORIG_CAMERAS_NPZ)

    # Replicate preprocess_cameras.py 

    preprocess_cameras(PATH_TO_SAVE_CAMERAS_NPZ)