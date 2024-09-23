# Preprocess Cameras (`preprocess_cameras.py`)

The `preprocess_cameras.py` script is designed to preprocess multi-view camera data by aligning and normalizing the 3D object coordinates using camera projection matrices and object masks. The main purpose of the script is to prepare the data for 3D reconstruction tasks, such as using NeRF or SDF-based methods, by normalizing the scene and ensuring that the object is properly centered and scaled in 3D space.

## Key Components:

### Helper Functions:

- **glob_imgs(path)**:
  - Collects all image files (with common extensions like `.png`, `.jpg`, etc.) from the specified directory.
  - This is used to gather the mask images that represent the silhouettes of the object in each view.

- **get_Ps(cameras, number_of_cameras)**:
  - Extracts the 3x4 projection matrices (P) from the provided camera data for all views. These matrices transform 3D world coordinates into 2D image coordinates.
  - The projection matrices are stored in the format `world_mat_%d` inside the `.npz` camera file.

- **get_fundamental_matrix(P_1, P_2)**:
  - Computes the fundamental matrix (F) between two camera views. The fundamental matrix relates points in one camera view to epipolar lines in another view.
  - This matrix is crucial for multi-view geometry because it helps identify corresponding points between views.

### Depth Calculation:

- **get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j)**:
  - For a given 2D point `(curx, cury)` in the image of camera 0, this function calculates the minimum and maximum possible depth of that point by considering the silhouette from another camera view (P_j).
  - It projects the point onto the epipolar line in the second camera view and checks where it intersects the silhouette using triangulation to compute the depth values.

- **get_fundamental_matrices(P_0, Ps)**:
  - Computes the fundamental matrices that transform points from camera 0 into corresponding epipolar lines in other camera views.
  - These matrices help to track corresponding points across different views.

### Mask and Visual Hull Processing:

- **get_all_mask_points(masks_dir)**:
  - This function loads all the mask images from the directory and extracts the coordinates of the object (foreground) in each image.
  - The masks are binary images representing the silhouette of the object in each view, and they are crucial for refining the visual hull and calculating the object’s bounding box.

- **refine_visual_hull(masks, Ps, scale, center)**:
  - This function refines the visual hull of the object by projecting a 3D grid of points into each camera view and checking how many views the points are visible in.
  - Only points that appear in a minimum number of camera views are retained, and the bounding box (centroid and scale) of the object is updated accordingly. This step improves the accuracy of the normalization process.

### Normalization Function:

- **get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras, masks_all)**:
  - This is the core function for computing the normalization transformation for the object. It calculates the centroid and scale of the object to ensure that the object fits within a unit bounding box and is centered.

#### Steps:

- **Initial Point Sampling**: Selects a subset of 2D points from the first camera view and attempts to find their corresponding depth ranges in other views.
- **Triangulation**: For points visible in multiple camera views, triangulation is used to compute the 3D coordinates.
- **Centroid and Scale Calculation**: The centroid of the object is computed as the mean of the 3D points, and the scale is computed based on the standard deviation of the points.
  - Optionally, the visual hull can be refined using the `refine_visual_hull` function, which improves the bounding box accuracy.
- **Normalization Matrix**: The normalization transformation is saved as a 4x4 matrix. This matrix translates and scales the object in 3D space so that it is centered and normalized.

### Main Processing (`get_normalization` function):

- This function is responsible for processing the entire dataset:
  - It loads camera matrices and object masks.
  - It calls `get_normalization_function` to compute the normalization transformation for the object.
  - The normalization matrix is then applied to the camera projection matrices, and the updated matrices are saved in a `.npz` file.
  - Depending on the `use_linear_init` flag, a different number of normalization points are used, which accounts for potential noise in the camera data.

## Command-Line Interface:

The script can be run with different command-line arguments:

- `--source_dir`: Directory containing the camera data and masks.
- `--dtu`: If set, the script processes all scenes in the DTU dataset.
- `--use_linear_init`: If set, the script uses linear initialization for cameras, which involves using more points to handle noisy camera data.

- When running the script with `--dtu`, it processes all scenes in the DTU dataset by applying normalization to each scene individually.

## Summary of Steps:

1. **Load Data**: The script loads camera projection matrices and mask images from the source directory.
2. **Fundamental Matrix Calculation**: It computes the fundamental matrices that map points between camera views.
3. **Depth and 3D Point Calculation**: For each selected 2D point in the reference camera, it calculates the 3D coordinates by triangulating across multiple camera views.
4. **Normalization**: It computes a normalization matrix that centers and scales the object based on the calculated 3D points. This matrix is saved along with the updated camera matrices.
5. **Optional Visual Hull Refinement**: Refines the visual hull to improve the accuracy of the normalization.

## Purpose:

The script processes camera matrices and object masks to compute a normalization transformation that aligns and scales the 3D object in the scene. This ensures that the object is properly centered and fits within a unit bounding box, which is essential for subsequent 3D reconstruction or rendering tasks. The output is used in neural rendering pipelines like NeRF or SDF-based models for training and validation.
