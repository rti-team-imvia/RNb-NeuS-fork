# SDM to IDR Converter (`sdm2idr.py`)

The `sdm2idr.py` script is designed to convert multi-view 3D data from the SDM-UniPS format into the IDR (Implicit Differentiable Renderer) format. The purpose of the script is to take the data from the SDM-UniPS dataset, including camera calibration, masks, normal maps, and albedo maps, and organize it into a structure that can be used by the IDR framework for tasks like 3D reconstruction or neural rendering.

## How it Works:

### Input Data Handling:

- **Input Folders**: The script first collects the input data from two main sources:
  - `input_dir`: Contains `.data` folders, each corresponding to a specific camera view. Each folder contains normal maps and albedo maps.
  - `source_dir`: Contains camera calibration data (`Calib_Results.mat`) and masks for each camera view.

- The script searches the `input_dir` for all `.data` folders and determines the number of views (`n_views`).

### Camera Calibration Loading:

- **Loading Calibration Data**: The camera calibration information is loaded from a `.mat` file (`Calib_Results.mat`) using `scipy.io.loadmat`. This file typically contains:
  - `KK`: The intrinsic camera matrix.
  - `Rc_{i}`: The rotation matrix for camera i, which transforms points from the world coordinate system to the camera's coordinate system.
  - `Tc_{i}`: The translation vector for camera i, representing the camera's position in world coordinates.

- **Intrinsic Matrix (K)**: The intrinsic matrix `KK` is extended into a 4x4 matrix by appending a bottom row `[0 0 0 1]`, making it suitable for homogeneous coordinates.

- **Extrinsic Matrices (R_w2c and T_w2c)**: The world-to-camera rotation and translation matrices (`Rc_` and `Tc_`) are extracted for each camera. These matrices are combined into a 4x4 extrinsic matrix `RT_w2c` for each view.

- **Projection Matrix Calculation**: For each view, the projection matrix is calculated by multiplying the intrinsic matrix (K) with the extrinsic matrix (`RT_w2c`). This projection matrix transforms points from world space to camera image space.

### Setting Up Output Directories:

The script creates three directories in the `output_dir`:

- `mask/`: Stores the binary masks (silhouettes) of the object in each view.
- `normal/`: Stores the normal maps for each view.
- `albedo/`: Stores the albedo (base color) images for each view.

### Copying and Organizing Data:

- **Masks**: The script copies the mask images from the `source_dir` for each view and renames them into a standardized format (`000.png`, `001.png`, etc.).
- **Normal Maps**: Normal maps for each view are copied from the `.data` folders within the `input_dir`.
- **Albedo Maps**: Albedo (base color) images for each view are also copied from the `.data` folders.

### Saving Camera Data:

- The projection matrices for each camera view are stored in a dictionary (`proj_dict`), where each key corresponds to the world matrix (`world_mat_i`) of a specific camera view. This dictionary is saved as a compressed numpy `.npz` file (`cameras.npz`) in the `output_dir`. The saved file will contain all the projection matrices that the IDR framework will use to transform points into the camera views.

## Breakdown of Key Functions:

### `main(args)`:

The main function orchestrates the entire process:
- Loads the camera calibration data and computes the projection matrices for each view.
- Sets up the necessary output directories.
- Copies the masks, normal maps, and albedo maps from their respective locations to the output directories.
- Saves the projection matrices in a `.npz` file for use by the IDR framework.

## Command-Line Interface (CLI):

### Arguments:

- `--input_dir`: Path to the folder containing the `.data` folders with normal and albedo maps.
- `--source_dir`: Path to the folder containing camera calibration data and masks.
- `--output_dir`: Path where the converted data will be saved.

### Usage:

The script is intended to be run from the command line, and it uses the provided arguments to locate the input data, perform the conversion, and save the output in the specified directories.

## Purpose:

### Data Conversion:

This script converts multi-view camera data from the SDM-UniPS format into the IDR format by organizing the data (masks, normal maps, albedo maps) and saving the projection matrices. This enables the data to be used in the IDR framework for 3D reconstruction tasks, which require well-defined camera parameters and image data for accurate rendering.

In summary, the script processes multi-view image data and camera calibration information and prepares it for use in a 3D reconstruction framework like IDR by copying necessary files, calculating projection matrices, and organizing them into the expected format.
