# Experiment Runner (`exp_runner.py`)

The `exp_runner.py` script is a utility for managing and running experiments in a 3D reconstruction framework using neural rendering techniques like **NeRF** (Neural Radiance Fields) and **SDF** (Signed Distance Fields). It facilitates training, validation, and rendering processes by orchestrating data loading, model training, and rendering procedures. Below is a detailed breakdown of how it works:

## Main Components

### Initialization of the Experiment (`__init__` method)

- **Configuration Loading**: The script loads the configuration file (`conf_path`), which contains parameters for the experiment, such as dataset paths, model architecture, training hyperparameters, and experiment-specific settings (e.g., `batch_size`, `learning_rate`).
  
- **Dataset Initialization**: It initializes the `Dataset` class (from `dataset.py`), which handles loading images, masks, camera parameters, and normal/albedo maps.

- **Training Parameters**: It sets up the training parameters from the configuration, such as:
  - `end_iter`: Number of iterations to train.
  - `warm_up_iter`: Number of iterations for warm-up.
  - `learning_rate`, `learning_rate_alpha`: Initial and decay rates for the learning rate.
  - `batch_size`: Number of rays sampled per iteration.

- **Networks**: Initializes the various neural networks used:
  - **NeRF**: Neural Radiance Field network for modeling volumetric scenes.
  - **SDFNetwork**: Signed Distance Field network for modeling the 3D geometry.
  - **SingleVarianceNetwork**: A network used to predict variance for rendering (often related to uncertainty).
  - **RenderingNetwork**: Used for rendering colors based on geometry and lighting.

- **Optimizer**: Sets up the optimizer (Adam) with the parameters from all the networks.

### Training Process (`train_rnb` method)

- **Training Loop**: The main training loop runs for a number of iterations (`end_iter`). It performs:
  - **Ray Sampling**: Samples rays from images using the dataset's `ps_gen_random_rays_at_view_on_all_lights` function, which generates random rays and their associated data (colors, pixels, etc.).
  - **Raymarching and Rendering**: Uses the `NeuSRenderer` (from `renderer.py`) to render the scene along the sampled rays by computing the color, surface normal, and depth along each ray.
  - **Loss Calculation**: Computes losses for:
    - **Color Loss**: L1 loss between the predicted and true RGB values.
    - **Eikonal Loss**: Enforces smoothness in the surface by penalizing deviation in the surface normal gradients.
    - **Mask Loss**: Penalizes the difference between predicted and true masks.
  - **Backpropagation**: The optimizer updates the network weights using the computed gradients.
  - **Logging and Saving**: At intervals, the script:
    - Logs training statistics like losses and learning rates to TensorBoard.
    - Saves checkpoints with the current state of the models and optimizer.

### Checkpoint Management

- **Load Checkpoint (`load_checkpoint` method)**: Loads a saved checkpoint from a previous run to resume training. This restores the model weights and optimizer state.
  
- **Save Checkpoint (`save_checkpoint` method)**: Saves the model parameters and optimizer state to a file after a certain number of iterations.

### Validation

- **Image Validation (`validate_image` method)**: Periodically renders images from novel views and compares them against ground truth images to validate model performance.
  
- **Mesh Validation (`validate_mesh` method)**: Extracts the mesh of the object being rendered using the Signed Distance Field (SDF) and saves it as a `.ply` file. This allows the reconstruction of the 3D shape of the scene.
  
- **Mesh Texture Validation (`validate_mesh_texture` method)**: Renders the object with texture (albedo and shading) and saves the mesh with vertex colors.

- **Novel View Rendering (`render_novel_image` method)**: Renders an interpolated view between two camera positions by generating rays between the two views and rendering the scene.

### Utility Functions

- **Learning Rate Scheduling (`update_learning_rate`)**: Adjusts the learning rate based on the current iteration, applying a cosine decay to the initial learning rate after a warm-up period.
  
- **Checkpoint Backup (`file_backup`)**: Backs up configuration files and code files to the experiment directory for debugging and record-keeping.

### Command-Line Interface (CLI)

The script can be run from the command line with different modes:
- `train_rnb`: Main training mode which trains the model using the RnB (Reflectance and Normal-Based) approach.
- `validate_mesh`: Validates the generated mesh at the current state of the model.
- `validate_mesh_texture`: Validates the mesh with textures (colors).
- `validate_image_ps`: Validates image predictions.

## Summary

- **Experiment Runner**: This script is the core controller for running experiments, handling the full training loop, data sampling, loss computation, and checkpointing.
  
- **Multi-Model Training**: It orchestrates the training of several networks, including NeRF, SDF networks, and rendering networks, to jointly learn the geometry, texture, and lighting of a 3D scene.
  
- **Validation**: The script periodically validates the model by rendering images and meshes and comparing them to the ground truth.
  
- **CLI**: It provides a flexible command-line interface for training, validating, and rendering, making it easy to run experiments and inspect intermediate results.

This script is crucial for running large-scale 3D neural rendering experiments using RnB and NeRF approaches, enabling researchers to train and evaluate models on 3D reconstruction tasks.
