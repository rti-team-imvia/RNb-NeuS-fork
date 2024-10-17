import numpy as np

def parse_cameras_v2(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Step 1: Get the number of cameras
    num_cameras = int(lines[16].strip())  # Number of cameras on line 16
    print(f"Number of cameras: {num_cameras}")

    # A dictionary to store the world_mat and scale_mat for each camera
    camera_matrices = {}

    # Start processing from line 18, each entry has 13 lines + 1 empty line between them
    line_index = 18
    for i in range(num_cameras):
        # Extract the relevant lines for this entry
        entry_lines = lines[line_index:line_index + 13]

        # Step 2.1: Get the image number from the path on the second line
        image_path = entry_lines[1].strip()
        image_number = image_path.split('\\')[-1].split('.')[0]  # Extract "015" from the path
        index = int(image_number) // 15  # Determine the corresponding index (000=0, 015=1,...)

        # Step 2.6: Get the 3-vec Translation T (last column of world_mat)
        translation_T = list(map(float, entry_lines[4].split()))

        # Step 2.10: Get the 3x3 Matrix format of R (lines 10-12)
        matrix_R = [
            list(map(float, entry_lines[8].split())),
            list(map(float, entry_lines[9].split())),
            list(map(float, entry_lines[10].split()))
        ]

        # Step 2.12: Construct world_mat by concatenating R and T, and adding the last row [0, 0, 0, 1]
        world_mat = np.array([
            matrix_R[0] + [translation_T[0]],
            matrix_R[1] + [translation_T[1]],
            matrix_R[2] + [translation_T[2]],
            [0, 0, 0, 1]
        ])

        # Step 2.12: Create the scale_mat as the 4x4 identity matrix
        scale_mat = np.eye(4)

        # Store the world_mat and scale_mat in the dictionary with the index as the key
        camera_matrices[index] = {'world_mat': world_mat, 'scale_mat': scale_mat}

        # Move to the next entry (skip 1 line between entries)
        line_index += 14

    # Step: Organize camera_matrices in ascending order of indices
    organized_camera_matrices = {}
    for i in range(num_cameras):
        organized_camera_matrices[f'world_mat_{i}'] = camera_matrices[i]['world_mat']
        organized_camera_matrices[f'scale_mat_{i}'] = camera_matrices[i]['scale_mat']

    return organized_camera_matrices

# Usage
file_path = r"C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/in/head_cs_3D_in_009/cameras_v2.txt"
output_file = r"C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/in/head_cs_3D_in_009/cameras_vsfm_head_cs_v2.npz"
camera_data = parse_cameras_v2(file_path)

# Save the world_mat and scale_mat to .npy files (optional)
np.savez(output_file, **camera_data)
