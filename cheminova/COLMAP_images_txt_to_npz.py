import numpy as np

WORLD_MAP_DIVIDED_BY_10 = True
SCALE_MAP_SAME_AS_BEAR = False

def quaternion_to_matrix(qw, qx, qy, qz, tx, ty, tz):
    # Compute the rotation matrix from quaternion
    r11 = 1 - 2*qy**2 - 2*qz**2
    r12 = 2*qx*qy - 2*qz*qw
    r13 = 2*qx*qz + 2*qy*qw
    r21 = 2*qx*qy + 2*qz*qw
    r22 = 1 - 2*qx**2 - 2*qz**2
    r23 = 2*qy*qz - 2*qx*qw
    r31 = 2*qx*qz - 2*qy*qw
    r32 = 2*qy*qz + 2*qx*qw
    r33 = 1 - 2*qx**2 - 2*qy**2

    if WORLD_MAP_DIVIDED_BY_10:
        # World Matrix with last column divided by 10
        world_mat = np.array([
            [r11, r12, r13, tx/10],
            [r21, r22, r23, ty/10],
            [r31, r32, r33, tz/10],
            [0, 0, 0, 1]
        ])    
    else:
        # World matrix
        world_mat = np.array([
            [r11, r12, r13, tx],
            [r21, r22, r23, ty],
            [r31, r32, r33, tz],
            [0, 0, 0, 1]
        ])

    return world_mat

def process_images_txt(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Step 1: Get the number of images from line 4
    num_images = int(lines[3].split()[4].replace(",", ""))  # Should get 24
    
    # Initialize empty lists to store the world and scale matrices in order
    world_mats = [None] * num_images
    scale_mats = [None] * num_images

    # Step 2: Iterate from line 5 and skip every second line
    for i in range(4, len(lines), 2):
        line = lines[i].strip()
        parts = line.split()
        
        # Extract image info
        image_id = int(parts[0]) - 1  # We will zero-index the image_id
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[-1]

        # Get image index from the file name, for example, "004.JPG" -> 4
        image_index = int(name.split('.')[0])

        # Step 4: Create the world matrix for this image
        world_mat = quaternion_to_matrix(qw, qx, qy, qz, tx, ty, tz)

        if SCALE_MAP_SAME_AS_BEAR:
            # Use the same scale matrix as in bearPNG
            scale_mat = np.array([
                [84.4945297241211, 0, 0, 87.13922119140625],
                [0, 84.4945297241211, 0, 83.05925750732422],
                [0, 0, 84.4945297241211, 36.135494232177734],
                [0, 0, 0, 1]
            ])            

        else:
            # Create the scale matrix (identity matrix with 1s and a translation factor of 1 in the last column)
            scale_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        # Store matrices at the correct index
        world_mats[image_index] = world_mat
        scale_mats[image_index] = scale_mat

    # Prepare a dictionary with the matrices in the correct order
    matrices_in_order = {}
    for idx in range(num_images):
        matrices_in_order[f'world_mat_{idx}'] = world_mats[idx]
        matrices_in_order[f'scale_mat_{idx}'] = scale_mats[idx]

    # Step 6: Save to an npz file
    np.savez(output_file, **matrices_in_order)

# File paths
input_file = r'C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/COLMAP_in/head_cs/shared_intrinsics/sparse/0/images.txt'
output_file = r'C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/in/head_cs_3D_in_008/cameras_COLMAP_head_cs_v2.npz'

# Process the file and create cameras.npz
process_images_txt(input_file, output_file)
