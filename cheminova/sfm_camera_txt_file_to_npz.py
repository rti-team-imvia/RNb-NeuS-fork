import numpy as np

def parse_camera_file(file_path):
    cameras = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Find the line with the number of cameras
        for i, line in enumerate(lines):
            if line.startswith("# The nubmer of cameras in this reconstruction"):
                num_cameras = int(lines[i+1])
                start_index = i + 2
                break
        
        current_camera = {}
        r_count = 0
        quaternion_seen = False
        for line in lines[start_index:]:
            line = line.strip()
            if line == '':
                if current_camera:
                    cameras.append(current_camera)
                    current_camera = {}
                    r_count = 0
                    quaternion_seen = False
            elif line.endswith('.jpg'):
                if 'filename' not in current_camera:
                    current_camera['filename'] = line
                else:
                    current_camera['original_filename'] = line
            elif len(line.split()) == 1 and line[0].isdigit():  # Focal length
                current_camera['focal_length'] = float(line)
            elif len(line.split()) == 2:  # Principal point
                current_camera['principal_point'] = [float(x) for x in line.split()]
            elif len(line.split()) == 3:
                if 'T' not in current_camera:
                    current_camera['T'] = [float(x) for x in line.split()]
                elif 'C' not in current_camera:
                    current_camera['C'] = [float(x) for x in line.split()]
                elif quaternion_seen and r_count < 3:
                    if 'R' not in current_camera:
                        current_camera['R'] = []
                    current_camera['R'].append([float(x) for x in line.split()])
                    r_count += 1
            elif len(line.split()) == 4:  # Quaternion
                current_camera['Q'] = [float(x) for x in line.split()]
                quaternion_seen = True
    
    if current_camera:  # Add the last camera
        cameras.append(current_camera)
    
    return cameras

def create_camera_matrix(camera):
    R = np.array(camera['R'])
    T = np.array(camera['T']).reshape(3, 1)
    
    print(f"R shape: {R.shape}, T shape: {T.shape}")  # Debug print
    print(f"R matrix content: {R}")  # Debug print
    
    if R.shape != (3, 3):
        print(f"Warning: R matrix is not 3x3. Actual shape: {R.shape}")
        return None
    
    camera_matrix = np.hstack((R, T))
    camera_matrix = np.vstack((camera_matrix, [0, 0, 0, 1]))
    return camera_matrix

def main():
    file_path = r'C:/Users/Deivid/Downloads/cameras_v2.txt'  # Update this path if needed
    cameras = parse_camera_file(file_path)
    
    scale_matrix = np.array([[1, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]])

    camera_matrices = {}
    for idx, camera in enumerate(cameras):
        matrix = create_camera_matrix(camera)
        if matrix is not None:
            camera_matrices[f'world_mat_{idx}'] = matrix
        else:
            print(f"Skipping camera_{idx} due to invalid matrix")

        camera_matrices[f'scale_mat_{idx}'] = scale_matrix
    
    np.savez('camera_matrices.npz', **camera_matrices)
    print(f"Saved {len(camera_matrices)} camera matrices to camera_matrices.npz")

if __name__ == "__main__":
    main()