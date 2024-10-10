import os
import shutil
import re
import time
import argparse
from pathlib import Path

def natural_sort(l):
    """Sorts a list in human order, converting Path objects to strings."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]  # Ensure Path is a string
    return sorted(l, key=alphanum_key)

def find_cameras_file(root_path):
    """Recursively search for the 'cameras.npz' file starting from root_path."""
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file == 'cameras.npz':
                return os.path.join(dirpath, file)
    return None

def main():
    parser = argparse.ArgumentParser(description="Organize 3D data files.")
    
    # Accept the input folder path via console argument
    parser.add_argument('--input', required=True, help="Input folder path where the data resides.")
    
    # Accept the output folder path via console argument
    parser.add_argument('--output', required=True, help="Output folder path where the 3D_in folder will be created.")
    
    args = parser.parse_args()
    
    # Cross-platform path handling
    input_folder = Path(args.input).resolve()
    output_base_folder = Path(args.output).resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: The provided input path '{input_folder}' does not exist or is not a directory.")
        return

    if not output_base_folder.exists() or not output_base_folder.is_dir():
        print(f"Error: The provided output path '{output_base_folder}' does not exist or is not a directory.")
        return

    print('================================================================')
    print('                     Running Organize data to RNb               ')
    print('================================================================') 
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    # Get the last folder name for output folder creation
    last_folder_name = input_folder.name
    
    # Create the output folder path at the user-specified location
    output_folder_name = last_folder_name + '_3D_in'
    output_folder = output_base_folder / output_folder_name
    
    # Create the output folder and subfolders (albedo, mask, normal)
    output_folder.mkdir(exist_ok=True)
    albedo_folder = output_folder / 'albedo'
    mask_folder = output_folder / 'mask'
    normal_folder = output_folder / 'normal'
    
    albedo_folder.mkdir(exist_ok=True)
    mask_folder.mkdir(exist_ok=True)
    normal_folder.mkdir(exist_ok=True)
    
    # Get list of subfolders (view_XX) in the input folder
    subfolders = [f for f in input_folder.iterdir() if f.is_dir()]
    subfolders = natural_sort(subfolders)
    
    cameras_copied = False  # Flag to copy the cameras.npz file only once

    # Process each subfolder (view_XX)
    for idx, subfolder in enumerate(subfolders):
        
        # Path to the 'SDM_out' folder inside the 'view_XX' folder
        sdm_out_folder = subfolder / 'SDM_out'
        
        # Paths to source files in the 'view_XX' and 'SDM_out' folders
        mask_src = subfolder / 'mask.png'
        albedo_src = sdm_out_folder / 'baseColor.png'
        normal_src = sdm_out_folder / 'normal.png'
        
        # Prepare destination filenames with zero-padded indices
        idx_str = f"{idx:03d}.png"
        albedo_dst = albedo_folder / idx_str
        mask_dst = mask_folder / idx_str
        normal_dst = normal_folder / idx_str
        
        # Copy the mask.png file
        try:
            shutil.copyfile(mask_src, mask_dst)
            print(f"Copied mask.png to {mask_dst}")
        except FileNotFoundError:
            print(f"mask.png not found in {subfolder}")
        
        # Copy the baseColor.png as albedo from the 'SDM_out' folder
        try:
            shutil.copyfile(albedo_src, albedo_dst)
            print(f"Copied baseColor.png to {albedo_dst}")
        except FileNotFoundError:
            print(f"baseColor.png not found in {sdm_out_folder}")
        
        # Copy the normal.png file from the 'SDM_out' folder
        try:
            shutil.copyfile(normal_src, normal_dst)
            print(f"Copied normal.png to {normal_dst}")
        except FileNotFoundError:
            print(f"normal.png not found in {sdm_out_folder}")
        
        # Find and copy the cameras.npz file (only once)
        if not cameras_copied:
            cameras_file_path = find_cameras_file(subfolder)
            if cameras_file_path:
                cameras_dst = output_folder / 'cameras.npz'
                try:
                    shutil.copyfile(cameras_file_path, cameras_dst)
                    print(f"Copied cameras.npz to {cameras_dst}")
                    cameras_copied = True  # Only copy once
                except Exception as e:
                    print(f"Failed to copy cameras.npz: {e}")
            else:
                print(f"cameras.npz not found in {subfolder}")
    
    if not cameras_copied:
        print("No cameras.npz file was copied since none was found.")

if __name__ == '__main__':
    main()
