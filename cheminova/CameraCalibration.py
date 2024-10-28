import cv2
import numpy as np
import os
import argparse
import glob
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Camera Calibration using checkerboard images.")
    parser.add_argument("--square_size", type=float, required=True, help="Size of each checkerboard square in millimeters.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to folder containing calibration images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save calibration results.")
    parser.add_argument("--checkerboard", type=tuple, default=(7, 10), help="Checkerboard dimensions (rows, columns).")

    args = parser.parse_args()

    # Display parameters for this experiment
    for arg in vars(args):
        print(f'{arg} : {getattr(args, arg)}')
    print('================================================================')
    print('================================================================') 

    return args

def main():
    args = parse_args()
    square_size = args.square_size
    checkerboard_size = args.checkerboard
    input_path = Path(args.input_path)  # Using pathlib for improved path handling
    output_path = Path(args.output_path)
    save_path = output_path / "CameraCalibration"

    save_path.mkdir(parents=True, exist_ok=True)

    # Define the 3D object points for the checkerboard corners
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    image_files = list(input_path.glob("*.jpg"))  # Using pathlib's glob for better file handling

    for fname in image_files:
        img = cv2.imread(str(fname))
        if img is None:
            print(f"Warning: Could not open {fname}. Skipping this file.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and save the checkerboard corners on the image
            img = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            corner_img_path = save_path / f"corners_{fname.name}"
            cv2.imwrite(str(corner_img_path), img)

    # Perform camera calibration if there are valid points
    if objpoints and imgpoints:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save calibration parameters
        np.savez(save_path / "camera_params.npz", K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

        # Show reprojection errors
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print("Total reprojection error:", total_error / len(objpoints))

        # Save an example of undistorted image
        example_img = cv2.imread(str(image_files[0]))
        if example_img is not None:
            undistorted_img = cv2.undistort(example_img, K, dist)
            cv2.imwrite(str(save_path / "undistorted_example.jpg"), undistorted_img)
            print("Saved undistorted example image as 'undistorted_example.jpg'.")
        else:
            print("Warning: Could not read the example image for undistortion.")

if __name__ == "__main__":
    print('================================================================')
    print('                     Running Camera calibration                 ')
    print('================================================================') 
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    main()
