import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

# Global variables to store the images and difference for easy access in the callback
global_img1 = None
global_img2 = None
global_difference = None

def compare_images(img1, img2):
    """Calculate and display comparison metrics between two images."""
    # Compute absolute difference
    difference = cv2.absdiff(img1, img2)
    mse_value = np.mean((img1 - img2) ** 2)

    # Return results
    return difference, mse_value

def on_click(event):
    """Callback function to handle mouse click events and display pixel values."""
    if event.inaxes is not None:
        # Get the x and y coordinates of the click
        x, y = int(event.xdata), int(event.ydata)

        # Fetch the pixel values for all three images (img1, img2, difference)
        pixel_img1 = global_img1[y, x]
        pixel_img2 = global_img2[y, x]
        pixel_difference = global_difference[y, x]

        # Print the pixel values
        print(f"Coordinates: ({x}, {y})")
        print(f"  Image 1 Pixel Value: {pixel_img1}")
        print(f"  Image 2 Pixel Value: {pixel_img2}")
        print(f"  Difference Pixel Value: {pixel_difference}")

def show_images(img1, img2, difference):
    """Display the two images and their difference side by side with clickable event."""
    global global_img1, global_img2, global_difference
    global_img1, global_img2, global_difference = img1, img2, difference

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1")
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2")

    # Convert the difference to grayscale for heatmap and ensure the size is consistent
    heatmap = np.mean(difference, axis=2)  # Average RGB channels to get a single-channel image
    im = axes[2].imshow(heatmap, cmap='hot', extent=[0, img1.shape[1], img1.shape[0], 0])  # Set extent to match size
    axes[2].set_title("Difference (Heatmap)")
    fig.colorbar(im, ax=axes[2])  # Add colorbar for heatmap

    for ax in axes:
        ax.axis("off")

    # Connect the click event handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the plot and wait for the window to be closed by the user
    plt.show(block=True)

def compare_folders(folder1, folder2):
    """Compare images from two folders and display results."""
    img_files = sorted(os.listdir(folder1))

    for img_file in img_files:
        img1_path = Path(folder1) / img_file
        img2_path = Path(folder2) / img_file

        # Check if corresponding image exists in both folders
        if not img1_path.exists() or not img2_path.exists():
            print(f"Image {img_file} does not exist in both folders, skipping...")
            continue

        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            print(f"Error reading {img_file}, skipping...")
            continue

        # Compare images
        difference, mse_value = compare_images(img1, img2)

        # Show images and comparison
        print(f"\nComparing {img_file}:")
        print(f"  MSE: {mse_value:.2f}")

        # Display images
        show_images(img1, img2, difference)

def main():
    parser = argparse.ArgumentParser(description="Compare images in two folders.")
    parser.add_argument('--input_folder1', type=str, required=True, help="Path to the first folder of images.")
    parser.add_argument('--input_folder2', type=str, required=True, help="Path to the second folder of images.")

    args = parser.parse_args()

    folder1 = Path(args.input_folder1)
    folder2 = Path(args.input_folder2)

    print('================================================================')
    print('                     Running compare_image_similarities         ')
    print('================================================================') 
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    # Verify that both folders exist
    if not folder1.exists() or not folder2.exists():
        print("One or both folders do not exist. Please check the paths and try again.")
        return

    # Compare the folders
    compare_folders(folder1, folder2)

if __name__ == "__main__":
    main()

# python cheminova/compare_image_similarities.py --input_folder1 "C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/normal" --input_folder2 "C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/bearPNG_3D_in/normal"
# python cheminova/compare_image_similarities.py --input_folder1 "C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/mask" --input_folder2 "C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/bearPNG_3D_in/mask"
# python cheminova/compare_image_similarities.py --input_folder1 "C:/Users/Deivid/Documents/repos/RNb-NeuS-fork/DiLiGenT-MV/bearPNG/albedo" --input_folder2 "C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/bearPNG_3D_in/albedo"