import os
import numpy as np
from PIL import Image

# Define the path to the reflectance maps
reflectance_path = r"C:/Users/Deivid/Documents/rti-data/Palermo_3D/3D/bearPNG_3D_in/albedo"

# Get the list of reflectance maps
reflectance_files = sorted([f for f in os.listdir(reflectance_path) if f.endswith('.png')])

# Load the reflectance maps into a list
reflectance_maps = []
for file in reflectance_files:
    image_path = os.path.join(reflectance_path, file)
    image = Image.open(image_path)
    reflectance_maps.append(np.array(image, dtype=np.float64))

# Iterate over neighboring pairs to compute reflectance ratios
scale_factors = []
for i in range(len(reflectance_maps) - 1):
    reflectance_1 = reflectance_maps[i]
    reflectance_2 = reflectance_maps[i + 1]
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6
    ratios = reflectance_2 / (reflectance_1 + epsilon)
    
    # Store the median of the ratios as the scale factor
    median_ratio = np.median(ratios)
    scale_factors.append(median_ratio)

# Apply the median scale factor to each reflectance map
scaled_reflectance_maps = []
for i, reflectance_map in enumerate(reflectance_maps):
    if i < len(scale_factors):
        scale_factor = scale_factors[i]
    else:
        scale_factor = 1  # No scaling for the last image
    
    scaled_map = reflectance_map * scale_factor
    scaled_map = np.clip(scaled_map, 0, 65535)  # Clip to valid range for 48-bit depth
    scaled_reflectance_maps.append(scaled_map)

# Save the scaled reflectance maps back to disk
for i, scaled_map in enumerate(scaled_reflectance_maps):
    output_path = os.path.join(reflectance_path, f"scaled_{reflectance_files[i]}")
    scaled_image = Image.fromarray(scaled_map.astype(np.uint16))
    scaled_image.save(output_path)

print("Reflectance maps scaling complete.")