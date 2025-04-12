import os
import random
from PIL import Image

IMG_WIDTH = 7442
IMG_HEIGHT = 17506

def generate_128_slices(tiffs_location, output, start_slice=0, num_slices=128, crop_size=128):
    if not os.path.exists(output):
        os.makedirs(output)

    # List and sort all TIFF files in the input folder
    all_files = sorted([f for f in os.listdir(tiffs_location) if f.endswith(".tif")])

    # Ensure there are enough files for slicing
    if len(all_files) < start_slice + num_slices:
        print("Not enough files to generate the slices. Exiting.")
        return

    # Define the cropping region (center crop)
    x = 4000
    y = 8000

    # Loop through the first 128 files starting at start_idx
    for z in range(num_slices):
        file_idx = start_slice + z
        file_path = os.path.join(tiffs_location, all_files[file_idx])

        try:
            with Image.open(file_path) as img:
                # Crop the image to the desired 128x128 square
                cropped_img = img.crop((x, y, x + crop_size, y + crop_size))

                # Save the cropped slice
                slice_name = f"slice_{z:03d}.tif"
                slice_path = os.path.join(output, slice_name)
                cropped_img.save(slice_path)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"128 slices saved to {output}")

# Example usage
generate_128_slices(
    tiffs_location="C:/Users/Imad Baida/Desktop/647nm_Lectin",
    output="C:/Users/Imad Baida/Desktop/slices",
    start_slice=1000,  # Starting slice index
    num_slices=128,  # Number of slices to process
    crop_size=128   # Crop size for each slice (128x128 pixels)
)
