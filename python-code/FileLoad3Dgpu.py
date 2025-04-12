#!/usr/bin/env python3

from skimage import io
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
import glob
import cv2
import zarr
import cupy as cp
from skimage import io, exposure
from make_dataset import makeX

root = tk.Tk()  # Create the main application window
root.withdraw()  # Hide the empty window, so only the file explorer opens

dataPath = ""  # Path of the folder containing TIFF files
savePath = ""  # Path of the folder to save the Zarr files
fileList = []  # List of TIFF files
imageInfo = []  # Array containing image information


def fetch_files():
    global dataPath, fileList

    while not (fileList):  # While there are no files in the fileList array

        dataPath = filedialog.askdirectory(
            title="Select a folder containing the TIFF files"
        )  # Prompt user to select a folder with TIFF images

        if not (dataPath):  # If no folder is selected
            exit # Exit program

        fileList = glob.glob(
            os.path.join(dataPath, "*.tif")
        )  # Append list of TIFF files to fileList array

        if not (
            fileList
        ):  # If no TIFF files are found in the fileList array (ie the selected folder)
            choice = tk.messagebox.askyesno(
                "No TIFF files",
                "No TIFF files found in directory. Do you want to try another folder?",
            )  # Give option to try again
            if not (choice):  # If user chooses No
                tk.messagebox.showinfo(
                    "Exit", "User opted to exit. No TIFF files found."
                )  # Show info message
                break  # Exit program

        save_files()


def save_files():
    global savePath
    savePath = filedialog.askdirectory(
        title="Select a folder for saving Zarr files."
    )  # Prompt user to select a save folder for the Zarr files

    if not (savePath):  # If no folder is selected
        exit

    # Create the Zarr root directory in the selected folder
    savePath = os.path.join(savePath, "volume.zarr")


def get_image_info():
    global fileList
    global imageInfo

    if not fileList:
        return None

    img = cv2.imread(fileList[0], cv2.IMREAD_UNCHANGED)

    if img is None:
        tk.messagebox.showerror("Error", "Image cannot be read.")
        return None

    numRows, numCols = img.shape[:2]
    numSlices = len(fileList)

    imageInfo.append(numRows)
    imageInfo.append(numCols)
    imageInfo.append(numSlices)


def calculate_chunk_size():
    global imageInfo

    targetChunkSize = np.array([1000, 1000, 1000])
    numChunks = np.ceil(imageInfo / targetChunkSize).astype(int)
    actualChunkSize = np.array(
        [
            np.ceil(imageInfo[0] / numChunks[0]),
            np.ceil(imageInfo[1] / numChunks[1]),
            np.ceil(imageInfo[2] / numChunks[2]),
        ]
    )
    return actualChunkSize


def adjust_contrast():
    global fileList
    global savePath
    global imageInfo

    if not fileList:
        return None

    # Define a custom contrast adjustment function using CuPy
    def rescale_intensity_gpu(image, in_range=(0, 255), out_range=(0, 255)):
        """Rescale the intensity of a CuPy array."""
        in_min, in_max = in_range
        out_min, out_max = out_range
        # Perform the contrast adjustment on GPU
        image = cp.clip((image - in_min) / (in_max - in_min), 0, 1)
        image = image * (out_max - out_min) + out_min
        return cp.array(image, dtype=cp.uint8)  # Return as uint8 CuPy array

    # Create a Zarr file and datasets for the volume and labels
    zarr_store = zarr.open(savePath, mode='w')
    zarr_volume = zarr_store.create('volume', shape=(
        imageInfo[2], imageInfo[0], imageInfo[1]), dtype='uint8', )
    zarr_labels = zarr_store.create('labels', shape=(
        imageInfo[2], imageInfo[0], imageInfo[1]), dtype='uint8', )
    zarr_predictions = zarr_store.create(
        "predictions",
        shape=(imageInfo[2], imageInfo[0], imageInfo[1]),
        dtype="uint8",
    )

    for i, filePath in enumerate(fileList):
        print(f'Processing slice {i+1}/{len(fileList)}')

        # Load the image slice and transfer to GPU
        image = io.imread(filePath)
        image_gpu = cp.asarray(image)  # Transfer image to GPU

        # Perform contrast adjustment on GPU
        contrast_adjusted_image_gpu = rescale_intensity_gpu(image_gpu)

        # Transfer back to CPU for saving to Zarr
        zarr_volume[i, :, :] = cp.asnumpy(contrast_adjusted_image_gpu)

    print("All slices processed and saved.")


def load_and_convert_to_zarr_GPU():
    """High-level function to fetch files, get image info, and adjust contrast."""
    fetch_files()
    get_image_info()
    adjust_contrast()

def main():
    print("Running NVIDIA GPU loader standalone...")
    load_and_convert_to_zarr_GPU()
    makeX(savePath)

if __name__ == "__main__":
    main()