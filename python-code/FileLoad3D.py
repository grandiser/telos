#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
import glob
import cv2
import zarr
from skimage import io, exposure
import nibabel as nib
from make_dataset import makeX


root = tk.Tk()  # Create the main application window
root.withdraw()  # Hide the empty window, so only the file explorer opens

dataPath = ""  # Path of the folder containing TIFF files
savePath = ""  # Path of the folder to save the HDF5 files
fileList = []  # List of TIFF files
imageInfo = []  # Array containing image information

def fetch_files():
    global dataPath, fileList

    while not (fileList):  # While there are no files in the fileList array

        dataPath = filedialog.askdirectory(
            title="Select a folder containing the TIFF files"
        )  # Prompt user to select a folder with TIFF images

        if not (dataPath):  # If no folder is selected
            tk.messagebox.showerror(
                "Error", "No folder selected for TIFF files. Exiting Script."
            )  # Show error message
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
    # Use asksaveasfilename to allow the user to choose the file name
    savePath = filedialog.askdirectory(
        title="Select a folder for saving Zarr files."
    )  # Prompt user to select a file name

    if not savePath:  # If no file name is selected
        tk.messagebox.showerror(
            "Error", "No file name selected for saving HDF5 file. Exiting Script."
        )  # Show error message
        exit
    savePath = os.path.join(savePath, "volume.zarr")  # Append file extension


def get_image_info():
    global fileList
    global imageInfo

    if not fileList:
        tk.messagebox.showerror("Error", "No images in chosen directory.")
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

    # Ensure it's a NumPy array before calculations
    imageInfo = np.array(imageInfo)


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

    zarr_store = zarr.open(savePath, mode="w")
    zarr_volume = zarr_store.create(
        "volume",
        shape=(imageInfo[2], imageInfo[0], imageInfo[1]),
        dtype="uint8",
    )
    zarr_labels = zarr_store.create(
        "labels",
        shape=(imageInfo[2], imageInfo[0], imageInfo[1]),
        dtype="uint8",
    )
    zarr_predictions = zarr_store.create(
        "predictions",
        shape=(imageInfo[2], imageInfo[0], imageInfo[1]),
        dtype="uint8",
    )
    zarr_predictions[0][0][0] = 1
    for i, filePath in enumerate(fileList):
        print(f"Processing slice {i+1}/{len(fileList)}")

        # Load the image slice
        image = io.imread(filePath)

        # Perform contrast adjustment

        in_min, in_max = 0, 255
        out_min, out_max = 0, 255

        # Perform the contrast adjustment on CPU
        image = np.clip((image - in_min) / (in_max - in_min), 0, 1)
        image = image * (out_max - out_min) + out_min

        # Return the rescaled image as uint8
        contrast_adjusted_image = np.array(image, dtype=np.uint8)

        # Transfer back to CPU for saving to HDF5
        zarr_volume[i, :, :] = np.array(
            contrast_adjusted_image
        )  # Convert to NumPy for HDF5 storage

    print("All slices processed and saved.")


def nii_to_zarr():
    """Selects a directory containing .nii files and converts them all to .zarr format."""
    # Open folder selection dialog
    folder_path = filedialog.askdirectory(
        title="Select a folder containing NIfTI (.nii) files")
    save_path = filedialog.askdirectory(
        title="Select a folder to save the .zarr files")

    if not folder_path:  # If no folder is selected, exit
        print("No folder selected. Exiting.")
        return

    # Get all .nii files in the selected directory
    nii_files = [f for f in os.listdir(folder_path) if f.endswith(".nii")]

    if not nii_files:
        messagebox.showerror(
            "Error", "No NIfTI (.nii) files found in the selected folder.")
        return

    print(f"Found {len(nii_files)} NIfTI files in {folder_path}.")

    # Process each .nii file
    for nii_file in nii_files:
        nii_path = os.path.join(folder_path, nii_file)
        print(f"Processing: {nii_path}")

        # Load the NIfTI file
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata(
            dtype=np.float32)  # Convert to NumPy array

        print(nii_data.shape)

        # Remove the last dimension
        if (len(nii_data.shape) == 4):
            nii_data = np.squeeze(nii_data, -1)

        # Swap the axes so we have square images
        nii_data = np.swapaxes(nii_data, 0, 2)

        # Define the output Zarr file path
        zarr_output_path = os.path.join(
            save_path, os.path.splitext(nii_file)[0] + ".zarr")

        # Create a Zarr store
        zarr_store = zarr.open(zarr_output_path, mode='w')

        # Save the volume data (chunk size can be adjusted)
        zarr_store.create(
            'volume',
            shape=nii_data.shape,
            dtype=nii_data.dtype,
            # chunks=(1, 1000, 1000)  # Adjust chunk size based on expected processing needs
        )

        # Write the data to Zarr
        zarr_store['volume'][:] = nii_data

        print(f"Converted {nii_file} to {zarr_output_path} successfully!")

    messagebox.showinfo(
        "Success", f"Converted {len(nii_files)} NIfTI files to Zarr in {folder_path}.")
    return


def load_and_convert_to_zarr():
    fetch_files()
    get_image_info()
    adjust_contrast()

def main():
    print("Running CPU loader standalone...")
    load_and_convert_to_zarr()
    makeX(savePath)

if __name__ == "__main__":
    main()
