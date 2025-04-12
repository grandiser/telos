#!/usr/bin/env python3

from skimage import io
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os
import glob
import cv2
import zarr
import pyopencl as cl
from skimage import io, exposure
import Vesselness as vs
from make_dataset import makeX

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
    savePath = filedialog.askdirectory(
        title="Select a folder for saving Zarr files."
    )  # Prompt user to select a save folder for the Zarr files

    if not (savePath):  # If no folder is selected
        tk.messagebox.showerror(
            "Error", "No folder selected for saving Zarr files. Exiting Script."
        )  # Show error message
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

    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=[devices[0]])  # Use the first GPU device
    queue = cl.CommandQueue(context)

    kernel_code = """
    __kernel void rescale_intensity(__global uchar *image, const float in_min, const float in_max,
                                    const float out_min, const float out_max, const int length) {
        int gid = get_global_id(0);
        if (gid < length) {
            float value = (float)(image[gid] - in_min) / (in_max - in_min);
            value = clamp(value, 0.0f, 1.0f);
            image[gid] = (uchar)(value * (out_max - out_min) + out_min);
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

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

        # Load the image slice
        image = io.imread(filePath).astype(np.uint8)
        flat_image = image.ravel()  # Flatten the image for OpenCL processing

        # Create buffers and transfer data to the GPU
        gpu_image = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_image)

        # Kernel arguments
        in_min, in_max = 0.0, 255.0
        out_min, out_max = 0.0, 255.0
        length = flat_image.size

        # Execute the kernel
        kernel = program.rescale_intensity
        kernel(queue, (length,), None, gpu_image, np.float32(in_min), np.float32(in_max),
               np.float32(out_min), np.float32(out_max), np.int32(length))

        # Transfer the result back to the CPU
        adjusted_image = np.empty_like(flat_image)
        cl.enqueue_copy(queue, adjusted_image, gpu_image)

        # Reshape and save to Zarr
        zarr_volume[i, :, :] = adjusted_image.reshape(image.shape)

    print("Preprocessing complete.")


def load_and_convert_to_zarr_AMD():
    """High-level function to fetch files, get image info, and adjust contrast."""
    
    root = tk.Tk()  # Create the main application window
    root.withdraw()  # Hide the empty window, so only the file explorer opens

    fetch_files()
    get_image_info()
    adjust_contrast()

def main():
    print("Running AMD GPU loader standalone...")
    load_and_convert_to_zarr_AMD()
    makeX(savePath)

if __name__ == "__main__":
    main()