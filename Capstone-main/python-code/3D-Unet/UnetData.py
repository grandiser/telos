import glob
import os

import nibabel as nib
import numpy as np
from patchify import patchify

### USE .NII FILES - C01 REPRESENTS THE ORIGINAL IMAGE, C00 IS THE PROCESSED IMAGE, CGT IS THE GROUND TRUTH ###

### MODIFY THESE FILE PATHS TO YOUR TRAINING AND TEST DATA - THE REST OF THE CODE WILL TAKE CARE OF SPLITTING THEM INTO THE IMAGE AND THE LABELS. ###
TRAINING_DATA_PATH = r"D:\capstone\TRUTH\real_training"
TEST_DATA_PATH = r"D:\capstone\TRUTH\real_test"

class UnetData:
    """
    Class that handles the pre-processing of the data for the 3D-Unet Model.
    1- Sorts the data into appropriate lists
    2- Loads volumes into np arrays and sets dimensions properly
    3- Patchifies the volumes into smaller chunks
    """
    def __init__(self, training_data_path: str, test_data_path: str, patch_size=(48, 48, 48, 1), step=(24, 24, 24, 1)):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path

        self.X_TRAIN_PATHS = []
        self.Y_TRAIN_PATHS = []

        self.X_TEST_PATHS = []

        self.X_RAW_TRAIN_PATHS = []
        self.X_RAW_TEST_PATHS = []

        self.X_TRAIN = []
        self.Y_TRAIN = []

        self.X_TEST = []

        self.X_TRAIN_RAW = []
        self.X_TEST_RAW = []

        self.X_TRAIN_PATCHES = []
        self.Y_TRAIN_PATCHES = []
        self.X_TEST_PATCHES = []

        self.patch_size = patch_size
        self.step = step

    def sort_training_data(self):
        for image in glob.glob(os.path.join(self.training_data_path, "*.nii")):
            if "CGT" in image:
                self.Y_TRAIN_PATHS.append(image)
            elif "C00" in image:
                self.X_TRAIN_PATHS.append(image)
            elif "C01" in image:
                self.X_RAW_TRAIN_PATHS.append(image)
            else:
                continue

    def sort_testing_data(self):
        for image in glob.glob(os.path.join(self.test_data_path, "*.nii")):
            if "C00" in image:
                self.X_TEST_PATHS.append(image)
            elif "C01" in image:
                self.X_RAW_TEST_PATHS.append(image)
            else:
                continue

    def select_list_from_path(self, paths: list):
        if paths == self.X_TRAIN_PATHS:
            return self.X_TRAIN

        elif paths == self.Y_TRAIN_PATHS:
            return self.Y_TRAIN

        elif paths == self.X_RAW_TRAIN_PATHS:
            return self.X_TRAIN_RAW

        elif paths == self.X_TEST_PATHS:
            return self.X_TEST

        elif paths == self.X_RAW_TEST_PATHS:
            return self.X_TEST_RAW

        else:
            return f"Paths list provided: {paths} is wrong"

    def load_numpy_and_expand_x(self, x_paths: list):
        selected_list = self.select_list_from_path(x_paths)

        images = [nib.load(image).get_fdata() for image in x_paths]
        images = np.array(images)  # Convert to a NumPy array

        # Normalize in one step (avoids the loop)
        max_vals = np.max(images, axis=(1, 2, 3), keepdims=True)  # Get max per image
        images = np.where(max_vals > 0, images / max_vals, images)  # Avoid division by zero

        selected_list.extend(images)

    def load_numpy_and_expand_y(self, y_paths: list):
        selected_list = self.select_list_from_path(y_paths)

        labels = [nib.load(label).get_fdata() for label in y_paths]
        labels = [(label > 0).astype(np.uint8) for label in labels]

        processed_labels = []
        for label in labels:
            if label.ndim == 3:
                label = np.expand_dims(label, axis=-1)
            processed_labels.append(label)

        processed_labels = np.array(processed_labels)
        selected_list.extend(processed_labels)

    def patch_volumes_training(self, x_volumes, y_volumes: list):
        for x_volume, y_volume in zip(x_volumes, y_volumes):
            x_volume_patch = patchify(x_volume, self.patch_size, step=self.step)
            x_volume_patch = x_volume_patch.reshape(-1, *self.patch_size)

            y_volume_patch = patchify(y_volume, self.patch_size, step=self.step)
            y_volume_patch = y_volume_patch.reshape(-1, *self.patch_size)

            self.X_TRAIN_PATCHES.append(x_volume_patch)
            self.Y_TRAIN_PATCHES.append(y_volume_patch)

        self.X_TRAIN_PATCHES = np.concatenate(self.X_TRAIN_PATCHES, axis=0)
        self.Y_TRAIN_PATCHES = np.concatenate(self.Y_TRAIN_PATCHES, axis=0)

        return self.X_TRAIN_PATCHES, self.Y_TRAIN_PATCHES

    def patch_volumes_testing(self, x_volumes: list):
        for x_volume in x_volumes:
            x_volume_patch = patchify(x_volume, self.patch_size, step=self.step)
            x_volume_patch = x_volume_patch.reshape(-1, *self.patch_size)

            self.X_TEST_PATCHES.append(x_volume_patch)

        self.X_TEST_PATCHES = np.concatenate(self.X_TEST_PATCHES, axis=0)

        return self.X_TEST_PATCHES


# unetdata = UnetData(TRAINING_DATA_PATH, TEST_DATA_PATH)
#
# unetdata.sort_training_data()
# unetdata.sort_testing_data()
# unetdata.load_numpy_and_expand_x(unetdata.X_TRAIN_PATHS)
# unetdata.load_numpy_and_expand_x(unetdata.X_TEST_PATHS)
# unetdata.load_numpy_and_expand_y(unetdata.Y_TRAIN_PATHS)
#
# X_train, Y_train = unetdata.patch_volumes_training(unetdata.X_TRAIN, unetdata.Y_TRAIN)
# X_test = unetdata.patch_volumes_testing(unetdata.X_TEST)

