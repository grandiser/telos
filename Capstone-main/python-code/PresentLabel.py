from tkinter import filedialog
import napari
import zarr
from magicgui import magicgui
import dask.array as da
from sklearn.linear_model import SGDClassifier
from dask_ml.wrappers import Incremental
from napari.utils.notifications import show_info, notification_manager
import os
from qtpy import QtWidgets
from qtpy.QtWidgets import QMessageBox, QFileDialog
from qtpy.QtCore import Qt
import warnings

# Make napari still work while data is loading
os.environ["NAPARI_ASYNC"] = "1"
warnings.filterwarnings("ignore", message=".*default color.*")

X = None
file_path = None
volume_shape = None
predictions_layer = None
viewer = None
zarr_file = None


@magicgui(call_button="Predict")
def predict():
    show_info("Predicting all labels based on given ones. This may take a while.")
    global file_path
    global X
    global volume_shape
    global predictions_layer
    global viewer
    global zarr_file

    y = da.from_zarr(file_path, "labels").ravel()

    mask = y != 0
    X_train = X[mask].compute()
    y_train = y[mask].compute()

    sgd = Incremental(SGDClassifier())
    sgd.fit(X_train, y_train, classes=[1, 2])
    sgd_out = sgd.predict(X) - 1
    # If the user labelled it, overwrite the prediction
    sgd_out[y == 1] = 0
    sgd_out[y == 2] = 1
    sgd_out = sgd_out.reshape(volume_shape)
    da.to_zarr(sgd_out, file_path, "predictions", overwrite=True)

    viewer.layers.remove('Predictions')
    predictions_layer = viewer.add_labels(
        zarr_file["predictions"], name="Predictions", colormap={1: "blue"}
    )
    predictions_layer.editable = False
    viewer.layers.selection.active = viewer.layers['Blood vessel labels']
    show_info("The predictions have been updated.")

def present_and_label():
    global file_path
    global X
    global volume_shape
    global predictions_layer
    global viewer
    global zarr_file

    # Open the Zarr file and get the dataset
    file_path = QFileDialog.getExistingDirectory(
        None,
        "Select Zarr File",
        "",
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
    )

    if not file_path:
        print("No directory selected")
        return

    # Open the Zarr file and access the selected datasets
    zarr_file = zarr.open(file_path, mode="r+")
    volume_shape = zarr_file["volume"].shape
    X = da.from_zarr(file_path, "X")

    # Show notifications in UI instead of console
    notification_manager.install_hooks()

    # Create Napari viewer and add the lazy-loaded data
    viewer = napari.view_image(zarr_file["volume"], name="Blood vessel image")

    labels_layer = viewer.add_labels(
        zarr_file["labels"], name="Blood vessel labels", colormap={1: "red", 2: "green"}
    )

    predictions_layer = viewer.add_labels(
        zarr_file["predictions"], name="Predictions", colormap={1: "blue"}
    )
    predictions_layer.editable = False
    viewer.layers.selection.active = viewer.layers['Blood vessel labels']

    viewer.window.add_dock_widget(predict, area="right")

    napari.run()

def main():
    print("Presenting and labeling chosen Zarr dataset")
    present_and_label()


if __name__ == "__main__":
    main()