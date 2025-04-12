# Machine Learning for Blood Vessel Segmentation

This project provides tools for preprocessing, filtering, labeling, and classifying high-resolution 3D cortical blood vessel datasets. It leverages semi-supervised learning, Zarr (Format 2), Dask, Napari, and various image processing techniques.

## 🔧 Features

- Efficient loading of large volumetric datasets in Zarr format
- Filtering and per-voxel feature expansion using Frangi, Sato, Meijering vesselness filters
- Semi-supervised learning using SGDClassifier (via Dask-ML)
- Napari GUI integration for labeling and previewing predictions

## 🧰 Requirements

Clone the repository using
```
git clone https://github.com/LandoRaDag/Capstone.git
```
Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the following command to launch the main GUI:
```
python python-code/main.py
```
Within the interface, you may:

1. Preprocessing
Convert your .tiff file into a Zarr dataset. This step is compute-heavy — enabling GPU acceleration with CuPy (for NVIDIA) or PyOpenCL (for AMD) is recommended for speed.

2. Feature Expansion
Embedded within preprocessing. Filters from make_dataset.py (Frangi, Sato, etc.) are applied to generate multi-channel voxel-level features. These features become the basis for classification.

3. Annotation & Segmentation
Napari will open for manual labeling. Use the "Labels" layer: Red (1) for background, Green (2) for vessels

Once labeled, click the Predict button. An SGDClassifier will train and classify the rest of the image. Predictions will appear in the "predictions" layer.

## 📝 Notes
Designed for research and prototyping, not production use.

Labels generated can be used to train downstream models like U-Net.

Feel free to expand or modify the pipeline to suit your specific imaging tasks

## 👥 Authors
Developed by Charles Blancas, Landon Ra Dagenais, Imad Baida, and Mohamed Elsamadouny and team as part of a McGill capstone project in computer engineering and machine learning, under supervision from Professor Amir Shmuel of the Neuro Lab.

