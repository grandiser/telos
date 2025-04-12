import dask.array as da
import dask_image.ndfilters as dif
import dask_ml.cluster
from skimage.filters import frangi,sato,meijering
from dask import delayed
import numpy as np

def apply_2d_laplace(volume, sigma=2.5):
    # Apply 2D Laplacian to each Z slice
    filtered_slices = [dif.gaussian_laplace(volume[z, :, :], sigma=sigma)[None, :, :]
                       for z in range(volume.shape[0])]
    return da.concatenate(filtered_slices, axis=0)

def apply_2d_sato_delayed(volume):
    slices = [
        delayed(sato)(volume[z].compute(), black_ridges=False)[None, :, :]
        for z in range(volume.shape[0])
    ]
    stacked = delayed(np.concatenate)(slices, axis=0)
    return da.from_delayed(stacked, shape=(volume.shape[0],) + volume.shape[1:], dtype=np.float64)

def makeX(zarr_path):
    
    volume = da.from_zarr(zarr_path, "volume")
    laplacian = apply_2d_laplace(volume, sigma=2.5)

    # Other vesselness features can be used
    #yfrangi = volume.map_overlap(lambda x: frangi(x, black_ridges=False), depth=1)
    ysato = apply_2d_sato_delayed(volume)
    #ymeijiring = volume.map_overlap(lambda x: meijering(x, black_ridges=False),depth=1)

    # Modify this array to change the features used to train the model
    arrs = [volume, laplacian, ysato]
    arrs = [arr.ravel() for arr in arrs]
    stacked = da.stack(arrs, -1)
    da.to_zarr(stacked, zarr_path, "X", overwrite=True)
    print("Generating features")
    X = stacked.rechunk({1: stacked.shape[1]})
    print("Features saved")