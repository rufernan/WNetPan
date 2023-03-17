import numpy as np
import rasterio

def load_data(file_name, scaling_factor=1, norm=True, add_batch_axis=True):
       
    if scaling_factor==1:
        data = rasterio.open(file_name).read()

    else:
        with rasterio.open(file_name) as dataset:

            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scaling_factor),
                    int(dataset.width * scaling_factor)
                ),
                resampling=rasterio.enums.Resampling.gauss
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )
    
    data = data.astype('uint16') # (bands, rows, cols)
    
    if norm:
        data = data.astype('double')/np.max(data)

    if add_batch_axis:
        data = data[np.newaxis, :] # (1, bands, rows, cols)
    
    return data
