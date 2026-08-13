import os
import numpy as np
import webknossos as wk
import fsspec
import zarr

import fsspec
import zarr
import numpy as np


def read_webk_dataset(url,
                      token,
                      annotation_id,
                      bbox_number=0,
                      annotation_name='Volume',
                      image_name='first_letters_segment',
                      patch_bbox=None,
                      return_data_or_label='label',
                      **kwargs):

    url = url
    token = token
    annotation_id = annotation_id

    with wk.webknossos_context(url, token):
        # there is likely a better way to do this , wk has a dataset.open_as_remote , but it doesnt
        # directly have user bounding boxes, which we need -- will do more looking into it
        # for wk data we will always return a dictionary containing both image and label

        annotation = wk.Annotation.download(f"{annotation_id}")
        dataset = annotation.get_remote_annotation_dataset()
        anno_mag_view = dataset.layers[f'{annotation_name}'].get_finest_mag()
        mag = anno_mag_view.mag
        vol_mag_view = dataset.layers[f'{image_name}'].mags[mag]
        bbox = annotation.user_bounding_boxes[bbox_number]

        if patch_bbox is None:
            anno_data = anno_mag_view.read(
                absolute_bounding_box=bbox
            )

            image_data = vol_mag_view.read(
                absolute_bounding_box=bbox
            )

        else:
            anno_data = anno_mag_view.read(
                absolute_bounding_box=patch_bbox
            )

            image_data = vol_mag_view.read(
                absolute_bounding_box=patch_bbox
            )

        # wk data shape is (c, x, y, z) but my entire pipeline assumes (c, z, y, x)
        # all our data is greyscale, and all my augs/etc from this point require
        # no addtl channels so we will squeeze the data and then transp the axes

        anno_data = np.squeeze(anno_data, axis=0)
        image_data = np.squeeze(image_data, axis=0)
        anno_data = np.transpose(anno_data, (2, 1, 0))
        image_data = np.transpose(image_data, (2, 1, 0))

        return {
            'data': image_data,
            'label': anno_data
        }

def slice_and_reorder_volume(array, shape_format, start_pos, patch_size):
    """
    array: zarr or numpy array with shape as indicated by shape_format
    shape_format: one of ["zyx", "xyz", "cxyz", ...]
    start_pos: [z, y, x]
    patch_size: [dz, dy, dx]
    Returns a NumPy array of shape (dz, dy, dx).
    """
    z, y, x = start_pos
    dz, dy, dx = patch_size

    if shape_format == 'zyx':
        # Already z, y, x. Just do normal indexing.
        patch = array[z:z+dz, y:y+dy, x:x+dx]

    elif shape_format == 'xyz':
        # Data is x,y,z so slice accordingly.
        patch = array[x:x+dx, y:y+dy, z:z+dz]
        # Now we reorder (x, y, z) -> (z, y, x)
        patch = patch.transpose(2, 1, 0)

    elif shape_format == 'cxyz':
        # Data is (c, x, y, z). Suppose c=1, or you want the first channel if c>1.
        # Slice (channel, x-range, y-range, z-range)
        # Adjust as needed if you have multiple channels you want to keep.
        patch = array[0, x:x+dx, y:y+dy, z:z+dz]
        # Now patch is (x, y, z). Reorder to (z, y, x).
        patch = patch.transpose(2, 1, 0)

    elif shape_format == 'xyzc':
        # Some users might store channels last. You can handle that as well.
        # Let's say the shape is (x, y, z, c).
        # If c=1, we can drop it:
        patch = array[x:x+dx, y:y+dy, z:z+dz, 0]
        patch = patch.transpose(2, 1, 0)

    elif shape_format == 'zyxc':
        # data is (Z, Y, X, C)
        # If you want the result as (C, Z, Y, X), do:
        patch = array[z:z + dz, y:y + dy, x:x + dx, :]
        patch = patch.transpose(3, 0, 1, 2)

    else:
        # Unknown format, or just default to "zyx" logic
        patch = array[z:z+dz, y:y+dy, x:x+dx]

    return patch






