import alphashape
import numpy as np
from PIL import Image, ImageDraw

from vesuvius.image_proc.helpers import arr_to_indices

def alpha_wrap_arr(arr, alpha=0.1):
    indices = arr_to_indices(arr)
    return alphashape.alphashape(indices, alpha)

def fill_alpha_shape(arr_shape,
                     alpha_shape,
                     fill_value=1,
                     fill_mode='inner'):

    if fill_mode not in ('inner', 'outer'):
        raise ValueError("fill_mode must be either 'inner' or 'outer'")

    background_value = 0 if fill_mode == 'inner' else fill_value
    polygon_value = fill_value if fill_mode == 'inner' else 0

    output_img = Image.new('L', (arr_shape[1], arr_shape[0]), background_value)
    draw = ImageDraw.Draw(output_img)

    if alpha_shape.geom_type == 'Polygon':
        coords = list(alpha_shape.exterior.coords)
        draw.polygon(coords, fill=polygon_value)
    elif alpha_shape.geom_type == 'MultiPolygon':
        for poly in alpha_shape.geoms:
            coords = list(poly.exterior.coords)
            draw.polygon(coords, fill=polygon_value)

    return np.array(output_img)
