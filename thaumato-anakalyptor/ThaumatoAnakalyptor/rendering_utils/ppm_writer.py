### Giorgio Angelotti - 2024

import numpy as np
import struct
from tqdm import tqdm

class PPMWriter(object):
    def __init__(self, filename, width, height):
        self.filename = filename
        # Setting width and height from the inputs
        self.info = {
            'width': width,
            'height': height,
            'dim': 6,
            'ordered': 'true',
            'type': 'double',
            'version': 1
        }

    def write_header(self):
        with open(self.filename, 'wb') as f:
            for key, value in self.info.items():
                f.write(f"{key}: {value}\n".encode('utf-8'))
            f.write(b'<>\n')

    def write_coords(self, positions, normals):
        assert positions.shape[0] == normals.shape[0], "Positions and normals must have the same number of entries"
        with open(self.filename, 'ab') as f:  # Append binary data
            for pos, norm in tqdm(zip(positions, normals), total=positions.shape[0], desc="Lines"):
                # Ensure the data is in double format
                buf = struct.pack('<' + 'd'*6, *pos, *norm)
                f.write(buf)

    def create_ppm(self, positions, normals):
        self.write_header()
        self.write_coords(positions, normals)