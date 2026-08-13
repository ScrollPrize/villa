# Copyright 2023 by kglspl, MIT (see LICENSE)
# Adjusted by Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023, MIT (see LICENSE)

import os
import struct

import numpy as np

class PPMParser(object):

    def __init__(self, filename, step=None):
        self.filename = filename
        self.step = step

    def open(self):
        print('Opening {}'.format(self.filename))
        self.f = open(self.filename, 'rb')
        self.info, self.header_size, self.header_content = PPMParser.vcps_parse_header(self.f)
        return self

    def close (self):
        print('Closing file.')
        self.f.close()

    def __enter__ (self):
        return self

    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()

    def im_shape(self):
        # output image depends on step:
        return (self._shrunk_dim(self.info['width']), self._shrunk_dim(self.info['height']))

    def _shrunk_dim(self, x):
        if self.step is None:
            return x
        else:
            return x // self.step if x % self.step == 0 else x // self.step + 1

    def im_zeros(self, dtype):
        # allocates exactly the size that is needed for resulting image, taking step into account:
        shape = self.im_shape()
        a = np.zeros(shape, dtype=dtype)
        return a

    @staticmethod
    def vcps_parse_header(f):
        info = {}
        header_size = 0
        header_content = b''
        while True:
            l_bytes = f.readline()
            header_size += len(l_bytes)
            header_content += l_bytes
            l = l_bytes.decode('utf-8').rstrip("\n")
            if l == '<>':
                break
            k, v = l.split(': ', 1)
            if v.isnumeric():
                v = int(v)
            info[k] = v
        return info, header_size, header_content

    def read_next_coords(self, step_empty=True):
        f = self.f
        im_width = self.info['width']
        step = self.step
        n = -1
        while True:
            n += 1

            buf = f.read(6*8)
            if not buf:
                break
            x, y, z, nx, ny, nz = struct.unpack('<dddddd', buf)

            if step_empty and int(x) == 0:
                continue

            imx, imy = n % im_width, n // im_width

            if step is not None:
                # step most of the data and adjust image coordinates if we are using step:
                if imx % step or imy % step:
                    continue

                yield imx // step, imy // step, x, y, z, nx, ny, nz, buf
            else:
                yield imx, imy, x, y, z, nx, ny, nz, buf

    def get_3d_coords(self, imx, imy):
        f = self.f
        im_width = self.info['width']
        pos = self.header_size + (imy * im_width + imx) * 6 * 8
        #print(f"Seeking to: {pos}")
        f.seek(pos, os.SEEK_SET)

        buf = f.read(6*8)
        if not buf:
            return None, None, None, None, None, None
        x, y, z, nx, ny, nz = struct.unpack('<dddddd', buf)
        return x, y, z, nx, ny, nz
    
    def classify_entries_to_cubes(self, cube_size):
        """
        Classifies entries into cubes based on their 3D coordinates.

        :param cube_size: Size of each cube in the grid (assuming cubic grid).
        :return: A dictionary with cube coordinates as keys and list of entries as values.
        """
        cubes = {}
        for imx, imy, x, y, z, nx, ny, nz, buf in self.read_next_coords():
            cube_coord = tuple((int(x // cube_size), int(y // cube_size), int(z // cube_size)))
            if cube_coord not in cubes:
                cubes[cube_coord] = []
            cubes[cube_coord].append((int(imx), int(imy), buf))
        return cubes
