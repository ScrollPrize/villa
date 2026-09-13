"""
OBJ Reader/Writer by James Gregson
Modified by Seth Parker
http://jamesgregson.ca/loadsave-wavefront-obj-files-in-python.html
"""
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolygon


class WavefrontOBJ:
    def __init__(self):
        self.path = None  # path of loaded object
        self.mtllibs = []  # .mtl files references via mtllib
        self.mtls = []  # materials referenced
        self.mtlid = []  # indices into self.mtls for each polygon
        self.vertices = []  # vertices as a Nx3 or Nx6 array (per vtx colors)
        self.normals = []  # normals
        self.texcoords = []  # texture coordinates
        self.polygons = []  # M*Nv*3 array, Nv=# of vertices, stored as vid,tid,nid (None for N/A)


def load_mtllib(filename: Union[str, Path]) -> Dict:
    with Path(filename).open('r') as mtlf:
        mtl = {}
        current_mtl = None
        current_data = None
        for line in mtlf:
            # Remove everything after comment delimiter
            line = line.partition('#')[0]
            # Split on whitespace
            toks = line.split()
            if not toks:
                continue
            if toks[0] == 'newmtl':
                # Save parsed mtl data to the final dict
                if current_mtl is not None:
                    mtl[current_mtl] = current_data
                # Set up a new mtl
                current_mtl = toks[1]
                current_data = {}
            else:
                if len(toks[1:]) == 1:
                    current_data[toks[0]] = toks[1]
                else:
                    current_data[toks[0]] = toks[1:]

        # Save parsed mtl data to the final dict
        if current_mtl is not None:
            mtl[current_mtl] = current_data

        return mtl


# This duplicates an MTL into a new file. The input mtl file is iterated,
# external texture images are copied to new file names derived from the output
# file name, and the mtl file is updated to point to the new files
def duplicate_mtllib(in_path, out_path, textures=None):
    num_kd = 0
    # Open both files
    with in_path.open('r') as in_file, out_path.open('w') as out_file:
        # Copy each line from in to out
        for line in in_file:
            # Handle map_Kd lines
            if line.startswith('map_Kd'):
                # Get the image path
                line_prefix, line_suffix = line.strip().rsplit(' ', 1)
                kd_src = Path(line_suffix)

                # Append an index if we've already seen a map_Kd line
                if num_kd > 0:
                    kd_dst = out_path.stem + f'{num_kd}{kd_src.suffix}'

                # The 1st map_Kd path is the same stem as the output mtl
                else:
                    kd_dst = out_path.with_suffix(kd_src.suffix).name

                # Spaces in the filename aren't allowed
                kd_dst = kd_dst.replace(' ', '_')

                # Write to the output and copy the kd file
                out_file.write(f'{line_prefix} {kd_dst}\n')
                if textures is not None and kd_src.name in textures.keys():
                    cv2.imwrite(str(out_path.parent / kd_dst),
                                textures[kd_src.name])
                else:
                    shutil.copy(in_path.parent / kd_src,
                                out_path.parent / kd_dst)
                num_kd += 1

            # Handle every other line
            else:
                out_file.write(line)

    if num_kd > 1:
        print(f'Warning: Found {num_kd} map_Kd lines in {str(in_path)}')


def load_obj(filename: Union[str, Path], triangulate=False) -> WavefrontOBJ:
    """Reads a .obj file from disk and returns a WavefrontOBJ instance

    Handles only very rudimentary reading and contains no error handling!

    Does not handle:
        - relative indexing
        - subobjects or groups
        - lines, splines, beziers, etc.
    """

    # parses a vertex record as either vid, vid/tid, vid//nid or vid/tid/nid
    # and returns a 3-tuple where unparsed values are replaced with None
    def parse_vertex(vstr):
        vals = vstr.split('/')
        vid = int(vals[0]) - 1
        tid = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
        nid = int(vals[2]) - 1 if len(vals) > 2 else None
        return vid, tid, nid

    with Path(filename).open('r') as objf:
        obj = WavefrontOBJ()
        obj.path = Path(filename)
        cur_mat = None
        for line in objf:
            toks = line.split()
            if not toks:
                continue
            if toks[0] == 'v':
                obj.vertices.append([float(v) for v in toks[1:]])
            elif toks[0] == 'vn':
                obj.normals.append([float(v) for v in toks[1:]])
            elif toks[0] == 'vt':
                # transform to be relative to bottom-left origin
                uv = [float(v) for v in toks[1:]]
                uv[1] = abs(uv[1] - 1.)
                obj.texcoords.append(uv)
            elif toks[0] == 'f':
                poly = [parse_vertex(vstr) for vstr in toks[1:]]
                if triangulate:
                    for i in range(2, len(poly)):
                        obj.mtlid.append(cur_mat)
                        obj.polygons.append((poly[0], poly[i - 1], poly[i]))
                else:
                    obj.mtlid.append(cur_mat)
                    obj.polygons.append(poly)
            elif toks[0] == 'mtllib':
                obj.mtllibs.append(toks[1])
            elif toks[0] == 'usemtl':
                if toks[1] not in obj.mtls:
                    obj.mtls.append(toks[1])
                cur_mat = obj.mtls.index(toks[1])
        return obj


def save_obj(obj: WavefrontOBJ, filename: Union[str, Path], _prec='.7f',
             _unique_mtl=True, _textures=None):
    """Saves a WavefrontOBJ object to a file

    @param obj: WavefrontOBJ mesh
    @param filename: Output location
    @param _prec: Format string for float to string formatting. Meant to
    control decimal precision.
    @param _unique_mtl: If True, mtl files and texture images in obj will be
    rewritten using a named derived from the output filename. If False, the obj
    file will reference the original mtl and texture files.
    @param _textures: Optional dictionary for controlling texture image
    replacement: {str: np.ndarray, ...}. When copying mtl files, if a map_Kd
    value matches one of the dictionary's keys, the corresponding np.ndarray
    (presumably a texture image) will be written to disk instead of copying the
    original texture image.
    """
    filename = Path(filename)

    # Duplicate mtl's and images to the new path
    if _unique_mtl:
        pad = len(str(len(obj.mtllibs)))
        for idx, mtl_file in enumerate(obj.mtllibs):
            # Input file
            mtl_in = obj.path.parent / mtl_file
            if not mtl_in.exists():
                print(f'Warning: Could not duplicate .mtl file {mtl_file}. '
                      f'Source file does not exist.')
                continue

            # Output file name
            mtl_out = filename.stem.replace(' ', '_')
            if len(obj.mtllibs) > 1:
                mtl_out += f'{idx:0{pad}}'
            mtl_out += '.mtl'
            # Assign to OBJ
            obj.mtllibs[idx] = mtl_out
            # Output file path
            mtl_out = filename.parent / mtl_out
            # Copy the mtl
            duplicate_mtllib(mtl_in, mtl_out, textures=_textures)

    with filename.open('w') as ofile:
        ofile.write(f'# Exported by: Gregson OBJ IO (EduceLab pgs-recon)\n')
        for mlib in obj.mtllibs:
            ofile.write(f'mtllib {mlib}\n')
        for vtx in obj.vertices:
            ofile.write(f'v {" ".join([f"{v:{_prec}}" for v in vtx])}\n')
        for tex in obj.texcoords:
            # transform to be relative to bottom-left origin
            tex[1] = abs(tex[1] - 1.)
            ofile.write(f'vt {" ".join([f"{vt:{_prec}}" for vt in tex])}\n')
        for nrm in obj.normals:
            ofile.write(f'vn {" ".join([f"{vn:{_prec}}" for vn in nrm])}\n')
        if not obj.mtlid:
            obj.mtlid = [None] * len(obj.polygons)

        # Sort polygon indices based on their material assignments
        if any(x is not None for x in obj.mtlid):
            poly_idx = [-np.Inf if x is None else x for x in obj.mtlid]
            poly_idx = np.argsort(poly_idx, kind='stable').tolist()
        else:
            poly_idx = range(len(obj.mtlid))

        # Iterate over sorted polygons
        cur_mat = None
        have_nrm = len(obj.normals) > 0
        have_uvs = len(obj.texcoords) > 0
        for pid in poly_idx:
            if obj.mtlid[pid] != cur_mat:
                cur_mat = obj.mtlid[pid]
                ofile.write(f'usemtl {obj.mtls[cur_mat]}\n')
            pstr = ['f ']
            for v in obj.polygons[pid]:
                p = v[0] + 1
                t = v[1] + 1 if have_uvs and v[1] is not None else ''
                n = v[2] + 1 if have_nrm and v[2] is not None else ''
                vstr = f'{p}/{t}/{n} '
                vstr = vstr.replace('/ ', ' ').replace('/ ', ' ')
                pstr.append(vstr)
            ofile.write(f'{"".join(pstr)}\n')


# Merge data from a into b, replacing missing entries in b
# If copy is True, returns an updated deepcopy of b. Otherwise, updates and
# returns b
def merge_meshes(a: WavefrontOBJ, b: WavefrontOBJ, copy: bool = False):
    if copy:
        tgt = deepcopy(b)
    else:
        tgt = b

    tgt.path = a.path if tgt.path is None else tgt.path
    tgt.mtllibs = a.mtllibs if len(tgt.mtllibs) == 0 else tgt.mtllibs
    tgt.mtls = a.mtls if len(tgt.mtls) == 0 else tgt.mtls
    tgt.mtlid = a.mtlid if len(tgt.mtlid) == 0 else tgt.mtlid
    tgt.vertices = a.vertices if len(tgt.vertices) == 0 else tgt.vertices
    tgt.normals = a.normals if len(tgt.normals) == 0 else tgt.normals
    tgt.texcoords = a.texcoords if len(tgt.texcoords) == 0 else tgt.texcoords

    if len(tgt.polygons) == 0:
        tgt.polygons = a.polygons
    elif len(tgt.polygons) == len(a.polygons):
        for idx, (src_f, tgt_f) in enumerate(zip(a.polygons, tgt.polygons)):
            new_f = list()
            for src_v, tgt_v in zip(src_f, tgt_f):
                new_v = tuple(
                    s if t is None else t for s, t in zip(src_v, tgt_v))
                new_f.append(new_v)
            tgt.polygons[idx] = new_f
    return tgt


# Convert a WavefrontOBJ to vtkPolyData
# Note: Only transfers vertices, faces, and vertex normals
def mesh_to_polydata(obj: WavefrontOBJ):
    polydata = vtkPolyData()

    # Vertices
    pts = vtkPoints()
    for v in obj.vertices:
        pts.InsertNextPoint(v[:3])
    polydata.SetPoints(pts)

    # Faces
    polys = vtkCellArray()
    normals = vtkFloatArray()
    normals.SetNumberOfComponents(3)
    normals.SetName('Normals')
    for p in obj.polygons:
        poly = vtkPolygon()
        for v in p:
            poly.GetPointIds().InsertNextId(v[0])
            # Add normal if we've got one
            if v[2] is not None:
                n = obj.normals[v[2]]
                normals.InsertTuple(v[2], n)
        polys.InsertNextCell(poly)
    polydata.SetPolys(polys)
    if normals.GetNumberOfTuples() > 0:
        polydata.GetPointData().SetNormals(normals)
    return polydata


# Convert a vtkPolyData to a WavefrontOBJ
# Note: Only transfers vertices, faces, and vertex normals
def polydata_to_mesh(polydata: vtkPolyData, src_mesh: WavefrontOBJ = None):
    obj = WavefrontOBJ()
    for idx in range(polydata.GetNumberOfPoints()):
        obj.vertices.append(list(polydata.GetPoints().GetPoint(idx)))

    have_normals = polydata.GetPointData().HasArray('Normals') != 0
    if have_normals:
        normals = polydata.GetPointData().GetArray('Normals')
        for idx in range(normals.GetNumberOfTuples()):
            obj.normals.append(list(normals.GetTuple(idx)))

    cells = polydata.GetPolys()
    id_list = vtkIdList()
    cells.InitTraversal()
    while cells.GetNextCell(id_list):
        cell_list = list()
        for i in range(id_list.GetNumberOfIds()):
            vid = id_list.GetId(i)
            if have_normals:
                cell_list.append((vid, None, vid))
            else:
                cell_list.append((vid, None, None))
        obj.polygons.append(cell_list)

    # copy missing items from src_mesh to output obj
    if src_mesh is not None:
        obj = merge_meshes(src_mesh, obj)

    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True, help='Input OBJ')
    parser.add_argument('--output-file', '-o', help='Output OBJ')
    args = parser.parse_args()

    print(f'Loading mesh: {args.input_file}')
    mesh = load_obj(args.input_file)

    print(f'Vertices: {len(mesh.vertices)}')
    print(f'Normals: {len(mesh.normals)}')
    print(f'Faces: {len(mesh.polygons)}')
    print(f'UV coordinates: {len(mesh.texcoords)}')
    print(f'Material files: {mesh.mtllibs}')
    print(f'Materials used: {mesh.mtls}')

    if args.output_file is not None:
        print(f'Saving mesh: {args.output_file}')
        save_obj(mesh, args.output_file)


if __name__ == '__main__':
    main()
