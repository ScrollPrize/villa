"""Shared README augmentation recipes and CT-derived sample construction."""
import ast
from dataclasses import dataclass
import re

import numpy as np


@dataclass
class Workload:
    name: str
    sample: str = "image"
    output: str = "image"
    channel: int = 0
    note: str = ""

def recipes(readme):
    entries = re.findall(r'<summary><strong>([^<]+)</strong></summary>.*?```python\n(.*?)\n```',
                         readme.read_text(), re.DOTALL)
    if not entries or len(dict(entries)) != len(entries):
        raise ValueError("README must contain uniquely named augmentation examples")
    return dict(entries)

def constructor(name, recipe, reference, augment):
    """Only A.<matching public constructor>(literal arguments, reference) is accepted."""
    call = ast.parse(recipe.strip(), mode="eval").body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "A"
            and call.func.attr == name and name in augment.__all__):
        raise ValueError(f"unsupported README recipe for {name}")

    def value(node):
        if isinstance(node, ast.Name) and node.id == "reference":
            return reference
        return ast.literal_eval(node)

    if any(kw.arg is None for kw in call.keywords):
        raise ValueError("keyword unpacking is not supported")
    return getattr(augment, name)(*(value(node) for node in call.args),
                                 **{kw.arg: value(kw.value) for kw in call.keywords})

def workload(name):
    if name in ("Morphology", "RemoveComponents"):
        return Workload(name, "labels", "mask", note="Foreground mask derived from CT > 0.5; not manual labels.")
    if name == "RandomBinaryOperator":
        return Workload(name, "binary", note="Binary input derived from CT > 0.5; not manual labels.")
    if name == "ChannelMisalignment":
        return Workload(name, "channels", channel=1, note="Second of two identical CT channels.")
    if name == "StripGeometry":
        return Workload(name, "strip", note="2D companion: central Z plane from the CT crop.")
    if name == "SurfacePerturb":
        return Workload(name, "surface", "surface", note="Synthetic coordinate surface with CT-derived height; not an anatomical surface.")
    return Workload(name, "both" if name in ("Affine", "CropPad", "Elastic", "Flip", "Rotate90", "Transpose",
                                             "ThickSlice", "SheetCompression", "ForegroundCrop", "ForegroundDropout", "ObjectDropout") else "image")

def planar_recipe(name, recipe):
    """Adapt the catalog's explicit 3D settings for a planar benchmark.

    Keep YX sizes/directions, map a Z-only operation to Y, and use a Y-aligned
    oriented blur. Return the actual recipe so reports record these choices.
    """
    call = ast.parse(recipe.strip(), mode='eval').body
    spatial = {'shape', 'size', 'unit_size', 'tile_shape', 'direction', 'center'}
    for kw in call.keywords:
        if kw.arg in spatial and isinstance(kw.value, (ast.Tuple, ast.List)) and len(kw.value.elts) == 3:
            kw.value.elts = kw.value.elts[-2:]
        elif kw.arg in ('axes', 'order') and isinstance(kw.value, ast.Constant):
            kw.value.value = kw.value.value.replace('z', '') or 'y'
        elif name == 'OrientedBlur' and kw.arg == 'axis':
            kw.value = ast.parse('(1, 0)', mode='eval').body
        elif name == 'Smear' and kw.arg == 'shift':
            kw.value.elts = kw.value.elts[:1]
    if name in ('SliceIllumination', 'Smear'):
        call.keywords.append(ast.keyword(arg='axis', value=ast.Constant('y')))
    return ast.unparse(call)

def inputs(image):
    labels = (image > .5).astype(np.int16)
    plane = image if image.ndim == 3 else np.ascontiguousarray(image[:, image.shape[1] // 2])
    y, x = np.indices(plane.shape[1:], dtype=np.float32)
    return dict(image=dict(image=image), both=dict(image=image, mask=labels),
                binary=dict(image=labels.astype(np.float32)), labels=dict(mask=labels),
                channels=dict(image=np.concatenate([image, image])), strip=dict(image=plane),
                surface=dict(surface=np.stack([plane[0] * 2, y, x])))
