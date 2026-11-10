"""Regenerate the small SLD parity fixture from a downloaded, pinned Nabuxx source.

Run: .venv/bin/python tests/reference/generate_sld_bm18.py /tmp/diffusion_digest.py
Source: https://gitlab.esrf.fr/night_rail/applications/tools/nabuxx/-/raw/
515e0df11cf34f669ae1cf918f219a8454c90e1c/nabuxx/app/diffusion_digest.py

Runs the upstream kernel and deconvolution functions, with NumPy replacing only
the PyFFTW transforms/plans. No upstream installation, HDF5 access or GPU needed.
The fixture is a numerical algorithm reference, not a complete Night Rail run.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def generate(source):
    code = Path(source).read_text()
    digest = hashlib.sha256(code.encode()).hexdigest()
    if digest != "a3f81ca9aded9abb0bf1bbd7ee9fa7c66c61877e4b9604b97d8e5c1bb91cd733":
        raise ValueError("Source differs from the pinned Nabuxx file; review it before updating the reference")
    names = {"_as_tuple", "_normalize_diffusion_terms", "_normalize_replica_kernel_params", "_replica_is_active",
             "_add_shifted_replica_fft_layout", "get_kernel_fft", "deconvolve_simple", "next_smooth_size"}
    functions = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Not the expected Nabuxx diffusion source")

    def context(dims):
        inp = np.empty(dims, np.complex64)
        out = np.empty(dims, np.complex64)

        def forward():
            out[:] = np.fft.fftn(inp)

        def inverse(**_):
            inp[:] = np.fft.ifftn(out)
            return inp

        return dict(in_buf=inp, out_buf=out, plan_fwd=forward, plan_inv=inverse)

    fft = SimpleNamespace(fftn=lambda a, **_: np.fft.fftn(a))
    ns = dict(np=np, math=__import__("math"), _kernel_mem={}, _get_fft2_complex_ctx=context,
              pyfftw=SimpleNamespace(interfaces=SimpleNamespace(numpy_fft=fft)))
    exec(compile(ast.Module(body=functions, type_ignores=[]), source, "exec"), ns)
    y, x = np.mgrid[:17, :21]
    flat = (1 + .4 * np.exp(-((y - 8) / 5) ** 2) + .1 * x / 21).astype(np.float32)
    radio = flat * (1 - .7 * np.exp(-((x - 7) ** 2 + (y - 9) ** 2) / 14))
    radio[:, :3] *= .4
    impulse = np.zeros_like(flat)
    impulse[8, 10] = 1
    frames = np.stack([radio, flat, impulse])
    cases = [dict(fraction=[.12, .08], scale_l=[1.5, 4.], border_x=3, border_y=2),
             dict(fraction=[.6], scale_l=[20.], border_x=0, border_y=0),
             dict(fraction=[.03, 1.2], scale_l=[.001, 12.], border_x=5, border_y=0)]
    payload = dict(source_commit="515e0df11cf34f669ae1cf918f219a8454c90e1c",
                   source_sha256=digest, fft_backend="numpy (complex64 buffers)",
                   frames=frames.tolist(), cases=[])
    for case in cases:
        expected = [ns["deconvolve_simple"](frame, fraction=case["fraction"], scale_l=case["scale_l"],
                    mask_border=case["border_x"], mask_border_v=case["border_y"]) for frame in frames]
        payload["cases"].append(dict(model=case, corrected=np.stack(expected).tolist()))
    target = Path(__file__).with_name("sld_bm18.json")
    target.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(target)


if __name__ == "__main__":
    generate(sys.argv[1])
