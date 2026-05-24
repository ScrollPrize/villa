import contextlib
import io
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "registration_pipe.py"
REGISTRATION_MODULE_PATH = MODULE_DIR / "registration.py"


class _AssignedTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def device(self):
        return types.SimpleNamespace(type="cpu")

    def __add__(self, other):
        if isinstance(other, _AssignedTensor):
            other = other.value
        return _AssignedTensor(self.value + other)


def _load_registration_module():
    module_name = "_test_registration"
    sys.modules.pop(module_name, None)

    class _Device:
        def __init__(self, name):
            self.type = name

        def __str__(self):
            return self.type

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _AssignedTensor
    torch_stub.float32 = "float32"
    torch_stub.long = "long"
    torch_stub.manual_seed = lambda _seed: None
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda name: _Device(name)
    torch_stub.tensor = lambda value, **_kwargs: _AssignedTensor(value)
    torch_stub.empty = lambda shape, **_kwargs: _AssignedTensor(np.empty(shape, dtype=np.float32))
    torch_stub.zeros_like = lambda value, **_kwargs: _AssignedTensor(np.zeros_like(value.value, dtype=np.float32))
    torch_stub.optim = types.SimpleNamespace(
        AdamW=lambda *_args, **_kwargs: types.SimpleNamespace(
            zero_grad=lambda: None,
        ),
        lr_scheduler=types.SimpleNamespace(
            StepLR=lambda *_args, **_kwargs: types.SimpleNamespace(step=lambda: None),
        ),
    )
    torch_stub.amp = types.SimpleNamespace(
        GradScaler=lambda **_kwargs: types.SimpleNamespace(
            scale=lambda value: types.SimpleNamespace(backward=lambda: None),
            step=lambda _optimizer: None,
            update=lambda: None,
        ),
        autocast=lambda **_kwargs: mock.MagicMock(),
    )

    torch_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_nn_stub.functional = torch_functional_stub

    open3d_stub = types.ModuleType("open3d")
    open3d_stub.geometry = types.SimpleNamespace(LineSet=object, PointCloud=object)
    tqdm_stub = types.ModuleType("tqdm")

    class _Tqdm:
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def close(self):
            return None

        def set_postfix(self, *_args, **_kwargs):
            return None

        def write(self, *_args, **_kwargs):
            return None

    tqdm_stub.tqdm = _Tqdm

    spec = importlib.util.spec_from_file_location(module_name, REGISTRATION_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "torch.nn": torch_nn_stub,
            "torch.nn.functional": torch_functional_stub,
            "open3d": open3d_stub,
            "tqdm": tqdm_stub,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _load_registration_pipe():
    module_name = "_test_registration_pipe"
    sys.modules.pop(module_name, None)

    class _Geometry:
        def __init__(self):
            self.points = None
            self.lines = None
            self.colors = None

        def paint_uniform_color(self, _color):
            return None

    open3d_stub = types.ModuleType("open3d")
    open3d_stub.geometry = types.SimpleNamespace(
        PointCloud=_Geometry,
        LineSet=_Geometry,
        TriangleMesh=_Geometry,
    )
    open3d_stub.utility = types.SimpleNamespace(
        Vector3dVector=lambda value: value,
        Vector2iVector=lambda value: value,
    )
    open3d_stub.io = types.SimpleNamespace(
        read_triangle_mesh=lambda *_args, **_kwargs: None,
        write_triangle_mesh=lambda *_args, **_kwargs: True,
    )
    open3d_stub.visualization = types.SimpleNamespace(
        draw_geometries=lambda *_args, **_kwargs: None,
    )

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.device = lambda name: name
    torch_stub.float32 = "float32"
    torch_stub.tensor = lambda value, **_kwargs: np.asarray(value, dtype=np.float32)

    extract_stub = types.ModuleType("extract_skeleton_tif")
    extract_stub.extract_skeleton_from_tif = lambda *_args, **_kwargs: {}

    registration_stub = types.ModuleType("registration")
    registration_stub.MeshSurface = object
    registration_stub.optimize_all_registration = lambda *_args, **_kwargs: []
    registration_stub.assign_skeletons_to_mesh = lambda *_args, **_kwargs: None

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "open3d": open3d_stub,
            "torch": torch_stub,
            "extract_skeleton_tif": extract_stub,
            "registration": registration_stub,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class RegistrationPipeScaleControlsTest(unittest.TestCase):
    def test_parse_mesh_label_and_fiber_accepts_only_supported_fiber_suffixes(self):
        registration_pipe = _load_registration_pipe()

        self.assertEqual(registration_pipe.parse_mesh_label_and_fiber("12_hz.obj"), (12, "hz"))
        self.assertEqual(registration_pipe.parse_mesh_label_and_fiber("nested/34_vt.obj"), (34, "vt"))

        for mesh_file in ["12_diag.obj", "bad.obj", "12.obj", "12_hz_extra.obj"]:
            with self.subTest(mesh_file=mesh_file):
                with self.assertRaisesRegex(ValueError, "<label>_<hz\\|vt>\\.obj"):
                    registration_pipe.parse_mesh_label_and_fiber(mesh_file)

    def test_curve_type_for_fiber_rejects_unknown_fiber_type_before_auto_mapping(self):
        registration_pipe = _load_registration_pipe()

        with self.assertRaisesRegex(ValueError, "fiber_type must be 'hz' or 'vt'"):
            registration_pipe.curve_type_for_fiber("diag", "auto")

    def test_process_one_mesh_rejects_invalid_mesh_filename_before_running_pipeline(self):
        registration_pipe = _load_registration_pipe()
        global_args = types.SimpleNamespace(
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=0.0,
            lambda_self=0.0,
            lambda_lap=0.0,
            lambda_arap=0.0,
            lambda_sdf=0.0,
            batch_size=1,
            tau=1.0,
            delta=1.0,
            beta=1.0,
            tau_sdf=1.0,
            lambda_inter=0.0,
            delta_inter=1.0,
            beta_inter=1.0,
            target_triangles=100,
            skel_origin="0,0,0",
            skel_axis="xyz",
            skeleton_axis_order="zyx",
            curve_type="auto",
            target_skel_points=0,
            visualize=False,
            mesh_root="meshes",
        )

        with mock.patch.object(registration_pipe, "unified_pipeline", return_value=[]) as unified_pipeline:
            with self.assertRaisesRegex(ValueError, "<label>_<hz\\|vt>\\.obj"):
                registration_pipe.process_one_mesh(
                    "meshes/cube/7_diag.obj",
                    "fiber.tif",
                    "mask.tif",
                    "registered",
                    global_args,
                )

        unified_pipeline.assert_not_called()

    def test_optimizer_preserves_preassigned_mesh_skeleton_curves(self):
        registration = _load_registration_module()

        mesh_a_curve = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mesh_b_curve = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        mesh_a = types.SimpleNamespace(
            vertices_np=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            skeleton_curves=[mesh_a_curve],
            vertices=_AssignedTensor([[0.0, 0.0, 0.0]]),
            displacement=_AssignedTensor([[0.0, 0.0, 0.0]]),
        )
        mesh_b = types.SimpleNamespace(
            vertices_np=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            skeleton_curves=[mesh_b_curve],
            vertices=_AssignedTensor([[0.0, 0.0, 0.0]]),
            displacement=_AssignedTensor([[0.0, 0.0, 0.0]]),
        )

        registration.optimize_all_registration(
            [mesh_a, mesh_b],
            [mesh_a_curve, mesh_b_curve],
            num_iters=0,
        )

        np.testing.assert_allclose(mesh_a.skeleton_points.value, mesh_a_curve)
        np.testing.assert_allclose(mesh_b.skeleton_points.value, mesh_b_curve)

    def test_optimizer_rejects_explicit_empty_mesh_skeleton_curves(self):
        registration = _load_registration_module()

        global_curve = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        mesh = types.SimpleNamespace(
            vertices_np=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            skeleton_curves=[],
            vertices=_AssignedTensor([[0.0, 0.0, 0.0]]),
            displacement=_AssignedTensor([[0.0, 0.0, 0.0]]),
        )

        with self.assertRaisesRegex(ValueError, "mesh has no skeleton curves"):
            registration.optimize_all_registration(
                [mesh],
                [global_curve],
                num_iters=0,
            )

    def test_prepare_skeleton_curves_filters_reorders_and_simplifies(self):
        registration_pipe = _load_registration_pipe()

        vertical_curve = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 6.0, 9.0],
                [4.0, 8.0, 12.0],
            ],
            dtype=np.float32,
        )
        horizontal_curve = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 22.0, 33.0],
                [12.0, 24.0, 36.0],
            ],
            dtype=np.float32,
        )

        prepared = registration_pipe.prepare_skeleton_curves(
            {
                "vertical": [vertical_curve],
                "horizontal": [horizontal_curve],
            },
            curve_type="vertical",
            origin=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            skel_axis="zyx",
            target_points=2,
        )

        self.assertEqual(list(prepared.keys()), ["vertical"])
        self.assertEqual(len(prepared["vertical"]), 1)
        np.testing.assert_allclose(
            prepared["vertical"][0],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [9.0, 6.0, 3.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_parser_defaults_keep_skeleton_classification_and_registration_axes_consistent(self):
        registration_pipe = _load_registration_pipe()
        captured = {}
        argv = [
            "registration_pipe.py",
            "--mesh_root",
            "meshes",
            "--tif_root",
            "tifs",
            "--cube_label_root",
            "labels",
            "--output_root",
            "registered",
        ]

        def fake_process_group(_mesh_files, _tif_file, _cube_label_file, _output_root, args):
            captured["skel_axis"] = args.skel_axis
            captured["skeleton_axis_order"] = args.skeleton_axis_order

        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(registration_pipe.os, "walk", return_value=[("meshes/cube_a", [], ["1_hz.obj"])]), \
             mock.patch.object(registration_pipe.os.path, "isfile", return_value=True), \
             mock.patch.object(registration_pipe, "process_mesh_group", side_effect=fake_process_group):
            registration_pipe.main()

        self.assertEqual(captured["skeleton_axis_order"], "zyx")
        self.assertEqual(captured["skel_axis"], "zyx")

    def test_unified_pipeline_passes_only_selected_simplified_curves_to_optimizer(self):
        registration_pipe = _load_registration_pipe()

        vertical_curve = np.stack(
            [
                np.arange(5, dtype=np.float32),
                np.arange(5, dtype=np.float32) + 10.0,
                np.arange(5, dtype=np.float32) + 20.0,
            ],
            axis=1,
        )
        horizontal_curve = np.stack(
            [
                np.arange(4, dtype=np.float32) + 100.0,
                np.arange(4, dtype=np.float32) + 200.0,
                np.arange(4, dtype=np.float32) + 300.0,
            ],
            axis=1,
        )
        captured = {}

        class _ToDevice:
            def to(self, _device):
                return self

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMesh:
            def __init__(self):
                self.vertices_np = np.zeros((3, 3), dtype=np.float32)
                self.displacement = _TensorLike(np.zeros((3, 3), dtype=np.float32))
                self.vertices = _TensorLike(np.zeros((3, 3), dtype=np.float32))
                self.skeleton_points = None

        def fake_optimize(meshes, global_skel_curves, **_kwargs):
            captured["curves"] = global_skel_curves
            return []

        args = types.SimpleNamespace(
            mesh="42_hz.obj",
            tif="fiber.tif",
            cube_label="mask.tif",
            skel_origin="0,0,0",
            skel_axis="xyz",
            curve_type="horizontal",
            target_skel_points=2,
            visualize=False,
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            batch_size=8,
        )

        with mock.patch.object(registration_pipe, "load_mesh_from_file", return_value=[_FakeMesh()]), \
             mock.patch.object(
                 registration_pipe,
                 "extract_skeleton_from_tif",
                 return_value={"vertical": [vertical_curve], "horizontal": [horizontal_curve]},
             ), \
             mock.patch.object(registration_pipe, "assign_skeletons_to_mesh", return_value=_ToDevice()), \
             mock.patch.object(registration_pipe, "optimize_all_registration", side_effect=fake_optimize):
            registration_pipe.unified_pipeline(args)

        self.assertEqual(len(captured["curves"]), 1)
        np.testing.assert_allclose(
            captured["curves"][0],
            np.array(
                [
                    [100.0, 200.0, 300.0],
                    [103.0, 203.0, 303.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_unified_pipeline_does_not_print_raw_displacement_objects(self):
        registration_pipe = _load_registration_pipe()

        class _ToDevice:
            def to(self, _device):
                return self

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def __repr__(self):
                return "DISPLACEMENT_DEBUG_SENTINEL"

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMesh:
            def __init__(self):
                self.vertices_np = np.zeros((1, 3), dtype=np.float32)
                self.vertices = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.displacement = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.skeleton_points = None

        def fake_optimize(meshes, _global_skel_curves, **_kwargs):
            return [mesh.displacement for mesh in meshes]

        args = types.SimpleNamespace(
            mesh="42_hz.obj",
            tif="fiber.tif",
            cube_label="mask.tif",
            skel_origin="0,0,0",
            skel_axis="xyz",
            curve_type="horizontal",
            target_skel_points=0,
            visualize=False,
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            batch_size=8,
        )

        curve = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        stdout = io.StringIO()
        with mock.patch.object(registration_pipe, "load_mesh_from_file", return_value=[_FakeMesh()]), \
             mock.patch.object(
                 registration_pipe,
                 "extract_skeleton_from_tif",
                 return_value={"vertical": [], "horizontal": [curve]},
             ), \
             mock.patch.object(registration_pipe, "assign_skeletons_to_mesh", return_value=_ToDevice()), \
             mock.patch.object(registration_pipe, "optimize_all_registration", side_effect=fake_optimize), \
             contextlib.redirect_stdout(stdout):
            registration_pipe.unified_pipeline(args)

        self.assertNotIn("DISPLACEMENT_DEBUG_SENTINEL", stdout.getvalue())

    def test_unified_pipeline_registers_multiple_meshes_together(self):
        registration_pipe = _load_registration_pipe()

        class _ToDevice:
            def to(self, _device):
                return self

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMesh:
            def __init__(self, z):
                self.vertices_np = np.array([[0.0, 0.0, z]], dtype=np.float32)
                self.displacement = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.vertices = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.skeleton_points = None

        captured = {"labels": []}

        def fake_load_mesh(mesh_file):
            z = 1.0 if mesh_file == "1_hz.obj" else 2.0
            return [_FakeMesh(z)]

        def fake_extract(_tif, _cube_label, **kwargs):
            label = kwargs["label"]
            captured["labels"].append(label)
            curve = np.array(
                [
                    [float(label), 0.0, 0.0],
                    [float(label), 1.0, 0.0],
                    [float(label), 2.0, 0.0],
                ],
                dtype=np.float32,
            )
            return {"vertical": [], "horizontal": [curve]}

        def fake_optimize(meshes, global_skel_curves, **_kwargs):
            captured["mesh_count"] = len(meshes)
            captured["curve_count"] = len(global_skel_curves)
            captured["mesh_curves"] = [mesh.skeleton_curves for mesh in meshes]
            return []

        args = types.SimpleNamespace(
            meshes=["1_hz.obj", "2_hz.obj"],
            tif="fiber.tif",
            cube_label="mask.tif",
            skel_origin="0,0,0",
            skel_axis="xyz",
            curve_type="horizontal",
            target_skel_points=2,
            visualize=False,
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            batch_size=8,
        )

        with mock.patch.object(registration_pipe, "load_mesh_from_file", side_effect=fake_load_mesh), \
             mock.patch.object(registration_pipe, "extract_skeleton_from_tif", side_effect=fake_extract), \
             mock.patch.object(registration_pipe, "assign_skeletons_to_mesh", return_value=_ToDevice()), \
             mock.patch.object(registration_pipe, "optimize_all_registration", side_effect=fake_optimize):
            registration_pipe.unified_pipeline(args)

        self.assertEqual(captured["labels"], [1, 2])
        self.assertEqual(captured["mesh_count"], 2)
        self.assertEqual(captured["curve_count"], 2)
        np.testing.assert_allclose(captured["mesh_curves"][0][0], np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(captured["mesh_curves"][1][0], np.array([[2.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32))

    def test_unified_pipeline_skips_meshes_without_matching_curves(self):
        registration_pipe = _load_registration_pipe()

        class _ToDevice:
            def to(self, _device):
                return self

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMesh:
            def __init__(self):
                self.vertices_np = np.zeros((1, 3), dtype=np.float32)
                self.displacement = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.vertices = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.skeleton_points = None

        captured = {}

        def fake_extract(_tif, _cube_label, **kwargs):
            if kwargs["label"] == 1:
                return {
                    "vertical": [],
                    "horizontal": [np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)],
                }
            return {"vertical": [], "horizontal": []}

        def fake_optimize(meshes, global_skel_curves, **_kwargs):
            captured["mesh_count"] = len(meshes)
            captured["curve_count"] = len(global_skel_curves)
            captured["sources"] = [mesh._source_mesh_file for mesh in meshes]
            return []

        args = types.SimpleNamespace(
            meshes=["1_hz.obj", "2_hz.obj"],
            tif="fiber.tif",
            cube_label="mask.tif",
            skel_origin="0,0,0",
            skel_axis="xyz",
            skeleton_axis_order="zyx",
            curve_type="horizontal",
            target_skel_points=0,
            visualize=False,
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            batch_size=8,
        )

        with mock.patch.object(registration_pipe, "load_mesh_from_file", side_effect=lambda _path: [_FakeMesh()]), \
             mock.patch.object(registration_pipe, "extract_skeleton_from_tif", side_effect=fake_extract), \
             mock.patch.object(registration_pipe, "assign_skeletons_to_mesh", return_value=_ToDevice()), \
             mock.patch.object(registration_pipe, "optimize_all_registration", side_effect=fake_optimize):
            registration_pipe.unified_pipeline(args)

        self.assertEqual(captured["mesh_count"], 1)
        self.assertEqual(captured["curve_count"], 1)
        self.assertEqual(captured["sources"], ["1_hz.obj"])

    def test_unified_pipeline_auto_curve_type_uses_mesh_fiber_type_and_axis_order(self):
        registration_pipe = _load_registration_pipe()

        class _ToDevice:
            def to(self, _device):
                return self

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMesh:
            def __init__(self):
                self.vertices_np = np.zeros((1, 3), dtype=np.float32)
                self.displacement = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.vertices = _TensorLike(np.zeros((1, 3), dtype=np.float32))
                self.skeleton_points = None

        captured = {"extract_calls": []}

        def fake_extract(_tif, _cube_label, **kwargs):
            captured["extract_calls"].append(kwargs)
            label = kwargs["label"]
            return {
                "vertical": [np.array([[10.0 + label, 0.0, 0.0], [10.0 + label, 1.0, 0.0]], dtype=np.float32)],
                "horizontal": [np.array([[20.0 + label, 0.0, 0.0], [20.0 + label, 1.0, 0.0]], dtype=np.float32)],
            }

        def fake_optimize(meshes, global_skel_curves, **_kwargs):
            captured["mesh_curves"] = [mesh.skeleton_curves for mesh in meshes]
            captured["curve_count"] = len(global_skel_curves)
            return []

        args = types.SimpleNamespace(
            meshes=["1_hz.obj", "2_vt.obj"],
            tif="fiber.tif",
            cube_label="mask.tif",
            skel_origin="0,0,0",
            skel_axis="xyz",
            skeleton_axis_order="xyz",
            curve_type="auto",
            target_skel_points=0,
            visualize=False,
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            batch_size=8,
        )

        with mock.patch.object(registration_pipe, "load_mesh_from_file", side_effect=lambda _path: [_FakeMesh()]), \
             mock.patch.object(registration_pipe, "extract_skeleton_from_tif", side_effect=fake_extract), \
             mock.patch.object(registration_pipe, "assign_skeletons_to_mesh", return_value=_ToDevice()), \
             mock.patch.object(registration_pipe, "optimize_all_registration", side_effect=fake_optimize):
            registration_pipe.unified_pipeline(args)

        self.assertEqual([call["fiber_type"] for call in captured["extract_calls"]], ["hz", "vt"])
        self.assertEqual([call["axis_order"] for call in captured["extract_calls"]], ["xyz", "xyz"])
        self.assertEqual(captured["curve_count"], 2)
        np.testing.assert_allclose(captured["mesh_curves"][0][0], np.array([[21.0, 0.0, 0.0], [21.0, 1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(captured["mesh_curves"][1][0], np.array([[12.0, 0.0, 0.0], [12.0, 1.0, 0.0]], dtype=np.float32))

    def test_main_groups_meshes_by_cube_context(self):
        registration_pipe = _load_registration_pipe()
        captured = {}

        def fake_process_group(mesh_files, tif_file, cube_label_file, output_root, _args):
            captured["mesh_files"] = mesh_files
            captured["tif_file"] = tif_file
            captured["cube_label_file"] = cube_label_file
            captured["output_root"] = output_root

        argv = [
            "registration_pipe.py",
            "--mesh_root",
            "meshes",
            "--tif_root",
            "tifs",
            "--cube_label_root",
            "labels",
            "--output_root",
            "registered",
        ]
        mesh_dir = "meshes/cube_a"

        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(registration_pipe.os, "walk", return_value=[(mesh_dir, [], ["2_hz.obj", "1_hz.obj"])]), \
             mock.patch.object(registration_pipe.os.path, "isfile", return_value=True), \
             mock.patch.object(registration_pipe, "process_one_mesh") as process_one_mesh, \
             mock.patch.object(registration_pipe, "process_mesh_group", side_effect=fake_process_group, create=True):
            registration_pipe.main()

        process_one_mesh.assert_not_called()
        self.assertEqual(
            captured["mesh_files"],
            [
                registration_pipe.os.path.join(mesh_dir, "1_hz.obj"),
                registration_pipe.os.path.join(mesh_dir, "2_hz.obj"),
            ],
        )
        self.assertEqual(captured["tif_file"], registration_pipe.os.path.join("tifs", "cube_a.tif"))
        self.assertEqual(
            captured["cube_label_file"],
            registration_pipe.os.path.join("labels", "cube_a", "cube_a_mask.tif"),
        )
        self.assertEqual(captured["output_root"], "registered")

    def test_process_mesh_group_saves_registered_mesh_to_its_source_path(self):
        registration_pipe = _load_registration_pipe()

        class _TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def __add__(self, other):
                if isinstance(other, _TensorLike):
                    other = other.value
                return _TensorLike(self.value + other)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class _FakeMeshSurface:
            def __init__(self, source_file):
                self._source_mesh_file = source_file
                self.vertices = _TensorLike(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
                self.displacement = _TensorLike(np.array([[0.5, 0.0, 0.0]], dtype=np.float32))

        class _OriginalMesh:
            def __init__(self):
                self.vertices = None

        global_args = types.SimpleNamespace(
            mesh_root="meshes",
            num_iters=1,
            lr=1e-2,
            lambda_data=1.0,
            lambda_disp=1e-3,
            lambda_elastic=1.0,
            lambda_self=1e-1,
            lambda_lap=5e-1,
            lambda_arap=2.0,
            lambda_sdf=1.0,
            batch_size=8,
            tau=0.01,
            delta=0.05,
            beta=10.0,
            tau_sdf=0.01,
            lambda_inter=0.1,
            delta_inter=0.05,
            beta_inter=10.0,
            target_triangles=1000,
            skel_origin="0,0,0",
            skel_axis="xyz",
            curve_type="horizontal",
            target_skel_points=2,
            visualize=False,
        )
        mesh_files = [
            registration_pipe.os.path.join("meshes", "cube_a", "1_hz.obj"),
            registration_pipe.os.path.join("meshes", "cube_a", "2_hz.obj"),
        ]
        registered_mesh = _FakeMeshSurface(mesh_files[1])
        captured = {}

        def fake_write_mesh(out_file, mesh):
            captured["out_file"] = out_file
            captured["vertices"] = mesh.vertices
            return True

        with mock.patch.object(registration_pipe, "unified_pipeline", return_value=[registered_mesh]), \
             mock.patch.object(registration_pipe.o3d.io, "read_triangle_mesh", return_value=_OriginalMesh()) as read_mesh, \
             mock.patch.object(registration_pipe.o3d.io, "write_triangle_mesh", side_effect=fake_write_mesh):
            registration_pipe.process_mesh_group(mesh_files, "fiber.tif", "mask.tif", "registered", global_args)

        read_mesh.assert_called_once_with(mesh_files[1])
        self.assertEqual(
            captured["out_file"],
            registration_pipe.os.path.join("registered", "cube_a", "2_hz_registered.obj"),
        )
        np.testing.assert_allclose(captured["vertices"], np.array([[1.5, 2.0, 3.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
