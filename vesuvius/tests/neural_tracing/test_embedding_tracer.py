"""CPU contracts for opt-in signed embedding inference products."""
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import FiberTrace3DPredictAdapter
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model


def config():
    return {"patch_shape_zyx": [16]*3, "embedding": {"enabled": True},
            "model_3d": {"features_per_stage": [8, 16], "normalization": "group", "output_stride": 2}}


def test_pilot_explicit_fiber_selection():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_pilot import select_pilot_fiber
    source = {"fibers": ["short.json", "hard.json"]}
    length = lambda path: {"short.json": 1, "hard.json": 10}[path]
    assert select_pilot_fiber(source, None, length) == "short.json"
    assert select_pilot_fiber(source, "hard.json", length) == "hard.json"
    with pytest.raises(ValueError, match="exactly one"):
        select_pilot_fiber(source, "unknown.json", length)


def test_analysis_causal_history_and_common_references():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_route_analysis import history_indices, contrast, line_point
    arcs = np.arange(0., 401., 8.)
    assert history_indices(arcs, 40) == list(range(8, 40, 4))
    assert history_indices(arcs, 0) == []
    np.testing.assert_array_equal(line_point(np.array([[0., 0, 0], [32., 0, 0]]), np.array([0., 32.]), 8), [8, 0, 0])
    rows = [dict(status="ok", embedding=[1., 0.]), dict(status="ok", embedding=[0., 1.]),
            dict(status="missing_coverage")]
    result = contrast(rows, 0, 1, [0, 2])
    assert result["gap"] == 1 and result["references"] == 1 and not result["complete"]
    assert contrast(rows, 0, 1, [2]) is None


def test_native_cached_query_reports_missing_and_invalid(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    from lasagna.tiled_predict3d import _create_omezarr
    binary = Path(__file__).resolve().parents[3] / "volume-cartographer/build/dev-quickbuild-gcc/bin/vc_fiber_trace_metric"
    if not binary.exists():
        pytest.skip("native diagnostic build not available")
    groups = {}
    for channel, dtype, value in [("presence", np.uint8, 255), ("embedding_000", np.float32, -2), ("embedding_001", np.float32, 1)]:
        root = tmp_path / channel
        array = _create_omezarr(str(root), (4, 4, 4), 0, 1, 2, channel, dtype=dtype)["0"]
        array[:] = value
        if channel.startswith("embedding"):
            array[:, :, 1] = -value  # Opposite corners cancel without absent fill chunks.
        groups[channel] = dict(zarr=f"{channel}/0", scaledown=0, channels=[channel])
    (tmp_path / "presence/0/1/0/0").unlink()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(version=2, source_to_base=1, groups=groups)))
    points = tmp_path / "points.json"
    points.write_text(json.dumps([[2.5, 2.5, 2.5], [.5, .5, .5], [.5, .5, 2.5], [-1, 0, 0]]))
    result = subprocess.run([str(binary), "sample-embeddings", str(manifest), str(points)], capture_output=True, text=True, check=True, timeout=30)
    rows = json.loads(result.stdout)
    assert [r["status"] for r in rows] == ["ok", "invalid_vector", "missing_coverage", "missing_coverage"]
    np.testing.assert_allclose(rows[0]["embedding"], np.array([-2., 1.])/np.sqrt(5), atol=1e-6)
    assert rows[0]["presence"] == 1


def test_signed_embeddings_same_forward_preserves_prediction():
    torch.manual_seed(17)
    raw = config()
    model = build_fiber_trace_3d_model(raw).eval()
    old = FiberTrace3DPredictAdapter(raw)
    new = FiberTrace3DPredictAdapter(raw, emit_embeddings=True)
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        legacy = old.run_tile_inference(model, x, device=torch.device("cpu"))
        signed = new.run_tile_inference(model, x, device=torch.device("cpu"))
        expected = model(x, return_embeddings=True)
    torch.testing.assert_close(signed[:, :7], legacy, rtol=0, atol=0)
    torch.testing.assert_close(signed[:, 7:], expected["embeddings"], rtol=0, atol=0)
    assert signed[:, 7:].min() < 0
    products = new.product_tensors_from_output(signed)
    product = new.output_products[-1]
    assert product.value_range is None and np.dtype(product.dtype) == np.float32
    values = new.finalize_product_slab(product, products[product.name][0].numpy())
    np.testing.assert_array_equal(values["embedding_000"], signed[0, 7].numpy())
    assert len(old.output_products) == 1 and len(new.output_products) == 2


def test_missing_embedding_config_and_nonfinite_output_error():
    with pytest.raises(ValueError, match="embedding-enabled"):
        FiberTrace3DPredictAdapter({}, emit_embeddings=True)
    adapter = FiberTrace3DPredictAdapter(config(), emit_embeddings=True)
    product = adapter.output_products[-1]
    with pytest.raises(ValueError, match="invalid embedding"):
        adapter.finalize_product_slab(product, np.full((16, 2, 2, 2), np.nan))


def test_split_embedding_inference_requires_explicit_bank():
    raw = config()
    raw["embedding"] = {
        "enabled": True,
        "channels": 16,
        "short_range_channels": 8,
        "objective": "local_contrastive",
        "paired_patches": True,
    }
    with pytest.raises(ValueError, match="requires embedding_bank"):
        FiberTrace3DPredictAdapter(raw, emit_embeddings=True)
    for bank in ("short", "long"):
        adapter = FiberTrace3DPredictAdapter(raw, emit_embeddings=True, embedding_bank=bank)
        assert adapter.embedding_channels == 8
        assert len(adapter.output_products[-1].channels) == 8
    with pytest.raises(ValueError, match="only valid"):
        FiberTrace3DPredictAdapter(config(), emit_embeddings=True, embedding_bank="short")


@pytest.mark.parametrize("optimum", [0, 3, 5, 8])
def test_weight_search(optimum):
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import search_weight
    seen = []
    def execute(v):
        tick = round(v['weight'] * 20)
        assert tick not in seen and tick >= 0
        seen.append(tick)
        assert (v['spacing'], v['window'], v['falloff']) == (8, 256, 'gaussian')
        return dict(v, completed=True, failures=(tick-optimum)**2)
    assert search_weight(execute)['weight'] == optimum / 20
    assert seen[:3] == [5, 4, 6]


def test_weight_search_ties_and_incomplete():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import search_weight
    assert search_weight(lambda v: dict(v, completed=True, failures=5))['weight'] == .25
    with pytest.raises(RuntimeError, match='incomplete'):
        search_weight(lambda v: dict(v, completed=False))


def test_sweep_grid():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import variants
    rows = list(variants())
    assert len(rows) == 38 and len({r['name'] for r in rows}) == 38
    assert rows[1]['count'] == 8
    assert [r['name'] for r in variants(True)] == ['off', 'original']
    assert all(r['window'] >= r['spacing'] for r in rows)
    weighted = list(variants(weight=.30))
    assert weighted[:2] == rows[:2]
    assert all(r['weight'] == .30 for r in weighted[2:])
    for invalid in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            list(variants(weight=invalid))


def test_sweep_fill_uses_demand_ownership_not_whole_sweep_grant(tmp_path, monkeypatch):
    import json
    import sys
    from vesuvius.neural_tracing.fiber_trace_3d import embedding_trace_sweep as sweep
    control = tmp_path / "control.json"
    manager = dict(exclusive=False, priority=0, cpu_cores=16, vram_bytes_per_gpu=20*1024**3)
    control.write_text(json.dumps({"embedding_trace_sweep": {"fill_missing": True, "gpu_manager": manager}}))
    monkeypatch.setattr(sys, "argv", ["sweep", str(control)])
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return 17
    monkeypatch.setattr("vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation.launch_managed", launch)
    seen = []
    monkeypatch.setattr(sweep, "run", lambda settings: seen.append(settings))
    sweep.main()
    assert not calls
    assert seen[0]["gpu_manager"] == manager
    monkeypatch.setattr(sys, "argv", ["sweep", str(control), "--unmanaged"])
    with pytest.raises(SystemExit):
        sweep.main()


@pytest.mark.parametrize('mode,aggregation',[('anchors-lru','mean'),('anchors-lru','normalized-mean'),('appearance','best'),('appearance','support-softmin')])
def test_configurable_embedding_search(mode,aggregation):
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import configured_variants, search_weight
    history = dict(spacing=128, window=4096, count=64, falloff='linear', mode=mode, half_life=512, anchors=4, ema_mix=.5, aggregation=aggregation,novelty=.1,temperature=.2)
    rows = configured_variants({'candidates': [dict(history, weight=.4)]})
    assert [r['name'] for r in rows] == ['off', 'original', 'candidate-000']
    assert rows[-1]['window'] == 4096
    seen = []
    def execute(v):
        seen.append(v)
        return dict(v, completed=True, failures=round(abs(v['weight']-.5)*100))
    assert search_weight(execute, history=history, start_weight=.4)['weight'] == .5
    assert all(all(v[k] == value for k, value in history.items()) for v in seen)
    for bad in [dict(history, weight=-1), dict(history, weight=.25, count=0),
                dict(history, weight=.25, step_size=4), dict(history, weight=.25, window=float('nan'))]:
        with pytest.raises(ValueError):
            configured_variants({'candidates': [bad]})
    with pytest.raises(ValueError):
        search_weight(execute, start_weight=.123)


def test_family_refinement_cycles_and_retains_each_family():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import refine_families
    calls=[]
    def execute(v):
        calls.append(v)
        # EMA mix changes the preferred weight, requiring a second cycle.
        target=.5 if v['ema_mix']==.5 else .25
        return dict(v,completed=True,failures=round(100*abs(v['weight']-target))+int(v['ema_mix']==0)*30)
    seed=dict(weight=.25,spacing=128,window=1024,count=8,falloff='uniform',mode='appearance',ema_mix=0)
    families=[dict(label='a',seed=seed,axes={'weight':[.25,.5],'ema_mix':[0,.5]}),
              dict(label='b',seed=dict(seed,mode='anchors-lru'),axes={'weight':[.25]})]
    emitted=[]
    results=refine_families(execute,families,on_result=emitted.append)
    assert emitted==results
    assert results[0]['parameters']['weight']==.5
    assert results[0]['parameters']['ema_mix']==.5
    assert results[1]['parameters']['mode']=='anchors-lru'
    assert len({r['name'] for r in calls})==len(calls)
    def weight_only(v):
        return dict(v,completed=True,failures=round(abs(v['weight']-.35)*100))
    fine=refine_families(weight_only,[dict(label='fine',seed=seed,axes={'weight':[.25,.5]},neighbor_weight=True)])
    assert fine[0]['parameters']['weight']==.35


def test_lru_touch_policy_validation():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import configured_variants
    candidate = dict(weight=.1, spacing=512, window=1024, count=8,
                     falloff='uniform', mode='anchors-lru', touch='hit')
    assert configured_variants({'candidates':[candidate]})[-1]['touch'] == 'hit'
    for override in ({'touch':'unknown'}, {'mode':'appearance'}, {'mode':'ema'}):
        with pytest.raises(ValueError, match='touch'):
            configured_variants({'candidates':[dict(candidate, **override)]})


def test_product_storage_preserves_signed_float32(tmp_path):
    from lasagna.tiled_predict3d import OutputProductSpec, OutputChannelSpec, create_product_omezarr_groups
    product = OutputProductSpec(name="embedding", level=0, scaledown=1,
        channels=[OutputChannelSpec("embedding_000", relative_path=str(tmp_path / "embedding.zarr"))],
        dtype=np.float32, value_range=None, chunk_size=2)
    options = dict(products=[product], base_shape_zyx=(4, 4, 4), n_levels=1, ome_chunk=2)
    groups = create_product_omezarr_groups(**options)
    array = groups["embedding_000"]["0"]
    assert array.dtype == np.dtype("float32")
    array[:] = -0.375
    reopened = create_product_omezarr_groups(**options)["embedding_000"]["0"]
    np.testing.assert_array_equal(reopened[:], np.full((4, 4, 4), -.375, np.float32))


def test_embedding_reference_variants():
    from vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep import configured_variants
    base = dict(weight=.25, spacing=128, window=1024, count=8, falloff='uniform')
    for mode in ('mean', 'best', 'seed', 'ema', 'ema-seed', 'ema-gated', 'ema-seed-gated', 'ema-delayed'):
        row = configured_variants({'candidates': [dict(base, mode=mode, half_life=128)]})[-1]
        assert row['mode'] == mode and row['half_life'] == 128
    row = configured_variants({'candidates': [dict(base, mode='ema-delayed', delay=64)]})[-1]
    assert row['delay'] == 64
    for delay in (-1, 2049, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            configured_variants({'candidates': [dict(base, mode='ema-delayed', delay=delay)]})
    for update in (dict(mode='max'), dict(mode='best', falloff='gaussian'), dict(half_life=0), dict(half_life=float('nan'))):
        with pytest.raises(ValueError):
            configured_variants({'candidates': [{**base, **update}]})
