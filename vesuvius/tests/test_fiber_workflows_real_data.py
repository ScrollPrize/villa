"""Read-only restoration check against externally supplied historical checkpoints.

Set FIBER_LAYOUT_REFERENCE_SPEC to a three-arm benchmark spec with historical
solves and expected reports. No data are bundled, optimized or rewritten.
"""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_layout_benchmark as benchmark
from vesuvius.neural_tracing.fiber_trace_3d import fiber_forced_selectable_audit as audit


@pytest.mark.parametrize('arm',benchmark.ARMS)
def test_historical_scores_and_recomputed_full_objective(arm):
    spec_path=os.environ.get('FIBER_LAYOUT_REFERENCE_SPEC')
    if not spec_path:pytest.skip('Set FIBER_LAYOUT_REFERENCE_SPEC to external historical benchmark inputs')
    spec=json.loads(Path(spec_path).read_text());entry=spec['arms'][arm]
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        result=benchmark.score(entry,entry['historical_solve'],spec['reference_root'])
        expected=json.loads(Path(entry['expected_report']).read_text())
        benchmark.validate_expected(result,expected)
        actual=audit.evaluate(audit.load(Path(entry['historical_solve']).parent))
        assert actual['terms'].keys()==expected['final']['terms'].keys()
        for name,value in actual['terms'].items():
            np.testing.assert_allclose(value,expected['final']['terms'][name],rtol=0,atol=1e-10,err_msg=name)
        np.testing.assert_allclose(actual['loss'],expected['final']['loss'],rtol=0,atol=1e-10)
    finally:
        torch.set_num_threads(threads)
