from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from vesuvius.neural_tracing.fiber_trace_3d.reference_chunk_geometry import reference_chunks, reference_chunk_job


def test_parallel_reference_chunks_match_serial():
    jobs = [([[0, 0, 0], [63, 63, 63]], 1, (64, 64, 64), 8),
            ([[-5, 12, 10], [25, 12, 10]], 2, (20, 20, 20), 4),
            ([[7, 7, 7], [8, 8, 8]], 1, (16, 16, 16), 8)]
    expected = [reference_chunks(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        actual = list(pool.map(reference_chunk_job, jobs))
    assert actual == expected
    assert actual[2] == {(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)}
