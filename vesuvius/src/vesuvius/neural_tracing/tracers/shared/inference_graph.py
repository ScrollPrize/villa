"""One bounded CUDA graph for fixed-shape, deterministic inference forwards.

No padding, model compilation, precision/backend changes, or cross-row mixing.
Other shapes use the original callable. Output clones prevent later replays from
overwriting earlier chunks. The first three matching inputs are checked exactly.
"""
import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class InferenceGraph:
    def __init__(self, model, rows):
        self.model, self.rows = model, rows
        self.graph = None
        self.checked = 0
        self.disabled = False
        self.suspended = False

    @staticmethod
    def signature(leaves):
        return tuple((tuple(v.shape), v.dtype, v.device, tuple(v.stride()))
                     if torch.is_tensor(v) else (type(v), v) for v in leaves)

    def __call__(self, x, hist, hmask, **sampling):
        args = (x, hist, hmask)
        eager = lambda: self.model(*args, **sampling)
        if self.disabled or self.suspended or not hist.is_cuda or len(hist) != self.rows or self.model.training:
            return eager()
        leaves, spec = tree_flatten((args, sampling))
        signature = self.signature(leaves)
        if self.graph is not None and (signature != self.key or spec != self.spec):
            return eager()
        if self.graph is None:
            self.key, self.spec = signature, spec
            self.static = [v.clone() if torch.is_tensor(v) else v for v in leaves]
            static_args, static_kwargs = tree_unflatten(self.static, spec)
            stream = torch.cuda.Stream(device=hist.device)
            stream.wait_stream(torch.cuda.current_stream(hist.device))
            # Capture parameter casts too: outer autocast's cached casts would
            # otherwise be temporary allocations outside the graph's pool.
            with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                                dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.model(*static_args, **static_kwargs)
                torch.cuda.current_stream(hist.device).wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self.model(*static_args, **static_kwargs)
            print(f'inference graph: captured rows={self.rows}', flush=True)
        else:
            for dst, src in zip(self.static, leaves):
                if torch.is_tensor(dst):
                    dst.copy_(src)
        self.graph.replay()
        values, output_spec = tree_flatten(self.output)
        result = tree_unflatten([v.clone() if torch.is_tensor(v) else v for v in values], output_spec)
        if self.checked < 3:
            expected = eager()
            expected_values, expected_spec = tree_flatten(expected)
            result_values, _ = tree_flatten(result)
            equal = expected_spec == output_spec and all(
                torch.equal(a, b) if torch.is_tensor(a) else a == b
                for a, b in zip(expected_values, result_values))
            if not equal:
                self.disabled = True
                self.graph = None
                self.static = self.output = None
                print('inference graph: exact comparison failed; retaining eager inference', flush=True)
                return expected
            self.checked += 1
            print(f'inference graph: exact output check {self.checked}/3 passed', flush=True)
            if self.checked == 3:
                # Three warmed trials, including input copies and output clones.
                timing = {}
                for name, call in [('eager', eager), ('graph', lambda: self(x, hist, hmask, **sampling))]:
                    elapsed = []
                    for _ in range(3):
                        torch.cuda.synchronize(hist.device)
                        start = time.perf_counter()
                        call()
                        torch.cuda.synchronize(hist.device)
                        elapsed.append(time.perf_counter()-start)
                    timing[name] = elapsed
                print(f'inference graph: same-input seconds rows={self.rows} {timing}', flush=True)
        return result
