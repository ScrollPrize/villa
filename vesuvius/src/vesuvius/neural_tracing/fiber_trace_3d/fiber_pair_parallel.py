"""Data-parallel pair evaluation with one shared optimizer and autograd graph.

Only pair arithmetic is distributed. UV geometry, unary regularization, cavity
updates and Adam remain on the primary device. Copies retain autograd edges;
there is no replica optimizer, detached state, or change in batch weighting.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import torch


class ParallelPairs:
    def __init__(self, function, devices):
        self.function = function
        self.devices = [torch.device(d) for d in devices]
        self.pool = ThreadPoolExecutor(max_workers=len(self.devices))
        self.audited = False

    def close(self):
        self.pool.shutdown(wait=True)

    def __call__(self, *args, **kwargs):
        # Diagnostics stay eager on the primary device, preserving their API
        # and avoiding mutation of a shared diagnostic output across workers.
        if not torch.is_grad_enabled() or len(args[0]) < len(self.devices):
            return self.function(*args, **kwargs)
        if kwargs.get('diagnostics') is not None or args[10].get('pair_audit'):
            raise ValueError('Run pair audits/diagnostics on a single GPU')
        if args[10].get('pair_parallel_audit') and not self.audited:
            self.audited = True
            self.audit(args, kwargs)
        source = args[0].device
        count = len(args[0])
        tasks = []
        for rank, device in enumerate(self.devices):
            start, end = count * rank // len(self.devices), count * (rank + 1) // len(self.devices)
            # Copies are issued on the caller's stream, then the worker waits
            # explicitly. This also supports non-default caller CUDA streams.
            local = tuple((value if index == 6 else value[start:end]).to(device)
                          if isinstance(value, torch.Tensor) else value
                          for index, value in enumerate(args))
            options = {}
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    value = value[start:end].to(device)
                elif value is not None and name in ('offsets_a', 'offsets_b'):
                    value = tuple(v[start:end].to(device) for v in value)
                options[name] = value
            event = None
            if device.type == 'cuda':
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
            tasks.append(self.pool.submit(self._evaluate, device, event, local, options))
        # Concatenation, rather than mean-of-means, handles uneven partitions.
        results = []
        for task in tasks:
            result, completed = task.result()
            if completed is not None:
                # A Python future completes after enqueue, not after CUDA.
                torch.cuda.current_stream(source).wait_event(completed)
                for value in result:
                    stream = torch.cuda.current_stream(value.device)
                    stream.wait_event(completed)
                    value.record_stream(stream)
            results.append(result)
        return tuple(torch.cat([result[i].to(source) for result in results])
                     for i in range(len(results[0])))

    def _evaluate(self, device, event, args, kwargs):
        context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        with context, torch.enable_grad():
            if event is not None:
                stream = torch.cuda.current_stream(device)
                stream.wait_event(event)
                def record(value):
                    if isinstance(value, torch.Tensor): value.record_stream(stream)
                    elif isinstance(value, (list, tuple)):
                        for item in value: record(item)
                for value in (*args, *kwargs.values()): record(value)
            result = self.function(*args, **kwargs)
            completed = None
            if device.type == 'cuda':
                completed = torch.cuda.Event()
                completed.record(torch.cuda.current_stream(device))
            return result, completed

    def audit(self, args, kwargs):
        """Small actual-data value/gradient regression on a non-default stream."""
        from pathlib import Path
        from lasagna.inference_provenance import atomic_write
        device = args[0].device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            params = []
            def clone(value, shared=False):
                if not isinstance(value, torch.Tensor): return value
                result = (value if shared else value[:17]).detach().clone()
                if result.is_floating_point():
                    result.requires_grad_()
                    params.append(result)
                return result
            local = tuple(clone(v, i == 6) for i, v in enumerate(args))
            options = {k: clone(v) for k, v in kwargs.items()}
            expected = self.function(*local, **options)
            actual = self(*local, **options)
            eg = torch.autograd.grad(sum(v.sum() for v in expected), params, allow_unused=True)
            ag = torch.autograd.grad(sum(v.sum() for v in actual), params, allow_unused=True)
            for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            differences = []
            for a, b in zip(eg, ag):
                if a is None or b is None:
                    if a is not b: raise AssertionError('Parallel gradient connectivity differs')
                    continue
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
                differences.append(float((a-b).abs().max()))
            record = dict(events=len(local[0]),devices=len(self.devices),nondefault_stream=True,
                          cost_max_abs=[float((a-b).abs().max()) for a,b in zip(expected,actual)],
                          gradient_max_abs=differences)
        torch.cuda.current_stream(device).wait_stream(stream)
        atomic_write(Path(args[10]['output_root'])/'pair_parallel_audit.json', record, durable=False)
