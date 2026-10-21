"""torch.compile settings of the tracers."""


def compile_options(backend=None, *, fullgraph=False):
    """Static shapes; with inductor, eager low-precision rounding is emulated (``emulate_precision_casts``), without
    which compiled bf16 graphs produced corrupted encoder gradients. ``backend`` lets tests inspect graph capture."""
    options = dict(dynamic=False)
    if fullgraph:
        options['fullgraph'] = True
    if backend is None or backend == 'inductor':
        options['options'] = dict(emulate_precision_casts=True)
    if backend is not None:
        options['backend'] = backend
    return options
