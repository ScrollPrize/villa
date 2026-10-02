"""Read saved batch settings across the gradient-accumulation CLI rename."""


def normalize_batch_options(options):
    """Return new-style options without mutating a saved config/checkpoint."""
    options = dict(options)
    if 'microbatch' in options:
        batch = options.pop('microbatch')
        effective = options['batch']
        if batch < 1 or effective < 1 or effective % batch:
            raise ValueError('Legacy microbatch must be positive and divide effective batch')
        grad_steps = effective // batch
        if 'grad_steps' in options and options['grad_steps'] != grad_steps:
            raise ValueError('Conflicting legacy and current gradient accumulation settings')
        options.update(batch=batch, grad_steps=grad_steps)
    return options
