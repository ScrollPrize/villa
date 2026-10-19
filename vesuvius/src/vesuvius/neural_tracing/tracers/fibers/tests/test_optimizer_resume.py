"""AdamW checkpoint placement and continued-update equivalence."""
import copy
from types import SimpleNamespace

import pytest
import torch

from vesuvius.neural_tracing.tracers.fibers.train.step import initialize_training_optimizer


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mode', ['default', 'foreach', 'single', 'fused', 'capturable'])
def test_optimizer_resume_counter_placement_and_updates(tmp_path, device, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cpu' and mode == 'capturable':
        pytest.skip('Capturable AdamW requires a supported accelerator')
    torch.manual_seed(19)
    model = torch.nn.ParameterDict({
        'scalar': torch.nn.Parameter(torch.tensor(1., device=device)),
        'weight': torch.nn.Parameter(torch.randn(2, 3, 4, 5, device=device)),
        'unused': torch.nn.Parameter(torch.ones(2, device=device)),
    })
    options = {'default': {}, 'foreach': {'foreach': True}, 'single': {'foreach': False},
               'fused': {'fused': True}, 'capturable': {'capturable': True}}[mode]
    opt = torch.optim.AdamW(model.parameters(), lr=.001, **options)
    for name, param in model.items():
        param.grad = None if name == 'unused' else torch.full_like(param, .25)
    opt.step()
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save(dict(model=model.state_dict(), ema=model.state_dict(), optimizer=opt.state_dict(),
                    step=1, lr_restart_step=0), checkpoint)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if device == 'cuda':
        assert all(state['step'].is_cuda for state in ck['optimizer']['state'].values())
    restored = copy.deepcopy(model)
    restored['weight'].data = restored['weight'].data.contiguous(memory_format=torch.channels_last)
    args = SimpleNamespace(lr=.001, reset_optimizer=False)
    resumed, done, origin = initialize_training_optimizer(restored, copy.deepcopy(restored), args, ck)
    assert (done, origin) == (1, 0)
    assert restored['unused'] not in resumed.state
    expected_device = device if mode in ('fused', 'capturable') else 'cpu'
    for name in ('scalar', 'weight'):
        state = resumed.state[restored[name]]
        assert state['step'].device.type == expected_device
        assert state['step'].item() == 1
        for key in ('exp_avg', 'exp_avg_sq'):
            assert state[key].device.type == device
            assert state[key].stride() == restored[name].stride()
            torch.testing.assert_close(state[key], opt.state[model[name]][key], rtol=0, atol=0)
    # Includes scalar parameters and a changed convolution memory format.
    # Both optimizers must increment exactly once and preserve all updates.
    for update in range(3):
        for name in model:
            gradient = torch.full_like(model[name], .5 + update) if name != 'unused' else None
            model[name].grad = gradient
            restored[name].grad = None if gradient is None else gradient.clone().contiguous(
                memory_format=torch.channels_last if gradient.ndim == 4 else torch.contiguous_format)
        opt.step()
        resumed.step()
        for name in model:
            torch.testing.assert_close(restored[name], model[name], rtol=0, atol=0)
            if name != 'unused':
                assert resumed.state[restored[name]]['step'].item() == update + 2
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(resumed.state[restored[name]][key], opt.state[model[name]][key],
                                               rtol=0, atol=0)
