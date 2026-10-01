import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from slab_fixtures import cfg
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.refinement_resume import STAGE, expand_refinement_checkpoint, fork_run
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser, checkpoint_config, initialize_training_optimizer
from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state
from types import SimpleNamespace


def checkpoint(tmp_path):
    model = build_model(cfg(recurrent_refinement_steps=1))
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.0003)
    for p in model.parameters(): p.grad = torch.ones_like(p)
    opt.step()
    args = vars(build_parser().parse_args(['--name','source','--fiber-zarrs','z','--fibers','f','--ct','c','--manifest','m',
        '--out-root',str(tmp_path),'--lr','.0003','--batch','16','--grad-steps','1','--workers','10',
        '--warmup','1000','--recurrent-refinement-steps','1']))
    return dict(architecture=model.architecture,model_cfg=model.cfg.to_dict(),model=model.state_dict(),ema=ema.state_dict(),
                optimizer=opt.state_dict(),rng=training_rng_state(),step=17000,samples_seen=272000,lr_restart_step=0,
                training_options=args)


def test_expansion_preserves_weights_moments_rng_and_can_update(tmp_path):
    original = checkpoint(tmp_path)
    rng = torch.get_rng_state().clone()
    migrated = expand_refinement_checkpoint(original,3)
    assert torch.equal(rng,torch.get_rng_state())
    for section in ('model','ema'):
        for key,value in original[section].items():
            if key == STAGE:
                torch.testing.assert_close(migrated[section][key],value.expand(3,-1),rtol=0,atol=0)
                assert value.shape[0] == 1
            else: torch.testing.assert_close(migrated[section][key],value,rtol=0,atol=0)
    for index,state in original['optimizer']['state'].items():
        for key,value in state.items():
            actual=migrated['optimizer']['state'][index][key]
            if actual.shape != value.shape:
                torch.testing.assert_close(actual[:1],value,rtol=0,atol=0)
                assert torch.count_nonzero(actual[1:]) == 0
            else: torch.testing.assert_close(actual,value,rtol=0,atol=0)
    torch.testing.assert_close(migrated['rng']['torch'],original['rng']['torch'],rtol=0,atol=0)
    model = build_model(checkpoint_config(migrated))
    opt,done,origin = initialize_training_optimizer(model,copy.deepcopy(model),SimpleNamespace(lr=.0003,reset_optimizer=False),migrated)
    assert (done,origin)==(17000,0)
    for p in model.parameters():p.grad=torch.ones_like(p)
    opt.step()  # Exercises migrated Adam tensors, including both newly added rows.
    assert torch.isfinite(model.refinement_stage.weight).all()


@pytest.mark.parametrize('legacy_prefetch',[False,True])
@pytest.mark.parametrize('legacy_batch',[False,True])
def test_fork_uses_checkpoint_settings_fixture_and_live_replay(tmp_path,legacy_prefetch,legacy_batch):
    source=tmp_path/'source'; source.mkdir()
    ck=checkpoint(tmp_path)
    if legacy_batch:
        ck['training_options'].pop('grad_steps')
        ck['training_options']['microbatch'] = 16
    prefetch_keys={'remote_prefetch_connections','remote_prefetch_queue_size','remote_prefetch_timeout','remote_prefetch_lookahead'}
    if legacy_prefetch:
        for key in prefetch_keys:ck['training_options'].pop(key)
    fixture=b'fixture bytes preserved exactly'
    (source/'monitor_recovery.npz').write_bytes(fixture)
    ck['monitor_recovery_sha256']=hashlib.sha256(fixture).hexdigest()
    path=source/'ckpt_017000.pt';torch.save(ck,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    # Deliberately stale config must never be consulted.
    (source/'config.json').write_text('{"lr":0.001,"microbatch":8}')
    replay=source/'cache';replay.mkdir();(source/'dagger').mkdir()
    (source/'dagger/replay.json').write_text(json.dumps([str(replay)]))
    info=fork_run(path,'refine3',3)
    dest=tmp_path/'refine3';saved=json.loads((dest/'config.json').read_text())
    assert (saved['lr'],saved['batch'],saved['grad_steps'],saved['workers'])==(.0003,16,1,10)
    assert saved['recurrent_refinement_steps']==saved['model_cfg']['recurrent_refinement_steps']==3
    assert (dest/'monitor_recovery.npz').read_bytes()==fixture
    assert json.loads((dest/'dagger/replay.json').read_text())==[str(replay)]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert saved['remote_prefetch_connections']==0
    assert set(info['changes'])=={'name','resume','recurrent_refinement_steps'} | (prefetch_keys if legacy_prefetch else set())
    with pytest.raises(FileExistsError):fork_run(path,'refine3',3)


@pytest.mark.parametrize('steps',[0,1,1.5])
def test_reject_nonexpansion(tmp_path,steps):
    with pytest.raises(ValueError):expand_refinement_checkpoint(checkpoint(tmp_path),steps)
