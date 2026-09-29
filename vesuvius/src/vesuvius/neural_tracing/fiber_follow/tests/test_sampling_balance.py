"""Whole-stream crop budgets and departed-only weighting preserve memory order."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_trajectory_memory import cfg
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import SwitchCropBudget, sequence_batches
from vesuvius.neural_tracing.fiber_follow.regression.supervision import memory_probe_terms


def stream(source, length, identity=0):
    return [dict(source=source, observation=i, identity_seed=identity) for i in range(length)]


def test_budget_uses_crop_counts_preserves_other_sources_and_whole_long_streams():
    budget=SwitchCropBudget(.15)
    accepted_switch=0
    for i in range(500):
        ordinary=stream(i%3,19+i%7,i)
        switch=stream(3,129,i)
        retained=budget.admit([ordinary,switch])
        assert retained[0] is ordinary
        if len(retained)==2:
            assert retained[1] is switch and len(retained[1])==129
            accepted_switch+=1
        assert budget.switch <= .15*(budget.ordinary+budget.switch)+1e-8
    assert accepted_switch>5 and budget.rejected_streams>0
    assert abs(budget.switch/(budget.ordinary+budget.switch)-.15)<.01
    assert SwitchCropBudget(0.).admit([stream(3,2),stream(5,2)])[0][-1]['source']==5


def test_burst_is_budgeted_as_one_complete_stream_not_truncated_or_interleaved(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.feature_sequences as module
    monkeypatch.setattr(module,'stream_rows',lambda item,builder,band:item)
    class Builder:
        def __init__(self):
            self.cfg=cfg(feature_switch_crop_fraction=.15,feature_memory_revision=2)
            self.encoded=[]
        def __call__(self,rows,vol):
            self.encoded.extend((r['source'],r['observation']) for r in rows)
            return dict(hist=torch.zeros(len(rows),1,3),source=torch.tensor([r['source'] for r in rows]),
                        observation=torch.tensor([r['observation'] for r in rows]))
    builder=Builder()
    batches=[]
    # No credit: entire switch stream rejected before image encoding.
    assert list(sequence_batches(builder,[stream(3,129)],None))==[]
    assert builder.encoded==[]
    # Earn room from ordinary groups, then complete a long switch in order.
    for i in range(6):
        batches.extend(sequence_batches(builder,[stream(0,128,i)],None,worker=2))
    batches.extend(sequence_batches(builder,[stream(3,129,7)],None,worker=2))
    seen={}
    resets={};ends={}
    for chunk in batches:
        for b in chunk['feature_sequence']:
            for sid,obs,reset,end in zip(b['stream_id'].tolist(),b['observation'].tolist(),
                                        b['stream_reset'].tolist(),b['stream_end'].tolist()):
                seen.setdefault(sid,[]).append(obs)
                resets[sid]=resets.get(sid,0)+reset
                ends[sid]=ends.get(sid,0)+end
    assert len(seen)==7 and all(v==list(range(len(v))) for v in seen.values())
    assert all(resets[k]==ends[k]==1 for k in seen)
    assert builder.encoded[-129:]==[(3,i) for i in range(129)]
    assert all(sid>>48==2 for sid in seen)


def test_departed_weight_changes_only_negative_bce_gradient_and_not_metrics():
    b=dict(memory_target_identity=torch.tensor([[1.,0.,0.]]),
           memory_target_identity_mask=torch.tensor([[True,True,False]]),
           memory_target_offset=torch.ones(1,3,3),
           memory_target_offset_mask=torch.tensor([[True,False,False]]))
    results=[]
    for weight in (1.,4.):
        p=torch.zeros(1,3,4,requires_grad=True)
        terms=memory_probe_terms(dict(memory_probe=p),b,departed_weight=weight)
        (terms['memory_identity_per_state'].sum()+terms['memory_offset_per_state'].sum()).backward()
        results.append((terms,p.grad))
    normal,weighted=results
    assert weighted[1][0,1,0]==4*normal[1][0,1,0]
    assert weighted[1][0,0,0]==normal[1][0,0,0]
    assert weighted[1][0,2,0]==0
    torch.testing.assert_close(weighted[1][...,1:],normal[1][...,1:])
    for key in ('memory_identity_count','memory_identity_correct','memory_departed_count','memory_departed_correct'):
        assert normal[0][key]==weighted[0][key]


@pytest.mark.parametrize('field,value',[('memory_departed_weight',0.),('memory_departed_weight',float('nan')),
 ('feature_switch_crop_fraction',1.),('feature_switch_crop_fraction',-.1),('feature_switch_crop_fraction',float('nan'))])
def test_invalid_balance_config(field,value):
    with pytest.raises(ValueError):
        cfg(**{field:value})
