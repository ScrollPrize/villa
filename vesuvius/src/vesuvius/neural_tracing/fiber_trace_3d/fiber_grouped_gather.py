"""Opt-in deterministic grouped gradient reduction for small width tables.

Forward values are ordinary gathers. Backward groups equal destinations then
reduces each segment once, avoiding the general advanced-index backward kernel.
The grouping is dynamic because the learned winding coordinates can move.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import torch

_mode=ContextVar('fiber_width_gather_mode',default='original')


@contextmanager
def width_gather_mode(mode):
    if mode not in ('original','select','grouped','packed'):raise ValueError('Unknown width gather mode')
    token=_mode.set(mode)
    try:yield
    finally:_mode.reset(token)


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,index):
        flat=index.flatten()
        order=torch.argsort(flat,stable=True)
        lengths=torch.bincount(flat,minlength=len(values))
        ctx.save_for_backward(order,lengths)
        return values.index_select(0,flat).reshape(index.shape)

    @staticmethod
    def backward(ctx,grad):
        order,lengths=ctx.saved_tensors
        sorted_grad=grad.flatten().index_select(0,order)
        return torch.segment_reduce(sorted_grad,'sum',lengths=lengths),None


def width_lookup(values,index):
    mode=_mode.get()
    if mode=='original':return values[index]
    if mode=='grouped' and torch.is_grad_enabled() and values.requires_grad:
        return _Grouped.apply(values,index)
    return values.index_select(0,index.flatten()).reshape(index.shape)


from .fiber_geometry_cache import cached_geometry


@cached_geometry
def width_records(widths,boundaries):
    # Built once per objective/dtype, not once per lookup. Both columns share
    # their index domain and autograd gather/scatter.
    return torch.stack((boundaries[:-1],widths),-1)


def width_pair(widths,boundaries,index):
    if _mode.get()!='packed':
        return width_lookup(boundaries,index),width_lookup(widths,index)
    values=width_records(widths,boundaries)
    result=values.index_select(0,index.flatten()).reshape(*index.shape,2)
    return result.unbind(-1)
