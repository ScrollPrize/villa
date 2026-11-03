"""Reuse geometry only within one immutable objective evaluation.

The context owns all cached tensors and is discarded before optimizer updates.
Never cache across updates, window remaps, or grad/no-grad evaluations.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import torch

_cache=ContextVar('fiber_geometry_evaluation',default=None)
_profile=ContextVar('fiber_geometry_profile',default=False)


@contextmanager
def geometry_evaluation(enabled=True,profile=False):
    token=_cache.set({} if enabled else None)
    pt=_profile.set(profile)
    try:
        yield
    finally:
        _cache.reset(token)
        _profile.reset(pt)


def profile_geometry(fn):
    @wraps(fn)
    def call(*args,**kwargs):
        if not _profile.get():return fn(*args,**kwargs)
        with torch.profiler.record_function('geometry/'+fn.__qualname__):
            return fn(*args,**kwargs)
    return call


def cached_geometry(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        cache=_cache.get()
        if cache is None:
            return method(self,*args,**kwargs)
        key=(method,self,torch.is_grad_enabled(),args,tuple(kwargs.items()))
        if key not in cache:
            cache[key]=method(self,*args,**kwargs)
        return cache[key]
    return call


def row_gather(values,indices):
    return values.index_select(0,indices) if _cache.get() is not None else values[indices]


@cached_geometry
def width_boundaries(widths):
    """Share the prefix map and its backward only within one objective."""
    return torch.cat((widths.new_zeros(1),widths.cumsum(0)))


def paired_gather(values,rows,columns):
    if _cache.get() is None:
        return values[rows[:,None],columns]
    index=(rows[:,None]*values.shape[1]+columns).flatten()
    return values.flatten(0,1).index_select(0,index).reshape(*columns.shape,*values.shape[2:])
