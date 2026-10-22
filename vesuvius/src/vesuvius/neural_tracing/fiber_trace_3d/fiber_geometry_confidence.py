"""Opt-in confidence gates for local geometry, not assignment evidence."""


def confidence_gate(probability, config):
    low=float(config.get('geometry_confidence_low', .5))
    high=float(config.get('geometry_confidence_high', .9))
    if not (0 <= low < high <= 1):
        raise ValueError('geometry confidence thresholds require 0 <= low < high <= 1')
    t=((probability.detach()-low)/(high-low)).clamp(0,1)
    return t.square()*(3-2*t)


def meeting_confidence_gate(pa,pb,config):
    """One gate, either per-endpoint product or on the joint probability.

    Absent setting preserves historical static/dynamic experiment behavior.
    The separate probability mass multiplier remains the caller's responsibility.
    """
    mode=config.get('meeting_gate_mode','joint' if config.get('bucket_uv_dynamic_same_sheet',False) else 'endpoints')
    if mode not in ('joint','endpoints'):raise ValueError('meeting_gate_mode must be joint or endpoints')
    if not config.get('geometry_confidence_gated',False) and not config.get('bucket_uv_dynamic_same_sheet',False):
        return (pa*pb)*0+1
    return confidence_gate(pa*pb,config) if mode=='joint' else confidence_gate(pa,config)*confidence_gate(pb,config)
