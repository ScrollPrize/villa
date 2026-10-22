"""Geometric layer order, independent of measured Lasagna winding distance."""
def order_margin(delta, sign, family_delta, parallel, leeway=.5):
    """Positive means permitted order. H=0, V=1; sign + means B outside A.

    Outward H->V requires delta >= -leeway; outward V->H requires
    delta >= 1-leeway. Parallel order retains its zero boundary.
    Inputs may be NumPy arrays or Torch tensors. No measurement calibration.
    """
    if not 0 <= leeway <= .5:
        raise ValueError('sign_order_leeway must be in [0, .5]')
    outward_family = sign * family_delta
    threshold = (~parallel) * ((1 - outward_family) * .5 - leeway)
    return sign * delta - threshold


def ordering(mapped, widths, d, config, saved=None):
    import torch
    from . import fiber_implicit_winding as density
    # Be safe for callers passing raw distribution centers: reports describe
    # one integer-label realization, never the fractional mean as geometry.
    mapped=density.map_state(mapped)
    i=torch.as_tensor(d['i']);j=torch.as_tensor(d['j'])
    family=torch.as_tensor(d['family'])
    if config.get('shared_curve_geometry'):
        from .fiber_shared_curve import saved_points
        if saved is None:raise ValueError('Shared-curve sign audit requires saved geometry')
        _,pa,_=saved_points(d,saved,mapped.numpy(),widths.numpy(),d['i'],d['ds_a'],
            shared_only=config.get('pair_predicted_winding',False))
        _,pb,_=saved_points(d,saved,mapped.numpy(),widths.numpy(),d['j'],d['ds_b'],
            shared_only=config.get('pair_predicted_winding',False))
        pa,pb=torch.as_tensor(pa),torch.as_tensor(pb)
    else:
        _,pa,_=density.soft.points(mapped[i],family[i],torch.as_tensor(d['ds_a']),widths)
        _,pb,_=density.soft.points(mapped[j],family[j],torch.as_tensor(d['ds_b']),widths)
    separation=mapped[j,0]-mapped[i,0]+pb-pa
    return order_margin(separation,torch.as_tensor(d['sign']),family[j]-family[i],
        torch.as_tensor(d['parallel']).bool(),config.get('sign_order_leeway',.5))
