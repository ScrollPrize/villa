"""Fixed cyclic coordinate search and exact optimistic screening bounds."""
from .fiber_paired_weight_search import better


def optimistic_rank(free):
    return dict(passed=True,
                perp_sign_sum=free['signs']['perpendicular']['violations'],
                perp_l1_sum=free['score']['gt_perpendicular']['l1'])


def can_improve(free, incumbent, tolerance=1e-6):
    # The uncomputed GT-assisted contribution has lower bound zero for both
    # counts and L1. Never infer or extrapolate its actual result.
    return better(optimistic_rank(free), incumbent, tolerance)


def cyclic_refine(initial, evaluate, evaluate_many, parameters, factors, bounds,
                  accepted, tolerance=1e-6, absolute_values=None, batch_size=5,
                  progress=None):
    current=dict(initial); best=evaluate(current)
    sweeps=0
    while True:
        sweeps+=1; sweep_improved=False
        for name in parameters:
            # Only this coordinate restarts on acceptance, never the parameter loop.
            while True:
                lo,hi=bounds[name]
                values=[float(f'{current[name]*f:.12g}') for grid in factors for f in grid]
                values+=list((absolute_values or {}).get(name,[]))
                values=[v for v in dict.fromkeys(values) if v!=current[name] and lo<=v<=hi]
                improved=False
                for start in range(0,len(values),batch_size):
                    candidates=[dict(current,**{name:v}) for v in values[start:start+batch_size]]
                    if progress:progress(sweeps,name,candidates,best)
                    results=evaluate_many(candidates,best['ranking'])
                    winner=best
                    for result in results:
                        if better(result['ranking'],winner['ranking'],tolerance):winner=result
                    if winner is not best:
                        old=best;best=winner;current=dict(best['parameters'])
                        accepted(name,current,best,old)
                        improved=sweep_improved=True
                        break
                if not improved:break
        if not sweep_improved:return best
