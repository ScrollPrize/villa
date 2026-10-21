"""Canonical inward solution gauge; historical unmarked solutions were outward."""
import numpy as np


def inward_solution(solution):
    result={key:np.asarray(solution[key]).copy() for key in solution}
    convention=str(result.get('normal_convention',np.asarray('outward')).item())
    if convention not in ('inward','outward'):
        raise ValueError(f'Unknown normal convention: {convention}')
    if convention=='outward':
        result['normal_sign']=-result['normal_sign']
        result['sign_logit']=-result['sign_logit']
        result['sign_probability']=1-result['sign_probability']
    result['normal_convention']=np.asarray('inward')
    return result
