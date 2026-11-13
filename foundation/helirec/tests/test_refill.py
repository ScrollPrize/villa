"""Exposure-integrated refill shots, including the two row-time signs."""
import numpy as np
import pytest
from helirec import _core
from helirec.geometry import fit_refill_shots


def refill(sign=1, missing=False, noise=2e-5, shots=18):
    rows, frames, T, E, pre = 200, 420, .033, .027, 100
    tau, theta, raster = sign*.6*T/rows, .003, 11.37*T
    vc = np.repeat((np.arange(rows//20)+.5)*20-.5, 20)
    u = (vc-(rows-1)/2)/(rows/2)
    x = np.zeros((frames,rows))
    primitive = lambda s: -np.expm1(-np.maximum(s,0)/theta)
    for j in range(shots):
        if missing and j in (3,8,12): continue
        start = pre*T + .009 + j*raster
        s = np.arange(frames)[:,None]*T + tau*vc[None,:] - start
        x -= .012*(1+.05*u-.03*u*u)*(primitive(s+E)-primitive(s))
    x += np.random.default_rng(79).normal(0,noise,x.shape)
    return x, rows, T, E, pre, tau


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("missing", [False, True])
def test_refill_recovery_and_covariance(sign, missing):
    x,n,T,E,pre,tau = refill(sign, missing)
    result = fit_refill_shots(x,n,T,E,pre)
    assert result is not None and result["usable"]
    assert result["x"][0] == pytest.approx(tau, rel=.005)
    assert result["x"][1] == pytest.approx(.003, rel=.02)
    assert result["x"][3] == pytest.approx(11.37*T, rel=1e-4)
    assert result["uncertainty_available"] and 0 < result["row_time_se"] < abs(tau)*.01
    assert result["cost"] == pytest.approx(.5*np.sum(result["residual"]**2), rel=1e-10)
    assert len(result["shots"]) >= 12


def test_insufficient_shots_invalid_noise_and_invalid_inputs():
    x,n,T,E,pre,_ = refill(shots=5)
    assert fit_refill_shots(x,n,T,E,pre) is None
    assert _core.calibration_fit_refill_shots(x,T,E,pre)["termination"] == "INSUFFICIENT_SUPPORT"
    x,n,T,E,pre,_ = refill(noise=0)
    r = _core.calibration_fit_refill_shots(x,T,E,pre)
    assert not r["usable"] and r["termination"] == "INVALID_NOISE"
    assert not r["uncertainty_available"] and np.isnan(r["row_time_se"])
    for period, exposure, leading in [(0,E,pre),(T,-E,pre),(T,E,len(x)+1)]:
        with pytest.raises(Exception): _core.calibration_fit_refill_shots(x,period,exposure,leading)
    x[10,2] = np.nan
    with pytest.raises(Exception): _core.calibration_fit_refill_shots(x,T,E,pre)


def test_usable_fit_can_have_unavailable_uncertainty():
    # Identical row profiles and shots at one exposure phase do not identify
    # every timing direction, despite a finite minimum of the dip residuals.
    x = np.random.default_rng(6).normal(0, 2e-5, (420, 1))
    for j in range(18):
        x[100+15*j] -= .012
    r = _core.calibration_fit_refill_shots(np.repeat(x, 200, axis=1), .033, .027, 100)
    assert r["usable"] and not r["uncertainty_available"] and np.isnan(r["row_time_se"])
