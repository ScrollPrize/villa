import numpy as np
import pytest
from helirec import _core


def test_wiener_transfer_and_smear_known_answers():
    f = np.linspace(.02,.48,60)
    truth = [23.,.35,.7,2.2]
    k2 = (2*np.pi*f)**2
    gain = (1+truth[0]*k2)*(truth[1]*np.exp(-.5*truth[2]**2*k2)+(1-truth[1])*np.exp(-.5*truth[3]**2*k2))
    fit = _core.calibration_fit_transfer(f.tolist(),gain.tolist())
    assert fit["usable"] and fit["solver_version"]
    assert fit["x"] == pytest.approx(truth, rel=1e-5)
    assert fit["cost"] < 1e-14
    np.testing.assert_allclose(_core.calibration_transfer_model(truth,f.tolist()),gain,rtol=1e-14)
    for smear in [.15,1.3,1.9]:
        result = _core.calibration_fit_smear(f.tolist(),np.sinc(smear*f).tolist())
        assert result["usable"] and result["x"][0] == pytest.approx(smear,abs=1e-6)
    for freq, values in [([],[]),([0]*5,[1]*5),([.1]*5,[1]*5),([.1,.2,.3,.4],[1,0,1,1])]:
        with pytest.raises(Exception): _core.calibration_fit_transfer(freq,values)
