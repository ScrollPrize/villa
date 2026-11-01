#pragma once

#include <QString>

namespace vc3d {

// The Python interpreter for services VC3D starts: `preferred` when set, else
// the environment VC3D was started from - the active virtualenv
// ($VIRTUAL_ENV/bin/python), the active conda env ($CONDA_PREFIX/bin/python),
// then python3 / python on PATH. Virtualenv paths are kept as given (not
// canonicalized), so the venv's site-packages stay in effect.
QString pythonExecutableForServices(const QString& preferred = {});

}  // namespace vc3d
