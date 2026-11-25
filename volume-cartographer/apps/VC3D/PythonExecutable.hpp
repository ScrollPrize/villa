#pragma once

#include <QString>

namespace vc3d {

// The Python tools installed for VC3D outside any shell environment: one
// virtual environment per tool family under envs/ (spiral, lasagna, vesuvius)
// and the villa sources their services run from under villa/. VC3D falls back
// to them when nothing else names an interpreter, as when it is opened from
// Finder, whose launchd PATH has no conda or Homebrew Python.
//
// The root is $VC3D_PYTHON_TOOLS, else python-tools/ in VC3D's application
// data directory (~/Library/Application Support/Vesuvius Challenge/VC3D on
// macOS).
QString installedPythonToolsRoot();

// The interpreter of the installed environment `name`, or empty when it is
// not installed or its install has not finished (no .vc3d-installed marker).
QString installedPythonEnvironment(const QString& name);

// `relative` inside the installed villa sources, or empty when it does not
// exist.
QString installedPythonToolsSource(const QString& relative);

// The Python interpreter for services VC3D starts: `preferred` when set, else
// the environment VC3D was started from - the active virtualenv
// ($VIRTUAL_ENV/bin/python), the active conda env ($CONDA_PREFIX/bin/python),
// then the installed vesuvius environment, then python3 / python on PATH.
// Virtualenv paths are kept as given (not canonicalized), so the venv's
// site-packages stay in effect.
QString pythonExecutableForServices(const QString& preferred = {});

}  // namespace vc3d
