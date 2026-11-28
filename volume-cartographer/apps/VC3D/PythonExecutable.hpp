#pragma once

#include <QProcessEnvironment>
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

// The interpreter the user chose for the services VC3D starts, or empty:
// `preferred` when it is an executable file, else the python of the active
// virtualenv or conda env. Unlike pythonExecutableForServices it never falls
// back to the installed Python tools or to PATH, so a feature can tell
// whether the user set up a Python of their own.
QString chosenPythonForServices(const QString& preferred = {});

// Records `name` as VC3D was launched with it (set or unset), before VC3D
// overrides it for its own process, such as the OMP_NUM_THREADS=1 cap at
// startup. Only the first call for a name counts. Safe to call before static
// initialization (from .preinit_array): it uses constant-initialized storage.
void rememberLaunchEnvironmentValue(const char* name);

// The environment for the Python services VC3D starts: VC3D's own, with every
// variable passed to rememberLaunchEnvironmentValue() back at its launch value,
// so a service's PyTorch and NumPy get the thread counts the user chose rather
// than the caps VC3D sets for itself. Native tools keep VC3D's caps.
QProcessEnvironment pythonServiceEnvironment();

}  // namespace vc3d
