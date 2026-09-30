#pragma once

#include <QString>

namespace vc3d
{

// The first Python that runs among PYTHON_EXECUTABLE, the active conda
// environment, ~/miniconda3, ~/anaconda3 and the system Python; "python3" if
// none does. Each candidate is tried with a one-second timeout.
QString findPythonExecutable();

}  // namespace vc3d
