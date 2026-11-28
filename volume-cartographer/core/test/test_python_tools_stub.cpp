// The Python tools prompt for tests that compile a feature's panel but not the
// installer: these builds ship no installer, so features always go ahead
#include "PythonToolsInstaller.hpp"

#include <QObject>

namespace vc3d {

bool requirePythonTools(const QString&, const QString&, const QString&, QWidget*, std::function<void()>)
{
    return true;
}

QString pythonToolsMissingMessage(const QString& feature)
{
    return QObject::tr("%1 needs the Python tools.").arg(feature);
}

}  // namespace vc3d
