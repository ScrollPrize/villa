#pragma once

#include <QString>
#include <QStringList>

namespace vc3d {

// The executable of command-line tool `name`: from `envVar` when set, the
// tools/<name>_path (or tools/<name>) setting, `extraPaths`, beside the VC3D
// executable or in the usual install layouts around it, then PATH. Empty when
// there is none.
QString findToolExecutable(const QString& name, const QStringList& extraPaths = {}, const QString& envVar = {});

} // namespace vc3d
