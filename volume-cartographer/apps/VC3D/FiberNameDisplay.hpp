#pragma once

#include <QFontMetrics>
#include <QString>

namespace vc3d {

QString displayStemForFiberFile(QString fileName);
QString adaptFiberNameToWidth(const QString& name, const QFontMetrics& metrics, int width);

} // namespace vc3d
