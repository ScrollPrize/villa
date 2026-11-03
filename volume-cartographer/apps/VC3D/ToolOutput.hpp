#pragma once

#include <QString>
#include <QStringList>

namespace vc3d {

// A tool's output as complete lines: output arrives in arbitrary chunks, so
// the unterminated tail waits for the next chunk (or finish()).
class ToolOutputLines {
public:
    QStringList add(const QString& chunk)
    {
        QStringList lines = (partial_ + chunk).split('\n');
        partial_ = lines.takeLast();
        return lines;
    }
    QStringList finish()
    {
        QStringList lines;
        if (!partial_.isEmpty()) lines << partial_;
        partial_.clear();
        return lines;
    }

private:
    QString partial_;
};

// The surface a store-mode tool reports creating ("surface_uuid: <uuid>"); empty for other lines.
inline QString reportedSurfaceUuid(const QString& line)
{
    const QString trimmed = line.trimmed();
    return trimmed.startsWith(QStringLiteral("surface_uuid:")) ? trimmed.mid(13).trimmed() : QString();
}

} // namespace vc3d
