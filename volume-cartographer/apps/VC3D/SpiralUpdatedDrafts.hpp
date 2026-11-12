#pragma once

#include <QHash>
#include <QString>
#include <QStringList>
#include <QtGlobal>

#include <cstdint>
#include <functional>

namespace vc3d {

// Which of the drafts saved to the attached store when an input update
// began (draft id -> the version its object had then) the fit holds now
// that the update finished, so they are shown as its inputs and dropped as
// drawings. None when the update failed, or when it named no position
// (`requested` < 0): then the service's store is not the one attached
// here, so the drafts never reached the fit. A draft whose object moved on
// (saved again meanwhile) or whose drawing changed stays a drawing whose
// next save replaces its object.
template<class Version, class Current, class Unchanged>
inline QStringList spiralDraftsIncludedByUpdate(
    const QString& updateError, qint64 requested, const QHash<QString, Version>& saved,
    const Current& currentVersion, const Unchanged& drawingUnchanged)
{
    QStringList ids;
    if (!updateError.isEmpty() || requested < 0) return ids;
    for (auto it = saved.cbegin(); it != saved.cend(); ++it)
        if (currentVersion(it.key()) == it.value() && drawingUnchanged(it.key())) ids << it.key();
    return ids;
}

} // namespace vc3d
