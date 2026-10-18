#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

// Inputs of vc_merge_tifxyz shared by the Merge tifxyz dialog and the
// dialog-free merge launch: the surface grid, its merge.json form, and the
// output name the tool derives from it.
namespace vc3d::merge_tifxyz {

// A row-major grid of segment directory names; an empty name is a hole.
using Grid = QVector<QStringList>;

// Alphabetically first non-empty name.
QString alphaFirst(const QStringList& names);

// `base`, or the first free `<base>_v<n>` under <volpkgDir>/paths.
QString resolveOutputName(const QString& volpkgDir, const QString& base);

// Pretty merge.json text. Solid grids use whitespace-delimited string rows;
// grids with holes use JSON array rows with null cells.
QString gridJsonText(const Grid& rows);

// Rejects a grid with fewer than two distinct surfaces, names missing from
// `availableSegments` (under `pathsDir`), names with whitespace, a reference
// surface outside the grid, or RANSAC thresholds with min >= max. Returns an
// empty string when valid.
QString validate(const Grid& rows,
                 const QStringList& availableSegments,
                 const QString& pathsDir,
                 const QString& refSurface,
                 double ransacMinThresh,
                 double ransacMaxThresh);

// Writes <volpkgDir>/<output name>.merge.json and returns its path, or an
// empty string with `error` set.
QString writeMergeJson(const QString& volpkgDir, const Grid& rows, QString* error);

}  // namespace vc3d::merge_tifxyz
