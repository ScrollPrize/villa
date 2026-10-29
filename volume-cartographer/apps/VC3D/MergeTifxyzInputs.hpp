#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

// Inputs of vc_merge_tifxyz shared by the Merge Surfaces dialog and the
// dialog-free merge launch: the surface grid and its merge.json form.
namespace vc3d::merge_tifxyz {

// A row-major grid of surface UUIDs; an empty cell is a hole.
using Grid = QVector<QStringList>;

// The grid's distinct surfaces, in row-major order of first appearance.
QStringList distinctSurfaces(const Grid& rows);

// Pretty merge.json text. Solid grids use whitespace-delimited string rows;
// grids with holes use JSON array rows with null cells.
QString gridJsonText(const Grid& rows);

// Rejects a grid with fewer than two distinct surfaces, surfaces missing from
// `availableSurfaces` (the current volume's), names with whitespace, a
// reference surface outside the grid, or RANSAC thresholds with min >= max.
// Returns an empty string when valid.
QString validate(const Grid& rows,
                 const QStringList& availableSurfaces,
                 const QString& refSurface,
                 double ransacMinThresh,
                 double ransacMaxThresh);

}  // namespace vc3d::merge_tifxyz
