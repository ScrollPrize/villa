#pragma once

#include <QString>

#include <cmath>
#include <optional>

// Fiber lengths as the panels display them. Lengths are measured in
// annotation-frame voxels (vc::atlas::fiberLineLengthVx over the stored line
// points); they read in centimetres only when the annotation frame's voxel
// size is known, and in voxels otherwise. There is deliberately no assumed
// voxel size here: a guessed conversion would present every figure as
// physical when it is not (see AnnotationFrame::voxelSizeUm).
namespace vc3d::fiber_length
{

constexpr double kUmPerCm = 10000.0;

// Decimals shown in centimetres: two, like the Fiber Map's distance ruler.
constexpr int kCmDecimals = 2;

// A voxel size that can carry a length into centimetres.
inline bool hasPhysicalScale(std::optional<double> voxelSizeUm)
{
    return voxelSizeUm && std::isfinite(*voxelSizeUm) && *voxelSizeUm > 0.0;
}

// The unit the panels report in: "cm" with a usable voxel size, else "vx".
inline QString unitLabel(std::optional<double> voxelSizeUm)
{
    return hasPhysicalScale(voxelSizeUm) ? QStringLiteral("cm") : QStringLiteral("vx");
}

// A voxel length in the display unit (centimetres, or the voxels themselves).
inline double displayValue(double lengthVx, std::optional<double> voxelSizeUm)
{
    if (hasPhysicalScale(voxelSizeUm)) {
        return lengthVx * *voxelSizeUm / kUmPerCm;
    }
    return lengthVx;
}

// The bare number for a table cell whose header names the unit: centimetres
// to kCmDecimals, voxels to vxDecimals. Non-finite lengths read as "-".
inline QString formatValue(double lengthVx,
                           std::optional<double> voxelSizeUm,
                           int vxDecimals = 1)
{
    if (!std::isfinite(lengthVx)) {
        return QStringLiteral("-");
    }
    const bool physical = hasPhysicalScale(voxelSizeUm);
    return QString::number(displayValue(lengthVx, voxelSizeUm), 'f',
                           physical ? kCmDecimals : vxDecimals);
}

// Value and unit together, for labels: "1.23 cm" or "412.0 vx".
inline QString formatLength(double lengthVx,
                            std::optional<double> voxelSizeUm,
                            int vxDecimals = 1)
{
    return QStringLiteral("%1 %2")
        .arg(formatValue(lengthVx, voxelSizeUm, vxDecimals), unitLabel(voxelSizeUm));
}

// A column header naming the unit its cells are in: "len (cm)" / "len (vx)".
inline QString columnHeader(const QString& base, std::optional<double> voxelSizeUm)
{
    return QStringLiteral("%1 (%2)").arg(base, unitLabel(voxelSizeUm));
}

}  // namespace vc3d::fiber_length
