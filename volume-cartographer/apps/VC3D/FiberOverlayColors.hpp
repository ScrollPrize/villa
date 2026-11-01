#pragma once

#include <QColor>

namespace vc3d
{

// Shared colors for AFV slice overlays and the 3D fiber view.
inline QColor horizontalFiberColor()
{
    return QColor(64, 165, 255);
}

inline QColor verticalFiberColor()
{
    return QColor(255, 173, 67);
}

inline QColor selectedFiberColor()
{
    return QColor(255, 255, 102);
}

}  // namespace vc3d
