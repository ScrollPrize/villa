#pragma once

#include <QColor>

#include <opencv2/core/types.hpp>

#include <array>
#include <cstddef>

namespace vc3d {

inline QColor surfaceOverlayColor(std::size_t index)
{
    static const std::array<QColor, 12> palette = {
        QColor(80, 180, 255), QColor(180, 80, 220), QColor(80, 220, 200),
        QColor(220, 80, 180), QColor(80, 130, 255), QColor(160, 80, 255),
        QColor(80, 255, 220), QColor(255, 80, 200), QColor(120, 220, 80),
        QColor(80, 180, 120), QColor(150, 200, 255), QColor(200, 150, 230),
    };
    return palette[index % palette.size()];
}

inline cv::Vec3b surfaceOverlayColorBgr(std::size_t index)
{
    const QColor color = surfaceOverlayColor(index);
    return {static_cast<uchar>(color.blue()),
            static_cast<uchar>(color.green()),
            static_cast<uchar>(color.red())};
}

} // namespace vc3d
