#pragma once

// Pure helpers behind segmentation.set_settings, kept free of widgets so the
// contract test can check them without an application.

#include <utility>

#include <QJsonObject>
#include <QString>

#include "agent_bridge/AgentBridgeError.hpp"

// The Z range bounds a segmentation.set_settings request applies. A bound the
// request leaves out keeps the panel's value. A request that makes min exceed
// max is refused with -32602 and data.param "zRange". Bounds the panel already
// holds are accepted back as they are, even when min exceeds max: the panel's
// two spin boxes do not order each other, and get_settings reports them
// unchanged so that its reply can be passed to set_settings.
inline std::pair<int, int> agentBridgeZRangeUpdate(const QJsonObject& zRange,
                                                   int currentMin,
                                                   int currentMax)
{
    const int zMin = zRange.value(QStringLiteral("min")).toInt(currentMin);
    const int zMax = zRange.value(QStringLiteral("max")).toInt(currentMax);
    const bool keepsPanelBounds = zMin == currentMin && zMax == currentMax;
    if (zMin > zMax && !keepsPanelBounds) {
        throw AgentBridgeError{
            -32602, QStringLiteral("zRange.min must not exceed zRange.max"),
            QJsonObject{{QStringLiteral("param"), QStringLiteral("zRange")}}};
    }
    return {zMin, zMax};
}
