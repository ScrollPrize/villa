#pragma once

#include "elements/VolumeSelector.hpp"
#include <QString>

class QWidget;

struct GrowPatchSeedParams {
    QString volumeId;          // vpkg volume id; empty => current volume
    int     iterations{200};   // Default: 1–100000; sheet modes: 1–1024 cells per side.
    double  minAreaCm{0.002};  // >= 0
    QString growthMethod{QStringLiteral("default")};
    double meshSpacing{20.0};
    double integrationStep{2.0};
    QString integrator{QStringLiteral("midpoint")};
    int workers{1};
    QString outputTag{QStringLiteral("paths")};  // tag of the new surface
};

bool selectGrowPatchSeedParams(QWidget* parent,
                              const QVector<VolumeSelector::VolumeOption>& volumes,
                              GrowPatchSeedParams& params);
