#pragma once

#include <QString>

#include <opencv2/core/matx.hpp>

#include <cmath>
#include <vector>

struct AtlasControlPointResult {
    QString fiberId;
    QString objectId;
    int sourceIndex = -1;
    int controlIndex = -1;
    int layerIndex = -1;
    bool valid = false;
    float distance = NAN;
    float signedDelta = NAN;
    QString snapStatus;
    bool snapValid = false;
    bool hasSnapStatus = false;
    float snapSignedDelta = NAN;
    cv::Vec3f snapTargetXyz{NAN, NAN, NAN};
    cv::Vec3f targetXyz{NAN, NAN, NAN};
    cv::Vec3f meshXyz{NAN, NAN, NAN};
    float modelH = NAN;
    float modelW = NAN;
};

using AtlasControlPointResults = std::vector<AtlasControlPointResult>;
