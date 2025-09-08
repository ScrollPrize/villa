#include "PathData.hpp"
#include <QPainterPath>
#include <QPointF>
#include <cmath>



PathData PathData::densify(float samplingInterval) const 
{
    if (points.size() < 2) {
        // Return a copy if path has less than 2 points
        return *this;
    }
    
    // Create a QPainterPath using the same logic as CVolumeViewer::renderPaths()
    QPainterPath painterPath;
    bool firstPoint = true;
    
    for (const auto& pt : points) {
        if (firstPoint) {
            painterPath.moveTo(pt[0], pt[1]);
            firstPoint = false;
        } else {
            painterPath.lineTo(pt[0], pt[1]);
        }
    }
    
    // Get the total length of the path
    float totalLength = painterPath.length();
    if (totalLength <= 0) {
        return *this;
    }
    
    // Calculate number of samples based on sampling interval
    int numSamples = static_cast<int>(std::ceil(totalLength / samplingInterval));
    if (numSamples < 2) {
        return *this;
    }
    
    // Sample points along the path
    std::vector<cv::Vec3f> densifiedPoints;
    densifiedPoints.reserve(numSamples);
    
    for (int i = 0; i < numSamples; i++) {
        float percent = static_cast<float>(i) / (numSamples - 1);
        QPointF sampledPoint = painterPath.pointAtPercent(percent);
        
        // For Z coordinate, interpolate between the original points
        float z = interpolateZ(percent, totalLength, painterPath);
        
        densifiedPoints.emplace_back(sampledPoint.x(), sampledPoint.y(), z);
    }
    
    // Create new PathData with densified points
    PathData result = *this;
    result.points = std::move(densifiedPoints);
    
    return result;
}

float PathData::interpolateZ(float percent, float totalLength, const QPainterPath& path) const
{
    if (points.size() < 2) {
        return points.empty() ? 0.0f : points[0][2];
    }
    
    // Find the segment length at the given percent
    float targetLength = percent * totalLength;
    
    // Walk through original points to find which segment we're in
    float accumulatedLength = 0.0f;
    for (size_t i = 1; i < points.size(); i++) {
        cv::Vec3f p1 = points[i-1];
        cv::Vec3f p2 = points[i];
        
        float segmentLength = std::sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + 
                                       (p2[1] - p1[1]) * (p2[1] - p1[1]));
        
        if (accumulatedLength + segmentLength >= targetLength) {
            // We're in this segment, interpolate Z
            float segmentPercent = (targetLength - accumulatedLength) / segmentLength;
            return p1[2] + segmentPercent * (p2[2] - p1[2]);
        }
        
        accumulatedLength += segmentLength;
    }
    
    // If we get here, return the Z of the last point
    return points.back()[2];
}


