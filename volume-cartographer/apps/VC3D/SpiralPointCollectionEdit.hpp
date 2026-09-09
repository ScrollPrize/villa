#pragma once

#include <QJsonDocument>
#include <QJsonObject>
#include <QPointF>
#include <QString>

#include <opencv2/core/types.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace vc3d::spiral {

struct EditablePclPoint {
    cv::Vec3f previewPosition{0.0f, 0.0f, 0.0f};
    QJsonObject sourcePayload;
    bool sourcePoint = true;
    // Q-placed points retain their exact coordinate on the preview surface so
    // drawing them does not depend on projecting their rounded 3D position
    // back through the surface index. Imported points have no such coordinate.
    std::optional<QPointF> previewSurfacePosition;
};

// One editable collection. Point order, preview geometry, and the source JSON
// payload intentionally live in the same vector so an edit cannot reorder one
// without the others.
struct EditablePclDraft {
    QString collectionId;
    QJsonObject topLevel;
    QJsonObject sourceCollection;
    std::vector<EditablePclPoint> points;
    double sourceToPreviewScale = 1.0;
    QString sourceRevision;
    bool editable = false;
    bool dirty = false;
    bool deleted = false;
    bool dirtyBeforeDelete = false;
    bool submissionBlocked = false;

    void appendPreviewPoint(
        const cv::Vec3f& previewPosition,
        std::optional<QPointF> previewSurfacePosition = std::nullopt);
    bool erase(std::size_t index);
    void reverse();
    void setDeleted(bool value);
    // Retained storage for projection callers whose cache keys include the
    // vector's address. Rebuilt from `points` on every access so public draft
    // edits cannot leave stale coordinates behind, while the backing storage
    // remains owned by this collection instead of a render-loop temporary.
    [[nodiscard]] const std::vector<cv::Vec3f>& projectionPositions() const;
    [[nodiscard]] bool isIncompleteNewCollection() const
    {
        return collectionId.isEmpty() && points.size() < 2;
    }
    QJsonDocument replacementDocument() const;

private:
    mutable std::vector<cv::Vec3f> _projectionPositions;
};

constexpr qreal kEditablePclHighlightScale = 1.4;
constexpr qreal editablePclPointRadius(qreal baseRadius, bool highlighted)
{
    return highlighted ? baseRadius * kEditablePclHighlightScale : baseRadius;
}

// Imports collections and points in numeric JSON-key order. Collections whose
// point or collection links could be invalidated by renumbering are read-only.
std::vector<EditablePclDraft> importEditablePcls(
    const QJsonDocument& document, double sourceToPreviewScale,
    const QString& sourceRevision, bool sourceEditable);

} // namespace vc3d::spiral
