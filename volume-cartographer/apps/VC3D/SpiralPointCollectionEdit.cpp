#include "SpiralPointCollectionEdit.hpp"

#include <QJsonArray>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace vc3d::spiral {
namespace {

std::optional<qulonglong> numericKey(const QString& key)
{
    bool ok = false;
    const qulonglong value = key.toULongLong(&ok, 10);
    return ok ? std::optional<qulonglong>(value) : std::nullopt;
}

QStringList numericKeys(const QJsonObject& object)
{
    QStringList keys;
    for (const QString& key : object.keys()) {
        if (numericKey(key)) keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end(), [](const QString& left, const QString& right) {
        const auto l = *numericKey(left);
        const auto r = *numericKey(right);
        return l != r ? l < r : left < right;
    });
    return keys;
}

bool nonEmptyArray(const QJsonValue& value)
{
    return value.isArray() && !value.toArray().isEmpty();
}

bool collectionHasAffectedLinks(const QJsonObject& collections,
                                const QString& targetId,
                                const QJsonObject& target)
{
    if (nonEmptyArray(target.value(QStringLiteral("windings_linked")))) return true;

    QSet<qulonglong> targetPointIds;
    const QJsonObject targetPoints = target.value(QStringLiteral("points")).toObject();
    for (const QString& pointId : numericKeys(targetPoints)) {
        targetPointIds.insert(*numericKey(pointId));
        if (nonEmptyArray(targetPoints.value(pointId).toObject()
                              .value(QStringLiteral("links")))) return true;
    }

    const auto targetNumeric = numericKey(targetId);
    for (auto it = collections.begin(); it != collections.end(); ++it) {
        if (it.key() == targetId || !it.value().isObject()) continue;
        const QJsonObject collection = it.value().toObject();
        for (const QJsonValue& linked :
             collection.value(QStringLiteral("windings_linked")).toArray()) {
            if (targetNumeric && linked.toVariant().toULongLong() == *targetNumeric)
                return true;
        }
        // Point ids are file-global in PointCollections v1. A link from any
        // other collection to an id owned by the target would be invalidated
        // by its contiguous renumbering.
        const QJsonObject points = collection.value(QStringLiteral("points")).toObject();
        for (auto point = points.begin(); point != points.end(); ++point) {
            for (const QJsonValue& linked :
                 point.value().toObject().value(QStringLiteral("links")).toArray()) {
                if (targetPointIds.contains(linked.toVariant().toULongLong()))
                    return true;
            }
        }
    }
    return false;
}

} // namespace

void EditablePclDraft::appendPreviewPoint(
    const cv::Vec3f& previewPosition,
    std::optional<QPointF> previewSurfacePosition)
{
    QJsonObject payload;
    const double inverseScale = 1.0 / sourceToPreviewScale;
    payload[QStringLiteral("p")] = QJsonArray{
        static_cast<double>(previewPosition[0]) * inverseScale,
        static_cast<double>(previewPosition[1]) * inverseScale,
        static_cast<double>(previewPosition[2]) * inverseScale,
    };
    payload[QStringLiteral("wind_a")] = QJsonValue::Null;
    points.push_back({previewPosition, std::move(payload), false,
                      std::move(previewSurfacePosition)});
    dirty = true;
}

bool EditablePclDraft::erase(std::size_t index)
{
    if (index >= points.size()) return false;
    points.erase(points.begin() + static_cast<std::ptrdiff_t>(index));
    dirty = true;
    return true;
}

void EditablePclDraft::reverse()
{
    std::reverse(points.begin(), points.end());
    dirty = true;
}

void EditablePclDraft::setDeleted(bool value)
{
    if (deleted == value) return;
    if (value) {
        dirtyBeforeDelete = dirty;
        deleted = true;
        dirty = true;
    } else {
        deleted = false;
        dirty = dirtyBeforeDelete;
    }
}

const std::vector<cv::Vec3f>& EditablePclDraft::projectionPositions() const
{
    _projectionPositions.clear();
    if (deleted) return _projectionPositions;
    _projectionPositions.reserve(points.size());
    for (const EditablePclPoint& point : points)
        _projectionPositions.push_back(point.previewPosition);
    return _projectionPositions;
}

QJsonDocument EditablePclDraft::replacementDocument() const
{
    QJsonObject pointsObject;
    qint64 firstCreationTime = 0;
    bool foundCreationTime = false;
    for (const EditablePclPoint& point : points) {
        const QJsonValue value = point.sourcePayload.value(QStringLiteral("creation_time"));
        if (!value.isDouble()) continue;
        const qint64 candidate = value.toInteger();
        if (!foundCreationTime || candidate < firstCreationTime) {
            firstCreationTime = candidate;
            foundCreationTime = true;
        }
    }
    if (!foundCreationTime) firstCreationTime = 0;
    const qint64 maximumBase = std::numeric_limits<qint64>::max()
        - static_cast<qint64>(points.size());
    firstCreationTime = std::min(firstCreationTime, maximumBase);

    for (std::size_t index = 0; index < points.size(); ++index) {
        QJsonObject payload = points[index].sourcePayload;
        payload[QStringLiteral("creation_time")] =
            firstCreationTime + static_cast<qint64>(index);
        pointsObject[QString::number(index)] = payload;
    }

    QJsonObject collection = sourceCollection;
    collection[QStringLiteral("points")] = pointsObject;
    QJsonObject root = topLevel;
    root[QStringLiteral("vc_pointcollections_json_version")] = QStringLiteral("1");
    root[QStringLiteral("collections")] =
        QJsonObject{{collectionId, collection}};
    return QJsonDocument(root);
}

std::vector<EditablePclDraft> importEditablePcls(
    const QJsonDocument& document, double sourceToPreviewScale,
    const QString& sourceRevision, bool sourceEditable)
{
    std::vector<EditablePclDraft> result;
    if (!document.isObject() || !std::isfinite(sourceToPreviewScale)
        || sourceToPreviewScale <= 0.0) return result;
    const QJsonObject root = document.object();
    const QJsonObject collections = root.value(QStringLiteral("collections")).toObject();
    for (const QString& collectionId : numericKeys(collections)) {
        const QJsonObject collection = collections.value(collectionId).toObject();
        const QJsonObject sourcePoints = collection.value(QStringLiteral("points")).toObject();
        EditablePclDraft draft;
        draft.collectionId = collectionId;
        draft.topLevel = root;
        draft.sourceCollection = collection;
        draft.sourceToPreviewScale = sourceToPreviewScale;
        draft.sourceRevision = sourceRevision;
        draft.editable = sourceEditable
            && !collectionHasAffectedLinks(collections, collectionId, collection);
        const QStringList orderedPointIds = numericKeys(sourcePoints);
        if (orderedPointIds.size() != sourcePoints.size()) draft.editable = false;
        for (const QString& pointId : orderedPointIds) {
            const QJsonObject payload = sourcePoints.value(pointId).toObject();
            const QJsonArray position = payload.value(QStringLiteral("p")).toArray();
            if (position.size() != 3) continue;
            const cv::Vec3f preview{
                static_cast<float>(position[0].toDouble() * sourceToPreviewScale),
                static_cast<float>(position[1].toDouble() * sourceToPreviewScale),
                static_cast<float>(position[2].toDouble() * sourceToPreviewScale),
            };
            if (!std::isfinite(preview[0]) || !std::isfinite(preview[1])
                || !std::isfinite(preview[2])) {
                draft.editable = false;
                continue;
            }
            draft.points.push_back({preview, payload, true});
        }
        if (draft.points.size() < 2) draft.editable = false;
        result.push_back(std::move(draft));
    }
    return result;
}

} // namespace vc3d::spiral
