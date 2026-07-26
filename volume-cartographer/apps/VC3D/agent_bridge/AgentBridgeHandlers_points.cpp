#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"

#include "vc/core/PointCollections.hpp"
#include "vc/ui/VCCollection.hpp"

// The points.* family: the live VCCollection editing surface (points,
// collections, winding fills, tags, IO). Handlers resolve the store via
// requirePointStore() and, where named, resolveCollectionId().

namespace {

// Resolve the live point-collection store or throw the same errors the whole
// points.* family shares: -32000 when no package is loaded, -32010 when the
// store itself is missing (should not happen once a vpkg is loaded).
VCCollection* requirePointStore(CState* state)
{
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    VCCollection* pc = state->pointCollection();
    if (!pc) {
        QJsonObject data;
        data["detail"] = "point collection store is unavailable";
        throw AgentBridgeError{-32010, "Internal error", data};
    }
    return pc;
}

[[noreturn]] void throwUnknownCollection(const QString& id)
{
    QJsonObject data;
    data["kind"] = "collection";
    data["id"] = id;
    throw AgentBridgeError{-32007, QStringLiteral("Unknown collection: %1").arg(id), data};
}

[[noreturn]] void throwUnknownPoint(uint64_t id)
{
    QJsonObject data;
    data["kind"] = "point";
    data["id"] = static_cast<double>(id);
    throw AgentBridgeError{-32007,
        QStringLiteral("Unknown point: %1").arg(id), data};
}

// Collection-identifying params accept EITHER `collection` (name) OR
// `collectionId` (number). The numeric id wins when both are present. Throws
// -32007 (kind:"collection") when the referenced collection does not exist.
uint64_t resolveCollectionId(VCCollection* pc, const QJsonObject& p)
{
    if (p.contains("collectionId")) {
        const uint64_t id =
            static_cast<uint64_t>(p.value("collectionId").toDouble());
        if (pc->getAllCollections().count(id) == 0)
            throwUnknownCollection(QString::number(id));
        return id;
    }

    if (!p.contains("collection"))
        throwParamError("collection", QStringLiteral("is required"));

    const QString name = p.value("collection").toString();
    const uint64_t id = pc->getCollectionId(name.toStdString());  // 0 when absent
    if (id == 0)
        throwUnknownCollection(name);
    return id;
}

// [r, g, b] float triple (colors travel as an array, not an {x,y,z} object, so
// jsonToVec3 does not apply here).
cv::Vec3f jsonRequireColor(const QJsonValue& v, const char* paramName)
{
    const QJsonArray a = v.toArray();
    if (a.size() != 3)
        throwParamError(paramName, QStringLiteral("must have exactly 3 components"));
    return cv::Vec3f(
        static_cast<float>(jsonRequireFiniteFloat(a.at(0), paramName)),
        static_cast<float>(jsonRequireFiniteFloat(a.at(1), paramName)),
        static_cast<float>(jsonRequireFiniteFloat(a.at(2), paramName)));
}

QJsonArray colorToJson(const cv::Vec3f& c)
{
    QJsonArray a;
    a.append(static_cast<double>(c[0]));
    a.append(static_cast<double>(c[1]));
    a.append(static_cast<double>(c[2]));
    return a;
}

std::vector<uint64_t> jsonRequireIdArray(const QJsonValue& v, const char* paramName)
{
    Q_UNUSED(paramName);
    std::vector<uint64_t> ids;
    for (const QJsonValue& e : v.toArray())
        ids.push_back(static_cast<uint64_t>(e.toDouble()));
    return ids;
}

PointCollections::WindingFillMode windingFillModeFromString(const QString& s)
{
    if (s == QLatin1String("none"))        return PointCollections::WindingFillMode::None;
    if (s == QLatin1String("incremental")) return PointCollections::WindingFillMode::Incremental;
    if (s == QLatin1String("decremental")) return PointCollections::WindingFillMode::Decremental;
    if (s == QLatin1String("constant"))    return PointCollections::WindingFillMode::Constant;
    QJsonObject data;
    data["param"] = QStringLiteral("mode");
    data["value"] = s;
    throw AgentBridgeError{-32602, QStringLiteral("Invalid mode: %1").arg(s), data};
}

// winding_annotation carries NaN for "unset"; on the wire that is JSON null.
QJsonValue windingToJson(float winding)
{
    return std::isnan(winding) ? QJsonValue(QJsonValue::Null)
                               : QJsonValue(static_cast<double>(winding));
}

}  // namespace


// ---------------------------------------------------------------------------
// points.commit / points.list — authoring + read-back
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsCommit(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString collection = p.value("collection").toString();
    if (collection.isEmpty()) {
        QJsonObject data;
        data["param"] = "collection";
        throw AgentBridgeError{-32602, "collection name is required", data};
    }

    const QJsonValue pointsv = p.value("points");
    if (!pointsv.isArray() || pointsv.toArray().isEmpty()) {
        QJsonObject data;
        data["param"] = "points";
        throw AgentBridgeError{-32602, "points must be a non-empty array", data};
    }

    std::vector<cv::Vec3f> pts;
    for (const QJsonValue& pv : pointsv.toArray())
        pts.push_back(jsonToVec3(pv, "points"));  // validates finiteness

    std::optional<double> winding;
    if (p.contains("winding")) {
        // jsonRequireFiniteFloat (not jsonRequireFinite): winding is a float, so
        // an out-of-float-range value is rejected, not narrowed to +/-inf below.
        const double w = jsonRequireFiniteFloat(p.value("winding"), "winding");
        winding = w;
    }

    VCCollection* pc = state->pointCollection();
    if (!pc) {
        QJsonObject data;
        data["detail"] = "point collection store is unavailable";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    const std::string col = collection.toStdString();
    QJsonArray pointIds;
    for (const cv::Vec3f& v : pts) {
        ColPoint cp = pc->addPoint(col, v);
        if (winding) {
            cp.winding_annotation = static_cast<float>(*winding);
            pc->updatePoint(cp);
        }
        pointIds.append(static_cast<double>(cp.id));
    }

    QJsonObject result;
    result["collectionId"] = static_cast<double>(pc->getCollectionId(col));
    result["pointIds"] = pointIds;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsList(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = state ? state->pointCollection() : nullptr;

    const QString filter = p.value("collection").toString();

    QJsonArray collections;
    if (pc) {
        if (!filter.isEmpty() && pc->getCollectionId(filter.toStdString()) == 0) {
            QJsonObject data;
            data["kind"] = "collection";
            data["id"] = filter;
            throw AgentBridgeError{-32007, QStringLiteral("Unknown collection: %1").arg(filter), data};
        }

        for (const auto& [id, coll] : pc->getAllCollections()) {
            if (!filter.isEmpty() && coll.name != filter.toStdString())
                continue;

            QJsonObject c;
            c["id"] = static_cast<double>(id);
            c["name"] = QString::fromStdString(coll.name);
            QJsonArray color;
            color.append(coll.color[0]);
            color.append(coll.color[1]);
            color.append(coll.color[2]);
            c["color"] = color;

            QJsonArray pointsArr;
            for (const auto& [pid, cp] : coll.points) {
                QJsonObject po;
                po["id"] = static_cast<double>(pid);
                po["position"] = vec3ToJson(cp.p);
                po["winding"] = windingToJson(cp.winding_annotation);
                pointsArr.append(po);
            }
            c["points"] = pointsArr;
            collections.append(c);
        }
    } else if (!filter.isEmpty()) {
        QJsonObject data;
        data["kind"] = "collection";
        data["id"] = filter;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown collection: %1").arg(filter), data};
    }

    QJsonObject result;
    result["collections"] = collections;
    return result;
}


// ---------------------------------------------------------------------------
// Collection lifecycle: create / rename / clear
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsAddCollection(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    // Omitted name -> a generated unique name (mirrors the "New collection"
    // button); a present name must be a string.
    QString name = p.value("name").toString();
    if (name.isEmpty())
        name = QString::fromStdString(pc->generateNewCollectionName());

    const uint64_t id = pc->addCollection(name.toStdString());

    QJsonObject result;
    result["collectionId"] = static_cast<double>(id);
    result["name"] = name;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsRenameCollection(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const QString newName = p.value("newName").toString();
    // Reject empty: it would make the collection unreachable by name
    // (resolveCollectionId's name lookup treats "" the same as absent).
    if (newName.isEmpty())
        throwParamError("newName", QStringLiteral("must not be empty"));
    pc->renameCollection(id, newName.toStdString());

    QJsonObject result;
    result["collectionId"] = static_cast<double>(id);
    result["name"] = newName;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsClearCollection(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    pc->clearCollection(resolveCollectionId(pc, p));

    QJsonObject result;
    result["cleared"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsClearAll(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);
    Q_UNUSED(params);

    pc->clearAll();

    QJsonObject result;
    result["cleared"] = true;
    return result;
}


// ---------------------------------------------------------------------------
// Point mutation: update / remove
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsUpdatePoint(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t pointId =
        static_cast<uint64_t>(p.value("pointId").toDouble());
    std::optional<ColPoint> cp = pc->getPoint(pointId);
    if (!cp)
        throwUnknownPoint(pointId);

    if (p.contains("position"))
        cp->p = jsonToVec3(p.value("position"), "position");
    if (p.contains("winding")) {
        // Explicit null clears the annotation (null <-> NaN convention).
        // jsonRequireFiniteFloat rejects an out-of-float-range value that would
        // narrow to +/-inf and corrupt that convention (isnan() no longer "unset").
        const QJsonValue wv = p.value("winding");
        cp->winding_annotation = wv.isNull()
            ? std::numeric_limits<float>::quiet_NaN()
            : static_cast<float>(jsonRequireFiniteFloat(wv, "winding"));
    }
    pc->updatePoint(*cp);

    QJsonObject result;
    result["id"] = static_cast<double>(cp->id);
    result["position"] = vec3ToJson(cp->p);
    result["winding"] = windingToJson(cp->winding_annotation);
    return result;
}


QJsonObject AgentBridgeServer::handlePointsRemovePoint(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t pointId =
        static_cast<uint64_t>(p.value("pointId").toDouble());
    if (!pc->getPoint(pointId))
        throwUnknownPoint(pointId);
    pc->removePoint(pointId);

    QJsonObject result;
    result["removed"] = true;
    return result;
}


// ---------------------------------------------------------------------------
// Collection attributes: color / metadata / tags / winding links
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsSetCollectionColor(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const cv::Vec3f color = jsonRequireColor(p.value("color"), "color");
    pc->setCollectionColor(id, color);

    QJsonObject result;
    result["collectionId"] = static_cast<double>(id);
    result["color"] = colorToJson(color);
    return result;
}


QJsonObject AgentBridgeServer::handlePointsSetCollectionMetadata(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    CollectionMetadata meta;
    meta.absolute_winding_number =
        p.value("absoluteWindingNumber").toBool();
    pc->setCollectionMetadata(id, meta);

    QJsonObject result;
    result["collectionId"] = static_cast<double>(id);
    result["absoluteWindingNumber"] = meta.absolute_winding_number;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsSetCollectionTag(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const QString key = p.value("key").toString();
    const QString value = p.value("value").toString();
    pc->setCollectionTag(id, key.toStdString(), value.toStdString());

    QJsonObject result;
    result["ok"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsRemoveCollectionTag(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const QString key = p.value("key").toString();
    pc->removeCollectionTag(id, key.toStdString());

    QJsonObject result;
    result["ok"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsSetWindingsLinked(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const std::vector<uint64_t> linked =
        jsonRequireIdArray(p.value("linkedCollectionIds"), "linkedCollectionIds");
    pc->setCollectionWindingsLinked(id, linked);

    QJsonArray echo;
    for (uint64_t l : linked)
        echo.append(static_cast<double>(l));
    QJsonObject result;
    result["collectionId"] = static_cast<double>(id);
    result["linkedCollectionIds"] = echo;
    return result;
}


// ---------------------------------------------------------------------------
// Winding fills
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsAutoFillWindings(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const PointCollections::WindingFillMode mode =
        windingFillModeFromString(p.value("mode").toString());
    // constant only matters for Constant mode; default 0 when omitted.
    // jsonRequireFiniteFloat rejects an out-of-float-range value before the cast
    // narrows it to +/-inf.
    const float constant = p.contains("constant")
        ? static_cast<float>(jsonRequireFiniteFloat(p.value("constant"), "constant"))
        : 0.0f;
    pc->autoFillWindingNumbers(id, mode, constant);

    QJsonObject result;
    result["ok"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsSetAutoFillMode(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const uint64_t id = resolveCollectionId(pc, p);
    const PointCollections::WindingFillMode mode =
        windingFillModeFromString(p.value("mode").toString());
    // Same float-narrowing concern as autoFillWindingNumbers above.
    const float constant = p.contains("constant")
        ? static_cast<float>(jsonRequireFiniteFloat(p.value("constant"), "constant"))
        : 0.0f;
    pc->setAutoFillMode(id, mode, constant);

    QJsonObject result;
    result["ok"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsResetWindings(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);
    Q_UNUSED(params);

    pc->resetWindingNumbers();

    QJsonObject result;
    result["ok"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handlePointsApplyAnchorOffset(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    // jsonRequireFiniteFloat (not jsonRequireFinite): the offsets are float
    // fields, so reject an out-of-float-range value rather than narrow to +/-inf.
    const float offsetX = static_cast<float>(jsonRequireFiniteFloat(p.value("offsetX"), "offsetX"));
    const float offsetY = static_cast<float>(jsonRequireFiniteFloat(p.value("offsetY"), "offsetY"));
    pc->applyAnchorOffset(offsetX, offsetY);

    QJsonObject result;
    result["ok"] = true;
    return result;
}


// ---------------------------------------------------------------------------
// Persistence: whole-collection JSON + per-segment correction paths
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handlePointsSaveJson(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const QString path = p.value("path").toString();

    QJsonObject result;
    result["saved"] = pc->saveToJSON(path.toStdString());
    return result;
}


QJsonObject AgentBridgeServer::handlePointsLoadJson(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const QString path = p.value("path").toString();

    QJsonObject result;
    result["loaded"] = pc->loadFromJSON(path.toStdString());
    return result;
}


QJsonObject AgentBridgeServer::handlePointsSaveSegmentPath(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const QString segmentPath = p.value("segmentPath").toString();

    QJsonObject result;
    result["saved"] = pc->saveToSegmentPath(std::filesystem::path(segmentPath.toStdString()));
    return result;
}


QJsonObject AgentBridgeServer::handlePointsLoadSegmentPath(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    VCCollection* pc = requirePointStore(state);

    const QString segmentPath = p.value("segmentPath").toString();

    QJsonObject result;
    result["loaded"] = pc->loadFromSegmentPath(std::filesystem::path(segmentPath.toStdString()));
    return result;
}
