#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <string>
#include <unordered_set>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"

#include <algorithm>

namespace {

// Review tags of a surface: its record tags (what the surface panel's tag
// checkboxes and filters use).
QJsonObject tagsJson(const std::vector<std::string>& recordTags)
{
    auto has = [&](const char* tag) {
        return std::find(recordTags.begin(), recordTags.end(), tag) != recordTags.end();
    };
    QJsonObject o;
    o["approved"] = has("approved");
    o["defective"] = has("defective");
    o["reviewed"] = has("reviewed");
    o["inspect"] = has("inspect");
    o["partial_review"] = has("partial_review");
    return o;
}

// Precedence summary. "defective" wins even over "approved" (a defective flag
// needs attention regardless of a stale approval); "partial_review" ranks below
// "reviewed" as the less complete state.
QString reviewStateOf(const QJsonObject& tags)
{
    if (tags.value("defective").toBool()) return QStringLiteral("defective");
    if (tags.value("approved").toBool()) return QStringLiteral("approved");
    if (tags.value("reviewed").toBool()) return QStringLiteral("reviewed");
    if (tags.value("partial_review").toBool()) return QStringLiteral("partial_review");
    if (tags.value("inspect").toBool()) return QStringLiteral("inspect");
    return QStringLiteral("unreviewed");
}

} // namespace


QJsonObject AgentBridgeServer::handleSegmentsReview(const QJsonValue& params)
{
    auto& catalog = requireSurfaceCatalog(_window->_surfaceCatalog.get());
    const QJsonObject p = params.toObject();

    // "filter" mirrors the surface panel's filter checkboxes: each true key ANDs
    // one more constraint; absent or false contributes nothing (no inversion).
    const QJsonObject filterObj = p.value("filter").toObject();
    const bool fUnreviewed = filterObj.value("unreviewed").toBool(false);
    const bool fApproved = filterObj.value("approved").toBool(false);
    const bool fDefective = filterObj.value("defective").toBool(false);
    const bool fHideDefective =
        filterObj.value("hideDefective").toBool(false);
    const bool fReviewed = filterObj.value("reviewed").toBool(false);
    const bool fInspect = filterObj.value("inspect").toBool(false);
    const bool fPartialReview =
        filterObj.value("partialReview").toBool(false);

    const std::string activeId = _window->_state ? _window->_state->activeSurfaceId() : std::string{};

    // "total" counts the listed surfaces (what segments.list returns); "returned"
    // counts what survives the filter object.
    int total = 0;
    QJsonArray segments;
    for (const auto& [uuid, row] : catalog.rows()) {
        ++total;
        const QJsonObject tags = tagsJson(row.record.tags);

        if (fUnreviewed && tags.value("reviewed").toBool())
            continue;
        if (fApproved && !tags.value("approved").toBool())
            continue;
        if (fDefective && !tags.value("defective").toBool())
            continue;
        if (fHideDefective && tags.value("defective").toBool())
            continue;
        if (fReviewed && !tags.value("reviewed").toBool())
            continue;
        if (fInspect && !tags.value("inspect").toBool())
            continue;
        if (fPartialReview && !tags.value("partial_review").toBool())
            continue;

        QJsonObject seg = surfaceRowJson(row, activeId);
        seg["review"] = tags;
        seg["reviewState"] = reviewStateOf(tags);
        segments.push_back(seg);
    }

    QJsonObject result;
    result["segments"] = segments;
    result["total"] = total;
    result["returned"] = static_cast<int>(segments.size());
    result["loading"] = catalog.loading();
    return result;
}
