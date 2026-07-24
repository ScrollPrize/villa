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

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/QuadSurface.hpp"

namespace {

// Tag snapshot for one segment. has_tag() reads meta["tags"], whose keys are
// the set tags -- what the surface panel's checkboxes toggle
// (SurfacePanelController::onTagCheckboxToggled) and filter against.
QJsonObject tagsJson(const utils::Json& meta)
{
    QJsonObject o;
    o["approved"] = vc::json::has_tag(meta, "approved");
    o["defective"] = vc::json::has_tag(meta, "defective");
    o["reviewed"] = vc::json::has_tag(meta, "reviewed");
    o["inspect"] = vc::json::has_tag(meta, "inspect");
    o["partial_review"] = vc::json::has_tag(meta, "partial_review");
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

// Resolve a segment's meta.json without forcing a full (TIFF) surface load.
// Preference order:
//  1. A live QuadSurface (state->surface(id), else vpkg.getSurface(id)) -- its
//     `meta` is mutated in place on tag edits and flushed to disk immediately,
//     so it is the freshest source for a segment touched this session.
//  2. A fresh meta.json read. NOT vpkg.segmentation(id)->metadata(): that copy
//     is parsed once at project-open and never refreshed, so it goes stale once
//     a tag-edited segment is unloaded. Re-parsing is a cheap small-file read
//     (no TIFF), so onlyLoaded=false stays a safe default. On a transient read
//     failure, fall back to the stale cached copy rather than drop the segment.
utils::Json resolveSegmentMeta(CState* state, VolumePkg& vpkg, const std::string& id)
{
    if (state) {
        if (auto surf = state->surface(id); surf && !surf->meta.is_null())
            return surf->meta;
    }
    if (auto qsurf = vpkg.getSurface(id); qsurf && !qsurf->meta.is_null())
        return qsurf->meta;
    if (auto seg = vpkg.segmentation(id)) {
        try {
            return vc::json::load_json_file(seg->path() / "meta.json");
        } catch (...) {
            return seg->metadata();
        }
    }
    return utils::Json::object();
}

} // namespace


QJsonObject AgentBridgeServer::handleSegmentsReview(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = params.toObject();
    const bool onlyLoaded = p.value("onlyLoaded").toBool(false);

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

    // "loaded" = id present in CState::surfaceNames(), computed exactly as
    // segments.list does so the two endpoints agree.
    const std::vector<std::string> loadedNames = state->surfaceNames();
    const std::unordered_set<std::string> loadedSet(loadedNames.begin(), loadedNames.end());
    const std::string activeId = state->activeSurfaceId();

    // "total" counts the onlyLoaded-scoped candidate set (what segments.list
    // would return for the same onlyLoaded); "returned" counts what survives
    // the filter object on top of that scope.
    int total = 0;
    QJsonArray segments;
    for (const std::string& id : vpkg->segmentationIDs()) {
        const bool loaded = loadedSet.count(id) > 0;
        if (onlyLoaded && !loaded)
            continue;
        ++total;

        const utils::Json meta = resolveSegmentMeta(state, *vpkg, id);
        const QJsonObject tags = tagsJson(meta);

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

        QJsonObject seg;
        seg["id"] = QString::fromStdString(id);

        QString path;
        try {
            if (auto s = vpkg->segmentation(id))
                path = QString::fromStdString(s->path().string());
        } catch (...) {
            // Metadata resolution may fail for a partially written segment; the
            // id is still reportable without a path (matches handleSegmentsList).
        }
        seg["path"] = path;
        seg["loaded"] = loaded;
        seg["active"] = (id == activeId);
        seg["tags"] = tags;
        seg["reviewState"] = reviewStateOf(tags);
        segments.push_back(seg);
    }

    QJsonObject result;
    result["segments"] = segments;
    result["total"] = total;
    result["returned"] = static_cast<int>(segments.size());
    return result;
}
