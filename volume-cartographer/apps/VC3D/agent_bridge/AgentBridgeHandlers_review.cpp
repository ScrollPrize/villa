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

// Build the {"approved","defective","reviewed","inspect","partial_review"} tag
// snapshot for one segment. has_tag()/tags_or_empty() (LoadJson.hpp) read
// meta["tags"], an object whose KEYS are the set tags -- exactly what the
// surface panel's checkboxes toggle (SurfacePanelController::onTagCheckboxToggled)
// and its filter checkboxes test (SurfacePanelController.cpp ~2727-2782).
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

// One-line precedence summary. "defective" wins even over "approved" because a
// segment flagged defective needs attention regardless of a stale approval;
// "partial_review" ranks below "reviewed" since the latter is the more
// complete state (mirrors the filter's `hasPartialReview = partial_review ||
// reviewed` grouping, but this summary still distinguishes the two so a caller
// doesn't have to reason about the OR itself).
QString reviewStateOf(const QJsonObject& tags)
{
    if (tags.value("defective").toBool()) return QStringLiteral("defective");
    if (tags.value("approved").toBool()) return QStringLiteral("approved");
    if (tags.value("reviewed").toBool()) return QStringLiteral("reviewed");
    if (tags.value("partial_review").toBool()) return QStringLiteral("partial_review");
    if (tags.value("inspect").toBool()) return QStringLiteral("inspect");
    return QStringLiteral("unreviewed");
}

// Resolve a segment's meta.json content without ever forcing a full (TIFF)
// surface load. Preference order:
//
//  1. A live QuadSurface: state->surface(id) (attached to a viewer) or,
//     failing that, vpkg.getSurface(id) (loaded earlier this session, e.g. by
//     the surface panel) -- because tag-checkbox edits mutate that object's
//     `meta` in place and are flushed to disk immediately
//     (SurfacePanelController::onTagCheckboxToggled), so it is the freshest
//     possible source for a segment touched this session.
//  2. A fresh read of meta.json straight off disk. We deliberately do NOT
//     use vpkg.segmentation(id)->metadata(): that is a copy parsed once at
//     project-open time (Segmentation ctor -> loadMetadata()) and never
//     updated afterwards, so it goes stale the moment a segment is loaded,
//     tag-edited (which writes only to the QuadSurface's meta + disk, per
//     path 1 above), and then unloaded again this session -- exactly the
//     nulled-`surface_` case path 1 can no longer see. Re-parsing meta.json
//     is a plain small-file read (same loader Segmentation::loadMetadata()
//     uses), not a TIFF/point-data load, so this stays cheap enough that
//     onlyLoaded=false remains safe as the default: there is still no hidden
//     O(N) heavy load here. If the file is transiently unreadable (e.g. a
//     concurrent writer mid-save), fall back to the stale cached copy rather
//     than dropping the segment from the listing.
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

    const QJsonObject p = paramsObject(params);
    const bool onlyLoaded = jsonOptionalBool(p, "onlyLoaded", false);

    // "filter" mirrors the surface panel's filter checkboxes (SurfacePanelController.cpp
    // ~2727-2782): each present-and-true key ANDs in one more constraint. A key
    // that is absent, or present-but-false, contributes nothing -- same as an
    // unchecked checkbox -- rather than inverting the predicate.
    QJsonObject filterObj;
    if (p.contains("filter")) {
        const QJsonValue fv = p.value("filter");
        if (!fv.isObject())
            throwParamError("filter", QStringLiteral("must be an object"));
        filterObj = fv.toObject();
    }
    const bool fUnreviewed = jsonOptionalBool(filterObj, "unreviewed", false);
    const bool fApproved = jsonOptionalBool(filterObj, "approved", false);
    const bool fDefective = jsonOptionalBool(filterObj, "defective", false);
    const bool fHideDefective = jsonOptionalBool(filterObj, "hideDefective", false);
    const bool fReviewed = jsonOptionalBool(filterObj, "reviewed", false);
    const bool fInspect = jsonOptionalBool(filterObj, "inspect", false);
    const bool fPartialReview = jsonOptionalBool(filterObj, "partialReview", false);

    // Loaded surface names live in CState; the on-disk segment ids come from the
    // package (same resolution as handleSegmentsList). "loaded" is deliberately
    // computed the same way segments.list computes it -- set membership against
    // CState::surfaceNames() -- so the two endpoints agree on what "loaded"
    // means for a given id. (A surface whose CState entry maps to a null
    // QuadSurface would still count as "loaded" here, same as in
    // handleSegmentsList; that's a shared, pre-existing nuance of CState's
    // surface map, not something specific to this listing.)
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
