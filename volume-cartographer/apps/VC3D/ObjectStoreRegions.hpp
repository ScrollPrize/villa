#pragma once
// Kind-generic object-store plumbing shared by the store overlays and catalogs
// (fibers, surfaces): background reads, session events filtered by record kind,
// and the per-viewer scheduling of slab region queries.
#include "vc/core/types/FiberView.hpp"
#include "ObjectStoreEvents.hpp"
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <QFutureWatcher>
#include <QObject>
#include <QString>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

namespace vc3d::store {

template <class T>
struct Outcome {
    T value{};
    QString error;
};

// Run `work` on the thread pool and hand its value, or the message of what it
// threw, to `done` on `parent`'s thread.
template <class T, class Work, class Done>
void background(QObject* parent, Work work, Done done)
{
    auto* watcher = new QFutureWatcher<Outcome<T>>(parent);
    QObject::connect(watcher, &QFutureWatcher<Outcome<T>>::finished, parent, [watcher, done]() {
        auto result = watcher->future().takeResult();
        watcher->deleteLater();
        done(std::move(result));
    });
    watcher->setFuture(QtConcurrent::run([work]() {
        Outcome<T> result;
        try {
            result.value = work();
        } catch (const std::exception& error) {
            result.error = QString::fromUtf8(error.what());
        }
        return result;
    }));
}

// Editing an object starts from the revision on screen: lease it on a hosted
// replica (leasing may first pull the hub's newer record), then check that the
// pin is still its current revision. Throws when it is not.
inline vc::objects::LeaseHold holdForEditing(vc::objects::ObjectStoreSession& session, const vc::objects::PinnedObject& pin)
{
    auto lease = session.holdLeases(pin.attachment, {pin.object.objectUuid});
    session.read(pin.attachment, [&](const auto& db, const auto& txn, const auto&) {
        const auto head = db.currentObjectRevision(txn, pin.object.objectUuid);
        if (!head || head->uuid != pin.objectRevision)
            throw std::runtime_error("It changed in the object store since it was opened; reopen it to edit");
    });
    return lease;
}

// --- Slab regions --------------------------------------------------------------
using Region = vc::fibers::ViewRegion;
using Bounds = vc::fibers::Bounds;

inline bool overlaps(const Bounds& a, const Bounds& b)
{
    for (size_t i = 0; i < 3; ++i)
        if (a.upper[i] < b.lower[i] || b.upper[i] < a.lower[i])
            return false;
    return true;
}
// The volume AABB of a region's local box.
inline Bounds regionBox(const Region& s)
{
    Bounds box;
    box.lower.fill(std::numeric_limits<double>::infinity());
    box.upper.fill(-std::numeric_limits<double>::infinity());
    for (double a : {s.local.lower[0], s.local.upper[0]})
        for (double b : {s.local.lower[1], s.local.upper[1]})
            for (double c : {s.local.lower[2], s.local.upper[2]})
                for (int i = 0; i < 3; ++i) {
                    const double p = s.origin[i] + a * s.u[i] + b * s.v[i] + c * s.n[i];
                    box.lower[i] = std::min(box.lower[i], p);
                    box.upper[i] = std::max(box.upper[i], p);
                }
    return box;
}
// Load a margin around the view so that pans, zooms and the small moves of a
// cut plane stay inside one complete region instead of re-querying each frame.
// The view is clipped from it exactly, so the margin never changes the drawing.
constexpr double kPrefetchSpanFraction = 0.25;
constexpr double kPrefetchDepthVx = 32;
inline Region prefetchRegion(Region s)
{
    const double du = kPrefetchSpanFraction * (s.local.upper[0] - s.local.lower[0]);
    const double dv = kPrefetchSpanFraction * (s.local.upper[1] - s.local.lower[1]);
    s.local.lower[0] -= du;
    s.local.upper[0] += du;
    s.local.lower[1] -= dv;
    s.local.upper[1] += dv;
    s.local.lower[2] -= kPrefetchDepthVx;
    s.local.upper[2] += kPrefetchDepthVx;
    s.box = regionBox(s);
    return s;
}
inline double viewSpan(const Region& s)
{
    return std::min(s.local.upper[0] - s.local.lower[0], s.local.upper[1] - s.local.lower[1]);
}
// The slab of half depth `halfDepth` around the view rectangle whose corners
// are `origin` (top left), `right` (top right) and `down` (bottom left), in
// the query's coordinates. Nothing for a degenerate rectangle.
inline std::optional<Region> viewRegion(const vc::fibers::Point& origin, const vc::fibers::Point& right,
                                        const vc::fibers::Point& down, double halfDepth)
{
    auto sub = [](const auto& a, const auto& b) { return vc::fibers::Point{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
    auto dot = [](const auto& a, const auto& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const auto x = sub(right, origin), y = sub(down, origin);
    const double width = std::sqrt(dot(x, x)), height = std::sqrt(dot(y, y));
    Region s;
    s.origin = origin;
    for (int i = 0; i < 3; ++i) {
        s.u[i] = x[i] / width;
        s.v[i] = y[i] / height;
    }
    s.n = {s.u[1] * s.v[2] - s.u[2] * s.v[1], s.u[2] * s.v[0] - s.u[0] * s.v[2], s.u[0] * s.v[1] - s.u[1] * s.v[0]};
    const double normal = std::sqrt(dot(s.n, s.n));
    if (!(width > 0) || !(height > 0) || !(normal > 0) || !std::isfinite(normal)) return std::nullopt;
    for (auto& c : s.n) c /= normal;
    s.local = {{0, 0, -halfDepth}, {width, height, halfDepth}};
    s.box = regionBox(s);
    return s;
}
inline vc::objects::RegionSlice regionSlice(const Region& s)
{
    vc::objects::RegionSlice slice;
    slice.origin = s.origin; slice.u = s.u; slice.v = s.v; slice.normal = s.n;
    slice.localBounds = {s.local.lower[0], s.local.lower[1], s.local.lower[2], s.local.upper[0], s.local.upper[1], s.local.upper[2]};
    return slice;
}

// One viewer's region queries: at most one in flight, kept while it can still
// draw part of the view, and none while a complete region covers the view.
// The payload a query produces belongs to its client.
struct RegionRequests {
    // requested is the view whose region was last queried; pending is the
    // region of the query in flight; coverage the installed region a view
    // inside needs no query for.
    std::optional<Region> requested, pending, coverage;
    // A region was installed since the last reset: its payload can be drawn.
    bool loaded{false};
    // Views at least this span (voxels) are queried without a margin: a
    // prefetch there exceeded the store's region budget.
    double prefetchFailedSpan{std::numeric_limits<double>::infinity()};
    std::shared_ptr<std::atomic_bool> cancelled;
    bool busy{false};
    bool complete{true};
    QString error;

    // Forget all requests and coverage (the data changed or went away).
    void reset()
    {
        if (cancelled) cancelled->store(true);
        requested.reset();
        pending.reset();
        coverage.reset();
        loaded = false;
        prefetchFailedSpan = std::numeric_limits<double>::infinity();
    }
    // Query again for the current view while keeping what is displayed.
    void refresh()
    {
        if (cancelled) cancelled->store(true);
        requested.reset();
        coverage.reset();
    }
    // The region to query for `view`, or nothing when the view is covered or
    // already requested. A started query is marked busy with a fresh token.
    std::optional<Region> start(const Region& view)
    {
        // A complete region (one that reached neither the display limit nor a
        // store budget) holds everything of any view inside its prism.
        if (complete && coverage && coverage->contains(view)) {
            if (busy && cancelled) cancelled->store(true);
            return std::nullopt;
        }
        if (requested && *requested == view) return std::nullopt;
        if (busy) {
            // Let the running query finish while it can still draw part of this
            // view: cancelling on every pan frame starved a moving view of all
            // results. Its completion requests the view again if still uncovered.
            if (cancelled && (!pending || !overlaps(pending->box, view.box)))
                cancelled->store(true);
            return std::nullopt;
        }
        const auto query = viewSpan(view) < prefetchFailedSpan ? prefetchRegion(view) : view;
        busy = true;
        requested = view;
        pending = query;
        error.clear();
        cancelled = std::make_shared<std::atomic_bool>(false);
        return query;
    }
    enum class Finished { Stale, Retry, Install };
    // A query for `view` (as `query`) ended. Stale: a newer query or a reset
    // superseded it, ignore it; Retry: re-request the view; Install: show it.
    Finished finish(const std::shared_ptr<std::atomic_bool>& token, bool current, const Region& view,
                    const Region& query, bool regionComplete, const QString& failure)
    {
        if (cancelled != token) return Finished::Stale;
        busy = false;
        pending.reset();
        if (!current || token->load()) {
            requested.reset();
            return Finished::Retry;
        }
        if (failure.isEmpty() && !regionComplete && query != view) {
            // The margin pushed the region past a store budget the view
            // alone may fit: never show less than the view's own query.
            prefetchFailedSpan = viewSpan(view);
            requested.reset();
            return Finished::Retry;
        }
        error = failure;
        if (!failure.isEmpty()) return Finished::Install;
        complete = regionComplete;
        coverage = query;
        loaded = true;
        return Finished::Install;
    }
};

} // namespace vc3d::store
