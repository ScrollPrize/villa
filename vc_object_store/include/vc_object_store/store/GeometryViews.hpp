#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
namespace detail { struct GeometryReadState; struct GeometryViewState; }

// Views share a read transaction and pin a geometry revision. Reads are bounded
// and preserve native scalar types. Closing their snapshot invalidates all its
// views; otherwise they keep the snapshot alive. Thread-confined, like Txn.
class GeometryView {
public:
    const GeometryRevision& revision() const;
    const std::string& objectRevision() const;
protected:
    explicit GeometryView(std::shared_ptr<detail::GeometryViewState>);
    geometry::Geometry loadGeometry() const;
    std::shared_ptr<detail::GeometryViewState> state_;
};

class PointCollectionView : public GeometryView {
public:
    int64_t size() const;
    std::vector<geometry::PointId> ids(int64_t start, int64_t count) const;
    std::vector<geometry::Point> points(int64_t start, int64_t count) const;
    // The ID index loads on first lookup; XYZ reads remain bounded.
    std::optional<geometry::Point> point(geometry::PointId) const;
    geometry::PointCollection load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};
class FiberView : public GeometryView {
public:
    int64_t controlPointCount() const;
    int64_t linePointCount() const;
    std::vector<geometry::Point> controlPoints(int64_t start, int64_t count) const;
    std::vector<geometry::Point> linePoints(int64_t start, int64_t count) const;
    geometry::Fiber load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};
class TrackView : public GeometryView {
public:
    int64_t size() const;
    std::vector<geometry::TrackPoint> points(int64_t start, int64_t count) const;
    geometry::Track load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};
class SurfaceView : public GeometryView {
public:
    int64_t rows() const;
    int64_t columns() const;
    std::array<double, 2> scale() const;
    geometry::SurfacePoint vertex(int64_t row, int64_t column) const;
    bool valid(int64_t row, int64_t column) const;
    // Returns an owned, row-major rectangle of points and validity with the same
    // scale; other channels are read with readGeometryRectangle.
    geometry::Surface readTile(int64_t row, int64_t column, int64_t rows, int64_t columns) const;
    geometry::Surface load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};

class UmbilicusView : public GeometryView {
public:
    int64_t size() const;
    std::vector<geometry::Point> points(int64_t start, int64_t count) const;
    geometry::Umbilicus load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};
using TypedGeometryView = std::variant<PointCollectionView, FiberView, TrackView, SurfaceView, UmbilicusView>;

class GeometrySnapshot {
public:
    // A borrowed store must outlive this snapshot and its views.
    explicit GeometrySnapshot(ObjectStore&);
    // Owning overload useful for application/binding lifetime management.
    explicit GeometrySnapshot(std::shared_ptr<ObjectStore>);
    // Views over the caller's transaction (e.g. inside ObjectStoreSession::read),
    // valid while it is.
    GeometrySnapshot(const ObjectStore&, const Txn&);
    void close();
    bool active() const;
    // The object's current geometry, or its pinned `geometryRevision`. A view of
    // a pinned revision has an empty objectRevision().
    PointCollectionView pointCollection(const std::string& objectUuid,
                                        const std::optional<std::string>& geometryRevision = std::nullopt) const;
    FiberView fiber(const std::string& objectUuid, const std::optional<std::string>& geometryRevision = std::nullopt) const;
    TrackView track(const std::string& objectUuid, const std::optional<std::string>& geometryRevision = std::nullopt) const;
    UmbilicusView umbilicus(const std::string& objectUuid,
                            const std::optional<std::string>& geometryRevision = std::nullopt) const;
    SurfaceView surface(const std::string& objectUuid, const std::optional<std::string>& geometryRevision = std::nullopt) const;
    TypedGeometryView geometry(const std::string& objectUuid,
                               const std::optional<std::string>& geometryRevision = std::nullopt) const;
private:
    std::shared_ptr<detail::GeometryViewState> view(const std::string&, const std::string& kind,
                                                    const std::optional<std::string>& geometryRevision) const;
    std::shared_ptr<detail::GeometryReadState> state_;
};

} // namespace vc::objects
