#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
namespace detail { struct GeometryReadState; struct GeometryViewState; }
class GeometryEdit;

// Views share a read transaction and pin a geometry revision. Reads are bounded
// and preserve native scalar types. Closing their snapshot invalidates all its
// views/edits; otherwise they keep the snapshot alive. Thread-confined, like Txn.
class GeometryView {
public:
    const GeometryRevision& revision() const;
    const std::string& objectRevision() const;
    GeometryEdit edit() const;
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
class PatchView : public GeometryView {
public:
    int64_t rows() const;
    int64_t columns() const;
    std::array<double, 2> scale() const;
    geometry::PatchPoint vertex(int64_t row, int64_t column) const;
    bool valid(int64_t row, int64_t column) const;
    // Returns an owned, row-major rectangle with the same scale.
    geometry::Patch readTile(int64_t row, int64_t column, int64_t rows, int64_t columns) const;
    geometry::Patch load() const;
private:
    friend class GeometrySnapshot;
    using GeometryView::GeometryView;
};

using TypedGeometryView = std::variant<PointCollectionView, FiberView, TrackView, PatchView>;

class GeometrySnapshot {
public:
    // A borrowed store must outlive this snapshot and its views/edits.
    explicit GeometrySnapshot(ObjectStore&);
    // Owning overload useful for application/binding lifetime management.
    explicit GeometrySnapshot(std::shared_ptr<ObjectStore>);
    void close();
    bool active() const;
    PointCollectionView pointCollection(const std::string& objectUuid) const;
    FiberView fiber(const std::string& objectUuid) const;
    TrackView track(const std::string& objectUuid) const;
    PatchView patch(const std::string& objectUuid) const;
    TypedGeometryView geometry(const std::string& objectUuid) const;
private:
    std::shared_ptr<detail::GeometryViewState> view(const std::string&, const std::string& kind) const;
    std::shared_ptr<detail::GeometryReadState> state_;
};

// Sparse edits prepared against a snapshot. No writes until commit; the object
// and geometry revisions are both checked. Existing relationship decisions and
// transaction semantics apply unchanged. An editor commits at most once.
class GeometryEdit {
public:
    void movePoint(geometry::PointId, const geometry::Point&);
    void moveControlPoint(int64_t index, const geometry::Point&);
    void moveLinePoint(int64_t index, const geometry::Point&);
    void moveTrackPoint(int64_t index, const geometry::TrackPoint&);
    void moveVertex(int64_t row, int64_t column, const geometry::PatchPoint&);
    std::vector<EdgeRecord> affectedRelationships() const;
    std::string commit(const std::vector<GeometryEdgeEdit>& decisions = {});
private:
    friend class GeometryView;
    explicit GeometryEdit(std::shared_ptr<detail::GeometryViewState>);
    void requireKind(const char*) const;
    std::shared_ptr<detail::GeometryViewState> state_;
    std::vector<GeometryArrayEdit> edits_;
    bool committed_ = false;
};
} // namespace vc::objects
