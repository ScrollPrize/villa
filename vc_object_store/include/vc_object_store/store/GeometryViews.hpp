#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
class TransformedGeometryView;
namespace detail { struct GeometryReadState; struct GeometryViewState; }

// Views share a read transaction and pin a geometry revision. Reads are bounded
// and preserve native scalar types. Closing their snapshot invalidates all its
// views; otherwise they keep the snapshot alive. Thread-confined, like Txn.
class GeometryView {
public:
    // This revision in `targetVolume`'s voxels; an older revision maps with the
    // coordinates of the latest object revision that held it.
    TransformedGeometryView viewTransformedAt(const std::string& targetVolume, double targetScale=1) const;
    std::vector<vcr::catalog::TransformResolution> availableTransforms() const;
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
    std::vector<geometry::PointId> controlPointIds(int64_t start, int64_t count) const;
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
    // scale; vertices without coordinates (validSurfacePoint) read as the
    // (-1,-1,-1) sentinel. Other channels are read with readGeometryRectangle.
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

struct TransformedSurfaceTile {
    int64_t rows=0, columns=0;
    std::vector<geometry::Point> points;
    std::vector<uint8_t> validity;
    std::array<double,2> scale{1,1};
};
// The frame a transformed view reads in: `volume`'s voxels at `scale`, mapped
// by `mapping` from the object's stored coordinates. A value to keep with what
// was read, so edits map back through the same frame.
struct ViewFrame {
    std::string volume;
    double scale = 1;
    VolumeAffine mapping;
};
// Uses the canonical view's transaction/lifetime, scalar types and stable IDs.
// Transformed coordinates are never serialized without an explicit inverse edit.
class TransformedGeometryView {
public:
    ViewFrame frame() const {return {target_, targetScale_, mapping_};}
    const GeometryRevision& revision() const;
    const std::string& objectRevision() const;
    const VolumeAffine& mapping() const {return mapping_;}
    const TypedGeometryView& canonical() const {return source_;}
    const std::string& targetVolume() const {return target_;}
    double targetScale() const {return targetScale_;}
    // Validate the pinned mapping and geometry before submitting inverse edits.
    void validateForWrite(const ObjectStore&, const Txn&) const;
    std::string writeCoordinates(ObjectStore&, WriteTxn&, const std::string& array, int64_t start,
        const std::vector<geometry::Point>& desired, const std::vector<GeometryEdgeEdit>& edges={}) const;
    std::string writeSurface(ObjectStore&, WriteTxn&, int64_t row, int64_t column, int64_t rows, int64_t columns,
        const std::vector<geometry::SurfacePoint>& desired, const std::vector<GeometryEdgeEdit>& edges={}) const;
    // Float32 surface consumer adapter. Validity remains separate from XYZ;
    // unchanged displayed vertices reuse the exact canonical float32 values.
    geometry::Surface canonicalSurface(const geometry::Surface& desired) const;
    // A whole edited fiber (points in the target volume) in canonical
    // coordinates. Control points keep their exact canonical values while
    // their displayed position is unchanged, matched by stable ID (by index
    // when desired has no IDs); line points match by index.
    geometry::Fiber canonicalFiber(const geometry::Fiber& desired) const;
    // A whole edited point collection (points in the target volume) in
    // canonical coordinates. Points keep their exact canonical values while
    // their displayed position is unchanged, matched by stable ID.
    geometry::PointCollection canonicalPointCollection(const geometry::PointCollection& desired) const;
    std::vector<GeometryArrayEdit> surfaceEdits(int64_t row, int64_t column,
        int64_t rows, int64_t columns, const std::vector<geometry::SurfacePoint>& desired) const;
    std::vector<geometry::Point> points(const std::string& array,int64_t start,int64_t count) const;
    TransformedSurfaceTile readTile(int64_t row,int64_t column,int64_t rows,int64_t columns) const;
    // Encode only explicitly changed positions; unchanged values reuse their
    // original bytes. Track edits round to nearest, with ties away from zero.
    std::vector<GeometryArrayEdit> coordinateEdits(const std::string& array,int64_t start,
        const std::vector<geometry::Point>& desired) const;
private:
    friend class GeometrySnapshot;
    friend class GeometryView;
    TransformedGeometryView(TypedGeometryView source,VolumeAffine mapping,std::string target,double targetScale)
        :source_(std::move(source)),mapping_(mapping),target_(std::move(target)),targetScale_(targetScale){}
    TypedGeometryView source_;
    VolumeAffine mapping_;
    std::string target_;
    double targetScale_;
};

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
    TransformedGeometryView viewTransformedAt(const std::string& objectUuid,const std::string& target,
        double targetScale=1, const VolumeTransforms* transforms=nullptr) const;
    // The object's geometry (current, or `geometryRevision`) in its stored
    // coordinates: the identity view, so edits in stored units take the same
    // path as edits through a transformed view.
    TransformedGeometryView viewStored(const std::string& objectUuid,
                                       const std::optional<std::string>& geometryRevision = std::nullopt) const;
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
