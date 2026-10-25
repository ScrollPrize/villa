#include "segmentation/SurfaceStoreEdit.hpp"

#include "ObjectStoreRegions.hpp"
#include "vc/core/util/QuadSurface.hpp"

using namespace vc::objects;

SurfaceStoreEdit::SurfaceStoreEdit(std::shared_ptr<ObjectStoreSession> session, PinnedObject pin)
    : _session(std::move(session)), _uuid(pin.object.objectUuid), _pin(pin), _origin(std::move(pin))
{
    if (!_pin.geometryRevision) throw std::runtime_error("The surface has no geometry");
}

void SurfaceStoreEdit::hold()
{
    if (!_session->writable(_pin.attachment)) throw std::runtime_error("The object store is attached read-only");
    auto lease = vc3d::store::holdForEditing(*_session, pin());
    std::lock_guard lock(_mutex);
    _lease = std::move(lease);
}

void SurfaceStoreEdit::release()
{
    std::lock_guard lock(_mutex);
    _lease.reset();
}

PinnedObject SurfaceStoreEdit::pin() const
{
    std::lock_guard lock(_mutex);
    return _pin;
}

void SurfaceStoreEdit::save(QuadSurface& surface, const cv::Rect& rect, const std::vector<std::string>& channels)
{
    // One save at a time: each starts from the previous one's revisions.
    std::lock_guard lock(_mutex);
    std::string geometry = *_pin.geometryRevision;
    const auto& points = *surface.rawPointsPtr();
    const auto receipt = _session->editKeepingRelationships(_pin.attachment, {{_uuid, _pin.objectRevision}}, {},
        [&](auto& db, auto& txn, const auto& keep) {
            const auto revision = db.geometryRevision(txn, geometry);
            if (!revision) throw std::runtime_error("surface geometry revision not found: " + geometry);
            const auto stored = std::find_if(revision->arrays.begin(), revision->arrays.end(),
                                             [](const auto& array) { return array.name == "points"; });
            if (stored == revision->arrays.end() || stored->shape[0] != points.rows || stored->shape[1] != points.cols) {
                const auto data = quad_surface_geometry(surface);
                geometry = db.putGeometry(txn, _uuid, data, geometry, keep(db.affectedGeometryEdges(txn, _uuid, data)));
                return;
            }
            if (!channels.empty()) geometry = write_quad_channels_to_store(db, txn, _uuid, surface, channels, {}, geometry);
            geometry = write_quad_rect_to_store(db, txn, _uuid, surface, rect, geometry, keep);
        });
    _pin.objectRevision = *receipt.at(_uuid);
    _pin.geometryRevision = geometry;
}
