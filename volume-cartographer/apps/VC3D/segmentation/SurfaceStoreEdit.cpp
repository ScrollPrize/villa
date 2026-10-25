#include "segmentation/SurfaceStoreEdit.hpp"

#include "ObjectStoreRegions.hpp"
#include <vc_object_store/store/GeometryViews.hpp>
#include "vc/core/util/QuadSurface.hpp"

using namespace vc::objects;

SurfaceStoreEdit::SurfaceStoreEdit(std::shared_ptr<ObjectStoreSession> session, PinnedObject pin,
                                 std::optional<std::string> viewBinding)
    : _session(std::move(session)), _uuid(pin.object.objectUuid), _pin(pin), _origin(std::move(pin)),
      _viewBinding(std::move(viewBinding))
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
    const auto binding = _viewBinding ? _viewBinding : surface.meta.contains("vc3d_view")
        ? std::optional(surface.meta.at("vc3d_view").dump()) : std::nullopt;
    const auto receipt = _session->editKeepingRelationships(_pin.attachment, {{_uuid, _pin.objectRevision}}, {},
        [&](auto& db, auto& txn, const auto& keep) {
            std::optional<TransformedGeometryView> transformed;
            if(binding) {
                const auto view=nlohmann::json::parse(*binding);
                transformed=GeometrySnapshot(db,txn).viewTransformedAt(_uuid,view.at("target"),view.at("scale"));
                if(transformed->mapping()!=VolumeAffine::fromJson(view.at("matrix")))
                    throw std::runtime_error("Volume registration changed; reopen the surface before saving");

            }
            const auto revision = db.geometryRevision(txn, geometry);
            if (!revision) throw std::runtime_error("surface geometry revision not found: " + geometry);
            const auto stored = std::find_if(revision->arrays.begin(), revision->arrays.end(),
                                             [](const auto& array) { return array.name == "points"; });
            if (stored == revision->arrays.end() || stored->shape[0] != points.rows || stored->shape[1] != points.cols) {
                auto data = quad_surface_geometry(surface);
                if(transformed) {
                    vc::geometry::Surface desired;
                    desired.rows=points.rows;desired.columns=points.cols;
                    for(const auto& p:points)desired.points.push_back({p[0],p[1],p[2]});
                    for(const auto& array:data.arrays)if(array.name=="valid")desired.validity=array.bytes;
                    const auto canonical=vc::geometry::encode(transformed->canonicalSurface(desired));
                    data.arrays.front()=canonical.arrays.front();
                    data.attributes=canonical.attributes;
                    for(const auto& array:canonical.arrays)if(array.name=="valid") {
                        auto it=std::find_if(data.arrays.begin(),data.arrays.end(),[](const auto& a){return a.name=="valid";});
                        if(it==data.arrays.end())data.arrays.push_back(array);else *it=array;
                    }
                }
                geometry = db.putGeometry(txn, _uuid, data, geometry, keep(db.affectedGeometryEdges(txn, _uuid, data)));
                return;
            }
            if (!channels.empty()) {
                geometry = write_quad_channels_to_store(db, txn, _uuid, surface, channels, {}, geometry);
                if(transformed) transformed=GeometrySnapshot(db,txn).viewTransformedAt(
                    _uuid,transformed->targetVolume(),transformed->targetScale());
            }
            geometry = write_quad_rect_to_store(db, txn, _uuid, surface, rect, geometry, keep, transformed ? &*transformed : nullptr);
        });
    _pin.objectRevision = *receipt.at(_uuid);
    _pin.geometryRevision = geometry;
    _viewBinding = binding;
}
