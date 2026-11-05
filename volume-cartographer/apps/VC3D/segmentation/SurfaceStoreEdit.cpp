#include "segmentation/SurfaceStoreEdit.hpp"

#include "ObjectStoreRegions.hpp"
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include "vc/core/util/QuadSurface.hpp"

using namespace vc::objects;

SurfaceStoreEdit::SurfaceStoreEdit(std::shared_ptr<ObjectStoreSession> session, PinnedObject pin,
                                 std::optional<vc::objects::ViewFrame> view)
    : _session(std::move(session)), _uuid(pin.object.objectUuid), _pin(pin), _origin(std::move(pin)),
      _view(std::move(view))
{
    if (!_pin.geometryRevision) throw std::runtime_error("The surface has no geometry");
}

PinnedObject SurfaceStoreEdit::pin() const
{
    std::lock_guard lock(_mutex);
    return _pin;
}

void SurfaceStoreEdit::save(QuadSurface& surface, const cv::Rect& rect, const std::vector<std::string>& channels)
{
    // One save at a time: each starts from the version the previous one committed.
    std::lock_guard lock(_mutex);
    const auto& points = *surface.rawPointsPtr();
    auto request = _session->newRequest(_pin.attachment);
    request.generation = _pin.generation;
    request.expectedObjects = {{_uuid, _pin.version}};
    _session->readStore(_pin.attachment, [&](const StoreReader& reader, const auto&) {
        const GeometrySnapshot snapshot(reader);
        std::optional<TransformedGeometryView> transformed;
        if (_view) {
            transformed = snapshot.surface(_uuid, *_pin.geometryRevision).viewTransformedAt(_view->volume, _view->scale);
            if (transformed->mapping() != _view->mapping)
                throw std::runtime_error("Volume registration changed; reopen the surface before saving");
            // A registration that changes before the commit conflicts.
            request.readDependencies = transformDependencies(reader);
        }
        const auto revision = reader.geometryRevision(*_pin.geometryRevision);
        if (!revision) throw std::runtime_error("surface geometry revision not found: " + *_pin.geometryRevision);
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
            request.operations.push_back(PutGeometryOp{_uuid, std::move(data), {}, true});
        } else {
            // Channels first; the vertex edit then applies to the validity they set.
            std::vector<uint8_t> validity;
            const std::vector<uint8_t>* replaced = nullptr;
            if (!channels.empty()) {
                auto op = quad_channels_operation(_uuid, surface, channels, {}, points.rows, points.cols);
                for (const auto& array : op.channels)
                    if (array.name == "valid") { validity = array.bytes; replaced = &validity; }
                request.operations.push_back(std::move(op));
            }
            const auto view = transformed ? *transformed : snapshot.viewStored(_uuid, *_pin.geometryRevision);
            if (auto edit = quad_rect_edit(view, surface, rect, replaced)) request.operations.push_back(std::move(*edit));
        }
        expectRelationshipEndpoints(reader, request.expectedObjects);
    });
    if (request.operations.empty()) return;
    const auto receipt = _session->submit(_pin.attachment, request);
    const auto committed = _session->objectsAtVersions(
        _pin.attachment, {{_pin.object.storeUuid, receipt.generation, _uuid, *receipt.objects.at(_uuid)}});
    if (!committed.front()) throw std::runtime_error("the saved surface version cannot be read back");
    _pin = pinOf(_pin.attachment, *committed.front());
}
