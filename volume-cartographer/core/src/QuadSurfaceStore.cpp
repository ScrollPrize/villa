// QuadSurface <-> vc_object_store surface objects.
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/SurfaceGrid.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/Uuid.hpp>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <bit>
#include <cstring>
#include <stdexcept>

static_assert(std::endian::native == std::endian::little, "stored geometry arrays are little-endian");

namespace {
using vc::objects::GeometryArray;

int cvDepth(const std::string& dtype)
{
    if (dtype == "|u1") return CV_8U;
    if (dtype == "<u2") return CV_16U;
    if (dtype == "<i4") return CV_32S;
    if (dtype == "<f4") return CV_32F;
    if (dtype == "<f8") return CV_64F;
    throw std::runtime_error("surface channel dtype has no OpenCV equivalent: " + dtype);
}

const char* storeDtype(int depth)
{
    switch (depth) {
    case CV_8U: return "|u1";
    case CV_16U: return "<u2";
    case CV_32S: return "<i4";
    case CV_32F: return "<f4";
    case CV_64F: return "<f8";
    }
    throw std::runtime_error("surface channel depth cannot be stored");
}

cv::Mat matFromBytes(const std::string& dtype, const std::vector<int64_t>& shape, const std::vector<uint8_t>& bytes)
{
    const int components = shape.size() == 3 ? int(shape[2]) : 1;
    cv::Mat mat(int(shape[0]), int(shape[1]), CV_MAKETYPE(cvDepth(dtype), components));
    if (mat.total() * mat.elemSize() != bytes.size()) throw std::runtime_error("surface channel size mismatch");
    std::memcpy(mat.data, bytes.data(), bytes.size());
    return mat;
}

GeometryArray arrayFromMat(const std::string& name, const cv::Mat& mat)
{
    const cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();
    GeometryArray array{name, storeDtype(contiguous.depth()), {contiguous.rows, contiguous.cols}, {}};
    if (contiguous.channels() != 1) array.shape.push_back(contiguous.channels());
    array.bytes.assign(contiguous.datastart, contiguous.dataend);
    return array;
}

// The "mask" channel as per-vertex validity (the store's rule; masks may be a
// whole multiple of the grid).
GeometryArray validFromMask(const cv::Mat& mask, int rows, int cols)
{
    if (mask.channels() != 1) throw std::runtime_error("surface mask must be one channel");
    cv::Mat_<double> values;
    mask.convertTo(values, CV_64F);
    return {"valid", "|u1", {rows, cols},
            vc::geometry::validityFromMask({values.ptr<double>(), values.total()}, values.rows, values.cols, rows, cols)};
}


std::vector<int64_t> pointsShape(const vc::objects::GeometryRevision& revision)
{
    for (const auto& array : revision.arrays)
        if (array.name == "points") return array.shape;
    throw std::runtime_error("stored surface has no points");
}

} // namespace

std::unique_ptr<QuadSurface> load_quad_from_store(const vc::objects::StoreReader& reader,
    const std::string& objectUuid, const std::optional<std::string>& geometryRevision, bool applyValidity, const vc::objects::TransformedGeometryView* transformed)
{
    const auto selected=geometryRevision?reader.objectAtGeometry(*geometryRevision):reader.currentObject(objectUuid);
    if(!selected||selected->objectUuid!=vc::objects::util::canonicalUuid(objectUuid))
        throw std::runtime_error("surface has no such geometry revision: "+objectUuid);
    return load_quad_from_store(reader,*selected,applyValidity,transformed);
}

std::unique_ptr<QuadSurface> load_quad_from_store(const vc::objects::StoreReader& reader,
    const vc::objects::ObjectRevision& selected,bool applyValidity,const vc::objects::TransformedGeometryView* transformed)
{
    auto grid=vc::objects::readSurfaceGrid(reader,selected,transformed,applyValidity);
    auto points=std::make_unique<cv::Mat_<cv::Vec3f>>(int(grid.rows),int(grid.columns));
    std::memcpy(points->data,grid.points.data(),grid.points.size()*sizeof(float));
    auto surface=std::make_unique<QuadSurface>(points.release(),cv::Vec2f(float(grid.scale[0]),float(grid.scale[1])));
    surface->id = grid.record.uuid;
    surface->meta = nlohmann::json::parse(grid.record.meta);
    surface->setComponents(surface_components(surface->meta));
    surface->meta["uuid"] = grid.record.uuid;
    surface->meta["name"] = grid.record.name;
    surface->tags = grid.record.tags;
    if(!grid.valid.empty()){
        cv::Mat_<uint8_t> mask(int(grid.rows),int(grid.columns));
        std::transform(grid.valid.begin(),grid.valid.end(),mask.begin(),[](uint8_t value){return value?255:0;});
        surface->setChannel("mask",mask);
    }
    for(const auto& channel:grid.channels)surface->setChannel(channel,cv::Mat());
    return surface;
}

std::function<cv::Mat(const std::string&)> store_channel_loader(
    std::shared_ptr<vc::objects::ObjectStoreSession> session, vc::objects::AttachmentIdentity attachment, std::string geometryRevision)
{
    return [session = std::move(session), attachment = std::move(attachment), revision = std::move(geometryRevision)](const std::string& name) {
        cv::Mat result;
        session->readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            const auto descriptor = reader.geometryRevision(revision);
            if (!descriptor) throw std::runtime_error("geometry revision not found: " + revision);
            for (const auto& array : descriptor->arrays)
                if (array.name == name)
                    result = matFromBytes(array.dtype, array.shape,
                        reader.readGeometryRectangle(revision, name, 0, 0, array.shape[0], array.shape[1]));
        });
        return result;
    };
}

std::vector<vc::objects::TransactionOperation> create_quad_operations(const std::string& uuid, QuadSurface& surface,
    const std::string& name, const std::vector<std::string>& tags, const nlohmann::json& coordinates,
    const vc::objects::SurfaceProvenance& provenance)
{
    return vc::objects::createSurfaceOperations(uuid, name, tags, coordinates, provenance, surface.meta,
                                                quad_surface_geometry(surface));
}

vc::geometry::Surface quad_surface_grid(QuadSurface& surface)
{
    const auto& points = *surface.rawPointsPtr();
    vc::geometry::Surface geometry{points.rows, points.cols, {surface._scale[0], surface._scale[1]}, {}};
    geometry.points.reserve(points.total());
    for (const auto& p : points)
        geometry.points.push_back(vc::geometry::validSurfacePoint(p) ? vc::geometry::SurfacePoint{p[0], p[1], p[2]} : vc::geometry::SurfacePoint{-1, -1, -1});
    const auto names = surface.channelNames();
    if (std::find(names.begin(), names.end(), "mask") != names.end())
        if (const auto mask = surface.channel("mask", SURF_CHANNEL_NORESIZE); !mask.empty())
            geometry.validity = validFromMask(mask, points.rows, points.cols).bytes;
    return geometry;
}

vc::objects::GeometryData quad_surface_geometry(QuadSurface& surface)
{
    auto geometry = quad_surface_grid(surface);
    for (const auto& name : surface.channelNames()) {
        if (name == "mask") continue;
        const auto channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (!channel.empty()) geometry.channels.push_back(arrayFromMat(name, channel));
    }
    return vc::geometry::encode(geometry);
}

std::optional<vc::objects::EditGeometryOp> quad_rect_edit(const vc::objects::TransformedGeometryView& view,
    QuadSurface& surface, const cv::Rect& rect, const std::vector<uint8_t>* validity)
{
    const auto& revision = view.revision();
    if (revision.kind != "surface") throw std::runtime_error("not a stored surface: " + revision.objectUuid);
    const auto shape = pointsShape(revision);
    const auto& points = *surface.rawPointsPtr();
    if (points.rows != shape[0] || points.cols != shape[1])
        throw std::runtime_error("surface grid differs from the stored grid; resize it first");
    const auto area = rect & cv::Rect(0, 0, points.cols, points.rows);
    if (area.empty()) return std::nullopt;
    std::vector<vc::geometry::SurfacePoint> desired;
    desired.reserve(size_t(area.width) * area.height);
    for (int r = area.y; r < area.y + area.height; ++r)
        for (int c = area.x; c < area.x + area.width; ++c) {
            const auto p = points(r, c);
            desired.push_back({p[0], p[1], p[2]});
        }
    auto edits = view.surfaceEdits(area.y, area.x, area.height, area.width, desired, validity);
    if (edits.empty()) return std::nullopt;
    return vc::objects::EditGeometryOp{revision.objectUuid, std::move(edits), {}, true};
}

vc::objects::SetSurfaceChannelsOp quad_channels_operation(const std::string& objectUuid, QuadSurface& surface,
    const std::vector<std::string>& names, const std::vector<std::string>& remove, int rows, int columns)
{
    vc::objects::SetSurfaceChannelsOp op;
    op.objectUuid = objectUuid;
    op.keepAffectedRecorded = true;
    for (const auto& name : names) {
        const auto channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) throw std::runtime_error("surface has no channel " + name);
        op.channels.push_back(name == "mask" ? validFromMask(channel, rows, columns) : arrayFromMat(name, channel));
    }
    for (const auto& name : remove) op.remove.push_back(name == "mask" ? "valid" : name);
    return op;
}
