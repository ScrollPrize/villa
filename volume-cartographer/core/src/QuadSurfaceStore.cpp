// QuadSurface <-> vc_object_store surface objects.
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
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
const cv::Vec3f kInvalid{-1.f, -1.f, -1.f};

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

vc::objects::GeometryRevision currentSurface(const vc::objects::ObjectStore& db, const vc::objects::Txn& txn,
                                             const std::string& objectUuid, const std::string& expected)
{
    const auto revision = db.currentGeometry(txn, objectUuid);
    if (!revision || revision->kind != "surface") throw std::runtime_error("not a stored surface: " + objectUuid);
    if (revision->uuid != vc::objects::util::canonicalUuid(expected))
        throw std::runtime_error("stored surface changed; reload before writing: " + objectUuid);
    return *revision;
}
} // namespace

std::unique_ptr<QuadSurface> load_quad_from_store(const vc::objects::ObjectStore& db, const vc::objects::Txn& txn,
    const std::string& objectUuid, const std::optional<std::string>& geometryRevision, bool applyValidity, const vc::objects::TransformedGeometryView* transformed)
{
    const auto object = db.objectByUuid(txn, objectUuid);
    if (!object || object->kind != "surface") throw std::runtime_error("not a surface object: " + objectUuid);
    const auto revision = geometryRevision ? db.geometryRevision(txn, *geometryRevision) : db.currentGeometry(txn, object->uuid);
    if (!revision || revision->objectUuid != object->uuid) throw std::runtime_error("surface has no such geometry revision: " + objectUuid);
    const auto view = vc::objects::GeometrySnapshot(db, txn).surface(object->uuid, revision->uuid);
    const int rows = int(view.rows()), cols = int(view.columns());
    const auto stored = view.readTile(0, 0, rows, cols);
    auto points = std::make_unique<cv::Mat_<cv::Vec3f>>(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        const auto& p = stored.points[size_t(i)];
        (*points)(i / cols, i % cols) = cv::Vec3f(p[0], p[1], p[2]);
    }
    cv::Mat_<uint8_t> mask;
    if (!stored.validity.empty()) {
        mask.create(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            mask(i / cols, i % cols) = stored.validity[size_t(i)] ? 255 : 0;
            if (applyValidity && !stored.validity[size_t(i)]) (*points)(i / cols, i % cols) = kInvalid;
        }
    }
    cv::Vec2f scale(float(view.scale()[0]), float(view.scale()[1]));
    if (transformed) {
        if (transformed->revision().uuid != revision->uuid) throw std::invalid_argument("transformed view revision differs from surface");
        const auto tile = transformed->readTile(0, 0, rows, cols);
        scale={float(tile.scale[0]),float(tile.scale[1])};
        for (int i=0; i<rows*cols; ++i) {
            const auto& p=tile.points[i];
            (*points)(i/cols,i%cols)=tile.validity[i] ? cv::Vec3f(float(p[0]),float(p[1]),float(p[2])) : kInvalid;
        }
    }
    auto surface = std::make_unique<QuadSurface>(points.release(), scale);
    surface->id = object->uuid;
    surface->meta = utils::Json::parse(object->meta);
    surface->setComponents(surface_components(surface->meta));
    surface->meta["uuid"] = object->uuid;
    surface->meta["name"] = object->name;
    surface->tags = object->tags;
    if (!mask.empty()) surface->setChannel("mask", mask);
    for (const auto& array : revision->arrays)
        if (array.name != "points" && array.name != "valid") surface->setChannel(array.name, cv::Mat());
    return surface;
}

std::function<cv::Mat(const std::string&)> store_channel_loader(
    std::shared_ptr<vc::objects::ObjectStoreSession> session, vc::objects::AttachmentIdentity attachment, std::string geometryRevision)
{
    return [session = std::move(session), attachment = std::move(attachment), revision = std::move(geometryRevision)](const std::string& name) {
        cv::Mat result;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto descriptor = db.geometryRevision(txn, revision);
            if (!descriptor) throw std::runtime_error("geometry revision not found: " + revision);
            for (const auto& array : descriptor->arrays)
                if (array.name == name)
                    result = matFromBytes(array.dtype, array.shape,
                        db.readGeometryRectangle(txn, revision, name, 0, 0, array.shape[0], array.shape[1]));
        });
        return result;
    };
}

void create_quad_in_store(vc::objects::ObjectStore& db, vc::objects::WriteTxn& txn, const std::string& uuid, QuadSurface& surface,
    const std::string& name, const std::vector<std::string>& tags, const nlohmann::json& coordinates,
    const vc::objects::SurfaceProvenance& provenance)
{
    vc::objects::createSurface(db, txn, uuid, name, tags, coordinates, provenance, nlohmann::json::parse(surface.meta.dump()),
                               quad_surface_geometry(surface));
}

vc::objects::GeometryData quad_surface_geometry(QuadSurface& surface)
{
    const auto& points = *surface.rawPointsPtr();
    vc::geometry::Surface geometry{points.rows, points.cols, {surface._scale[0], surface._scale[1]}, {}};
    geometry.points.reserve(points.total());
    for (const auto& p : points)
        geometry.points.push_back(vc::geometry::validSurfacePoint(p) ? vc::geometry::SurfacePoint{p[0], p[1], p[2]} : vc::geometry::SurfacePoint{-1, -1, -1});
    for (const auto& name : surface.channelNames()) {
        const auto channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) continue;
        if (name == "mask") geometry.validity = validFromMask(channel, points.rows, points.cols).bytes;
        else geometry.channels.push_back(arrayFromMat(name, channel));
    }
    return vc::geometry::encode(geometry);
}

std::string write_quad_rect_to_store(vc::objects::ObjectStore& db, vc::objects::WriteTxn& txn, const std::string& objectUuid,
    QuadSurface& surface, const cv::Rect& rect, const std::string& expected,
    const vc::objects::ObjectStoreSession::KeepRecorded& decide, const vc::objects::TransformedGeometryView* transformed)
{
    const auto revision = currentSurface(db, txn, objectUuid, expected);
    const auto shape = pointsShape(revision);
    const auto& points = *surface.rawPointsPtr();
    if (points.rows != shape[0] || points.cols != shape[1])
        throw std::runtime_error("surface grid differs from the stored grid; resize it first");
    const auto area = rect & cv::Rect(0, 0, points.cols, points.rows);
    if (area.empty()) return revision.uuid;
    // Without a transformed view the surface holds stored coordinates; an
    // edit made through one is refused once the object or its registration
    // has moved on since the view was opened.
    std::optional<vc::objects::TransformedGeometryView> stored;
    if (transformed) {
        transformed->validateForWrite(db, txn);
    } else {
        stored.emplace(vc::objects::GeometrySnapshot(db, txn).viewStored(objectUuid, revision.uuid));
        transformed = &*stored;
    }
    std::vector<vc::geometry::SurfacePoint> desired;
    desired.reserve(size_t(area.width) * area.height);
    for (int r = area.y; r < area.y + area.height; ++r)
        for (int c = area.x; c < area.x + area.width; ++c) {
            const auto p = points(r, c);
            desired.push_back({p[0], p[1], p[2]});
        }
    const auto edits = transformed->surfaceEdits(area.y, area.x, area.height, area.width, desired);
    // One plan decides the affected relationships and commits.
    return decide ? db.editGeometry(txn, objectUuid, edits, revision.uuid, vc::objects::ObjectStore::EdgeDecider(decide))
                  : db.editGeometry(txn, objectUuid, edits, revision.uuid);
}

std::string write_quad_channels_to_store(vc::objects::ObjectStore& db, vc::objects::WriteTxn& txn, const std::string& objectUuid,
    QuadSurface& surface, const std::vector<std::string>& names, const std::vector<std::string>& remove,
    const std::string& expected, const std::vector<vc::objects::GeometryEdgeEdit>& edgeEdits)
{
    const auto revision = currentSurface(db, txn, objectUuid, expected);
    const auto shape = pointsShape(revision);
    std::vector<GeometryArray> channels;
    for (const auto& name : names) {
        const auto channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) throw std::runtime_error("surface has no channel " + name);
        channels.push_back(name == "mask" ? validFromMask(channel, int(shape[0]), int(shape[1])) : arrayFromMat(name, channel));
    }
    std::vector<std::string> removed;
    for (const auto& name : remove) removed.push_back(name == "mask" ? "valid" : name);
    return db.updateSurfaceChannels(txn, objectUuid, channels, removed, revision.uuid, edgeEdits);
}
