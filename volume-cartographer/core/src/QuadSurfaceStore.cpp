// QuadSurface <-> vc_object_store surface objects.
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/geometry/Geometry.hpp>
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

// The "mask" channel as per-vertex validity: a vertex is valid only if every
// mask pixel it covers is 255 (masks may be a whole multiple of the grid).
GeometryArray validFromMask(const cv::Mat& mask, int rows, int cols)
{
    if (mask.channels() != 1 || mask.rows % rows || mask.cols % cols)
        throw std::runtime_error("surface mask must be one channel and a whole multiple of the grid");
    cv::Mat_<uint8_t> values;
    mask.convertTo(values, CV_8U);
    const int sy = mask.rows / rows, sx = mask.cols / cols;
    std::vector<uint8_t> valid(size_t(rows) * size_t(cols), 255);
    for (int r = 0; r < mask.rows; ++r)
        for (int c = 0; c < mask.cols; ++c)
            if (values(r, c) < 255) valid[size_t(r / sy) * size_t(cols) + size_t(c / sx)] = 0;
    return {"valid", "|u1", {rows, cols}, std::move(valid)};
}

bool usable(const cv::Vec3f& p) { return vc::geometry::validSurfacePoint({p[0], p[1], p[2]}); }

std::vector<int64_t> pointsShape(const vc::objects::GeometryRevision& revision)
{
    for (const auto& array : revision.arrays)
        if (array.name == "points") return array.shape;
    throw std::runtime_error("stored surface has no points");
}

bool hasArray(const vc::objects::GeometryRevision& revision, const std::string& name)
{
    return std::any_of(revision.arrays.begin(), revision.arrays.end(), [&](const auto& a) { return a.name == name; });
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
    const std::string& objectUuid, const std::optional<std::string>& geometryRevision, bool applyValidity)
{
    const auto object = db.objectByUuid(txn, objectUuid);
    if (!object || object->kind != "surface") throw std::runtime_error("not a surface object: " + objectUuid);
    const auto revision = geometryRevision ? db.geometryRevision(txn, *geometryRevision) : db.currentGeometry(txn, object->uuid);
    if (!revision || revision->objectUuid != object->uuid) throw std::runtime_error("surface has no such geometry revision: " + objectUuid);
    const auto shape = pointsShape(*revision);
    const int rows = int(shape[0]), cols = int(shape[1]);
    auto points = std::make_unique<cv::Mat_<cv::Vec3f>>(rows, cols);
    const auto bytes = db.readGeometryRectangle(txn, revision->uuid, "points", 0, 0, rows, cols);
    std::memcpy(points->data, bytes.data(), bytes.size());
    for (auto& p : *points) if (!usable(p)) p = kInvalid;
    cv::Mat_<uint8_t> mask;
    if (hasArray(*revision, "valid")) {
        const auto valid = db.readGeometryRectangle(txn, revision->uuid, "valid", 0, 0, rows, cols);
        mask.create(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            mask(i / cols, i % cols) = valid[size_t(i)] ? 255 : 0;
            if (applyValidity && !valid[size_t(i)]) (*points)(i / cols, i % cols) = kInvalid;
        }
    }
    const auto attributes = nlohmann::json::parse(revision->attributes);
    const cv::Vec2f scale(attributes.at("scale").at(0).get<float>(), attributes.at("scale").at(1).get<float>());
    auto surface = std::make_unique<QuadSurface>(points.release(), scale);
    surface->id = object->uuid;
    surface->meta = utils::Json::parse(object->meta);
    surface->meta["uuid"] = object->uuid;
    surface->meta["name"] = object->name;
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

vc::objects::GeometryData quad_surface_geometry(QuadSurface& surface)
{
    auto* points = surface.rawPointsPtr();
    vc::objects::GeometryData data;
    data.kind = "surface";
    data.attributes = nlohmann::json{{"scale", {surface._scale[0], surface._scale[1]}}}.dump();
    cv::Mat_<cv::Vec3f> stored = points->clone();
    for (auto& p : stored) if (!usable(p)) p = kInvalid;
    GeometryArray xyz{"points", "<f4", {stored.rows, stored.cols, 3}, {}};
    xyz.bytes.assign(stored.datastart, stored.dataend);
    data.arrays.push_back(std::move(xyz));
    for (const auto& name : surface.channelNames()) {
        const auto channel = surface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) continue;
        data.arrays.push_back(name == "mask" ? validFromMask(channel, stored.rows, stored.cols) : arrayFromMat(name, channel));
    }
    return data;
}

std::string write_quad_rect_to_store(vc::objects::ObjectStore& db, vc::objects::WriteTxn& txn, const std::string& objectUuid,
    QuadSurface& surface, const cv::Rect& rect, const std::string& expected,
    const std::vector<vc::objects::GeometryEdgeEdit>& edgeEdits)
{
    const auto revision = currentSurface(db, txn, objectUuid, expected);
    const auto shape = pointsShape(revision);
    const auto& points = *surface.rawPointsPtr();
    if (points.rows != shape[0] || points.cols != shape[1])
        throw std::runtime_error("surface grid differs from the stored grid; resize it first");
    const auto area = rect & cv::Rect(0, 0, points.cols, points.rows);
    if (area.empty()) return revision.uuid;
    auto stored = db.readGeometryRectangle(txn, revision.uuid, "points", area.y, area.x, area.height, area.width);
    const bool hasValid = hasArray(revision, "valid");
    auto valid = hasValid ? db.readGeometryRectangle(txn, revision.uuid, "valid", area.y, area.x, area.height, area.width)
                          : std::vector<uint8_t>{};
    bool validChanged = false;
    for (int r = 0; r < area.height; ++r)
        for (int c = 0; c < area.width; ++c) {
            const size_t i = size_t(r) * size_t(area.width) + size_t(c);
            cv::Vec3f old;
            std::memcpy(old.val, stored.data() + i * 12, 12);
            // What this vertex looked like when loaded: masked vertices are sentinels.
            const cv::Vec3f loaded = usable(old) && (!hasValid || valid[i]) ? old : kInvalid;
            const cv::Vec3f now = usable(points(area.y + r, area.x + c)) ? points(area.y + r, area.x + c) : kInvalid;
            if (std::memcmp(now.val, loaded.val, 12) == 0) continue;
            std::memcpy(stored.data() + i * 12, now.val, 12);
            if (hasValid && now != kInvalid && !valid[i]) { valid[i] = 255; validChanged = true; }
        }
    std::vector<vc::objects::GeometryArrayEdit> edits;
    auto rows = [&](const std::string& name, const std::vector<uint8_t>& bytes, size_t pixel, int64_t components) {
        for (int r = 0; r < area.height; ++r)
            edits.push_back({name, ((area.y + r) * shape[1] + area.x) * components,
                {bytes.begin() + r * area.width * pixel, bytes.begin() + (r + 1) * area.width * pixel}});
    };
    rows("points", stored, 12, 3);
    if (validChanged) rows("valid", valid, 1, 1);
    return db.editGeometry(txn, objectUuid, edits, revision.uuid, edgeEdits);
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

std::string resize_store_surface(vc::objects::ObjectStore& db, vc::objects::WriteTxn& txn, const std::string& objectUuid,
    const cv::Rect& gridInStored, const std::string& expected, const std::vector<vc::objects::GeometryEdgeEdit>& edgeEdits)
{
    const auto revision = currentSurface(db, txn, objectUuid, expected);
    vc::objects::GeometrySurfaceResize resize;
    resize.row = gridInStored.y; resize.column = gridInStored.x;
    resize.rows = gridInStored.height; resize.columns = gridInStored.width;
    for (const auto& array : revision.arrays) {
        auto fillShape = array.shape;
        fillShape[0] = fillShape[1] = 1;
        const size_t pixel = cv::Mat(1, 1, CV_MAKETYPE(cvDepth(array.dtype), array.shape.size() == 3 ? int(array.shape[2]) : 1)).elemSize();
        GeometryArray fill{array.name, array.dtype, fillShape, std::vector<uint8_t>(pixel, 0)};
        if (array.name == "points") {
            fill.bytes.resize(12);
            std::memcpy(fill.bytes.data(), kInvalid.val, 12);
        } else if (array.name == "valid") fill.bytes = {255};
        resize.fill.push_back(std::move(fill));
    }
    return db.resizeSurface(txn, objectUuid, resize, revision.uuid, edgeEdits);
}
