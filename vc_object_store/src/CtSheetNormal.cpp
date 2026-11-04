#include "vc_object_store/CtSheetNormal.hpp"
#include <vc_reader/array.hpp>
#include <vc_reader/reader.hpp>
#include <vc_reader/zarr.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace vc {
struct CtSheetNormalSampler::Impl {
    vcr::zarr::OpenedArray ct;
    double scale;
    std::mutex mutex;
    std::map<std::array<double, 3>, CtSheetNormal> cache;
    std::deque<std::array<double, 3>> order;

    Impl(const std::filesystem::path& path, int level, double scale)
        : ct(vcr::zarr::openPyramidLevel(path.string(), level, options())), scale(scale) {}

    static vcr::zarr::OpenOptions options()
    {
        vcr::zarr::OpenOptions out;
        out.memoryCache = std::make_shared<vcr::MemoryCache>(std::size_t(64) << 20);
        return out;
    }
};

CtSheetNormalSampler::CtSheetNormalSampler(const std::filesystem::path& path, int level, double scale)
{
    if (level < 0 || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("invalid CT level or scale");
    impl_ = std::make_unique<Impl>(path, level, scale);
}

CtSheetNormalSampler::~CtSheetNormalSampler() = default;

CtSheetNormal CtSheetNormalSampler::sampleNormal(const cv::Vec3d& xyz,
                                                 const std::atomic_bool* cancel) const
{
    auto cancelled = [&] {
        if (cancel && cancel->load()) throw std::runtime_error("CT normals cancelled");
    };
    cancelled();
    std::lock_guard lock(impl_->mutex);
    cancelled();
    const std::array<double, 3> point{xyz[0] * impl_->scale, xyz[1] * impl_->scale, xyz[2] * impl_->scale};
    for (auto coordinate : point) {
        if (!std::isfinite(coordinate)) throw std::invalid_argument("CT query must be finite");
    }
    if (auto found = impl_->cache.find(point); found != impl_->cache.end()) return found->second;
    std::array<float, 4> sample{};
    vcr::sheetNormals(impl_->ct, point.data(), 1, sample.data());
    cancelled();
    CtSheetNormal result{{sample[0], sample[1], sample[2]}, sample[3] > 0, sample[3],
                         sample[3] > 0 ? "" : "invalid tensor or insufficient CT support"};
    if (impl_->cache.size() == 256) {
        impl_->cache.erase(impl_->order.front());
        impl_->order.pop_front();
    }
    impl_->order.push_back(point);
    impl_->cache.emplace(point, result);
    return result;
}

struct LasagnaSheetNormalSampler::Impl {
    vcr::CtReader nx, ny;
    double scale;
    std::mutex mutex;

    Impl(const std::filesystem::path& x, const std::filesystem::path& y, int level, double scale)
        : nx(x.string(), level), ny(y.string(), level), scale(scale) {}
};

LasagnaSheetNormalSampler::LasagnaSheetNormalSampler(const std::filesystem::path& normalX,
    const std::filesystem::path& normalY, int level, double scale)
{
    if (level < 0 || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("invalid lasagna normal level or scale");
    impl_ = std::make_unique<Impl>(normalX, normalY, level, scale);
    if (impl_->nx.shape() != impl_->ny.shape())
        throw std::invalid_argument("lasagna nx and ny volumes differ in shape");
}

LasagnaSheetNormalSampler::~LasagnaSheetNormalSampler() = default;

namespace {
// The voxel nearest a working XYZ point at `scale`, ZYX; nullopt outside `shape`.
template <typename Shape>
std::optional<std::array<int64_t, 3>> nearestLasagnaVoxel(const cv::Vec3d& xyz, double scale, const Shape& shape)
{
    std::array<int64_t, 3> zyx;
    for (int axis = 0; axis < 3; ++axis) {
        const double coordinate = xyz[2 - axis] * scale;
        if (!std::isfinite(coordinate)) throw std::invalid_argument("lasagna normal query must be finite");
        zyx[axis] = int64_t(std::llround(coordinate));
    }
    for (int axis = 0; axis < 3; ++axis)
        if (zyx[axis] < 0 || zyx[axis] >= int64_t(shape[axis])) return std::nullopt;
    return zyx;
}

CtSheetNormal decodeLasagnaNormal(uint8_t x, uint8_t y)
{
    if (x == 0 && y == 0) return {{0, 0, 0}, false, 0, "no lasagna normal"};
    const double dx = (x - 128.0) / 127.0, dy = (y - 128.0) / 127.0;
    const cv::Vec3d normal(dx, dy, std::sqrt(std::max(0.0, 1.0 - dx * dx - dy * dy)));
    return {normal / cv::norm(normal), true, 0, ""};
}

const CtSheetNormal kOutsideLasagna{{0, 0, 0}, false, 0, "outside lasagna normal volume"};
} // namespace

CtSheetNormal LasagnaSheetNormalSampler::sampleNormal(const cv::Vec3d& xyz,
                                                      const std::atomic_bool* cancel) const
{
    if (cancel && cancel->load()) throw std::runtime_error("lasagna normals cancelled");
    const auto zyx = nearestLasagnaVoxel(xyz, impl_->scale, impl_->nx.shape());
    if (!zyx) return kOutsideLasagna;
    uint8_t x = 0, y = 0;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->nx.readRegion(*zyx, {1, 1, 1}, std::span(&x, 1));
        impl_->ny.readRegion(*zyx, {1, 1, 1}, std::span(&y, 1));
    }
    return decodeLasagnaNormal(x, y);
}

namespace {
std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read lasagna pack file " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// A C-order little-endian int32 .npy array of `shape` (the pack's table.npy).
std::vector<int32_t> readTable(const std::filesystem::path& path, const std::array<int64_t, 3>& shape)
{
    static_assert(std::endian::native == std::endian::little, "lasagna packs are little-endian");
    const std::string data = readText(path);
    if (data.size() < 10 || data.compare(0, 6, "\x93NUMPY") != 0)
        throw std::runtime_error(path.string() + " is not a .npy file");
    const int major = static_cast<unsigned char>(data[6]);
    const std::size_t lengthBytes = major == 1 ? 2 : 4;
    std::size_t headerLength = 0;
    for (std::size_t i = 0; i < lengthBytes; ++i)
        headerLength |= std::size_t(static_cast<unsigned char>(data[8 + i])) << (8 * i);
    const std::size_t offset = 8 + lengthBytes + headerLength;
    const std::string header = data.substr(8 + lengthBytes, headerLength);
    std::string expectedShape = "(" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " +
                                std::to_string(shape[2]) + ")";
    if (header.find("'<i4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos ||
        header.find("'shape': " + expectedShape) == std::string::npos)
        throw std::runtime_error(path.string() + " is not the pack's " + expectedShape + " int32 brick table");
    const std::size_t count = std::size_t(shape[0] * shape[1] * shape[2]);
    if (data.size() != offset + count * sizeof(int32_t))
        throw std::runtime_error(path.string() + " has the wrong size");
    std::vector<int32_t> table(count);
    std::memcpy(table.data(), data.data() + offset, count * sizeof(int32_t));
    return table;
}

std::array<int64_t, 3> shapeOf(const nlohmann::json& meta, const char* key)
{
    const auto value = meta.at(key);
    return {value.at(0).get<int64_t>(), value.at(1).get<int64_t>(), value.at(2).get<int64_t>()};
}
} // namespace

struct PooledLasagnaSheetNormalSampler::Impl {
    std::string metadata;
    std::array<int64_t, 3> shape{}, brick{}, grid{};
    std::vector<int32_t> table;
    int64_t brickVoxels = 0;
    double scale = 1;
    std::mutex mutex;
    std::array<std::ifstream, 2> channels;
    // Recently read bricks by pool row: nx then ny.
    std::map<int32_t, std::vector<uint8_t>> bricks;
    std::deque<int32_t> order;
};

PooledLasagnaSheetNormalSampler::PooledLasagnaSheetNormalSampler(const std::filesystem::path& pack, double scale)
    : impl_(std::make_unique<Impl>())
{
    if (!(scale > 0) || !std::isfinite(scale)) throw std::invalid_argument("invalid lasagna normal scale");
    impl_->scale = scale;
    impl_->metadata = readText(pack / "meta.json");
    const auto meta = nlohmann::json::parse(impl_->metadata);
    if (meta.value("format", "") != "respool" || meta.value("version", 0) != 2 || meta.value("dtype", "") != "u1"
        || meta.at("channels").size() != 2)
        throw std::runtime_error(pack.string() + " is not a two-channel uint8 lasagna pack (respool v2)");
    impl_->shape = shapeOf(meta, "array_shape");
    impl_->brick = shapeOf(meta, "brick_shape");
    impl_->grid = shapeOf(meta, "grid_shape");
    impl_->brickVoxels = impl_->brick[0] * impl_->brick[1] * impl_->brick[2];
    impl_->table = readTable(pack / "table.npy", impl_->grid);
    const auto rows = meta.at("rows").get<int64_t>();
    for (int channel = 0; channel < 2; ++channel) {
        const auto path = pack / ("channel_" + std::to_string(channel) + ".u8");
        if (int64_t(std::filesystem::file_size(path)) != rows * impl_->brickVoxels)
            throw std::runtime_error(path.string() + " does not hold the pack's " + std::to_string(rows) + " bricks");
        impl_->channels[channel].open(path, std::ios::binary);
        if (!impl_->channels[channel]) throw std::runtime_error("cannot read " + path.string());
    }
}

PooledLasagnaSheetNormalSampler::~PooledLasagnaSheetNormalSampler() = default;

const std::string& PooledLasagnaSheetNormalSampler::metadata() const
{
    return impl_->metadata;
}

CtSheetNormal PooledLasagnaSheetNormalSampler::sampleNormal(const cv::Vec3d& xyz,
                                                            const std::atomic_bool* cancel) const
{
    if (cancel && cancel->load()) throw std::runtime_error("lasagna normals cancelled");
    const auto zyx = nearestLasagnaVoxel(xyz, impl_->scale, impl_->shape);
    if (!zyx) return kOutsideLasagna;
    const auto& v = *zyx;
    const auto& b = impl_->brick;
    const int64_t slot = ((v[0] / b[0]) * impl_->grid[1] + v[1] / b[1]) * impl_->grid[2] + v[2] / b[2];
    const int32_t row = impl_->table[std::size_t(slot)];
    if (row == 0) return decodeLasagnaNormal(0, 0);
    const int64_t local = ((v[0] % b[0]) * b[1] + v[1] % b[1]) * b[2] + v[2] % b[2];
    std::lock_guard lock(impl_->mutex);
    auto found = impl_->bricks.find(row);
    if (found == impl_->bricks.end()) {
        std::vector<uint8_t> brick(std::size_t(2 * impl_->brickVoxels));
        for (int channel = 0; channel < 2; ++channel) {
            auto& stream = impl_->channels[channel];
            stream.seekg(std::streamoff(row) * impl_->brickVoxels);
            stream.read(reinterpret_cast<char*>(brick.data() + channel * impl_->brickVoxels), impl_->brickVoxels);
            if (!stream) throw std::runtime_error("cannot read a lasagna pack brick");
        }
        if (impl_->bricks.size() == 128) {
            impl_->bricks.erase(impl_->order.front());
            impl_->order.pop_front();
        }
        impl_->order.push_back(row);
        found = impl_->bricks.emplace(row, std::move(brick)).first;
    }
    return decodeLasagnaNormal(found->second[std::size_t(local)],
                               found->second[std::size_t(impl_->brickVoxels + local)]);
}
} // namespace vc
