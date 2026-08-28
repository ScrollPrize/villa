#include <spiral_graph/track_spatial_index.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace spiral::trackio {
namespace {

constexpr std::array<char, 8> magic{'V', 'C', 'T', 'S', 'P', 'X', '1', '\0'};

#pragma pack(push, 1)
struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t cell_size;
    std::uint64_t point_count;
    std::uint64_t cell_count;
    std::uint64_t reserved[4];
};

struct Entry {
    std::int32_t z;
    std::int32_t y;
    std::int32_t x;
    std::uint64_t point;
};

struct Cell {
    std::int32_t z;
    std::int32_t y;
    std::int32_t x;
    std::uint64_t begin;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 64);
static_assert(sizeof(Entry) == 20);
static_assert(sizeof(Cell) == 20);

bool same_cell(const Entry& a, const Entry& b)
{
    return a.z == b.z && a.y == b.y && a.x == b.x;
}

bool entry_less(const Entry& a, const Entry& b)
{
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.point < b.point;
}

bool cell_less(const Cell& a, const Cell& b)
{
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

std::pair<std::uint64_t, std::uint64_t> coordinate_stamp(
    const PackedTrackStore& tracks)
{
    const auto path = tracks.path / "coordinates.i32";
    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(path, error);
    if (error) throw std::system_error(error, "cannot stat " + path.string());
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error) throw std::system_error(error, "cannot stat " + path.string());
    return {
        size,
        static_cast<std::uint64_t>(modified.time_since_epoch().count()),
    };
}

std::int32_t floor_div(std::int32_t coordinate, std::uint32_t divisor)
{
    const std::int64_t value = coordinate;
    const std::int64_t width = divisor;
    const std::int64_t quotient = value >= 0
        ? value / width : -((-value + width - 1) / width);
    if (quotient < std::numeric_limits<std::int32_t>::min()
        || quotient > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("spatial bin coordinate overflows int32");
    }
    return static_cast<std::int32_t>(quotient);
}

std::int32_t floor_bin(float coordinate, std::uint32_t divisor)
{
    const double value = std::floor(static_cast<double>(coordinate) / divisor);
    if (value < std::numeric_limits<std::int32_t>::min()
        || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("spatial query bin overflows int32");
    }
    return static_cast<std::int32_t>(value);
}

class Mapping {
public:
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    Mapping& operator=(Mapping&& other) noexcept
    {
        if (this != &other) {
            close();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    ~Mapping() { close(); }

    void open(const std::filesystem::path& path)
    {
        close();
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "cannot open " + path.string());
        }
        struct stat status {};
        if (::fstat(descriptor, &status) != 0 || status.st_size < 0) {
            const int error = errno;
            ::close(descriptor);
            throw std::system_error(error, std::generic_category(),
                                    "cannot stat " + path.string());
        }
        size_ = static_cast<std::size_t>(status.st_size);
        if (size_) {
            data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
            if (data_ == MAP_FAILED) {
                const int error = errno;
                data_ = nullptr;
                ::close(descriptor);
                throw std::system_error(error, std::generic_category(),
                                        "cannot map " + path.string());
            }
        }
        ::close(descriptor);
    }

    template <typename T>
    const T* exact(std::size_t count) const
    {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)
            || count * sizeof(T) != size_) {
            throw std::runtime_error("spatial index file has the wrong size");
        }
        return static_cast<const T*>(data_);
    }

private:
    void close() noexcept
    {
        if (data_) ::munmap(data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace

struct TrackSpatialIndex::Impl {
    std::filesystem::path path;
    Header header{};
    Mapping cells_mapping;
    Mapping points_mapping;
    const Cell* cells = nullptr;
    const std::uint64_t* points = nullptr;
};

TrackSpatialIndex::TrackSpatialIndex() : impl_(std::make_unique<Impl>()) {}
TrackSpatialIndex::~TrackSpatialIndex() = default;
TrackSpatialIndex::TrackSpatialIndex(TrackSpatialIndex&&) noexcept = default;
TrackSpatialIndex& TrackSpatialIndex::operator=(TrackSpatialIndex&&) noexcept = default;

SpatialIndexInfo TrackSpatialIndex::build(
    const PackedTrackStore& tracks,
    const std::filesystem::path& output,
    std::uint32_t cell_size,
    std::size_t memory_budget_bytes)
{
    if (cell_size == 0) throw std::invalid_argument("track index cell size must be positive");
    if (std::filesystem::is_regular_file(output / "header.bin")) {
        TrackSpatialIndex existing;
        existing.open(output);
        existing.validate_source(tracks);
        SpatialIndexInfo result = existing.info();
        if (result.cell_size != cell_size
            || result.point_count != static_cast<std::uint64_t>(tracks.point_count)) {
            throw std::runtime_error(
                "existing track index does not match requested store/options");
        }
        result.already_present = true;
        return result;
    }
    const std::size_t point_count = static_cast<std::size_t>(tracks.point_count);
    const auto source_stamp = coordinate_stamp(tracks);
    const std::size_t entries_per_run = std::max<std::size_t>(
        1, memory_budget_bytes / sizeof(Entry));
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temporary = output.string() + ".building-"
        + std::to_string(::getpid()) + "-" + std::to_string(nonce);
    std::filesystem::create_directories(temporary / "runs");
    std::vector<std::filesystem::path> runs;
    std::vector<Entry> entries;
    entries.reserve(std::min(entries_per_run, point_count));
    for (std::size_t begin = 0; begin < point_count; begin += entries_per_run) {
        const std::size_t end = std::min(point_count, begin + entries_per_run);
        entries.clear();
        for (std::size_t point = begin; point < end; ++point) {
            entries.push_back({
                floor_div(tracks.coords[3 * point], cell_size),
                floor_div(tracks.coords[3 * point + 1], cell_size),
                floor_div(tracks.coords[3 * point + 2], cell_size),
                point,
            });
        }
        std::sort(entries.begin(), entries.end(), entry_less);
        const auto run = temporary / "runs" / (std::to_string(runs.size()) + ".bin");
        std::ofstream stream(run, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(entries.data()),
                     static_cast<std::streamsize>(entries.size() * sizeof(Entry)));
        if (!stream) throw std::runtime_error("failed writing spatial-index run");
        runs.push_back(run);
    }

    struct Cursor {
        std::ifstream stream;
        Entry value{};
        bool valid = false;
    };
    std::vector<Cursor> cursors(runs.size());
    struct HeapItem { Entry value; std::size_t run; };
    const auto compare = [](const HeapItem& a, const HeapItem& b) {
        return entry_less(b.value, a.value);
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(compare)> heap(compare);
    for (std::size_t run = 0; run < runs.size(); ++run) {
        cursors[run].stream.open(runs[run], std::ios::binary);
        cursors[run].stream.read(
            reinterpret_cast<char*>(&cursors[run].value), sizeof(Entry));
        cursors[run].valid = static_cast<bool>(cursors[run].stream);
        if (cursors[run].valid) heap.push({cursors[run].value, run});
    }
    std::ofstream point_stream(temporary / "point_ids.u64", std::ios::binary | std::ios::trunc);
    std::ofstream cell_stream(temporary / "cells.bin", std::ios::binary | std::ios::trunc);
    std::uint64_t emitted = 0;
    std::uint64_t cell_count = 0;
    std::optional<Entry> current_cell;
    while (!heap.empty()) {
        const HeapItem item = heap.top();
        heap.pop();
        if (!current_cell || !same_cell(*current_cell, item.value)) {
            const Cell cell{item.value.z, item.value.y, item.value.x, emitted};
            cell_stream.write(reinterpret_cast<const char*>(&cell), sizeof(cell));
            current_cell = item.value;
            ++cell_count;
        }
        point_stream.write(
            reinterpret_cast<const char*>(&item.value.point), sizeof(item.value.point));
        ++emitted;
        Cursor& cursor = cursors[item.run];
        cursor.stream.read(reinterpret_cast<char*>(&cursor.value), sizeof(Entry));
        if (cursor.stream) heap.push({cursor.value, item.run});
    }
    if (!point_stream || !cell_stream || emitted != point_count) {
        throw std::runtime_error("failed merging spatial-index runs");
    }
    Header header{};
    std::memcpy(header.magic, magic.data(), magic.size());
    header.version = 1;
    header.cell_size = cell_size;
    header.point_count = point_count;
    header.cell_count = cell_count;
    if (coordinate_stamp(tracks) != source_stamp) {
        throw std::runtime_error("track coordinates changed while building the index");
    }
    header.reserved[0] = source_stamp.first;
    header.reserved[1] = source_stamp.second;
    {
        std::ofstream stream(temporary / "header.bin", std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (!stream) throw std::runtime_error("failed writing spatial-index header");
    }
    std::filesystem::remove_all(temporary / "runs");
    std::error_code error;
    std::filesystem::rename(temporary, output, error);
    if (error) throw std::system_error(error, "cannot publish track spatial index");
    return {point_count, cell_count, cell_size, false};
}

void TrackSpatialIndex::open(const std::filesystem::path& path)
{
    std::ifstream stream(path / "header.bin", std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open track spatial-index header");
    stream.read(reinterpret_cast<char*>(&impl_->header), sizeof(Header));
    if (!stream || std::memcmp(impl_->header.magic, magic.data(), magic.size()) != 0
        || impl_->header.version != 1 || impl_->header.cell_size == 0) {
        throw std::runtime_error("invalid track spatial-index header");
    }
    if (impl_->header.cell_count > std::numeric_limits<std::size_t>::max()
        || impl_->header.point_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("track spatial index is too large for this host");
    }
    impl_->cells_mapping.open(path / "cells.bin");
    impl_->points_mapping.open(path / "point_ids.u64");
    impl_->cells = impl_->cells_mapping.exact<Cell>(
        static_cast<std::size_t>(impl_->header.cell_count));
    impl_->points = impl_->points_mapping.exact<std::uint64_t>(
        static_cast<std::size_t>(impl_->header.point_count));
    impl_->path = path;
}

SpatialIndexInfo TrackSpatialIndex::info() const
{
    return {
        impl_->header.point_count,
        impl_->header.cell_count,
        impl_->header.cell_size,
        false,
    };
}

void TrackSpatialIndex::validate_source(const PackedTrackStore& tracks) const
{
    if (impl_->header.point_count
        != static_cast<std::uint64_t>(tracks.point_count)) {
        throw std::runtime_error("track spatial index point count does not match source");
    }
    const auto [coordinate_size, coordinate_mtime] = coordinate_stamp(tracks);
    if (impl_->header.reserved[0] != coordinate_size
        || impl_->header.reserved[1] != coordinate_mtime) {
        throw std::runtime_error(
            "track spatial index source fingerprint does not match; rebuild the index");
    }
}

void TrackSpatialIndex::query(
    const SpatialBounds& bounds,
    std::vector<std::uint64_t>& point_ids) const
{
    if (!impl_->cells) throw std::logic_error("track spatial index is not open");
    const auto z0 = floor_bin(bounds.z_min, impl_->header.cell_size);
    const auto y0 = floor_bin(bounds.y_min, impl_->header.cell_size);
    const auto x0 = floor_bin(bounds.x_min, impl_->header.cell_size);
    const auto z1 = floor_bin(bounds.z_max, impl_->header.cell_size);
    const auto y1 = floor_bin(bounds.y_max, impl_->header.cell_size);
    const auto x1 = floor_bin(bounds.x_max, impl_->header.cell_size);
    const Cell* begin = impl_->cells;
    const Cell* end = begin + impl_->header.cell_count;
    for (std::int64_t z = z0; z <= z1; ++z) {
        for (std::int64_t y = y0; y <= y1; ++y) {
            const Cell low{static_cast<std::int32_t>(z), static_cast<std::int32_t>(y), x0, 0};
            auto found = std::lower_bound(begin, end, low, cell_less);
            while (found != end && found->z == z && found->y == y && found->x <= x1) {
                const std::uint64_t first = found->begin;
                const std::uint64_t last = (found + 1 == end)
                    ? impl_->header.point_count : (found + 1)->begin;
                point_ids.insert(point_ids.end(), impl_->points + first, impl_->points + last);
                ++found;
            }
        }
    }
}

} // namespace spiral::trackio
