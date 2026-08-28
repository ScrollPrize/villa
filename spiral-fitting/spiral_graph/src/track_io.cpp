#include <spiral_graph/track_io.hpp>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace spiral::trackio {
namespace {

constexpr char magic[8] = {'V', 'C', 'T', 'R', 'K', '0', '1', '\0'};

struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint64_t track_count;
    std::uint64_t point_count;
    std::uint64_t reserved[4];
};
static_assert(sizeof(Header) == 64);

class Mapping {
public:
    Mapping() = default;
    explicit Mapping(const std::filesystem::path& path) { open(path); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&& other) noexcept
        : fd_(std::exchange(other.fd_, -1)),
          data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    Mapping& operator=(Mapping&& other) noexcept {
        if (this == &other) return *this;
        close();
        fd_ = std::exchange(other.fd_, -1);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }
    ~Mapping() { close(); }

    void open(const std::filesystem::path& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::system_error(
                errno, std::generic_category(), "cannot open " + path.string());
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0) {
            const int error = errno;
            close();
            throw std::system_error(
                error, std::generic_category(), "cannot stat " + path.string());
        }
        if (status.st_size < 0) {
            close();
            throw std::runtime_error("negative file size for " + path.string());
        }
        size_ = static_cast<std::size_t>(status.st_size);
        if (size_ == 0) return;
        void* value = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (value == MAP_FAILED) {
            const int error = errno;
            data_ = nullptr;
            close();
            throw std::system_error(
                error, std::generic_category(), "cannot map " + path.string());
        }
        data_ = static_cast<const std::uint8_t*>(value);
    }

    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    template <typename T>
    const T* exact(std::size_t count, std::string_view label) const {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)
            || size_ != count * sizeof(T)) {
            throw std::runtime_error(std::string(label) + " has an invalid byte size");
        }
        return reinterpret_cast<const T*>(data_);
    }

private:
    void close() noexcept {
        if (data_) ::munmap(const_cast<std::uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        data_ = nullptr;
        size_ = 0;
    }

    int fd_ = -1;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

std::uint16_t u16(const std::uint8_t* value) {
    std::uint16_t result;
    std::memcpy(&result, value, sizeof(result));
    return result;
}
std::uint32_t u32(const std::uint8_t* value) {
    std::uint32_t result;
    std::memcpy(&result, value, sizeof(result));
    return result;
}
std::uint64_t u64(const std::uint8_t* value) {
    std::uint64_t result;
    std::memcpy(&result, value, sizeof(result));
    return result;
}

struct NpyView {
    const std::uint8_t* data = nullptr;
    std::size_t count = 0;
    std::string descriptor;
};

std::string dictionary_string(const std::string& header, std::string_view key) {
    const std::size_t found = header.find(key);
    if (found == std::string::npos) throw std::runtime_error("malformed NPY header");
    const std::size_t colon = header.find(':', found + key.size());
    const std::size_t first = header.find('\'', colon);
    const std::size_t second = header.find('\'', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        throw std::runtime_error("malformed NPY dtype");
    }
    return header.substr(first + 1, second - first - 1);
}

class NpzMapping {
public:
    void open(const std::filesystem::path& path) {
        mapping_.open(path);
        members_.clear();
        const auto* data = mapping_.data();
        const std::size_t size = mapping_.size();
        if (size < 22) throw std::runtime_error(path.string() + " is not a ZIP file");
        std::size_t eocd = std::numeric_limits<std::size_t>::max();
        const std::size_t lower = size > 22 + 65536 ? size - 22 - 65536 : 0;
        for (std::size_t offset = size - 22;; --offset) {
            if (u32(data + offset) == 0x06054b50u) { eocd = offset; break; }
            if (offset == lower) break;
        }
        if (eocd == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("ZIP directory is missing from " + path.string());
        }
        std::uint64_t entries = u16(data + eocd + 10);
        std::uint64_t directory = u32(data + eocd + 16);
        if (entries == 0xffff || directory == 0xffffffffu) {
            if (eocd < 20 || u32(data + eocd - 20) != 0x07064b50u) {
                throw std::runtime_error("ZIP64 locator is missing from " + path.string());
            }
            const std::uint64_t zip64 = u64(data + eocd - 12);
            if (zip64 + 56 > size || u32(data + zip64) != 0x06064b50u) {
                throw std::runtime_error("ZIP64 directory is invalid in " + path.string());
            }
            entries = u64(data + zip64 + 32);
            directory = u64(data + zip64 + 48);
        }
        if (directory > size) throw std::runtime_error("ZIP directory offset is invalid");
        std::size_t cursor = static_cast<std::size_t>(directory);
        for (std::uint64_t ordinal = 0; ordinal < entries; ++ordinal) {
            if (cursor + 46 > size || u32(data + cursor) != 0x02014b50u) {
                throw std::runtime_error("invalid ZIP central directory in " + path.string());
            }
            const std::uint16_t method = u16(data + cursor + 10);
            const std::uint16_t name_size = u16(data + cursor + 28);
            const std::uint16_t extra_size = u16(data + cursor + 30);
            const std::uint16_t comment_size = u16(data + cursor + 32);
            if (cursor + 46ull + name_size + extra_size + comment_size > size) {
                throw std::runtime_error("truncated ZIP directory in " + path.string());
            }
            std::uint64_t local = u32(data + cursor + 42);
            const std::string name(
                reinterpret_cast<const char*>(data + cursor + 46), name_size);
            if (local == 0xffffffffu) {
                std::size_t extra = cursor + 46 + name_size;
                const std::size_t extra_end = extra + extra_size;
                bool found = false;
                while (extra + 4 <= extra_end) {
                    const std::uint16_t identifier = u16(data + extra);
                    const std::uint16_t block_size = u16(data + extra + 2);
                    if (extra + 4ull + block_size > extra_end) break;
                    if (identifier == 1) {
                        std::size_t field = extra + 4;
                        if (u32(data + cursor + 24) == 0xffffffffu) field += 8;
                        if (u32(data + cursor + 20) == 0xffffffffu) field += 8;
                        if (field + 8 > extra + 4ull + block_size) break;
                        local = u64(data + field);
                        found = true;
                        break;
                    }
                    extra += 4 + block_size;
                }
                if (!found) throw std::runtime_error("ZIP64 offset is missing for " + name);
            }
            cursor += 46 + name_size + extra_size + comment_size;
            if (name.size() < 4 || name.substr(name.size() - 4) != ".npy") continue;
            if (method != 0) {
                throw std::runtime_error(
                    name + " is compressed; crossings must be written with numpy.savez");
            }
            if (local + 30 > size || u32(data + local) != 0x04034b50u) {
                throw std::runtime_error("invalid ZIP local header for " + name);
            }
            const std::uint16_t local_name = u16(data + local + 26);
            const std::uint16_t local_extra = u16(data + local + 28);
            const std::size_t npy = static_cast<std::size_t>(local) + 30 + local_name + local_extra;
            if (npy + 10 > size || std::memcmp(data + npy, "\x93NUMPY", 6) != 0) {
                throw std::runtime_error("invalid NPY member " + name);
            }
            const std::uint8_t major = data[npy + 6];
            if (major < 1 || major > 3) throw std::runtime_error("unsupported NPY version");
            const std::size_t length_size = major == 1 ? 2 : 4;
            const std::uint64_t header_size = length_size == 2
                ? u16(data + npy + 8) : u32(data + npy + 8);
            const std::size_t header_at = npy + 8 + length_size;
            if (header_at + header_size > size) throw std::runtime_error("truncated NPY header");
            const std::string header(
                reinterpret_cast<const char*>(data + header_at),
                static_cast<std::size_t>(header_size));
            if (header.find("'fortran_order': False") == std::string::npos
                && header.find("\"fortran_order\": False") == std::string::npos) {
                throw std::runtime_error("Fortran-ordered NPY member " + name + " is unsupported");
            }
            const std::size_t shape_key = header.find("shape");
            const std::size_t left = header.find('(', shape_key);
            const std::size_t right = header.find(')', left);
            if (left == std::string::npos || right == std::string::npos) {
                throw std::runtime_error("malformed NPY shape for " + name);
            }
            NpyView view;
            view.descriptor = dictionary_string(header, "descr");
            view.count = 1;
            bool dimension = false;
            for (std::size_t at = left + 1; at < right;) {
                while (at < right && (header[at] < '0' || header[at] > '9')) ++at;
                if (at >= right) break;
                std::size_t end = at;
                while (end < right && header[end] >= '0' && header[end] <= '9') ++end;
                const std::size_t value = std::stoull(header.substr(at, end - at));
                if (view.count > std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(1, value)) {
                    throw std::runtime_error("NPY shape overflows size_t");
                }
                view.count *= value;
                dimension = true;
                at = end;
            }
            if (!dimension) view.count = 1;
            view.data = data + header_at + header_size;
            const auto element_size = [&]() -> std::size_t {
                if (view.descriptor.size() < 3) return 0;
                try { return std::stoull(view.descriptor.substr(2)); }
                catch (...) { return 0; }
            }();
            if (element_size == 0 || view.data < data
                || static_cast<std::size_t>(view.data - data) > size
                || view.count > (size - static_cast<std::size_t>(view.data - data)) / element_size) {
                throw std::runtime_error("NPY payload exceeds ZIP member for " + name);
            }
            members_.emplace(name.substr(0, name.size() - 4), std::move(view));
        }
    }

    const NpyView& member(std::string_view name, std::string_view descriptor) const {
        const auto found = members_.find(std::string(name));
        if (found == members_.end()) throw std::runtime_error("crossings is missing " + std::string(name));
        const std::string& actual = found->second.descriptor;
        // NumPy may emit '=' for native little-endian scalars.
        const bool compatible = actual == descriptor
            || (std::endian::native == std::endian::little && actual.size() == descriptor.size()
                && actual.substr(1) == descriptor.substr(1) && actual.front() == '=');
        if (!compatible) {
            throw std::runtime_error(
                "crossings " + std::string(name) + " has dtype " + actual
                + ", expected " + std::string(descriptor));
        }
        return found->second;
    }

private:
    Mapping mapping_;
    std::map<std::string, NpyView> members_;
};

template <typename T>
const T* view(const NpyView& value) {
    return reinterpret_cast<const T*>(value.data);
}

} // namespace

struct PackedTrackStore::Storage {
    Mapping header;
    Mapping coordinates;
    Mapping offsets;
    Mapping source_ids;
    Mapping families;
    Mapping arclengths;
    Mapping tortuosities;
    Mapping z_bounds;
};

void PackedTrackStore::open(
    const std::filesystem::path& root,
    const ProgressCallback& progress) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("packed track stores require a little-endian host");
    }
    auto storage = std::make_shared<Storage>();
    storage->header.open(root / "header.bin");
    if (storage->header.size() != sizeof(Header)) {
        throw std::runtime_error("track-store header has an invalid size");
    }
    Header header {};
    std::memcpy(&header, storage->header.data(), sizeof(header));
    if (std::memcmp(header.magic, magic, sizeof(magic)) != 0) {
        throw std::runtime_error("not a packed .vctracks store: " + root.string());
    }
    if (header.version != 1 || header.header_size != sizeof(Header)) {
        throw std::runtime_error("unsupported packed track-store version");
    }
    if (header.track_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        || header.point_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("packed track-store dimensions exceed INT64_MAX");
    }
    const std::size_t tracks = static_cast<std::size_t>(header.track_count);
    const std::size_t points = static_cast<std::size_t>(header.point_count);
    if (tracks != header.track_count || points != header.point_count) {
        throw std::runtime_error("packed track-store dimensions exceed this platform");
    }
    if (points > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::runtime_error("packed track-store point count overflows coordinate shape");
    }
    if (progress) progress("tracks.map", 0, tracks);
    storage->coordinates.open(root / "coordinates.i32");
    storage->offsets.open(root / "offsets.i64");
    storage->source_ids.open(root / "source_ids.u64");
    storage->families.open(root / "family_codes.i8");
    storage->arclengths.open(root / "arclengths.f64");
    storage->tortuosities.open(root / "tortuosities.f64");
    storage->z_bounds.open(root / "z_bounds.i32");
    const auto* new_coords = storage->coordinates.exact<std::int32_t>(points * 3, "coordinates.i32");
    const auto* new_offsets = storage->offsets.exact<std::int64_t>(tracks + 1, "offsets.i64");
    const auto* new_sources = storage->source_ids.exact<std::uint64_t>(tracks, "source_ids.u64");
    const auto* new_families = storage->families.exact<std::int8_t>(tracks, "family_codes.i8");
    const auto* new_arcs = storage->arclengths.exact<double>(tracks, "arclengths.f64");
    const auto* new_torts = storage->tortuosities.exact<double>(tracks, "tortuosities.f64");
    const auto* new_z_bounds = storage->z_bounds.exact<std::int32_t>(tracks * 2, "z_bounds.i32");
    if (new_offsets[0] != 0 || new_offsets[tracks] != static_cast<std::int64_t>(points)) {
        throw std::runtime_error("track-store offsets do not span coordinates");
    }
    if (progress) progress("tracks.validate", 0, tracks);
    for (std::size_t row = 0; row < tracks; ++row) {
        if (new_offsets[row] < 0 || new_offsets[row + 1] < new_offsets[row]
            || new_offsets[row + 1] > static_cast<std::int64_t>(points)) {
            throw std::runtime_error("track-store offsets are not monotonic");
        }
        if (new_offsets[row + 1] == new_offsets[row]) {
            throw std::runtime_error("track-store contains an empty track");
        }
        if (row && new_sources[row] <= new_sources[row - 1]) {
            throw std::runtime_error("track-store source IDs must be strictly increasing");
        }
        if (new_families[row] != 0 && new_families[row] != 1) {
            throw std::runtime_error("track-store family code must be 0 (H) or 1 (V)");
        }
        if (!std::isfinite(new_arcs[row]) || new_arcs[row] < 0.0) {
            throw std::runtime_error("track-store arclength is invalid");
        }
        if (new_z_bounds[2 * row] > new_z_bounds[2 * row + 1]) {
            throw std::runtime_error("track-store z bounds are inverted");
        }
        // z_bounds is derived acceleration metadata. Its shape/order is
        // validated above; recomputing it used to reread every coordinate
        // (more than 12 GB in the production store) before any work could
        // start. Consumers that use the bounds still apply exact geometric
        // tests to coordinates, so this redundant full-store pass provided no
        // safety for reconstruction.
        if (progress && ((row + 1) % 100'000 == 0 || row + 1 == tracks)) {
            progress("tracks.validate", row + 1, tracks);
        }
    }
    path = std::filesystem::absolute(root);
    track_count = static_cast<std::int64_t>(tracks);
    point_count = static_cast<std::int64_t>(points);
    coords = new_coords;
    offsets = new_offsets;
    source_ids = new_sources;
    fams = families = new_families;
    arcs = new_arcs;
    torts = new_torts;
    zbounds = z_bounds = new_z_bounds;
    storage_ = std::move(storage);
}

std::size_t PackedTrackStore::point_begin(std::size_t row) const {
    if (!storage_ || row >= static_cast<std::size_t>(track_count)) {
        throw std::out_of_range("track row is out of range");
    }
    return static_cast<std::size_t>(offsets[row]);
}

std::size_t PackedTrackStore::point_end(std::size_t row) const {
    if (!storage_ || row >= static_cast<std::size_t>(track_count)) {
        throw std::out_of_range("track row is out of range");
    }
    return static_cast<std::size_t>(offsets[row + 1]);
}

std::span<const std::int32_t> PackedTrackStore::track_coordinates(std::size_t row) const {
    const std::size_t begin = point_begin(row);
    const std::size_t end = point_end(row);
    return {coords + 3 * begin, 3 * (end - begin)};
}

struct CrossingStore::Storage {
    NpzMapping npz;
};

void CrossingStore::open(
    const std::filesystem::path& candidate,
    const PackedTrackStore& tracks,
    const ProgressCallback& progress)
{
    auto storage = std::make_shared<Storage>();
    if (progress) progress("crossings.map", 0, static_cast<std::size_t>(tracks.track_count));
    storage->npz.open(candidate);
    const NpyView& sources = storage->npz.member("source_ids", "<u8");
    const NpyView& off = storage->npz.member("offsets", "<i8");
    const NpyView& part = storage->npz.member("partners", "<i4");
    const NpyView& self = storage->npz.member("self_local", "<i4");
    const NpyView& other = storage->npz.member("partner_local", "<i4");
    const NpyView& position = storage->npz.member("positions", "<f8");
    const NpyView& clearance = storage->npz.member("clearances", "<f8");
    const std::size_t count = static_cast<std::size_t>(tracks.track_count);
    if (sources.count != count || off.count != count + 1) {
        throw std::runtime_error("crossings and track stores have different track counts");
    }
    if (self.count != part.count || other.count != part.count
        || position.count != part.count || clearance.count != part.count) {
        throw std::runtime_error("crossing record arrays are not parallel");
    }
    const auto* new_sources = view<std::uint64_t>(sources);
    const auto* new_offsets = view<std::int64_t>(off);
    const auto* new_partners = view<std::int32_t>(part);
    const auto* new_self = view<std::int32_t>(self);
    const auto* new_other = view<std::int32_t>(other);
    const auto* new_positions = view<double>(position);
    const auto* new_clearances = view<double>(clearance);
    if (new_offsets[0] != 0 || new_offsets[count] != static_cast<std::int64_t>(part.count)) {
        throw std::runtime_error("crossing offsets do not span the record arrays");
    }
    if (progress) progress("crossings.validate", 0, count);
    for (std::size_t row = 0; row < count; ++row) {
        if (new_sources[row] != tracks.source_ids[row]) {
            throw std::runtime_error("crossings source IDs are stale or row-misaligned");
        }
        if (new_offsets[row] < 0 || new_offsets[row + 1] < new_offsets[row]
            || new_offsets[row + 1] > static_cast<std::int64_t>(part.count)) {
            throw std::runtime_error("crossing offsets are not monotonic");
        }
        for (std::int64_t record = new_offsets[row]; record < new_offsets[row + 1]; ++record) {
            const std::size_t at = static_cast<std::size_t>(record);
            const std::int32_t partner = new_partners[at];
            if (partner < 0 || static_cast<std::size_t>(partner) >= count
                || static_cast<std::size_t>(partner) == row) {
                throw std::runtime_error("crossings contains an invalid partner row");
            }
            if (tracks.fams[row] == tracks.fams[partner]) {
                throw std::runtime_error("crossings contains a same-family edge");
            }
            const std::int64_t self_length = tracks.offsets[row + 1] - tracks.offsets[row];
            const std::int64_t partner_length = tracks.offsets[partner + 1] - tracks.offsets[partner];
            if (new_self[at] < 0 || new_self[at] >= self_length
                || new_other[at] < 0 || new_other[at] >= partner_length) {
                throw std::runtime_error("crossings contains an out-of-range local index");
            }
            if (!std::isfinite(new_positions[at]) || !std::isfinite(new_clearances[at])
                || new_clearances[at] < 0.0) {
                throw std::runtime_error("crossings contains non-finite geometry");
            }
        }
        if (progress && ((row + 1) % 100'000 == 0 || row + 1 == count)) {
            progress("crossings.validate", row + 1, count);
        }
    }
    path = std::filesystem::absolute(candidate);
    source_ids = new_sources;
    offsets = new_offsets;
    partners = new_partners;
    self_local = new_self;
    partner_local = new_other;
    positions = new_positions;
    clearances = new_clearances;
    track_count = count;
    records = part.count;
    storage_ = std::move(storage);
}

std::size_t CrossingStore::begin(std::size_t row) const {
    if (!storage_ || row >= track_count) throw std::out_of_range("track row is out of range");
    return static_cast<std::size_t>(offsets[row]);
}

std::size_t CrossingStore::end(std::size_t row) const {
    if (!storage_ || row >= track_count) throw std::out_of_range("track row is out of range");
    return static_cast<std::size_t>(offsets[row + 1]);
}

} // namespace spiral::trackio
