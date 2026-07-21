#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;
namespace fs = std::filesystem;

namespace {

using FloatCoordinates = nb::ndarray<nb::numpy, const float,
                                     nb::shape<-1, 3>, nb::c_contig>;
using Int64Vector = nb::ndarray<nb::numpy, const int64_t,
                                nb::ndim<1>, nb::c_contig>;

constexpr char TrackStoreMagic[8] = {'V', 'C', 'T', 'R', 'K', '0', '1', '\0'};
constexpr uint32_t TrackStoreVersion = 1;

struct TrackStoreHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t track_count;
    uint64_t point_count;
    uint64_t reserved[4];
};

static_assert(sizeof(TrackStoreHeader) == 64);

class Mapping {
public:
    Mapping() = default;

    explicit Mapping(const fs::path& path)
    {
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0)
            throw std::system_error(
                errno, std::generic_category(), "cannot open " + path.string());
        struct stat status {};
        if (::fstat(descriptor_, &status) != 0) {
            const int error = errno;
            ::close(descriptor_);
            descriptor_ = -1;
            throw std::system_error(
                error, std::generic_category(), "cannot stat " + path.string());
        }
        size_ = static_cast<size_t>(status.st_size);
        if (size_ != 0) {
            data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
            if (data_ == MAP_FAILED) {
                data_ = nullptr;
                const int error = errno;
                ::close(descriptor_);
                descriptor_ = -1;
                throw std::system_error(
                    error, std::generic_category(), "cannot map " + path.string());
            }
#ifdef MADV_SEQUENTIAL
            ::madvise(data_, size_, MADV_SEQUENTIAL);
#endif
        }
    }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    Mapping(Mapping&& other) noexcept
        : descriptor_(std::exchange(other.descriptor_, -1)),
          data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0))
    {
    }

    ~Mapping()
    {
        if (data_ != nullptr)
            ::munmap(data_, size_);
        if (descriptor_ >= 0)
            ::close(descriptor_);
    }

    const void* data() const { return data_; }
    size_t size() const { return size_; }

    template <typename T>
    const T* as(size_t count, const std::string& label) const
    {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)
            || size_ != count * sizeof(T))
            throw std::runtime_error(
                label + " has an invalid byte size (expected "
                + std::to_string(count * sizeof(T)) + ", got "
                + std::to_string(size_) + ")");
        return static_cast<const T*>(data_);
    }

private:
    int descriptor_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> own_1d(std::vector<T>&& values)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        held->data(), {held->size()}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> own_2d(
    std::vector<T>&& values, size_t rows, size_t columns)
{
    auto* held = new std::vector<T>(std::move(values));
    nb::capsule owner(held, [](void* pointer) noexcept {
        delete static_cast<std::vector<T>*>(pointer);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
        held->data(), {rows, columns}, owner);
}

TrackStoreHeader read_header(const fs::path& root)
{
    Mapping mapping(root / "header.bin");
    if (mapping.size() != sizeof(TrackStoreHeader))
        throw std::runtime_error("track-store header has an invalid size");
    TrackStoreHeader header {};
    std::memcpy(&header, mapping.data(), sizeof(header));
    if (std::memcmp(header.magic, TrackStoreMagic, sizeof(TrackStoreMagic)) != 0)
        throw std::runtime_error("not a VC packed track store");
    if (header.version != TrackStoreVersion
        || header.header_size != sizeof(TrackStoreHeader))
        throw std::runtime_error(
            "unsupported packed track-store version "
            + std::to_string(header.version));
    return header;
}

int effective_workers(int requested)
{
    if (requested < 1)
        throw std::runtime_error("workers must be positive");
#ifdef _OPENMP
    return std::min(requested, omp_get_num_procs());
#else
    return 1;
#endif
}

nb::dict load_track_store(
    const std::string& path, int64_t z_minimum, int64_t z_maximum, int workers)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("packed track stores currently require little endian");
    workers = effective_workers(workers);
    const fs::path root(path);
    const TrackStoreHeader header = read_header(root);
    const size_t track_count = static_cast<size_t>(header.track_count);
    const size_t point_count = static_cast<size_t>(header.point_count);
    if (header.track_count != track_count || header.point_count != point_count)
        throw std::runtime_error("track-store dimensions exceed this platform");

    Mapping coordinates_mapping(root / "coordinates.i32");
    Mapping offsets_mapping(root / "offsets.i64");
    Mapping source_ids_mapping(root / "source_ids.u64");
    Mapping family_codes_mapping(root / "family_codes.i8");
    Mapping z_bounds_mapping(root / "z_bounds.i32");
    Mapping arclengths_mapping(root / "arclengths.f64");
    Mapping tortuosities_mapping(root / "tortuosities.f64");
    const int32_t* coordinates = coordinates_mapping.as<int32_t>(
        point_count * 3, "coordinates.i32");
    const int64_t* offsets = offsets_mapping.as<int64_t>(
        track_count + 1, "offsets.i64");
    const uint64_t* source_ids = source_ids_mapping.as<uint64_t>(
        track_count, "source_ids.u64");
    const int8_t* family_codes = family_codes_mapping.as<int8_t>(
        track_count, "family_codes.i8");
    const int32_t* z_bounds = z_bounds_mapping.as<int32_t>(
        track_count * 2, "z_bounds.i32");
    const double* stored_arclengths = arclengths_mapping.as<double>(
        track_count, "arclengths.f64");
    const double* stored_tortuosities = tortuosities_mapping.as<double>(
        track_count, "tortuosities.f64");
    if (offsets[0] != 0 || offsets[track_count] != static_cast<int64_t>(point_count))
        throw std::runtime_error("track-store offsets do not match coordinates");

    std::vector<uint32_t> selected;
    selected.reserve(track_count);
    for (size_t track = 0; track < track_count; ++track) {
        if (offsets[track + 1] < offsets[track])
            throw std::runtime_error("track-store offsets are not monotonic");
        if (offsets[track + 1] == offsets[track])
            continue;
        if (z_bounds[2 * track] < z_minimum
            || z_bounds[2 * track + 1] >= z_maximum)
            continue;
        if (track > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("packed store exceeds UINT32_MAX tracks");
        selected.push_back(static_cast<uint32_t>(track));
    }

    std::vector<int64_t> selected_offsets(selected.size() + 1, 0);
    std::vector<uint64_t> selected_source_ids(selected.size());
    std::vector<int8_t> selected_family_codes(selected.size());
    std::vector<double> arclengths(selected.size(), 0.0);
    std::vector<double> tortuosities(selected.size(),
        std::numeric_limits<double>::infinity());
    for (size_t output = 0; output < selected.size(); ++output) {
        const size_t track = selected[output];
        selected_offsets[output + 1] = selected_offsets[output]
            + offsets[track + 1] - offsets[track];
        selected_source_ids[output] = source_ids[track];
        selected_family_codes[output] = family_codes[track];
        arclengths[output] = stored_arclengths[track];
        tortuosities[output] = stored_tortuosities[track];
    }
    const size_t selected_points = static_cast<size_t>(selected_offsets.back());
    if (selected_points > std::numeric_limits<size_t>::max() / 3)
        throw std::runtime_error("selected coordinates overflow size_t");
    std::vector<float> selected_coordinates(selected_points * 3);

    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
        for (int64_t output = 0;
             output < static_cast<int64_t>(selected.size()); ++output) {
            const size_t track = selected[static_cast<size_t>(output)];
            const int64_t input_begin = offsets[track];
            const int64_t input_end = offsets[track + 1];
            const int64_t output_begin = selected_offsets[static_cast<size_t>(output)];
            for (int64_t local = 0; local < input_end - input_begin; ++local) {
                for (size_t axis = 0; axis < 3; ++axis) {
                    selected_coordinates[
                        3 * static_cast<size_t>(output_begin + local) + axis]
                        = static_cast<float>(coordinates[
                            3 * static_cast<size_t>(input_begin + local) + axis]);
                }
            }
        }
    }

    nb::dict result;
    result["coordinates"] = own_2d(
        std::move(selected_coordinates), selected_points, 3);
    result["offsets"] = own_1d(std::move(selected_offsets));
    result["source_ids"] = own_1d(std::move(selected_source_ids));
    result["family_codes"] = own_1d(std::move(selected_family_codes));
    result["arclengths"] = own_1d(std::move(arclengths));
    result["tortuosities"] = own_1d(std::move(tortuosities));
    result["stored_track_count"] = header.track_count;
    result["stored_point_count"] = header.point_count;
    return result;
}

nb::dict inspect_track_store(const std::string& path)
{
    const TrackStoreHeader header = read_header(fs::path(path));
    nb::dict result;
    result["version"] = header.version;
    result["track_count"] = header.track_count;
    result["point_count"] = header.point_count;
    return result;
}

nb::dict compact_tracks(
    FloatCoordinates coordinates, Int64Vector offsets,
    Int64Vector selected, int workers)
{
    workers = effective_workers(workers);
    if (offsets.shape(0) == 0 || offsets(0) != 0
        || offsets(offsets.shape(0) - 1)
            != static_cast<int64_t>(coordinates.shape(0)))
        throw std::runtime_error("track offsets do not match coordinates");
    const size_t track_count = offsets.shape(0) - 1;
    std::vector<int64_t> output_offsets(selected.shape(0) + 1, 0);
    for (size_t output = 0; output < selected.shape(0); ++output) {
        const int64_t track = selected(output);
        if (track < 0 || static_cast<size_t>(track) >= track_count)
            throw std::runtime_error("selected track index is out of range");
        output_offsets[output + 1] = output_offsets[output]
            + offsets(track + 1) - offsets(track);
    }
    const size_t point_count = static_cast<size_t>(output_offsets.back());
    std::vector<float> output_coordinates(point_count * 3);
    {
        nb::gil_scoped_release release;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(workers)
        for (int64_t output = 0;
             output < static_cast<int64_t>(selected.shape(0)); ++output) {
            const int64_t track = selected(static_cast<size_t>(output));
            const int64_t begin = offsets(track);
            const int64_t length = offsets(track + 1) - begin;
            const int64_t destination = output_offsets[static_cast<size_t>(output)];
            std::copy_n(
                coordinates.data() + 3 * begin, 3 * length,
                output_coordinates.data() + 3 * destination);
        }
    }
    nb::dict result;
    result["coordinates"] = own_2d(
        std::move(output_coordinates), point_count, 3);
    result["offsets"] = own_1d(std::move(output_offsets));
    return result;
}

} // namespace

NB_MODULE(track_store, module)
{
    module.doc() = "Native loader for mmap-backed VC packed track stores.";
    module.def(
        "load", &load_track_store, nb::arg("path"),
        nb::arg("z_minimum") = std::numeric_limits<int64_t>::min(),
        nb::arg("z_maximum") = std::numeric_limits<int64_t>::max(),
        nb::arg("workers") = 1);
    module.def("inspect", &inspect_track_store, nb::arg("path"));
    module.def(
        "compact", &compact_tracks, nb::arg("coordinates"),
        nb::arg("offsets"), nb::arg("selected"), nb::arg("workers") = 1);
}
