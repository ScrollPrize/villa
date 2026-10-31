#pragma once
// Spatial crops of a Source: Zarr, PNG/JPEG/TIFF and explicit stacks, local or remote, with
// exact encodings and explicitly owned decode caches.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vc_reader/sources.hpp"
#include "vc_reader/zarr.hpp"

namespace vcr {
class Array;
class Sha256;
}

namespace vcr::sources {

// A decoded file is not what its format says it should be, or cannot be decoded.
struct ImageFormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// A read would exceed a decode or working budget.
struct DecodeBudgetError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The numpy typestr ("|u1", "<u2", ">f4") of a typestr or dtype name; native-endian multi-byte
// types come back in native order. std::invalid_argument for anything else.
[[nodiscard]] std::string canonicalDtype(std::string_view dtype);
// Its name as numpy prints it: "uint8", "float32"; a typestr in the other byte order is kept.
[[nodiscard]] std::string dtypeName(std::string_view typestr);

// The bytes of a number as an element of a dtype; std::invalid_argument when it cannot hold it.
[[nodiscard]] std::vector<std::uint8_t> elementBytes(const nlohmann::json& value, const std::string& typestr);

// An owned C-order block of elements.
struct Block {
    std::vector<std::int64_t> shape;
    std::string dtype;  // typestr
    std::shared_ptr<std::uint8_t[]> data;
    std::size_t nbytes = 0;
    [[nodiscard]] static Block allocate(std::vector<std::int64_t> shape, std::string dtype);
    [[nodiscard]] std::size_t itemsize() const;
};

// ---- selected grids ----

// Micrometers per spatial unit; nullopt for a unit this does not know.
[[nodiscard]] std::optional<double> micrometersPer(std::string_view unit);
// Embeds an explicit pixel-to-UV transform in XYZ affine arithmetic.
[[nodiscard]] VolumeTransform pixelAffine(const PixelAffine& rows);

struct StorageGrid {
    std::optional<VolumeTransform> transform;
    std::optional<std::string> units;  // "micrometer" when dimensional
};
// NGFF transformations in application order, then conversion to micrometers.
[[nodiscard]] StorageGrid storageTransform(const std::vector<std::string>& axes, const std::vector<std::optional<std::string>>& units,
                                           const std::vector<nlohmann::json>& groups);
// A level's transformations, placing scale-only levels as block-reduced from the finest
// dataset, as zarr::Group::placement does.
[[nodiscard]] nlohmann::json blockReducedTransforms(const nlohmann::json& transforms, const nlohmann::json& finest);
// The same registration on a selected grid, given original-from-selected.
[[nodiscard]] SourceRegistration regrid(const SourceRegistration& registration, const VolumeTransform& originalFromSelected,
                                        const Selection& selection);
// The source's registration on a selected grid; see the storage metadata arguments.
[[nodiscard]] std::optional<SourceRegistration> resolveRegistration(
    const Source& source, const Selection& selection, std::optional<VolumeTransform> selected = std::nullopt,
    const std::optional<std::string>& units = std::nullopt, const std::optional<VolumeTransform>& declared = std::nullopt,
    const std::optional<std::string>& declaredUnits = std::nullopt, std::optional<VolumeTransform> relative = std::nullopt);
// Exact voxel resolution after the published volume-to-volume mapping.
[[nodiscard]] bool matchesReference(const Source& source, const std::optional<SourceRegistration>& registration, const Frame& reference,
                                    const std::optional<std::vector<std::string>>& axes = std::nullopt);
// A complete array-to-volume placement; std::invalid_argument when it is not one.
[[nodiscard]] VolumeTransform arrayPlacement(const std::optional<SourceRegistration>& registration);

struct ResolvedSource {
    std::string location;
    Selection selection;
    std::vector<std::string> axes;
    std::vector<std::int64_t> shape;
    std::string dtype;  // dtypeName
    std::optional<SourceRegistration> registration;
    std::optional<VolumeTransform> storageFromArray;
    std::optional<std::string> storageUnits;
};

// ---- caches ----

inline constexpr std::size_t kDefaultDecodeBytes = std::size_t(256) << 20;

// One LRU and decode budget shared by all readers referencing this object. Decode
// operations on an owner are serialized, including conversion scratch. Independent owners
// may run concurrently. A forked process starts empty, with fresh locks.
class DecodedCache {
public:
    explicit DecodedCache(std::size_t maxBytes = kDefaultDecodeBytes, std::size_t maxDecodeBytes = kDefaultDecodeBytes);
    [[nodiscard]] std::size_t maxBytes() const { return maxBytes_; }
    [[nodiscard]] std::size_t maxDecodeBytes() const { return maxDecodeBytes_; }
    struct Stats {
        std::size_t nbytes = 0, entries = 0, hits = 0, misses = 0, peakWorkingBytes = 0;
    };
    [[nodiscard]] Stats stats();
    void clear();
    // A reservation of working bytes, serializing this owner until released.
    class Lease {
    public:
        Lease() = default;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();
        void release();
    private:
        friend class DecodedCache;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Borrows an immutable decoded block while reserving its working set: decodeBytes
    // covers decoder output and estimated codec workspace, scratchBytes the caller's
    // conversion buffers. Caller-owned outputs and retained entries are separate.
    using Load = std::function<Block()>;
    [[nodiscard]] std::pair<Block, Lease> use(const std::string& key, std::size_t decodeBytes, std::size_t scratchBytes, const Load& load);
    // Reserves scratch for a read that retains nothing.
    [[nodiscard]] Lease working(std::size_t nbytes);
private:
    struct State;
    [[nodiscard]] std::shared_ptr<State> state();
    std::size_t maxBytes_, maxDecodeBytes_;
    std::mutex stateMutex_;
    std::shared_ptr<State> state_;
};

// What opens one location: its access, caches and reader settings.
using OptionsFor = std::function<zarr::OpenOptions(const std::string& location)>;
// Environment credentials and no caches, for callers with no other settings.
[[nodiscard]] OptionsFor defaultOptions();

// Seekable image files through the reader's transport and disk budget. A remote file is
// cached at its URL's decoded-cache path, so mirrors laid out by bucket path are found in
// place; cached copies are trusted like chunks. Descriptors pin open inodes across eviction.
class EncodedFiles {
public:
    explicit EncodedFiles(OptionsFor options);
    ~EncodedFiles();
    EncodedFiles(const EncodedFiles&) = delete;
    EncodedFiles& operator=(const EncodedFiles&) = delete;
    struct Opened {
        int fd = -1;
        std::string token;  // identifies these bytes
        Opened() = default;
        Opened(int f, std::string t) : fd(f), token(std::move(t)) {}
        Opened(Opened&& other) noexcept : fd(other.fd), token(std::move(other.token)) { other.fd = -1; }
        Opened& operator=(Opened&&) = delete;
        ~Opened();
    };
    [[nodiscard]] Opened open(const std::string& location);
    [[nodiscard]] std::size_t bytesRead() const { return bytesRead_; }
    [[nodiscard]] std::size_t cacheHits() const { return cacheHits_; }
    // Where files live when nothing can be kept; empty until one is needed.
    [[nodiscard]] std::string temporaryDirectory() const;
    void close();
private:
    void process();
    std::string identity(const std::string& location, int fd);
    OptionsFor options_;
    std::mutex mutex_;
    long pid_;
    std::string temporary_;
    std::map<std::string, std::pair<std::string, std::string>> identities_;
    std::map<std::string, zarr::OpenOptions> resolved_;
    std::size_t bytesRead_ = 0, cacheHits_ = 0;
};

// ---- images ----

struct FileInfo {
    std::vector<std::int64_t> shape;
    std::vector<std::string> axes;
    std::string dtype;
    std::vector<std::int64_t> pageShape;
    std::vector<std::string> pageAxes;
    std::int64_t series = 0, level = 0;
};

// Checks declared axes against what storage says (nullopt: it does not say).
[[nodiscard]] std::vector<std::string> resolveAxes(const std::optional<std::vector<std::string>>& declared,
                                                   const std::vector<std::optional<std::string>>& observed,
                                                   const std::vector<std::int64_t>& shape);

// An axis selection of a page: one index, or the range [low, high).
struct PageSlice {
    std::int64_t low = 0, high = 0;
    bool single = false;
};

// Immutable file metadata; decoders exist only inside page reads. Tile and strip decoding
// is selective, and uncompressed data is read in place without a decode.
class ImageFile {
public:
    ImageFile(std::string location, const std::optional<std::string>& format, const Selection& selection,
              const std::optional<std::vector<std::string>>& axes, std::shared_ptr<EncodedFiles> files, bool namePageAxis = false);
    ~ImageFile();
    [[nodiscard]] const FileInfo& info() const { return info_; }
    [[nodiscard]] const std::string& format() const { return format_; }
    [[nodiscard]] const std::optional<VolumeTransform>& storageFromArray() const { return storageFromArray_; }
    [[nodiscard]] const std::optional<std::string>& storageUnits() const { return storageUnits_; }
    [[nodiscard]] const std::vector<std::string>& channelNames() const { return channelNames_; }
    [[nodiscard]] const nlohmann::json& palette() const { return palette_; }  // {mode, values, transparency}; null when none
    [[nodiscard]] std::int64_t levelCount() const { return levelCount_; }
    [[nodiscard]] std::vector<std::int64_t> prefixShape() const;
    [[nodiscard]] std::size_t bytesRead() const { return bytesRead_; }
    // The requested part of one page, in page-axis order without its single axes.
    [[nodiscard]] Block region(std::int64_t page, const std::vector<PageSlice>& slices, DecodedCache& cache, std::size_t scratchBytes);
private:
    struct Tiff;
    void openTiff(int fd, const Selection& selection, const std::optional<std::vector<std::string>>& axes, bool namePageAxis);
    void openRaster(int fd, const std::optional<std::vector<std::string>>& axes);
    std::string location_, format_, token_;
    std::shared_ptr<EncodedFiles> files_;
    FileInfo info_;
    std::optional<VolumeTransform> storageFromArray_;
    std::optional<std::string> storageUnits_;
    std::vector<std::string> channelNames_;
    nlohmann::json palette_;
    std::int64_t levelCount_ = 1;
    std::size_t encodedSize_ = 0, bytesRead_ = 0;
    std::unique_ptr<Tiff> tiff_;
};

// ---- the reader ----

// Zarr, PNG/JPEG/TIFF and explicit-stack crops of one source. Output preserves storage
// channel position and order; time selection removes the time axis. Encoding consumes a
// color channel axis and reports the encoded shape, axes and dtype. A forked process
// reopens what it inherited instead of using it.
class SourceReader {
public:
    struct Settings {
        OptionsFor options;  // default: defaultOptions()
        std::shared_ptr<DecodedCache> decodedCache;  // shared; default: an own one with these budgets
        std::optional<std::size_t> decodedCacheBytes, maxDecodeBytes;
    };
    SourceReader(Source source, Settings settings);
    ~SourceReader();
    SourceReader(const SourceReader&) = delete;
    SourceReader& operator=(const SourceReader&) = delete;

    [[nodiscard]] const Source& source() const { return source_; }
    [[nodiscard]] const std::vector<std::string>& axes();
    [[nodiscard]] const std::vector<std::int64_t>& shape();
    [[nodiscard]] const std::string& dtype();  // typestr
    [[nodiscard]] const std::vector<std::string>& spatialAxes();
    // A number: integer or float, in dtype's range.
    [[nodiscard]] const nlohmann::json& fillValue();
    [[nodiscard]] const nlohmann::json& palette();
    [[nodiscard]] const std::optional<SourceRegistration>& registration();
    [[nodiscard]] const ResolvedSource& resolved();
    [[nodiscard]] const std::shared_ptr<DecodedCache>& decodedCache() const { return cache_; }
    [[nodiscard]] std::shared_ptr<SourceReader> companion(const std::string& name);
    [[nodiscard]] std::vector<std::string> companionNames();

    // One owned crop, padded with the fill value in the encoded output domain.
    [[nodiscard]] Block read(const std::vector<std::int64_t>& start, const std::vector<std::int64_t>& size);
    [[nodiscard]] std::vector<Block> readMany(const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>>& regions);

    // The one array this reads directly, when it is a Zarr array of exactly its spatial axes,
    // unencoded, with the array's own fill: samplers may go straight to it.
    [[nodiscard]] Array* directArray();
    [[nodiscard]] bool isZarr();
    [[nodiscard]] const std::vector<std::string>& rawAxes();
    // The opened Zarr array, when there is one.
    [[nodiscard]] std::shared_ptr<zarr::OpenedArray> zarrArray();
    // The pyramid levels its Zarr group declares; zero when it has none.
    [[nodiscard]] std::size_t zarrDatasets();

    struct Stats {
        DecodedCache::Stats decodedCache;
        std::size_t remoteBytes = 0, encodedCacheHits = 0, imagePayloadBytes = 0;
        std::size_t zarrCacheBytes = 0, zarrChunkBytes = 0;
        nlohmann::json zarr;  // the array's own counters; null for image files
    };
    [[nodiscard]] Stats stats();
    [[nodiscard]] std::string temporaryDirectory() const;
    [[nodiscard]] bool loaded(std::int64_t stackPlane) const;  // whether a stack plane has been opened
    void close();
    [[nodiscard]] bool closed() const { return closed_; }

private:
    struct Backend;
    struct Lookup;
    void ready();
    void open();
    void reset();
    void openCompanions();
    ImageFile& file(std::int64_t index);
    Selection correspondingImage();
    void readZarr(const std::vector<std::pair<std::int64_t, std::int64_t>>& bounds, const std::vector<std::int64_t>& origin, Block& output);

    Source source_;
    OptionsFor options_;
    std::shared_ptr<DecodedCache> cache_;
    bool ownsCache_;
    std::shared_ptr<EncodedFiles> encoded_;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<Lookup> lookup_;
    std::map<std::int64_t, std::unique_ptr<ImageFile>> files_;
    std::map<std::string, std::shared_ptr<SourceReader>> companions_;
    std::vector<std::string> companionOrder_;
    long pid_;
    bool closed_ = false, stack_ = false, zarr_ = false, opened_ = false;
    std::vector<std::string> locations_;
    std::optional<std::vector<std::string>> fileAxes_;
    Selection imageSelection_;
    std::vector<std::string> rawAxes_, selectedAxes_, axes_, spatialAxes_;
    std::vector<std::int64_t> rawShape_, shape_;
    std::string dtype_, rawDtype_;
    std::optional<std::int64_t> time_;
    std::optional<std::vector<std::int64_t>> channels_;
    nlohmann::json fill_, palette_;
    std::vector<std::uint8_t> fillBytes_;
    std::optional<SourceRegistration> registration_;
    ResolvedSource resolved_;
};

// A plain image file as a source: its first series and level, and a page stack's anonymous axis named z.
// TIFF and PNG/JPEG, by suffix. Opens the file once to learn its axes.
[[nodiscard]] Source imageSource(const std::string& location, const OptionsFor& options = {});
// The whole of such a file, retaining nothing, with no decode budget. `axes` skips learning them.
[[nodiscard]] Block readImage(const std::string& location, const OptionsFor& options = {}, const std::optional<std::vector<std::string>>& axes = std::nullopt);
// One page of such a file: of a page stack, that page; of anything else, the whole image.
[[nodiscard]] Block readImagePage(const std::string& location, std::int64_t page, const OptionsFor& options = {});
// Where a local single-page TIFF keeps its values when they lie uncompressed in one native-order run,
// so they can be memory-mapped. std::invalid_argument for a file that cannot be.
struct TiffMapping {
    std::uint64_t offset = 0;
    std::vector<std::int64_t> shape;
    std::string dtype;
};
[[nodiscard]] TiffMapping tiffMapping(const std::string& path);

} // namespace vcr::sources
