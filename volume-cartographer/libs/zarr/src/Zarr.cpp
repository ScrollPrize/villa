#include "vc/zarr/Zarr.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

#include <blosc.h>

namespace vc::zarr {

// ─────────────────────────────────────────────────────────────────────────────
//  Datatype helpers
// ─────────────────────────────────────────────────────────────────────────────

Datatype dtypeFromZarr(const std::string& s)
{
    if (s == "|u1") return Datatype::uint8;
    if (s == "<u2") return Datatype::uint16;
    if (s == "<f4") return Datatype::float32;
    throw std::runtime_error("vc::zarr: unsupported zarr dtype '" + s + "'");
}

static const std::string kDtypeZarrU1 = "|u1";
static const std::string kDtypeZarrU2 = "<u2";
static const std::string kDtypeZarrF4 = "<f4";

const std::string& dtypeToZarr(Datatype dt)
{
    switch (dt) {
        case Datatype::uint8:   return kDtypeZarrU1;
        case Datatype::uint16:  return kDtypeZarrU2;
        case Datatype::float32: return kDtypeZarrF4;
    }
    throw std::runtime_error("vc::zarr: unknown Datatype");
}

std::size_t dtypeSize(Datatype dt)
{
    switch (dt) {
        case Datatype::uint8:   return 1;
        case Datatype::uint16:  return 2;
        case Datatype::float32: return 4;
    }
    throw std::runtime_error("vc::zarr: unknown Datatype");
}

Datatype dtypeFromName(const std::string& name)
{
    if (name == "uint8")   return Datatype::uint8;
    if (name == "uint16")  return Datatype::uint16;
    if (name == "float32") return Datatype::float32;
    throw std::runtime_error("vc::zarr: unsupported dtype name '" + name + "'");
}

static const std::string kNameU8  = "uint8";
static const std::string kNameU16 = "uint16";
static const std::string kNameF32 = "float32";

const std::string& dtypeToName(Datatype dt)
{
    switch (dt) {
        case Datatype::uint8:   return kNameU8;
        case Datatype::uint16:  return kNameU16;
        case Datatype::float32: return kNameF32;
    }
    throw std::runtime_error("vc::zarr: unknown Datatype");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Store
// ─────────────────────────────────────────────────────────────────────────────

Store::Store(std::filesystem::path root) : path_(std::move(root)) {}

Store::Store(const Store& parent, const std::string& child)
    : path_(parent.path_ / child) {}

bool Store::exists() const { return std::filesystem::exists(path_); }

std::vector<std::string> Store::keys() const
{
    std::vector<std::string> out;
    if (!std::filesystem::exists(path_)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(path_)) {
        if (entry.is_directory())
            out.emplace_back(entry.path().filename().string());
    }
    return out;
}

nlohmann::json Store::readAttrs() const
{
    auto p = path_ / ".zattrs";
    if (!std::filesystem::exists(p)) return nlohmann::json{};
    std::ifstream f(p);
    nlohmann::json j;
    f >> j;
    return j;
}

void Store::writeAttrs(const nlohmann::json& j) const
{
    auto p = path_ / ".zattrs";
    nlohmann::json existing;
    if (std::filesystem::exists(p)) {
        std::ifstream f(p);
        f >> existing;
    }
    for (auto it = j.begin(); it != j.end(); ++it)
        existing[it.key()] = it.value();
    std::ofstream f(p);
    f << existing;
}

Store Store::create(const std::filesystem::path& path, bool overwrite)
{
    if (std::filesystem::exists(path)) {
        if (overwrite)
            std::filesystem::remove_all(path);
        else if (std::filesystem::exists(path / ".zgroup"))
            return Store(path);  // already a zarr store
    }
    std::filesystem::create_directories(path);
    // Write .zgroup marker
    std::ofstream f(path / ".zgroup");
    f << R"({"zarr_format":2})" << std::endl;
    return Store(path);
}

Store Store::createGroup(const std::string& name) const
{
    auto p = path_ / name;
    std::filesystem::create_directories(p);
    std::ofstream f(p / ".zgroup");
    f << R"({"zarr_format":2})" << std::endl;
    return Store(p);
}

// ─────────────────────────────────────────────────────────────────────────────
//  DatasetMeta
// ─────────────────────────────────────────────────────────────────────────────

void DatasetMeta::fromJson(const nlohmann::json& j)
{
    dtype = dtypeFromZarr(j.at("dtype").get<std::string>());
    shape = ShapeType(j.at("shape").begin(), j.at("shape").end());
    chunks = ShapeType(j.at("chunks").begin(), j.at("chunks").end());

    // fill_value
    const auto& fv = j.at("fill_value");
    if (fv.is_null())
        fillValue = 0;
    else if (fv.is_string()) {
        auto s = fv.get<std::string>();
        if (s == "NaN") fillValue = std::numeric_limits<double>::quiet_NaN();
        else if (s == "Infinity") fillValue = std::numeric_limits<double>::infinity();
        else if (s == "-Infinity") fillValue = -std::numeric_limits<double>::infinity();
        else fillValue = 0;
    } else {
        fillValue = fv.get<double>();
    }

    // dimension_separator
    if (auto it = j.find("dimension_separator"); it != j.end())
        dimSeparator = it->get<std::string>();
    else
        dimSeparator = ".";

    // compressor
    const auto& comp = j.at("compressor");
    if (comp.is_null()) {
        bloscCodec = "";
        bloscLevel = 0;
        bloscShuffle = 0;
    } else {
        auto id = comp.value("id", std::string{});
        if (id == "blosc") {
            bloscCodec = comp.value("cname", std::string("zstd"));
            bloscLevel = comp.value("clevel", 1);
            bloscShuffle = comp.value("shuffle", 0);
        } else {
            throw std::runtime_error("vc::zarr: unsupported compressor '" + id + "'");
        }
    }
}

nlohmann::json DatasetMeta::toJson() const
{
    nlohmann::json j;
    j["zarr_format"] = 2;
    j["dtype"] = dtypeToZarr(dtype);
    j["shape"] = shape;
    j["chunks"] = chunks;
    j["fill_value"] = fillValue;
    j["order"] = "C";
    j["filters"] = nullptr;
    j["dimension_separator"] = dimSeparator;

    if (bloscCodec.empty()) {
        j["compressor"] = nullptr;
    } else {
        j["compressor"] = {
            {"id", "blosc"},
            {"cname", bloscCodec},
            {"clevel", bloscLevel},
            {"shuffle", bloscShuffle},
            {"blocksize", 0}
        };
    }
    return j;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dataset
// ─────────────────────────────────────────────────────────────────────────────

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

void Dataset::initChunkSize()
{
    chunkSize_ = std::accumulate(meta_.chunks.begin(), meta_.chunks.end(),
                                 std::size_t{1}, std::multiplies<>());
    chunksPerDim_.resize(meta_.shape.size());
    for (std::size_t d = 0; d < meta_.shape.size(); ++d) {
        chunksPerDim_[d] = (meta_.shape[d] + meta_.chunks[d] - 1) / meta_.chunks[d];
    }
}

std::unique_ptr<Dataset> Dataset::openImpl(const std::filesystem::path& dsPath,
                                           const std::string& dimSeparator)
{
    auto zarrayPath = dsPath / ".zarray";
    if (!std::filesystem::exists(zarrayPath))
        throw std::runtime_error("vc::zarr: no .zarray at " + dsPath.string());

    nlohmann::json j;
    {
        std::ifstream f(zarrayPath);
        f >> j;
    }

    auto ds = std::unique_ptr<Dataset>(new Dataset());
    ds->path_ = dsPath;
    ds->meta_.fromJson(j);

    // Override dimSeparator if explicitly provided
    if (!dimSeparator.empty())
        ds->meta_.dimSeparator = dimSeparator;

    ds->initChunkSize();
    return ds;
}

std::unique_ptr<Dataset> Dataset::open(const Store& parent, const std::string& name)
{
    return openImpl(parent.path() / name, "");
}

std::unique_ptr<Dataset> Dataset::open(const std::filesystem::path& dsPath)
{
    return openImpl(dsPath, "");
}

std::unique_ptr<Dataset> Dataset::open(const Store& parent, const std::string& name,
                                       const std::string& dimSeparator)
{
    return openImpl(parent.path() / name, dimSeparator);
}

std::unique_ptr<Dataset> Dataset::create(
    const Store& parent, const std::string& name,
    const std::string& dtype, const ShapeType& shape, const ShapeType& chunks,
    const std::string& compressor, const nlohmann::json& compOpts,
    double fillValue, const std::string& dimSeparator)
{
    auto dsPath = parent.path() / name;
    std::filesystem::create_directories(dsPath);

    DatasetMeta meta;
    meta.dtype = dtypeFromName(dtype);
    meta.shape = shape;
    meta.chunks = chunks;
    meta.fillValue = fillValue;
    meta.dimSeparator = dimSeparator;

    if (compressor == "blosc") {
        meta.bloscCodec = compOpts.value("cname", std::string("zstd"));
        meta.bloscLevel = compOpts.value("clevel", 1);
        meta.bloscShuffle = compOpts.value("shuffle", 0);
    } else if (compressor == "raw" || compressor.empty()) {
        meta.bloscCodec = "";
    } else {
        throw std::runtime_error("vc::zarr: unsupported compressor '" + compressor + "'");
    }

    // Write .zarray
    {
        std::ofstream f(dsPath / ".zarray");
        f << std::setw(4) << meta.toJson() << std::endl;
    }

    auto ds = std::unique_ptr<Dataset>(new Dataset());
    ds->path_ = dsPath;
    ds->meta_ = meta;
    ds->initChunkSize();
    return ds;
}

// --- Chunk path construction ---

void Dataset::chunkPath(const ShapeType& id, std::filesystem::path& out) const
{
    std::string key;
    for (std::size_t d = 0; d < id.size(); ++d) {
        if (d > 0) key += meta_.dimSeparator;
        key += std::to_string(id[d]);
    }
    out = path_ / key;
}

bool Dataset::chunkExists(const ShapeType& id) const
{
    std::filesystem::path p;
    chunkPath(id, p);
    return std::filesystem::exists(p);
}

void Dataset::readRawChunk(const ShapeType& id, std::vector<char>& out) const
{
    std::filesystem::path p;
    chunkPath(id, p);
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("vc::zarr: cannot open chunk " + p.string());
    f.seekg(0, std::ios::end);
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    f.read(out.data(), size);
}

void Dataset::decompress(const std::vector<char>& compressed, void* out,
                         std::size_t numElements) const
{
    if (meta_.bloscCodec.empty()) {
        // Raw / no compression
        std::size_t bytes = numElements * dtypeSize(meta_.dtype);
        if (compressed.size() < bytes)
            throw std::runtime_error("vc::zarr: raw chunk too small");
        std::memcpy(out, compressed.data(), bytes);
        return;
    }

    int ret = blosc_decompress_ctx(
        compressed.data(), out,
        numElements * dtypeSize(meta_.dtype),
        /*numinternalthreads=*/1
    );
    if (ret <= 0)
        throw std::runtime_error("vc::zarr: blosc decompression failed");
}

void Dataset::compress(const void* data, std::vector<char>& out,
                       std::size_t numElements) const
{
    std::size_t typeSize = dtypeSize(meta_.dtype);
    std::size_t nbytes = numElements * typeSize;

    if (meta_.bloscCodec.empty()) {
        out.resize(nbytes);
        std::memcpy(out.data(), data, nbytes);
        return;
    }

    out.resize(nbytes + BLOSC_MAX_OVERHEAD);
    int compressed = blosc_compress_ctx(
        meta_.bloscLevel, meta_.bloscShuffle,
        typeSize,
        nbytes, data,
        out.data(), out.size(),
        meta_.bloscCodec.c_str(),
        /*blocksize=*/0,
        /*numinternalthreads=*/1
    );
    if (compressed <= 0)
        throw std::runtime_error("vc::zarr: blosc compression failed");
    out.resize(static_cast<std::size_t>(compressed));
}

void Dataset::readChunk(const ShapeType& id, void* buf) const
{
    std::vector<char> raw;
    readRawChunk(id, raw);
    decompress(raw, buf, chunkSize_);
}

void Dataset::writeChunk(const ShapeType& id, const void* data) const
{
    std::vector<char> compressed;
    compress(data, compressed, chunkSize_);

    std::filesystem::path p;
    chunkPath(id, p);

    // Ensure parent directories for "/" separator
    if (meta_.dimSeparator == "/") {
        auto parent = p.parent_path();
        if (!std::filesystem::exists(parent)) {
            try { std::filesystem::create_directories(parent); }
            catch (const std::filesystem::filesystem_error&) {}
        }
    }

    std::ofstream f(p, std::ios::binary);
    f.write(compressed.data(), static_cast<std::streamsize>(compressed.size()));
}

void Dataset::getChunkShape(const ShapeType& id, ShapeType& out) const
{
    out.resize(meta_.shape.size());
    for (std::size_t d = 0; d < meta_.shape.size(); ++d) {
        out[d] = std::min(meta_.chunks[d],
                          meta_.shape[d] - id[d] * meta_.chunks[d]);
    }
}

void Dataset::checkRequestShape(const ShapeType& offset, const ShapeType& shape) const
{
    if (offset.size() != meta_.shape.size() || shape.size() != meta_.shape.size())
        throw std::runtime_error("vc::zarr: request has wrong dimension");
    for (std::size_t d = 0; d < meta_.shape.size(); ++d) {
        if (offset[d] + shape[d] > meta_.shape[d])
            throw std::runtime_error("vc::zarr: request out of range");
        if (shape[d] == 0)
            throw std::runtime_error("vc::zarr: request shape has zero entry");
    }
}

void Dataset::getBlocksOverlappingRoi(const ShapeType& roiBegin, const ShapeType& roiShape,
                                      std::vector<ShapeType>& blockList) const
{
    const std::size_t ndim = roiBegin.size();
    ShapeType minIds(ndim), maxIds(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        minIds[d] = roiBegin[d] / meta_.chunks[d];
        std::size_t endCoord = roiBegin[d] + roiShape[d];
        std::size_t endId = endCoord / meta_.chunks[d];
        maxIds[d] = (endCoord % meta_.chunks[d] == 0) ? endId - 1 : endId;
    }
    detail::makeRegularGrid(minIds, maxIds, blockList);
}

bool Dataset::getCoordinatesInRoi(const ShapeType& chunkCoord,
                                  const ShapeType& roiBegin, const ShapeType& roiShape,
                                  ShapeType& offsetInRoi, ShapeType& shapeInRoi,
                                  ShapeType& offsetInChunk) const
{
    const std::size_t ndim = roiBegin.size();
    offsetInRoi.resize(ndim);
    shapeInRoi.resize(ndim);
    offsetInChunk.resize(ndim);

    // Compute chunk begin and actual shape
    ShapeType chunkBegin(ndim), chunkShape(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        chunkBegin[d] = chunkCoord[d] * meta_.chunks[d];
        chunkShape[d] = std::min(meta_.chunks[d],
                                 meta_.shape[d] - chunkBegin[d]);
    }

    bool completeOvlp = true;
    for (std::size_t d = 0; d < ndim; ++d) {
        std::size_t chunkEnd = chunkBegin[d] + chunkShape[d];
        std::size_t roiEnd = roiBegin[d] + roiShape[d];
        int offDiff = static_cast<int>(chunkBegin[d]) - static_cast<int>(roiBegin[d]);
        int endDiff = static_cast<int>(roiEnd) - static_cast<int>(chunkEnd);

        if (offDiff < 0) {
            offsetInRoi[d] = 0;
            offsetInChunk[d] = static_cast<std::size_t>(-offDiff);
            completeOvlp = false;
            shapeInRoi[d] = (chunkEnd <= roiEnd)
                ? chunkEnd - roiBegin[d]
                : roiEnd - roiBegin[d];
        } else if (endDiff < 0) {
            offsetInRoi[d] = chunkBegin[d] - roiBegin[d];
            offsetInChunk[d] = 0;
            completeOvlp = false;
            shapeInRoi[d] = roiEnd - chunkBegin[d];
        } else {
            offsetInRoi[d] = chunkBegin[d] - roiBegin[d];
            offsetInChunk[d] = 0;
            shapeInRoi[d] = chunkShape[d];
        }
    }
    return completeOvlp;
}

} // namespace vc::zarr
