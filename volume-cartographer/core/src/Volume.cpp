#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/LoadJson.hpp"

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/filesystem/metadata.hxx"
#include "z5/multiarray/xtensor_access.hxx"

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

Volume::Volume(std::filesystem::path path) : path_(std::move(path))
{
    loadMetadata();

    _width = metadata_["width"].get<int>();
    _height = metadata_["height"].get<int>();
    _slices = metadata_["slices"].get<int>();

    std::vector<std::mutex> init_mutexes(_slices);


    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_["uuid"] = uuid;
    metadata_["name"] = name;
    metadata_["type"] = "vol";
    metadata_["width"] = _width;
    metadata_["height"] = _height;
    metadata_["slices"] = _slices;
    metadata_["voxelsize"] = double{};
    metadata_["min"] = double{};
    metadata_["max"] = double{};

    zarrOpen();
}

Volume::~Volume() = default;

void Volume::loadMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    if (std::filesystem::exists(metaPath)) {
        metadata_ = vc::json::load_json_file(metaPath);
    } else if (std::filesystem::exists(path_ / METADATA_FILE_ALT)) {
        auto altPath = path_ / METADATA_FILE_ALT;
        auto full = vc::json::load_json_file(altPath);
        if (!full.contains("scan")) {
            throw std::runtime_error(
                "metadata.json missing 'scan' key: " + altPath.string());
        }
        metadata_ = full["scan"];
        if (!metadata_.contains("format")) {
            metadata_["format"] = "zarr";
        }
        metaPath = altPath;
    } else {
        const auto baseName = path_.filename().string();
        metadata_["uuid"] = baseName;
        metadata_["name"] = baseName;
        metadata_["type"] = "vol";
        metadata_["format"] = "zarr";
        metadata_["width"] = 0;
        metadata_["height"] = 0;
        metadata_["slices"] = 0;
        metadata_["voxelsize"] = double{};
        metadata_["min"] = double{};
        metadata_["max"] = double{};
        metadata_["_generated_from_zarr"] = true;
        return;
    }
    vc::json::require_type(metadata_, "type", "vol", metaPath.string());
    vc::json::require_fields(metadata_, {"uuid", "width", "height", "slices"}, metaPath.string());
}

std::string Volume::id() const
{
    return metadata_["uuid"].get<std::string>();
}

std::string Volume::name() const
{
    return metadata_["name"].get<std::string>();
}

void Volume::setName(const std::string& n)
{
    metadata_["name"] = n;
}

void Volume::saveMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

bool Volume::checkDir(std::filesystem::path path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_FILE) ||
            std::filesystem::exists(path / METADATA_FILE_ALT) ||
            std::filesystem::exists(path / ".zgroup") ||
            std::filesystem::exists(path / ".zattrs"));
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_["format"].get<std::string>() != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);

    auto isPowerOfTwoScale = [](double v) {
        if (!(v > 0.0)) return false;
        const double lv = std::log2(v);
        const double rounded = std::round(lv);
        return std::abs(lv - rounded) < 1e-6;
    };

    auto scaleToLevel = [](double v) -> int {
        return static_cast<int>(std::llround(std::log2(v)));
    };

    struct Candidate {
        int level;
        std::string path;
    };
    std::vector<Candidate> candidates;

    bool usedOmeMultiscales = false;
    if (zarrGroup_.contains("multiscales") && zarrGroup_["multiscales"].is_array() &&
        !zarrGroup_["multiscales"].empty()) {
        const auto& ms0 = zarrGroup_["multiscales"][0];
        if (ms0.contains("datasets") && ms0["datasets"].is_array()) {
            usedOmeMultiscales = true;
            for (const auto& ds : ms0["datasets"]) {
                if (!ds.contains("path") || !ds["path"].is_string()) {
                    throw std::runtime_error("OME-Zarr dataset entry missing string 'path' in " + path_.string());
                }
                const std::string dsPath = ds["path"].get<std::string>();

                if (!ds.contains("coordinateTransformations") || !ds["coordinateTransformations"].is_array()) {
                    throw std::runtime_error("OME-Zarr dataset '" + dsPath + "' missing coordinateTransformations in " + path_.string());
                }

                std::optional<int> level;
                for (const auto& tr : ds["coordinateTransformations"]) {
                    if (!tr.is_object() || !tr.contains("type") || !tr["type"].is_string())
                        continue;
                    if (tr["type"].get<std::string>() != "scale")
                        continue;
                    if (!tr.contains("scale") || !tr["scale"].is_array() || tr["scale"].size() < 3) {
                        throw std::runtime_error("OME-Zarr dataset '" + dsPath + "' has invalid scale transformation in " + path_.string());
                    }

                    const double sz = tr["scale"][0].get<double>();
                    const double sy = tr["scale"][1].get<double>();
                    const double sx = tr["scale"][2].get<double>();
                    if (std::abs(sz - sy) > 1e-6 || std::abs(sz - sx) > 1e-6 || !isPowerOfTwoScale(sz)) {
                        throw std::runtime_error(
                            "unsupported OME-Zarr scale for dataset '" + dsPath +
                            "' (expected isotropic power-of-two, got [" +
                            std::to_string(sz) + "," + std::to_string(sy) + "," + std::to_string(sx) +
                            "]) in " + path_.string());
                    }
                    level = scaleToLevel(sz);
                    break;
                }

                if (!level.has_value()) {
                    throw std::runtime_error("OME-Zarr dataset '" + dsPath + "' has no scale transformation in " + path_.string());
                }
                candidates.push_back({*level, dsPath});
            }
        }
    }

    if (!usedOmeMultiscales) {
        std::vector<std::string> groups;
        zarrFile_->keys(groups);
        std::sort(groups.begin(), groups.end());
        for (const auto& name : groups) {
            try {
                const int level = std::stoi(name);
                if (level < 0) {
                    continue;
                }
                candidates.push_back({level, name});
            } catch (...) {
                // Ignore non-numeric groups in legacy fallback mode.
            }
        }
    }

    if (candidates.empty()) {
        throw std::runtime_error("no zarr datasets found in " + path_.string());
    }

    int maxLevel = -1;
    for (const auto& c : candidates) {
        maxLevel = std::max(maxLevel, c.level);
    }
    zarrDs_.clear();
    zarrDs_.resize(static_cast<size_t>(maxLevel + 1));

    for (const auto& c : candidates) {
        if (c.level < 0) {
            continue;
        }

        // Allow missing scales: ignore datasets declared in metadata but not physically present.
        if (!std::filesystem::exists(path_ / c.path / ".zarray")) {
            continue;
        }

        // Read metadata first to discover the dimension separator
        z5::filesystem::handle::Dataset tmp_handle(path_ / c.path, z5::FileMode::FileMode::r);
        z5::DatasetMetadata dsMeta;
        z5::filesystem::readMetadata(tmp_handle, dsMeta);

        // Re-create handle with correct delimiter so chunk keys resolve properly
        z5::filesystem::handle::Dataset ds_handle(group, c.path, dsMeta.zarrDelimiter);

        auto ds = z5::filesystem::openDataset(ds_handle);
        if (ds->getDtype() != z5::types::Datatype::uint8 && ds->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +c.path);

        zarrDs_[static_cast<size_t>(c.level)] = std::move(ds);
    }

    const bool generatedFromZarr = metadata_.value("_generated_from_zarr", false);

    if (generatedFromZarr) {
        auto ceilDivPow2 = [](int v, int level) -> int {
            const int64_t denom = int64_t{1} << level;
            return static_cast<int>((static_cast<int64_t>(v) + denom - 1) / denom);
        };

        bool hasReference = false;
        int baseSlices = 0;
        int baseHeight = 0;
        int baseWidth = 0;

        for (size_t level = 0; level < zarrDs_.size(); ++level) {
            const auto& ds = zarrDs_[level];
            if (!ds) {
                continue;
            }

            const auto& shape = ds->shape();
            const int levelInt = static_cast<int>(level);

            if (!hasReference) {
                const int64_t scale = int64_t{1} << levelInt;
                baseSlices = static_cast<int>(shape[0] * scale);
                baseHeight = static_cast<int>(shape[1] * scale);
                baseWidth = static_cast<int>(shape[2] * scale);
                hasReference = true;
            }

            const int expectedSlices = ceilDivPow2(baseSlices, levelInt);
            const int expectedHeight = ceilDivPow2(baseHeight, levelInt);
            const int expectedWidth = ceilDivPow2(baseWidth, levelInt);

            if (static_cast<int>(shape[0]) != expectedSlices ||
                static_cast<int>(shape[1]) != expectedHeight ||
                static_cast<int>(shape[2]) != expectedWidth) {
                throw std::runtime_error(
                    "zarr level " + std::to_string(levelInt) + " shape [z,y,x]=("
                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                    + ") does not match synthesized dimensions from first found scale (slices=" + std::to_string(baseSlices)
                    + ", height=" + std::to_string(baseHeight) + ", width=" + std::to_string(baseWidth)
                    + ") in " + path_.string());
            }
        }

        if (!hasReference) {
            throw std::runtime_error("no physical zarr dataset directories found in " + path_.string());
        }

        _slices = baseSlices;
        _height = baseHeight;
        _width = baseWidth;
        metadata_["slices"] = _slices;
        metadata_["height"] = _height;
        metadata_["width"] = _width;
    }

    // Verify each existing level shape against meta.json dimensions and level downscale.
    // zarr shape is [z, y, x] = [slices, height, width]
    if (!skipShapeCheck) {
        auto ceilDivPow2 = [](int v, int level) -> int {
            const int64_t denom = int64_t{1} << level;
            return static_cast<int>((static_cast<int64_t>(v) + denom - 1) / denom);
        };

        bool hasAnyPhysicalScale = false;
        for (size_t level = 0; level < zarrDs_.size(); ++level) {
            const auto& ds = zarrDs_[level];
            if (!ds) {
                continue;
            }
            hasAnyPhysicalScale = true;

            const auto& shape = ds->shape();
            const int expectedSlices = ceilDivPow2(_slices, static_cast<int>(level));
            const int expectedHeight = ceilDivPow2(_height, static_cast<int>(level));
            const int expectedWidth = ceilDivPow2(_width, static_cast<int>(level));

            if (static_cast<int>(shape[0]) != expectedSlices ||
                static_cast<int>(shape[1]) != expectedHeight ||
                static_cast<int>(shape[2]) != expectedWidth) {
                throw std::runtime_error(
                    "zarr level " + std::to_string(level) + " shape [z,y,x]=("
                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                    + ") does not match expected downscaled meta.json dimensions (slices=" + std::to_string(expectedSlices)
                    + ", height=" + std::to_string(expectedHeight) + ", width=" + std::to_string(expectedWidth)
                    + ") in " + path_.string());
            }
        }

        if (!hasAnyPhysicalScale) {
            throw std::runtime_error("no physical zarr dataset directories found in " + path_.string());
        }
    }
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path)
{
    return std::make_shared<Volume>(path);
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path, std::string uuid, std::string name)
{
    return std::make_shared<Volume>(path, uuid, name);
}

int Volume::sliceWidth() const { return _width; }
int Volume::sliceHeight() const { return _height; }
int Volume::numSlices() const { return _slices; }
std::array<int, 3> Volume::shape() const { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    return metadata_["voxelsize"].get<double>();
}

z5::Dataset *Volume::zarrDataset(int level) const {
    if (level < 0 || zarrDs_.empty())
        return nullptr;

    if (static_cast<size_t>(level) >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}
