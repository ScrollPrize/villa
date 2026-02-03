#include "vc/core/types/Volume.hpp"

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
    } else {
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
            std::filesystem::exists(path / METADATA_FILE_ALT));
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_["format"].get<std::string>() != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);

    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::sort(groups.begin(), groups.end());

    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        // Read metadata first to discover the dimension separator
        z5::filesystem::handle::Dataset tmp_handle(path_ / name, z5::FileMode::FileMode::r);
        z5::DatasetMetadata dsMeta;
        z5::filesystem::readMetadata(tmp_handle, dsMeta);

        // Re-create handle with correct delimiter so chunk keys resolve properly
        z5::filesystem::handle::Dataset ds_handle(group, name, dsMeta.zarrDelimiter);

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);

        // Verify level 0 shape matches meta.json dimensions
        // zarr shape is [z, y, x] = [slices, height, width]
        if (zarrDs_.size() == 1 && !skipShapeCheck) {
            const auto& shape = zarrDs_[0]->shape();
            if (static_cast<int>(shape[0]) != _slices ||
                static_cast<int>(shape[1]) != _height ||
                static_cast<int>(shape[2]) != _width) {
                throw std::runtime_error(
                    "zarr level 0 shape [z,y,x]=(" + std::to_string(shape[0]) + ", " +
                    std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
                    ") does not match meta.json dimensions (slices=" + std::to_string(_slices) +
                    ", height=" + std::to_string(_height) + ", width=" + std::to_string(_width) +
                    ") in " + path_.string());
            }
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
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}
