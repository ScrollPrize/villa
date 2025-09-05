#include "vc/core/types/Volume.hpp"

#include <opencv2/imgcodecs.hpp>

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"


Volume::Volume(std::filesystem::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    _width = metadata_.get<int>("width");
    _height = metadata_.get<int>("height");
    _slices = metadata_.get<int>("slices");

    std::vector<std::mutex> init_mutexes(_slices);


    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name))
{
    metadata_.set("type", "vol");
    metadata_.set("width", _width);
    metadata_.set("height", _height);
    metadata_.set("slices", _slices);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});    

    zarrOpen();
}

void Volume::zarrOpen()
{
    if (!metadata_.hasKey("format") || metadata_.get<std::string>("format") != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);
    
    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::sort(groups.begin(), groups.end());
    
    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        z5::filesystem::handle::Dataset ds_handle(group, name, nlohmann::json::parse(std::ifstream(path_/name/".zarray")).value<std::string>("dimension_separator","."));

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);
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
double Volume::voxelSize() const
{
    return metadata_.get<double>("voxelsize");
}

z5::Dataset *Volume::zarrDataset(int level) const {
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}
