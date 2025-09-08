#include "vc/core/types/DiskBasedObjectBaseClass.hpp"





static const std::filesystem::path METADATA_FILE = "meta.json";

// Load file from disk
DiskBasedObjectBaseClass::DiskBasedObjectBaseClass(
    std::filesystem::path path)
    : path_(std::move(path))
{
    metadata_ = Metadata(path_ / METADATA_FILE);
}

// Create new file on disk
DiskBasedObjectBaseClass::DiskBasedObjectBaseClass(
    std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_.setPath((path_ / METADATA_FILE));
    metadata_.set("uuid", uuid);
    metadata_.set("name", name);
}

//return wether dir could be a volume
bool DiskBasedObjectBaseClass::checkDir(std::filesystem::path path)
{
    return std::filesystem::is_directory(path) && std::filesystem::exists(path / METADATA_FILE);
}