#include "vc/core/types/Segmentation.hpp"



Segmentation::Segmentation(std::filesystem::path path)
    : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "seg") {
        throw std::runtime_error("File not of type: seg");
    }
}

Segmentation::Segmentation(std::filesystem::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name))
{
    metadata_.set("type", "seg");
    metadata_.set("vcps", std::string{});
    metadata_.set("vcano", std::string{});
    metadata_.set("volume", std::string{});
    metadata_.save();
}

std::shared_ptr<Segmentation> Segmentation::New(std::filesystem::path path)
{
    return std::make_shared<Segmentation>(path);
}

std::shared_ptr<Segmentation> Segmentation::New(std::filesystem::path path, std::string uuid, std::string name)
{
    return std::make_shared<Segmentation>(path, uuid, name);
}
