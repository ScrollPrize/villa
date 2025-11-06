#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Logging.hpp"

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
    metadata_.set("volume", std::string{});
    metadata_.save();
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path)
{
    return std::make_shared<Segmentation>(path);
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path, const std::string& uuid, const std::string& name)
{
    return std::make_shared<Segmentation>(path, uuid, name);
}

bool Segmentation::isSurfaceLoaded() const
{
    return surface_ != nullptr;
}

bool Segmentation::canLoadSurface() const
{
    return metadata_.hasKey("format") &&
           metadata_.get<std::string>("format") == "tifxyz";
}

std::shared_ptr<SurfaceMeta> Segmentation::loadSurface()
{
    if (surface_) {
        return surface_;
    }

    if (!canLoadSurface()) {
        return nullptr;
    }

    try {
        surface_ = std::make_shared<SurfaceMeta>(path_);
        surface_->surface();
        surface_->readOverlapping();
        return surface_;
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface for {}: {}", id(), e.what());
        surface_ = nullptr;
        return nullptr;
    }
}

std::shared_ptr<SurfaceMeta> Segmentation::getSurface() const
{
    return surface_;
}

void Segmentation::unloadSurface()
{
    surface_ = nullptr;
}