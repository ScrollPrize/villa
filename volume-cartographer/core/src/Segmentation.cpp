#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"

static const std::filesystem::path METADATA_FILE = "meta.json";

Segmentation::Segmentation(std::filesystem::path path)
    : path_(std::move(path))
{
    loadMetadata();
}

Segmentation::Segmentation(std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_["uuid"] = uuid;
    metadata_["name"] = name;
    metadata_["type"] = "seg";
    metadata_["volume"] = std::string{};
    saveMetadata();
}

void Segmentation::loadMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    metadata_ = vc::json::load_json_file(metaPath);
    vc::json::require_type(metadata_, "type", "seg", metaPath.string());
    vc::json::require_fields(metadata_, {"uuid"}, metaPath.string());
}

std::string Segmentation::id() const
{
    return metadata_["uuid"].get<std::string>();
}

std::string Segmentation::name() const
{
    return metadata_["name"].get<std::string>();
}

void Segmentation::setName(const std::string& n)
{
    metadata_["name"] = n;
}

void Segmentation::saveMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

bool Segmentation::checkDir(std::filesystem::path path)
{
    return std::filesystem::is_directory(path) && std::filesystem::exists(path / METADATA_FILE);
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
    return metadata_.contains("format") &&
           metadata_["format"].get<std::string>() == "tifxyz";
}

std::shared_ptr<QuadSurface> Segmentation::loadSurface()
{
    if (surface_) {
        return surface_;
    }

    if (!canLoadSurface()) {
        return nullptr;
    }

    try {
        // Load the surface directly (no SurfaceMeta wrapper)
        surface_ = load_quad_from_tifxyz(path_.string());

        // Load overlapping info and cache mask timestamp
        surface_->readOverlappingJson();
        surface_->refreshMaskTimestamp();

        return surface_;
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface for {}: {}", id(), e.what());
        surface_ = nullptr;
        return nullptr;
    }
}

std::shared_ptr<QuadSurface> Segmentation::getSurface() const
{
    return surface_;
}

void Segmentation::unloadSurface()
{
    surface_ = nullptr;
}