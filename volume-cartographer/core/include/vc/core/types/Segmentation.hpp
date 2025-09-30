#pragma once

#include <filesystem>
#include <memory>
#include "vc/core/types/DiskBasedObjectBaseClass.hpp"
#include "vc/core/util/Surface.hpp"

class Segmentation : public DiskBasedObjectBaseClass
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    // Surface management
    [[nodiscard]] bool isSurfaceLoaded() const;
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<SurfaceMeta> loadSurface();
    [[nodiscard]] std::shared_ptr<SurfaceMeta> getSurface() const;
    void unloadSurface();

private:
    std::shared_ptr<SurfaceMeta> surface_;
};