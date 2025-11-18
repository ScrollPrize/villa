#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceMeta.hpp"

class Segmentation
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void saveMetadata();

    // Surface management
    [[nodiscard]] bool isSurfaceLoaded() const;
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<SurfaceMeta> loadSurface();
    [[nodiscard]] std::shared_ptr<SurfaceMeta> getSurface() const;
    void unloadSurface();

    static bool checkDir(std::filesystem::path path);

private:
    std::filesystem::path path_;
    nlohmann::json metadata_;
    std::shared_ptr<SurfaceMeta> surface_;

    void loadMetadata();
};