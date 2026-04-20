#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include "utils/Json.hpp"
#include "vc/core/util/QuadSurface.hpp"

class Segmentation
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    ~Segmentation();
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const;
    void setId(const std::string& newId);
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void saveMetadata();
    void ensureScrollSource(const std::string& scrollName, const std::string& volumeUuid);

    // Surface management - returns QuadSurface directly (no SurfaceMeta wrapper)
    [[nodiscard]] bool isSurfaceLoaded() const;
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<QuadSurface> loadSurface();
    [[nodiscard]] std::shared_ptr<QuadSurface> getSurface() const;
    void unloadSurface();

    static bool checkDir(std::filesystem::path path);

private:
    std::filesystem::path path_;
    utils::Json metadata_;
    std::shared_ptr<QuadSurface> surface_;

    void loadMetadata();
};