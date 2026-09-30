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

    // VC3D's key for this segment: the directory name (Open Data catalog
    // segments: their catalog representation id).
    [[nodiscard]] std::string id() const;
    // meta.json "uuid": the persistent object identity, independent of the
    // directory name. Unchanged by rename; a copy needs a new one.
    [[nodiscard]] std::string uuid() const;
    void setUuid(const std::string& newUuid);
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    [[nodiscard]] const utils::Json& metadata() const noexcept { return metadata_; }
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
