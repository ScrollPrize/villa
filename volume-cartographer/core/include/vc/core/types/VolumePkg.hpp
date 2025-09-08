#pragma once

#include <cstddef>
#include <iostream>
#include <map>

#include <filesystem>
#include "vc/core/types/Metadata.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkgVersion.hpp"
#include "vc/core/util/Surface.hpp"

class VolumePkg
{
public:
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    static std::shared_ptr<VolumePkg> New(std::filesystem::path fileLocation);
    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] double materialThickness() const;
    [[nodiscard]] Metadata metadata() const;
    void saveMetadata();
    void saveMetadata(const std::filesystem::path& filePath);
    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    [[nodiscard]] std::vector<std::string> volumeNames() const;
    std::shared_ptr<Volume> newVolume(std::string name = "");
    [[nodiscard]] const std::shared_ptr<Volume> volume() const;
    std::shared_ptr<Volume> volume();
    [[nodiscard]] const std::shared_ptr<Volume> volume(const std::string& id) const;
    std::shared_ptr<Volume> volume(const std::string& id);
    [[nodiscard]] bool hasSegmentations() const;
    [[nodiscard]] std::size_t numberOfSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;
    [[nodiscard]] std::vector<std::string> segmentationNames() const;
    [[nodiscard]] const std::shared_ptr<Segmentation> segmentation(const std::string& id) const;

    std::vector<std::filesystem::path> segmentationFiles();

    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;

    void refreshSegmentations();

    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<SurfaceMeta> loadSurface(const std::string& id);
    std::shared_ptr<SurfaceMeta> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);


private:
    Metadata config_;
    std::filesystem::path rootDir_;
    std::map<std::string, std::shared_ptr<Volume>> volumes_;
    std::map<std::string, std::shared_ptr<Segmentation>> segmentations_;
    std::vector<std::filesystem::path> segmentation_files_;
    std::string currentSegmentationDir_ = "paths";
    std::map<std::string, std::string> segmentationDirectories_;

    std::map<std::string, std::shared_ptr<SurfaceMeta>> loadedSurfaces_;

    void loadSegmentationsFromDirectory(const std::string& dirName);
};

