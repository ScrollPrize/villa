#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"

class VolumePkg
{
public:
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    ~VolumePkg();
    static std::shared_ptr<VolumePkg> New(const std::filesystem::path& fileLocation);

    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    std::shared_ptr<Volume> volume();
    std::shared_ptr<Volume> volume(const std::string& id);
    [[nodiscard]] bool hasSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;

    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;
    [[nodiscard]] std::string getVolpkgDirectory() const;

    void refreshSegmentations();


    // Surface management
    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<SurfaceMeta> loadSurface(const std::string& id);
    std::shared_ptr<SurfaceMeta> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);
    bool addSingleSegmentation(const std::string& id);
    bool removeSingleSegmentation(const std::string& id);
    bool reloadSingleSegmentation(const std::string& id);

private:
    nlohmann::json config_;
    std::filesystem::path rootDir_;
    std::map<std::string, std::shared_ptr<Volume>> volumes_;
    std::map<std::string, std::shared_ptr<Segmentation>> segmentations_;
    std::string currentSegmentationDir_ = "paths";
    std::map<std::string, std::string> segmentationDirectories_;

    void loadSegmentationsFromDirectory(const std::string& dirName);
};