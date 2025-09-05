#pragma once

#include <filesystem>
#include <z5/dataset.hxx>
#include <z5/filesystem/handle.hxx>

#include "vc/core/types/DiskBasedObjectBaseClass.hpp"

#include "z5/types/types.hxx"


class Volume : public DiskBasedObjectBaseClass
{
public:
    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume() = default;


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, std::string uuid, std::string name);

    [[nodiscard]] int sliceWidth() const;
    [[nodiscard]] int sliceHeight() const;
    [[nodiscard]] int numSlices() const;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] z5::Dataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const;
    
protected:
    int _width{0};
    int _height{0};
    int _slices{0};

    std::unique_ptr<z5::filesystem::handle::File> zarrFile_;
    std::vector<std::unique_ptr<z5::Dataset>> zarrDs_;
    nlohmann::json zarrGroup_;
    void zarrOpen();
};

