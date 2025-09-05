#pragma once

#include <filesystem>
#include "vc/core/types/DiskBasedObjectBaseClass.hpp"

class Segmentation : public DiskBasedObjectBaseClass
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);
    static std::shared_ptr<Segmentation> New(std::filesystem::path path);
    static std::shared_ptr<Segmentation> New(std::filesystem::path path, std::string uuid, std::string name);
};

