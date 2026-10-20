#pragma once

#include "vc_object_store/Uuid.hpp"
#include <filesystem>
#include <stdexcept>
#include <string>

namespace vc::test {
struct TempDir {
    std::filesystem::path dir;

    explicit TempDir(const std::string& name)
        : dir(std::filesystem::temp_directory_path() /
              ("vc_object_test_" + name + "_" + objects::util::newUuid()))
    {
        if (!std::filesystem::create_directory(dir))
            throw std::runtime_error("test directory already exists: " + dir.string());
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    ~TempDir()
    {
        std::error_code error;
        std::filesystem::remove_all(dir, error);
    }
};
} // namespace vc::test
