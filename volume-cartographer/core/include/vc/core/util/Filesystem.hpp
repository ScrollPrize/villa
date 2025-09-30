#pragma once

#include <filesystem>

namespace fs = std::filesystem;

static inline fs::path create_temp_directory(const std::string& prefix = "tmp") {
    // Get system temp directory
    fs::path temp_dir = fs::temp_directory_path();

    // Generate a unique name
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);

    fs::path new_dir;
    do {
        new_dir = temp_dir / (prefix + "_" + std::to_string(dis(gen)));
    } while (fs::exists(new_dir));

    // Create the directory
    fs::create_directory(new_dir);
    return new_dir;
}