#pragma once

#include <filesystem>
#include <string>

namespace vc {

struct VolpkgConvertResult {
    bool ok = false;
    std::string message;
    std::filesystem::path output;
};

VolpkgConvertResult convertVolpkg(const std::string& input,
                                  const std::filesystem::path& outFile);

}
