#pragma once

#include "vc/lasagna/Manifest.hpp"

#include <filesystem>

namespace vc::lasagna {

class LasagnaDataset {
public:
    explicit LasagnaDataset(LasagnaDatasetManifest manifest);

    static LasagnaDataset open(const std::filesystem::path& manifestPath);

    [[nodiscard]] const LasagnaDatasetManifest& manifest() const noexcept;
    [[nodiscard]] bool hasNormalSource() const noexcept;
    [[nodiscard]] const std::filesystem::path& normalSourcePath() const;

private:
    LasagnaDatasetManifest manifest_;
};

} // namespace vc::lasagna
