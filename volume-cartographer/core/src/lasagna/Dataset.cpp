#include "vc/lasagna/Dataset.hpp"

#include <stdexcept>

namespace vc::lasagna {

LasagnaDataset::LasagnaDataset(LasagnaDatasetManifest manifest)
    : manifest_(std::move(manifest))
{
}

LasagnaDataset LasagnaDataset::open(const std::filesystem::path& manifestPath)
{
    return LasagnaDataset(LasagnaDatasetManifest::parseFile(manifestPath));
}

const LasagnaDatasetManifest& LasagnaDataset::manifest() const noexcept
{
    return manifest_;
}

bool LasagnaDataset::hasNormalSource() const noexcept
{
    return manifest_.hasNormalSource();
}

const std::filesystem::path& LasagnaDataset::normalSourcePath() const
{
    if (!manifest_.normalPath.has_value()) {
        throw std::runtime_error("Lasagna dataset manifest has no normal source path");
    }
    return *manifest_.normalPath;
}

} // namespace vc::lasagna
