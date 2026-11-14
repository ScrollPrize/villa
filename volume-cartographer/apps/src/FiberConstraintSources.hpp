#pragma once

#include "vc/lasagna/Dataset.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

// Normal axes and winding density need not come from the same prediction.
// Resolve each through the normal dataset reader and check its own provenance.
struct FiberConstraintSource {
    vc::lasagna::LasagnaDataset dataset;
    nlohmann::json provenance;
};

inline FiberConstraintSource openFiberConstraintSource(
    const nlohmann::json& config, const std::string& prefix,
    const std::string& scan, const vc::lasagna::LasagnaDatasetOpenOptions& options)
{
    auto dataset=vc::lasagna::LasagnaDataset::openLocation(config.at(prefix+"_manifest"),options);
    nlohmann::json provenance;
    std::ifstream stream(config.at(prefix+"_provenance").get<std::string>());
    if(!stream)throw std::invalid_argument("Cannot read "+prefix+" provenance");
    stream>>provenance;
    const auto slash=scan.find('/');
    if(slash==std::string::npos ||
       provenance.at("source").at("sample_id")!=scan.substr(0,slash) ||
       provenance.at("source").at("volume_id")!=scan.substr(slash+1))
        throw std::invalid_argument(prefix+" provenance does not name expected scan");
    const auto& path=dataset.manifest().manifestPath;
    bool found=false;
    for(const auto& artifact:provenance.at("artifacts")) {
        if(artifact.at("kind")!="manifest")continue;
        const auto recorded=std::filesystem::path(artifact.at("path").get<std::string>());
        if(recorded.filename()!=path.filename())continue;
        // Compound manifests live in STORE.zarr/.zattrs, not at the artifact root.
        if(path.filename()==".zattrs" && recorded.parent_path().filename()!=path.parent_path().filename())continue;
        found=true;
    }
    if(!found)throw std::invalid_argument(prefix+" manifest is not the recorded provenance artifact");
    if(!dataset.manifest().baseShapeZYX)throw std::invalid_argument(prefix+" manifest has no base shape");
    return {std::move(dataset),std::move(provenance)};
}
