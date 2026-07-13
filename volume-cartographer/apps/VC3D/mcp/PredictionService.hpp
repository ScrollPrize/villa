#pragma once

#include <nlohmann/json.hpp>

namespace vc::mcp
{

using Json = nlohmann::json;

Json inspectPrediction(const Json& request);
Json findSeedCandidates(const Json& request);

}  // namespace vc::mcp
