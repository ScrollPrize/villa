#pragma once

#include "CpuDiscovery.hpp"
#include "JobStore.hpp"

#include <fastmcpp/tools/manager.hpp>

#include <memory>

namespace vc::mcp
{
void registerDiscoveryTools(fastmcpp::tools::ToolManager& tools, const std::shared_ptr<JobStore>& store, const std::shared_ptr<CpuDiscovery>& discovery);
}
