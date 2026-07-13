#pragma once

#include "JobStore.hpp"

#include <fastmcpp/resources/manager.hpp>
#include <fastmcpp/tools/manager.hpp>

#include <memory>

namespace vc::mcp
{

void registerInspectionTools(fastmcpp::tools::ToolManager& tools,
                             fastmcpp::resources::ResourceManager& resources,
                             const std::shared_ptr<JobStore>& store);

}  // namespace vc::mcp
