#pragma once

#include "CpuDiscovery.hpp"
#include "JobStore.hpp"
#include "VolumeCartographer.hpp"

#include <fastmcpp/prompts/manager.hpp>
#include <fastmcpp/resources/manager.hpp>
#include <fastmcpp/server/server.hpp>
#include <fastmcpp/tools/manager.hpp>

#include <functional>
#include <memory>
#include <string>

namespace vc::mcp
{

struct ServerConfig {
    std::string vcVersion{"unknown"};
    std::string vcCommit{"unknown"};
    std::string containerDigest{"unverified"};
    std::shared_ptr<VolumeCartographer> worker;
    std::shared_ptr<CpuDiscovery> discovery;
};

class McpApplication
{
public:
    using Handler = std::function<Json(const Json&)>;

    explicit McpApplication(ServerConfig config = {});

    McpApplication(const McpApplication&) = delete;
    McpApplication& operator=(const McpApplication&) = delete;
    McpApplication(McpApplication&&) = delete;
    McpApplication& operator=(McpApplication&&) = delete;

    Json handle(const Json& request) const;
    Handler handler() const;
    std::shared_ptr<JobStore> store() const { return store_; }

private:
    ServerConfig config_;
    std::shared_ptr<JobStore> store_;
    fastmcpp::server::Server routing_;
    fastmcpp::tools::ToolManager tools_;
    fastmcpp::resources::ResourceManager resources_;
    fastmcpp::prompts::PromptManager prompts_;
    Handler handler_;
};

}  // namespace vc::mcp
