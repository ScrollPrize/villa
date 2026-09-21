#pragma once
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace vc::constraints::detail {
using QuarantineMoves=std::map<std::filesystem::path,std::filesystem::path>;
std::string workspaceText(const std::filesystem::path&);
std::string workspaceRevision(const std::string&);
QuarantineMoves planQuarantine(const std::filesystem::path& patchRoot,
                              const std::filesystem::path& workspace,
                              const std::vector<std::filesystem::path>& sources);
void publishQuarantine(const std::filesystem::path& workspace,const std::string& original,
                       const std::string& candidate,const QuarantineMoves&);
}
