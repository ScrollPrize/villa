#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace vc::project {

// The portable workspace description. Store data and credentials never live here.
class ProjectDocument {
public:
    static const std::vector<std::string>& kinds();
    static ProjectDocument load(const std::filesystem::path& path);
    static ProjectDocument create(const std::filesystem::path& path, const std::string& name,
                                  const std::filesystem::path& store, bool readOnly = false);
    const std::filesystem::path& path() const { return path_; }
    const nlohmann::json& data() const { return data_; }
    bool writable() const;
    // Validate and atomically persist before publishing the new state.
    void replace(nlohmann::json data);
    ProjectDocument saveAs(const std::filesystem::path& destination) const;
    std::string location(const std::filesystem::path& path) const;
    std::filesystem::path resolve(const std::string& location) const;
private:
    std::filesystem::path path_;
    nlohmann::json data_;
    static void validate(const nlohmann::json& data);
};
}
