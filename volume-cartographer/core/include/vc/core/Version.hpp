#pragma once

#include <string>

struct ProjectInfo {
    static auto Name() -> std::string;
    static auto VersionString() -> std::string;
    static auto NameAndVersion() -> std::string;
    static auto RepositoryHash() -> std::string;
    static auto RepositoryShortHash() -> std::string;
};
