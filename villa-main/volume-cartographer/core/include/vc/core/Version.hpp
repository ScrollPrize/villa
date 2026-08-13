#pragma once

/** @file */

#include <cstdint>
#include <string>

/** Provides programmatic access to codebase information */
struct ProjectInfo {
    /** Get the library name */
    static auto Name() -> std::string;
    /** Get the library version as a Major.Minor.Patch string */
    static auto VersionString() -> std::string;
    /** Get the library name and version string */
    static auto NameAndVersion() -> std::string;
    /** Get the library Major version number */
    static auto VersionMajor() -> std::uint32_t;
    /** Get the library Minor version number */
    static auto VersionMinor() -> std::uint32_t;
    /** Get the library Patch version number */
    static auto VersionPatch() -> std::uint32_t;
    /** Get the git repository URL */
    static auto RepositoryURL() -> std::string;
    /** Get the full hash for the current git commit */
    static auto RepositoryHash() -> std::string;
    /** Get the short hash for the current git commit */
    static auto RepositoryShortHash() -> std::string;
};
