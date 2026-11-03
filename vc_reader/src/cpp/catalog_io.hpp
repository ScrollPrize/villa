#pragma once
#include <filesystem>
#include <string>
namespace vcr::catalog {
std::string readCatalogFile(const std::filesystem::path&);
}
