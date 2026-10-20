#pragma once
#include "internal/Database.hpp"
namespace vc::objects::detail {
void createDependencyIndex(sqlite3*);
std::vector<EdgeRecord> structuralEdgeCandidates(sqlite3*, const std::string&, const std::string&, const std::vector<int64_t>&);
}
