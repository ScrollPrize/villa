#pragma once
#include "internal/Database.hpp"

namespace vc::objects::detail {
// Derived current-state index. Dirty rows are refreshed before publication and
// before source relationship lookup; history is never modified.
void createSourceIdentityIndex(sqlite3*);
void refreshSourceIdentityIndex(sqlite3*);
void updateSourcePointIdentities(sqlite3*, const std::string& object,
    const std::string& previous, const std::string& revision,
    const nlohmann::json& manifest, const std::vector<int64_t>& changed);
struct SourcePoint { std::string object; int64_t offset; };
std::optional<SourcePoint> sourcePoint(sqlite3*, const std::string& origin, int64_t id);
std::optional<std::string> sourceCollection(sqlite3*, const std::string& origin, const std::string& id);
bool indexedSourceRevision(sqlite3*, const std::string& object, const std::string& revision);
std::optional<int64_t> indexedSourcePointOffset(sqlite3*, const std::string& object, int64_t id);
void validateSourceIdentities(sqlite3*, const std::string& object);
}
