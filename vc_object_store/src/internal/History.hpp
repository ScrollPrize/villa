#pragma once
#include "internal/Database.hpp"

namespace vc::objects::detail {
void createHistorySchema(sqlite3*);
void publishHistory(sqlite3*, bool importedBindings = false);
void expectHistory(sqlite3*, const RevisionSelection&);
[[noreturn]] void historyConflict(sqlite3*, const std::string&);
void recordRelationshipValidation(sqlite3*, const EdgeRecord&, bool recordedOnly = false);
// Shared metadata dependency semantics for native and collection relationships.
nlohmann::json dependencyMetadata(const nlohmann::json&, const Anchor&);
// Identical dependency projection, without loading unrelated current-object data.
nlohmann::json currentDependencyMetadata(sqlite3*, const std::string& objectUuid, const Anchor&);
// Latest-head checks reuse the selected-revision dependency evaluator. Indexed
// membership probes avoid enumerating unrelated incident relationships.
bool currentRelationshipApplicable(const ObjectStore&, const Txn&, const std::string& edgeUuid);
void requirePublishedGraphRead(const Txn&);
}
