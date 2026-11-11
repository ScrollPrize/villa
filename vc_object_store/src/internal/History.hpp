#pragma once
#include "internal/Database.hpp"

namespace vc::objects::detail {
void createHistorySchema(sqlite3*);
std::optional<int64_t> retainedCreationPublication(sqlite3*, const std::string& object);
// Private conversion/export only: transform complete archived states, then
// rebuild their deltas and cached heads without changing revision identities.
// The caller owns the transaction and suspends mutation/immutability triggers.
void transformArchivedObjectRecords(sqlite3*,
    const std::function<void(const std::string& revision, nlohmann::json&)>&);
void beginCurrentWrite(sqlite3*);
void publishHistory(sqlite3*, bool baseline = false);
void expectHistory(sqlite3*, const RevisionSelection&);
[[noreturn]] void historyConflict(sqlite3*, const std::string&);
void recordRelationshipValidation(sqlite3*, const EdgeRecord&, bool recordedOnly = false);
// Shared metadata dependency semantics for native and collection relationships.
nlohmann::json dependencyMetadata(const nlohmann::json&, const Anchor&);
// Identical dependency projection, without loading unrelated current-object data.
nlohmann::json currentDependencyMetadata(sqlite3*, const std::string& objectUuid, const Anchor&);
// legacyTextUuid is used only by the separate schema-17 converter before migration.
nlohmann::json historicalDependencyMetadata(sqlite3*, const std::string& revision, const Anchor&, bool legacyTextUuid=false);
// Shared dependency projection for current/archived records and v1 conversion.
nlohmann::json dependencyFields(const nlohmann::json* metadata, bool retired,
    const std::string& lifecycleEpoch, const Anchor&);
// Latest-head checks reuse the selected-revision dependency evaluator. Indexed
// membership probes avoid enumerating unrelated incident relationships.
bool currentRelationshipApplicable(const ObjectStore&, const Txn&, const std::string& edgeUuid);
// The last published binding, archived or still current in a sealed source.
std::optional<RelationshipRevision> recordedRelationship(sqlite3*,const std::string& edgeUuid);
std::optional<std::string> currentObjectReference(sqlite3*,const std::string& objectUuid);
void requirePublishedGraphRead(const Txn&);
// One logical store's timeline (including its physical shards). The owner is
// explicit for consumers restoring/exporting a state from a referenced store.
RevisionSelection revisionSelectionAt(sqlite3*,const std::string& owner,int64_t publishedAtUs,bool includeRetired);
}
