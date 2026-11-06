#pragma once
// Numbered history, store commit sequence, change feed and receipts.
#include "internal/Database.hpp"

namespace vc::objects::detail {
// Fills RecordCommon::version/versionGeneration of records selected in a read
// transaction from their published heads (one prepared lookup per record).
void attachObjectVersions(sqlite3*, std::vector<ObjectRecord>&);
void attachRelationshipVersions(sqlite3*, std::vector<EdgeRecord>&);
// Tables shared by new stores and the offline converter. `generation` is the
// initial active history generation.
void createVersionSchema(sqlite3*, const std::string& generation, const std::string& reason = "created");

std::string activeGeneration(sqlite3*);
int64_t commitSequence(sqlite3*);
std::string storeUuidText(sqlite3*);
std::optional<int64_t> headVersion(sqlite3*, const std::string& objectUuid);
std::optional<int64_t> relationshipHeadVersion(sqlite3*, const std::string& edgeUuid);
// Version of an immutable object revision (internal UUID); 0 if unnumbered.
int64_t revisionVersion(sqlite3*, const std::string& revisionUuid);

// Allocates the commit lazily: a publication that numbers nothing leaves the
// commit sequence unchanged. Called only from publishHistory.
class CommitPublisher {
public:
    CommitPublisher(sqlite3* db, int64_t publishedAtUs);
    int64_t object(const std::string& objectUuid, const std::string& revisionUuid, bool deleted);
    // nullopt revision records the relationship's deletion.
    int64_t relationship(const std::string& edgeUuid, const std::optional<std::string>& revisionUuid);
    std::optional<int64_t> sequence() const { return sequence_; }
private:
    int64_t nextObject(const std::string& objectUuid);
    void begin();
    sqlite3* db_;
    std::string generation_;
    int64_t publishedAtUs_;
    std::optional<int64_t> sequence_;
};

// Called by Txn::release after publishHistory, inside the write transaction:
// fills the numbered receipt and persists it for a request UUID.
TransactionReceipt finishReceipt(sqlite3*, const StatementPool&, const VersionSelection& expectedObjects,
                                 const VersionSelection& expectedRelationships);
std::optional<TransactionReceipt> savedReceipt(sqlite3*, const std::string& requestUuid,
                                               std::string* digest = nullptr);
nlohmann::json receiptToJson(const TransactionReceipt&);
TransactionReceipt receiptFromJson(const nlohmann::json&);

// Begin-time checks for TransactionContext (writer lock held). Translates
// object expectations to revision expectations for the strict publish checks.
void checkTransactionContext(sqlite3*, const TransactionContext&, RevisionSelection& expectedRevisions);
}
