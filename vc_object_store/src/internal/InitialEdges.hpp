#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <memory>
#include <span>

namespace vc::objects::detail {
struct EdgeInsertRow;
// Private initial-state construction only. Caller owns a savepoint spanning
// the entire stream and must roll it back on failure. Existing endpoints must
// remain immutable unless invalidateEndpoint is called before their next use.
// finish restores deferred accelerators before graph reads/publication.
class InitialEdgeInserter {
public:
    InitialEdgeInserter(ObjectStore&, WriteTxn&, bool deferIndexes = true,
                        size_t maxCachedEndpoints = 65536,size_t sqlBatchRows = 1);
    ~InitialEdgeInserter();
    InitialEdgeInserter(const InitialEdgeInserter&) = delete;
    InitialEdgeInserter& operator=(const InitialEdgeInserter&) = delete;
    void append(std::span<const EdgeRecord>);
    // Endpoint ids/dependencies already resolved and checked by the collection
    // preparation boundary. This performs only native row preparation/writes.
    void appendResolved(std::span<const EdgeInsertRow>);
    void finish();
    void invalidateEndpoint(const std::string&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
