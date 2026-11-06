#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <memory>
#include <span>

namespace vc::objects::detail {
struct EdgeInsertRow;
// Trusted initial-state construction only. Caller owns the complete input and
// savepoint; endpoint existence, state, geometry and cross-table UUID identity
// are not rediscovered here. finish restores deferred lookup indexes only.
class InitialEdgeInserter {
public:
    InitialEdgeInserter(ObjectStore&, WriteTxn&, bool deferIndexes = true,
                        size_t sqlBatchRows = 1);
    ~InitialEdgeInserter();
    InitialEdgeInserter(const InitialEdgeInserter&) = delete;
    InitialEdgeInserter& operator=(const InitialEdgeInserter&) = delete;
    void append(std::span<EdgeRecord>);
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
