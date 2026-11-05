#pragma once
// Tests of store-level operations (edits, exports, ingestion) drive an
// ObjectStore directly: one checked write transaction, or one read.
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <atomic>
#include <utility>

// Commits fn(store, txn) in a write checked against `expected`; returns the
// revisions it committed.
template<class F>
vc::objects::RevisionSelection storeAct(vc::objects::ObjectStore& store, const vc::objects::RevisionSelection& expected, F&& fn)
{
    auto txn = store.write(expected);
    fn(store, txn);
    txn.commit();
    return txn.committedRevisions();
}

// fn(store, txn, cancelled) over one read transaction (never cancelled).
template<class F>
void storeRead(vc::objects::ObjectStore& store, F&& fn)
{
    static const std::atomic_bool never{false};
    auto txn = store.read();
    fn(std::as_const(store), txn, never);
}

// fn(reader, cancelled) over one read transaction (never cancelled).
template<class F>
void storeReader(vc::objects::ObjectStore& store, F&& fn)
{
    static const std::atomic_bool never{false};
    auto txn = store.read();
    const vc::objects::LocalStoreReader reader(store, txn);
    fn(reader, never);
}
