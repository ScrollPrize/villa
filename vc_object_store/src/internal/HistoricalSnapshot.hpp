#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <memory>
struct sqlite3;
namespace vc::objects::detail {
// Per-lease, RAM-only projection of an earlier retained source publication.
// Leaves the physical database and immutable payloads untouched.
std::shared_ptr<void> historicalSnapshot(sqlite3*,const StoreStatus&);
void requireRetainedPublication(sqlite3*,const StoreStatus&);
}
