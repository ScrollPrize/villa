#pragma once
// Selection records: the exact result of an object selection as of one
// commit, kept durably so a long job, an export or a training set can be read
// back later at the same versions while the store moves on.
//
// A record holds the query, the store, the commit sequence it is exact at,
// every object it matched (in listing order) at its numbered version, and,
// for volume/coordinate selections, the versions of the recorded transforms
// and route choices the matching used. Read the objects back with
// objectsAtVersions (local or hosted).
//
// The binary form is compact, about 24 bytes per object, and is meant to be
// kept as an artifact (objects/Artifacts.hpp):
//   "VCSEL\0\2\n"  8-byte magic (format version 2)
//   u32            header length (little-endian), then the header JSON:
//                  store, generation, commit_sequence, query, transforms,
//                  origins ([store, generation] pairs entries use), count
//   count x 24 B   object UUID (16 bytes), u32 index into origins,
//                  u32 version (little-endian)
// Only version 2 is supported; every reference carries its owning store,
// independent of the query store. Regenerate obsolete version-1 artifacts.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vc_object_store/objects/Transactions.hpp"

namespace vc::objects {

class ObjectStore;
class Txn;
struct ObjectSelection;

struct SelectionRecord {
    std::string storeUuid;
    // The store's active generation and commit sequence the listing is exact at.
    std::string generation;
    int64_t commitSequence = 0;
    // The selection as objectQueryRequestToJson(selection, nullopt) text.
    std::string queryJson;
    // Recorded volume_transform and volume_transform_choice objects in force,
    // in UUID order; empty unless the selection names a volume or coordinates.
    std::vector<ObjectVersion> transforms;
    // Every match, in listing order.
    std::vector<ObjectVersion> objects;
    bool operator==(const SelectionRecord&) const = default;
};

std::vector<uint8_t> encodeSelectionRecord(const SelectionRecord&);
// Throws std::invalid_argument on malformed bytes.
SelectionRecord decodeSelectionRecord(std::span<const uint8_t>);

// Whether a selection's membership depends on the recorded transforms.
bool selectionUsesTransforms(const ObjectSelection&);

// Captures a selection in one read snapshot (a ReadTxn: write transactions
// hold unpublished rows). The selection must not carry explicit transforms or
// a limit; a volume selection must not rely on client catalog registrations
// that the store has not recorded (record them first with
// recordCatalogTransforms), since the record could not name them.
// Hosted stores: HostedStore::recordSelection.
SelectionRecord recordSelection(const ObjectStore&, const Txn&, const ObjectSelection&);

} // namespace vc::objects
