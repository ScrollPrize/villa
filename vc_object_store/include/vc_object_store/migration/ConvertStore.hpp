#pragma once
// Offline conversion of schema-16 to schema-19 stores (standalone, hub, or
// settled hub replica) to the current schema (20). The only place that knows
// the previous formats; the runtime opens the current schema only. Schema 16
// (without surface data) is first brought to schema 17; schema 17
// is numbered first (schema 18), then every format is copied into a store
// built by the runtime's own schema definition. Schema 20 is schema 19
// without the hub's replica identity table.
//
// The source is never modified: it is read through one consistent SQLite
// snapshot into a new destination. Identities, recorded history, geometry,
// relationships and timestamps are copied unchanged; numbered versions and the
// commit sequence are added deterministically from recorded publication order;
// hub coordination tables are removed. A replica with unpublished, in-flight
// or conflicting state is refused: settle it with the old tooling first.
//
// Gaps (untimed baselines, relationship states without a recorded
// publication, geometry revisions that do not identify one object version,
// metadata holding old revision UUIDs, missing artifact files) are reported,
// never guessed; they do not by themselves block conversion.

#include <cstdint>
#include <filesystem>
#include <string>

namespace vc::objects::migration {

struct ConversionResult {
    // Report: counts, removed tables, explicit gaps, validation.
    std::string reportJson;
    // Old object/relationship revision UUID -> generation-qualified numbered
    // reference, for converting pins in external exports and job configs.
    // Entries from a schema-18 source also name their own generation.
    std::string mappingJson;
};

ConversionResult convertStore(const std::filesystem::path& source, const std::filesystem::path& destination);

} // namespace vc::objects::migration
