#pragma once
// Offline conversion of upstream schema-16 to schema-20 stores (standalone,
// hosted, or settled legacy hub replica) to the current schema. The only place that knows
// the previous formats; the runtime opens the current schema only. Schema 16
// (without surface data) is first brought to schema 17; schema 17
// is numbered first (schema 18), then every format is copied into a store
// built by the runtime's own schema definition. Unpublished experimental
// schema-18/v2 layouts are not supported, despite sharing a version number.
//
// The source is never modified: it is read through one consistent SQLite
// snapshot into a new destination. Identities, recorded history, geometry,
// relationships and timestamps are preserved; numbered versions and the
// commit sequence are added deterministically from recorded publication order;
// hub coordination tables are removed. A replica with unpublished, in-flight
// or conflicting state is refused: settle it with the old tooling first.
// Content-addressed artifacts referenced by any recorded version are copied
// and verified. Missing or corrupt referenced blobs fail conversion.
//
// Gaps (untimed baselines, relationship states without a recorded
// publication, geometry revisions that do not identify one object version,
// metadata holding old revision UUIDs, missing unversioned external files) are reported,
// never guessed; they do not by themselves block conversion.

#include <cstdint>
#include <filesystem>
#include <functional>
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

// Explicit offline replacement: exclusive ownership, validated sibling output,
// durable atomic database replacement. Existing source artifacts stay in place.
// On validation failure the source remains an upstream store; no partial
// converted database is published. Stop and reopen every client around this call.
// A report sink, if supplied, runs with the final report and mapping before
// replacing the source. A failed sink leaves the legacy database in place.
ConversionResult convertStoreInPlace(const std::filesystem::path& source,
    const std::function<void(const ConversionResult&)>& beforeReplace = {});

void writeConversionReports(const ConversionResult&, const std::filesystem::path& mapping,
    const std::filesystem::path& report);

} // namespace vc::objects::migration
