#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace vc::objects {
// Input provenance, NOT the commit which writes a result. A reference does not
// retain history or declare that a result is valid for a later source state.
struct ExtensionSource {
    std::string storeUuid, generation;
    int64_t commitSequence = 0;
    bool operator==(const ExtensionSource&) const = default;
};
std::string extensionSourcesJson(std::span<const ExtensionSource>);
std::vector<ExtensionSource> extensionSourcesFromJson(const std::string&);

using ExtensionValue = std::variant<std::monostate, int64_t, double, std::string, std::vector<uint8_t>>;
using ExtensionRow = std::vector<ExtensionValue>;
ExtensionValue extensionUuid(const std::string&);
std::string extensionUuid(const ExtensionValue&);
enum class ExtensionType { Integer, Real, Text, Blob };
struct ExtensionColumn {
    std::string name;
    ExtensionType type = ExtensionType::Real;
    bool nullable = false;
    bool operator==(const ExtensionColumn&) const = default;
};
struct ExtensionIndex {
    std::vector<std::string> columns;
    bool unique = false;
    bool operator==(const ExtensionIndex&) const = default;
};
struct ExtensionTable {
    std::string name; // ext_*; a physical SQL table, not a JSON/EAV payload
    std::string extension;
    int64_t formatVersion = 1; // independent of core schema 21
    std::string provisioning = "{}"; // extension-owned, small JSON object
    std::vector<ExtensionSource> basedOn; // common input snapshot for these rows
    std::vector<ExtensionColumn> columns;
    std::vector<std::string> primaryKey;
    std::vector<ExtensionIndex> indexes;
    // Optional colocated UUID routing: "objects" or "edges". First primary-key
    // column is that entity UUID. No per-row routing/endpoint directories.
    std::string keySpace;
    bool versioned = false; // manual fields use the owning object/edge history
};
struct ExtensionTableInfo {
    std::string tableUuid, name, extension;
    int64_t formatVersion = 1, lastModifiedCommit = 0;
    std::string provisioning = "{}";
    std::vector<ExtensionSource> basedOn;
    std::vector<ExtensionColumn> columns;
    std::vector<std::string> primaryKey;
    std::vector<ExtensionIndex> indexes;
    std::string keySpace;
    bool versioned = false;
};
struct InitialExtensionRow {
    std::string table;
    ExtensionRow values;
};
size_t extensionPayloadBytes(std::span<const InitialExtensionRow>);
} // namespace vc::objects
