#pragma once
#include "vc_object_store/objects/FiberImportExtension.hpp"
#include <map>

namespace vc::objects {

// All are optional: requesting a definition does not create tables. Importers
// create only sources/constraints; solve writers create the tables they use.
enum class FiberGraphTable {
    Runs, Inputs, FiberResults, Samples, Hypotheses, Widths, Positions,
    ConstraintResults, Metrics
};
enum class FiberGraphLayer : int64_t { Shared = 0, Intermediate = 1, Final = 2, Refined = 3 };

ExtensionTable fiberGraphSolveTable(FiberGraphTable);

// Named construction for workflow writers, using the same typed SQL values as
// the batch API. Missing nullable fields become NULL; unknown fields fail.
// For hot bulk paths, the declared column order can be used directly instead.
InitialExtensionRow fiberGraphSolveRow(FiberGraphTable,
    const std::map<std::string, ExtensionValue>& fields);

// Only explicit scalar settings, never a source report or JSON configuration
// tree. Algorithm identifiers go in method; file locations are not stored.
using FiberGraphSetting = std::variant<bool, int64_t, double>;
InitialExtensionRow fiberGraphRunRow(const std::string& runUuid,
    const std::string& stage, const std::string& method,
    const std::map<std::string, FiberGraphSetting>& settings = {});

// A run can use several input stores and earlier runs. Input references are
// typed rows, not lists copied into each physical table's provisioning.
InitialExtensionRow fiberGraphInputRow(const std::string& runUuid,
    const std::string& role, const ExtensionSource& input,
    const std::optional<std::string>& inputRunUuid = {});
} // namespace vc::objects
