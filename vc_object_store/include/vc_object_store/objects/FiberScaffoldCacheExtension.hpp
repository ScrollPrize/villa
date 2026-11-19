#pragma once
// Cached derived results of a scaffold solve (docs/extensions.md).
//
// A measured run spends its minutes on two things: solving each fiber's strip
// normals, and extracting the crossings of every H/V pair out of the volume.
// Both are functions of stored geometry and of the run's parameters, so a run
// that reads the same geometry with the same parameters may read the result
// instead of computing it again.
//
// Two immutable tables hold them, keyed so that a row is either *current* or
// irrelevant - nothing is ever invalidated by hand, and nothing is hashed:
// the store versions its records, and a row names the versions it was
// computed from.
//
// - the normals of one fiber, keyed by its UUID, the object version the
//   normals were solved from, and `inputs`: a readable statement of the
//   parameters and of every other versioned record that constrained the
//   solve (each anchoring link and its peer fiber, by version). The row has
//   an identity of its own, `normalsUuid`;
// - the crossings of one fiber *pair*, keyed by both UUIDs in order, both
//   object versions, the extraction parameters as `inputs`, and the
//   identities of the two normals rows the extraction read.
//
// A fiber edit advances its object version, so its own normals row and every
// crossing row naming it stop matching, and only those are recomputed. A
// changed parameter is a different `inputs`, so nothing matches and the run
// computes everything - which is what a different question deserves.
#include "vc_object_store/objects/Extensions.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::objects {
class StoreReader;

inline constexpr const char* kFiberScaffoldNormalsTable = "ext_fiber_strip_normals";
inline constexpr const char* kFiberScaffoldCrossingsTable = "ext_fiber_pair_crossings";
inline constexpr const char* kFiberScaffoldCacheExtension = "fiber-scaffold-cache";
inline constexpr int64_t kFiberScaffoldCacheFormat = 1;

// One fiber's cached strip normals. `normals` holds three float64 per line
// point of the fiber at `fiberVersion`, in its stored order - the solve's own
// precision, so a reusing run computes exactly what the solving run did.
// `signSource` is the scaffold's enumeration of what fixed the sign and
// `converged` the solve's own verdict, both carried through so a reusing run
// reports the same provenance.
struct FiberScaffoldNormalsRow {
    std::string fiberUuid;
    int64_t fiberVersion = 0;
    // This row's own identity; the crossings extracted with these normals
    // name it.
    std::string normalsUuid;
    // What the normals were solved from besides the fiber itself: the
    // parameters, and each anchoring record by version. A run that would
    // solve from anything else states a different text and does not match.
    std::string inputs;
    int64_t signSource = 0;
    int64_t sign = 0;
    bool converged = false;
    // 3 * linePointCount float64, little endian.
    std::vector<uint8_t> normals;
    std::string author, computedAt;
};

// One crossing of one pair, as the extraction found it. The fields are the
// scaffold's event, not the extractor's full constraint: the arcs, the points
// and the measured winding separation are what the winding assembly reads.
struct FiberScaffoldCrossingRow {
    std::string fiberA, fiberB;   // fiberA < fiberB
    int64_t versionA = 0, versionB = 0;
    // The extraction parameters, and the normals rows of fiberA and fiberB
    // the extraction read.
    std::string inputs;
    std::string normalsA, normalsB;
    int64_t eventIndex = 0;       // within this pair, in extraction order
    double arcA = 0, arcB = 0;
    std::array<double, 3> pointA{}, pointB{};
    double parallelScore = 0, perpendicularScore = 0;
    // NaN when the run measured no separation.
    double windingDistance = 0, parallelWindingDistance = 0;
    bool parallel = false;
    std::string author, computedAt;
};

// A pair with no crossing at all is cached too, as one row with
// `eventIndex == kFiberScaffoldNoCrossings`: otherwise an empty answer would
// be re-extracted on every run.
inline constexpr int64_t kFiberScaffoldNoCrossings = -1;

[[nodiscard]] ExtensionTable fiberScaffoldNormalsTable();
[[nodiscard]] ExtensionTable fiberScaffoldCrossingsTable();
[[nodiscard]] ExtensionRow fiberScaffoldNormalsRow(const FiberScaffoldNormalsRow&);
[[nodiscard]] ExtensionRow fiberScaffoldCrossingsRow(const FiberScaffoldCrossingRow&);

// Every cached normals row of the store, newest commit last.
[[nodiscard]] std::vector<FiberScaffoldNormalsRow> readFiberScaffoldNormals(const StoreReader&);
// Every cached crossing. Rows of one pair come back in `eventIndex` order.
[[nodiscard]] std::vector<FiberScaffoldCrossingRow> readFiberScaffoldCrossings(const StoreReader&);
} // namespace vc::objects
