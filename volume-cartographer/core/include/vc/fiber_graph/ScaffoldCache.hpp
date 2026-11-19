#pragma once
// Reuse of a scaffold run's derived results: the fibers' strip normals and
// the crossings of each pair (vc_object_store FiberScaffoldCacheExtension).
//
// Both are expensive and both are functions of stored geometry and of the
// run's parameters, so a run states the inputs it would compute from, looks
// the answer up, and computes only what is missing. Nothing is invalidated by
// hand and nothing is hashed: a row names the versions of the records it was
// computed from and states its parameters in the clear, so an edited fiber,
// an edited link or a changed parameter simply stops matching.
#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include "vc/fiber_graph/ScaffoldNormals.hpp"

#include <vc_object_store/objects/FiberScaffoldCacheExtension.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace vc::fiber_graph {

// What a run reuses, read from the store once. Lookups are by the row's own
// key, so a store holding rows of several parameter sets and several fiber
// versions answers each run with its own.
class ScaffoldCache {
public:
    ScaffoldCache() = default;
    explicit ScaffoldCache(const objects::StoreReader& reader);
    // The same cache over rows the caller already has (a test, or a run that
    // keeps what it just computed).
    ScaffoldCache(std::span<const objects::FiberScaffoldNormalsRow> normals,
                  std::span<const objects::FiberScaffoldCrossingRow> crossings);

    [[nodiscard]] bool empty() const { return _normals.empty() && _crossings.empty(); }

    // The cached normals of one fiber, or null.
    [[nodiscard]] const objects::FiberScaffoldNormalsRow* normals(const std::string& fiberUuid,
                                                                  std::int64_t fiberVersion,
                                                                  const std::string& inputs) const;
    // The cached crossings of one pair (possibly none, which is itself a
    // cached answer), or nullopt when the pair was never cached this way.
    // The fibers may be given in either order; `normalsA` and `normalsB` are
    // the normals rows of fiberA and fiberB as given.
    [[nodiscard]] std::optional<std::span<const objects::FiberScaffoldCrossingRow>> crossings(
        const std::string& fiberA, std::int64_t versionA, const std::string& normalsA, const std::string& fiberB,
        std::int64_t versionB, const std::string& normalsB, const std::string& inputs) const;

private:
    void sortPairs();
    using NormalKey = std::tuple<std::string, std::int64_t, std::string>;
    using PairKey = std::tuple<std::string, std::string, std::int64_t, std::int64_t, std::string, std::string,
                               std::string>;
    std::map<NormalKey, objects::FiberScaffoldNormalsRow> _normals;
    std::map<PairKey, std::vector<objects::FiberScaffoldCrossingRow>> _crossings;
};

// What a run wants written back. The caller appends these rows in its own
// transaction (the run already writes its result rows there), so the cache
// never opens one of its own.
struct ScaffoldCacheSink {
    std::string author, computedAt;
    // Typed rows: the caller encodes them with fiberScaffoldNormalsRow and
    // fiberScaffoldCrossingsRow when it commits, so nothing here depends on
    // the store's encoding.
    std::vector<objects::FiberScaffoldNormalsRow> normals;
    std::vector<objects::FiberScaffoldCrossingRow> crossings;
    [[nodiscard]] bool empty() const { return normals.empty() && crossings.empty(); }
};

// What one fiber's normals are solved from besides the fiber itself, as the
// text its row is keyed by: the frame the geometry was read in (`frame`: the
// graph's volume and scale), the strip and anchor parameters, the prediction
// a fused run reads (by its identity), and `anchors` - one statement per
// record that fixes a normal on this fiber, each naming that record's
// version. Two runs that would solve the same fiber the same way state the
// same text.
[[nodiscard]] std::string scaffoldNormalInputs(const ScaffoldNormalOptions&, const std::string& frame, bool fused,
                                               std::span<const std::string> anchors);
// The extraction parameters a pair's crossings are keyed by.
[[nodiscard]] std::string scaffoldCrossingInputs(const ScaffoldEventOptions&);

} // namespace vc::fiber_graph
