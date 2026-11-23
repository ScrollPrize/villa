#pragma once

#include "vc/fiber_tracer/FiberCrossingAxes.hpp"
#include "vc/fiber_tracer/FiberGapFalloff.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

// Normals inferred from the signed strip normals of OTHER fibers nearby.
//
// A solved strip normal is sampled at a regular arc spacing along every fiber
// (FiberStripSample). Signed by a sign stage, the samples of all fibers form a
// field; at a query sample the field states the principal axis of the weighted
// outer products of the neighbours' normals (perpendicular to the query's own
// tangent), signed by their weighted majority. The query fiber need not be one
// of the field's fibers: a manually annotated fiber is given its normal and
// facing from the automatic fibers around it this way.
namespace vc::fiber_tracer
{

// One sample of a solved strip normal: the unit normal at a point of the
// fiber, and the unit prediction axis there (zero where there is none).
struct FiberStripSample {
    double arc = 0.;
    cv::Vec3d point{0., 0., 0.};
    cv::Vec3d normal{0., 0., 0.};
    cv::Vec3d prediction{0., 0., 0.};
};

using FiberStripUuid = std::array<unsigned char, 16>;

// A strip samples file: the fibers it lists (in file order) with their samples.
struct FiberStripSampleFile {
    std::vector<FiberStripUuid> ids;
    std::vector<std::vector<FiberStripSample>> samples;
};

[[nodiscard]] FiberStripSampleFile readFiberStripSampleFile(const std::filesystem::path& path);
// Fibers without samples are left out of the file.
void writeFiberStripSampleFile(const std::filesystem::path& path, std::span<const FiberStripUuid> ids,
                               std::span<const std::vector<FiberStripSample>> samples);

// Weighting of one inference: the falloff of the neighbour offset, measured
// per axis in a frame (own tangent, side, normal) whose normal comes from
// `frame`: isotropic (no frame, scales[0] for all axes), own (the query's own
// solved normal), prediction (the query sample's prediction axis) or two_pass
// (an isotropic Gaussian estimate of firstSigma from the same neighbours).
// fiber: distances along the fibers. A neighbour fiber's samples belong to its
// nearest contact (local minimum of its gap to the query fiber); at a crossing
// contact (more than 30 degrees) the scales are (own arc from the query sample
// to the contact, gap at the contact, partner arc from the contact to the
// sample); along a parallel contact the offset is measured like the prediction
// frame with parallelScales (tangent, side, normal). A query sample without a
// prediction axis measures a parallel contact isotropically with
// parallelScales[0], or, with estimatedParallelFrame, in the frame of the
// isotropic first estimate (firstSigma) - for query fibers that carry no
// prediction at all.
// fiberCap scales each neighbour fiber's weights so they sum to its largest one.
enum class FiberNeighbourFrame { Isotropic, Own, Prediction, TwoPass, Fiber };
struct FiberNeighbourVariant {
    GapFalloff falloff = GapFalloff::Gaussian;
    FiberNeighbourFrame frame = FiberNeighbourFrame::Fiber;
    std::array<double, 3> scales{128., 32., 128.}, parallelScales{192., 192., 64.};
    double radius = 384., firstSigma = 64.;
    bool fiberCap = false;
    bool estimatedParallelFrame = false;
};

// What the field states at one query sample for one variant. `axis` is zero
// and `count` 0 where no neighbour has weight.
struct FiberNeighbourEstimate {
    cv::Vec3d axis{0., 0., 0.};
    double weight = 0.;       // summed neighbour weight
    double coherence = 0.;    // (largest - second eigenvalue) / trace
    double margin = 0.;       // |weighted signed sum along the axis| / weight
    double count = 0.;        // neighbour samples with weight
    double effectiveSamples = 0., effectiveFibers = 0.;  // (sum w)^2 / sum w^2, per sample and per fiber
};

struct FiberNeighbourCounts {
    std::size_t frameFallbacks = 0;  // framed variants that had no frame at a sample
    std::size_t crossingPairs = 0, parallelPairs = 0;  // first fiber-frame variant
};

class FiberNeighbourField
{
public:
    // `signs[f]` signs fiber f's sample normals; 0 leaves the fiber out as a
    // source (it can still be queried). `radius` bounds every later variant's radius.
    FiberNeighbourField(std::vector<std::vector<FiberStripSample>> samples, std::vector<int> signs, double radius);

    [[nodiscard]] std::size_t fibers() const { return samples_.size(); }
    [[nodiscard]] std::size_t sampleCount() const { return owner_.size(); }
    [[nodiscard]] std::size_t signedSampleCount() const { return keyed_.size(); }
    [[nodiscard]] double radius() const { return cell_; }
    [[nodiscard]] const std::vector<FiberStripSample>& fiber(std::size_t f) const { return samples_.at(f); }
    [[nodiscard]] int sign(std::size_t f) const { return signs_.at(f); }

    // One estimate per (query sample, variant), sample-major. The query's
    // samples are consecutive along its fiber, ascending in arc. `self` is the
    // field fiber the query is (never its own source), -1 for a foreign fiber;
    // `excluded` names further field fibers to leave out.
    [[nodiscard]] std::vector<FiberNeighbourEstimate> infer(std::span<const FiberStripSample> query,
                                                            std::span<const FiberNeighbourVariant> variants,
                                                            std::int64_t self = -1,
                                                            const std::set<std::uint32_t>* excluded = nullptr,
                                                            FiberNeighbourCounts* counts = nullptr) const;

    // The crossings of the query fiber by the field's fibers: every contact
    // (a local minimum of a field fiber's gap to the query, as in the fiber
    // frame above) where the two tangents meet at more than 30 degrees, with
    // own tangent x partner tangent from the two fibers' own samples. Sources
    // are the signed fibers; `self` and `excluded` as for infer().
    [[nodiscard]] std::vector<FiberCrossingAxis> crossings(std::span<const FiberStripSample> query,
                                                           std::int64_t self = -1,
                                                           const std::set<std::uint32_t>* excluded = nullptr) const;

private:
    // One signed sample near a query sample.
    struct Near { std::uint32_t owner, index; double distance; };
    // The contacts of a query with the field's fibers: per field sample near
    // the query its nearest query sample and gap (`nearest`), those samples in
    // index order (`samples`), and their runs of consecutive samples of one
    // fiber (`runs`: first and one past the last position in `samples`).
    struct Contacts {
        std::unordered_map<std::uint32_t, std::pair<double, std::size_t>> nearest;
        std::vector<std::uint32_t> samples;
        std::vector<std::pair<std::size_t, std::size_t>> runs;
        // The local gap minima of one run, as positions in `samples`.
        [[nodiscard]] std::vector<std::size_t> minima(std::size_t first, std::size_t last) const;
    };
    [[nodiscard]] std::uint64_t key(const cv::Vec3d& p, int dx = 0, int dy = 0, int dz = 0) const;
    [[nodiscard]] std::vector<std::vector<Near>> gather(std::span<const FiberStripSample> query, std::int64_t self,
                                                        const std::set<std::uint32_t>* excluded) const;
    [[nodiscard]] Contacts contacts(const std::vector<std::vector<Near>>& gathered) const;

    std::vector<std::vector<FiberStripSample>> samples_;
    std::vector<int> signs_;
    double cell_ = 0.;
    std::vector<std::size_t> begin_;
    std::vector<const FiberStripSample*> sample_;
    std::vector<std::uint32_t> owner_;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> keyed_;
    std::unordered_map<std::uint64_t, std::pair<std::size_t, std::size_t>> cells_;
};

// Signed soft targets of a strip solve from inferred normals (`inferred`
// ascending in arc, `normal` signed): per chain node the inferred normals are
// averaged with a Gaussian of `sigma` along the arc; the signed mean direction
// becomes the node's signedAxis with weight `weight` * min(1, summed Gaussian)
// * |mean| / summed Gaussian (agreement of the samples). Returns the number of
// nodes that got a target.
std::size_t setFiberNeighbourTargets(const FiberStripChain& chain, std::span<const FiberStripSample> inferred,
                                     double weight, double sigma, std::vector<FiberStripAxisTarget>& targets);

}  // namespace vc::fiber_tracer
