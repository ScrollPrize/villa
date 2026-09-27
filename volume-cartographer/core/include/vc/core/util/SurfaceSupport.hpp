#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

#include <opencv2/core.hpp>

namespace vc::surface
{
struct OnPredictionSupport {
    // Fraction of valid vertices landing on nonzero prediction voxels.
    double fraction = 1.0;
    std::size_t on = 0;
    std::size_t total = 0;
};

// A min_on_prediction_support threshold is valid only when it is a finite
// fraction in [0, 1]. Anything else is a configuration error: fail fast instead
// of silently disabling the check (negative or NaN can never trigger) or
// rejecting every surface (above 1). See #1675.
inline bool validSupportThreshold(double t)
{
    return std::isfinite(t) && t >= 0.0 && t <= 1.0;
}

// A vertex is valid when no component is the -1 invalid sentinel and all
// components are finite (std::lround on NaN/Inf is undefined behavior).
inline bool isValidVertex(const cv::Vec3f& p)
{
    return p[0] != -1.f && p[1] != -1.f && p[2] != -1.f && std::isfinite(p[0]) &&
           std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Axis-aligned bounds of the valid vertices in points (x, y, z order).
// Returns false when there are no valid vertices.
inline bool validVertexBounds(const cv::Mat_<cv::Vec3f>& points, cv::Vec3f& lo, cv::Vec3f& hi)
{
    bool any = false;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f p = points(r, c);
            if (!isValidVertex(p)) {
                continue;
            }
            if (!any) {
                lo = hi = p;
                any = true;
            } else {
                for (int i = 0; i < 3; ++i) {
                    lo[i] = std::min(lo[i], p[i]);
                    hi[i] = std::max(hi[i], p[i]);
                }
            }
        }
    }
    return any;
}

// True when the grown surface tracks the prediction no better than chance:
// its on-prediction fraction is no more than `margin` above the background
// rate measured on uniform random points in the surface's neighborhood. Absolute
// thresholds cannot catch this on dense predictions, where even a random
// surface scores high. The margin is needed because real bad surfaces can sit
// above the raw background rate: on 65 eye-labelled patches from 21 scrolls,
// bad (swirl-only) patches scored 25.9-34.8% against a ~20% background, while
// good (sheet-following) patches scored 44.8-62.7% -- a +11 to +23 point
// margin above background separates them, but a flat at-or-below-background
// comparison flags only 3/21 bad patches.
//
// A background with total == 0 was never measured (e.g. the surface had no
// valid vertices, so there was no neighborhood to sample); without a measured
// background there is nothing to compare against, so this returns false
// rather than comparing two default 1.0 fractions.
inline bool noBetterThanChance(const OnPredictionSupport& support,
                               const OnPredictionSupport& background,
                               double margin)
{
    return background.total != 0 && support.fraction <= background.fraction + margin;
}

// Outcome of the post-growth on-prediction acceptance decision. Accept means
// the surface passes and is saved; Warn means it should be reported but still
// saved; Reject means strict mode requires discarding it (exit non-zero).
enum class SupportVerdict { Accept, Warn, Reject };

// Decides what to do with a grown surface. A rejected check is advisory by
// default (Warn: report and keep the surface) and discards the surface only
// in strict mode. The two warning messages are printed by the caller; this is
// the pure branch logic, extracted so warn-vs-strict behavior is unit
// testable (see #1675).
inline SupportVerdict supportVerdict(bool rejected, bool strict)
{
    if (!rejected) {
        return SupportVerdict::Accept;
    }
    return strict ? SupportVerdict::Reject : SupportVerdict::Warn;
}

// A sampling failure leaves the surface unverified: strict mode treats that
// like a rejected check (discard it), default mode keeps it.
inline SupportVerdict samplingFailureVerdict(bool strict)
{
    return supportVerdict(/*rejected=*/true, strict);
}

// Strict-mode cleanup must never delete the target directory when
// --segment-name is used: then seg_dir IS the shared target directory, which
// the tool did not create and which may hold pre-existing segments. Only the
// default layout (a fresh per-run subfolder) may be removed.
inline bool strictCleanupMayDeleteSegDir(const std::string& segment_name)
{
    return segment_name.empty();
}

// A freshly claimed per-run segment directory. created is true only when this
// call atomically created dir; only then may a later cleanup delete it (a
// directory this run did not create may belong to a concurrent run). failed
// is true when every candidate name collided with an existing directory: the
// returned dir may belong to another run, so the caller must abort instead
// of saving into it (see #1906 review).
struct ClaimedRunDir {
    std::string name;
    std::filesystem::path dir;
    bool created = false;
    bool failed = false;
};

// Claim a fresh per-run segment directory under tgt_dir: atomically create
// tgt_dir/base_name, appending a numeric suffix while the name is taken. The
// default segment name is a millisecond-resolution timestamp, so two runs
// started in the same millisecond would otherwise share one directory, and a
// strict-mode rejection in one run could then delete the other run's
// snapshots. The atomic create_directory loop makes each run own a distinct
// directory instead. Never throws; when the directory cannot be created at
// all (e.g. a missing parent), the claim is returned uncreated and the
// caller keeps the old lazy behavior, except that cleanup will not delete a
// directory it does not own. When every candidate collides, failed is set
// and the fallback path must not be used: writing into it could overwrite
// another run's output.
inline ClaimedRunDir claimFreshRunDir(const std::filesystem::path& tgt_dir,
                                      const std::string& base_name,
                                      int max_attempts = 1000)
{
    ClaimedRunDir claim{base_name, tgt_dir / base_name, false, false};
    std::error_code ec;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (attempt > 0) {
            claim.name = base_name + "_" + std::to_string(attempt);
            claim.dir = tgt_dir / claim.name;
        }
        ec.clear();
        if (std::filesystem::create_directory(claim.dir, ec)) {
            claim.created = true;
            return claim;
        }
        if (ec) {
            // Not a name collision (e.g. missing parent or no permission):
            // leave downstream code to behave as before, unclaimed.
            claim.name = base_name;
            claim.dir = tgt_dir / base_name;
            return claim;
        }
        // The directory already existed: another (possibly concurrent) run
        // owns this name, so try the next suffix.
    }
    // Exhausted: every candidate was already taken, so the base directory
    // may belong to another run. Mark the claim failed so the caller aborts
    // before tracing or saving instead of falling back to a possibly
    // occupied path.
    claim.name = base_name;
    claim.dir = tgt_dir / base_name;
    claim.failed = true;
    return claim;
}

// Outcome of attempting rejection cleanup of a segment directory.
enum class SegDirCleanup { Deleted, Missing, Skipped, DeleteFailed };

// Delete seg_dir for a rejected surface, honoring the shared-directory guard:
// with --segment-name the tool writes into the shared target directory and
// must never remove it. Only pass owns_dir=true when this run atomically
// created seg_dir (see claimFreshRunDir); a directory owned by someone else
// (e.g. a concurrent run that won the same timestamp) is left alone.
// Never throws; a failed deletion is reported through the returned outcome
// so the caller can warn instead of escaping as an uncaught exception.
inline SegDirCleanup tryDiscardSegDir(const std::string& segment_name,
                                      const std::filesystem::path& seg_dir,
                                      bool owns_dir)
{
    if (!strictCleanupMayDeleteSegDir(segment_name) || !owns_dir) {
        return SegDirCleanup::Skipped;
    }
    std::error_code ec;
    if (!std::filesystem::exists(seg_dir, ec) || ec) {
        return ec ? SegDirCleanup::DeleteFailed : SegDirCleanup::Missing;
    }
    std::filesystem::remove_all(seg_dir, ec);
    return ec ? SegDirCleanup::DeleteFailed : SegDirCleanup::Deleted;
}

// Upper bound for uniform background sampling on one axis. The caller passes
// the dimension length and the (dilated, volume-clamped) neighborhood edge;
// the returned value is the largest sampling coordinate whose rounded voxel
// index stays in range: dimension length minus half a voxel. Coordinates in the
// last half-voxel would round to `shape`, fall outside the volume, and be
// counted as off-prediction, biasing the background rate downward near volume
// edges (for a one-voxel dimension, roughly half the samples). The caller
// treats hi <= lo as "nothing to sample" (covers empty volumes).
inline float backgroundSampleHiBound(int dim_length, float neighborhood_hi)
{
    return std::min(static_cast<float>(dim_length) - 0.5f, neighborhood_hi);
}

// Fraction of valid surface vertices that land on nonzero voxels of the
// prediction the surface was grown from, sampled nearest-neighbor in the
// prediction's native voxel frame.
//
// points holds (x, y, z) vertex positions; a vertex with any component == -1
// or non-finite is invalid and skipped. sample(z, y, x) returns the prediction
// voxel value at integer coordinates; the caller is responsible for bounds
// handling (the helper calls it for every rounded valid vertex).
//
// fraction is 1.0 when there are no valid vertices: an empty surface has
// nothing to judge. See #1675: on synthetic data, grown surfaces that cut
// across windings instead of following a sheet land at ~6-10%
// (indistinguishable from random points in the volume), while true sheets
// land at ~100%. On 65 eye-labelled real patches from 21 scrolls, bad
// (swirl-only) patches scored 25.9-34.8% and good (sheet-following) patches
// 44.8-62.7%, so the default 0.4 threshold separates the observed real sample
// (good n=5 is small; treat the default as a starting point, not a proven
// separator).
template <typename Sampler>
OnPredictionSupport onPredictionSupport(const cv::Mat_<cv::Vec3f>& points, Sampler&& sample)
{
    OnPredictionSupport res;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f p = points(r, c);
            // -1 marks invalid vertices; non-finite coordinates must also be
            // skipped before std::lround (which is UB on NaN/Inf), matching
            // QuadSurface's isValidPointSample.
            if (!isValidVertex(p)) {
                continue;
            }
            ++res.total;
            const int z = static_cast<int>(std::lround(p[2]));
            const int y = static_cast<int>(std::lround(p[1]));
            const int x = static_cast<int>(std::lround(p[0]));
            if (sample(z, y, x) != 0) {
                ++res.on;
            }
        }
    }
    if (res.total != 0) {
        res.fraction = static_cast<double>(res.on) / static_cast<double>(res.total);
    }
    return res;
}
}  // namespace vc::surface
