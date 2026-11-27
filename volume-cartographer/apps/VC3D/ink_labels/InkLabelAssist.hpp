#pragma once

// Assisted ink labeling: the ink prediction as a guide, flood fill, threshold
// suggestions and mask cleanup. GUI-free; every function reads only its
// arguments (a raster snapshot, a copy of the prediction), so the controller
// runs them on worker threads and applies the result as one mask operation.

#include "ink_labels/InkLabelRaster.hpp"

#include "vc/core/util/RenderCanvas.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vc3d::ink_labels {

// Largest region a fill or a suggestion pass looks at, and a cleanup pass.
inline constexpr int kMaxAssistWindow = 4096;
inline constexpr int kMaxCleanupWindow = 8192;

// The selected ink prediction of the Ink Detection dock, as the overlay draws
// it: one 8-bit value per prediction pixel (other depths min-max scaled to
// 0..255), stretched over the whole segment grid with its corner at grid
// (0, 0), then flipped.
struct PredictionGuide {
    cv::Mat_<uint8_t> scalar;
    bool hflip = false, vflip = false;
    std::string source;
    [[nodiscard]] bool empty() const { return scalar.empty(); }
};

// The guide's value under a label pixel position.
[[nodiscard]] uint8_t predictionAt(const PredictionGuide& guide, const vc::RenderCanvas& canvas, cv::Point2d labelPx);
// The guide sampled at the center of every label pixel of `labelRect`
// (which must lie inside the canvas).
[[nodiscard]] cv::Mat_<uint8_t> sampleGuide(const PredictionGuide& guide, const vc::RenderCanvas& canvas, const cv::Rect& labelRect);

// `bound` cut to the canvas and to a `maxSize` window centered on `center`.
[[nodiscard]] cv::Rect assistBound(const cv::Rect& bound, cv::Size canvas, cv::Point center, int maxSize);

enum class GuideKind : uint8_t { Prediction, Labels };

struct FloodOptions {
    int connectivity = 4;
    uint8_t threshold = 128;
    // Prediction guide: pass where value >= threshold (above) or <= threshold.
    bool above = true;
    // Also require (v & (ink | supervision)) == 0.
    bool unlabeledOnly = false;
};

struct MaskResult {
    cv::Point origin;
    cv::Mat_<uint8_t> mask;  // 255 inside the region
    // A fill reached the edge of its bound where the bound is not the
    // canvas edge: the region may go on outside the view.
    bool reachedBound = false;
    [[nodiscard]] uint64_t count() const;
};

// The connected region of passable pixels around `seed`, inside `bound` (cut
// to kMaxAssistWindow around the seed). Prediction guide: passable where the
// prediction is above (or below) the threshold. Labels guide: passable where
// the label value equals the seed's. nullopt if the seed is not passable or
// lies outside the bound.
[[nodiscard]] std::optional<MaskResult> floodRegion(const TileRaster& snapshot,
                                                    const PredictionGuide* guide,
                                                    GuideKind kind,
                                                    const vc::RenderCanvas& canvas,
                                                    cv::Point seed,
                                                    const cv::Rect& bound,
                                                    const FloodOptions& options);

struct Suggestion {
    int id = 0;
    int area = 0;
    cv::Rect bbox;  // label pixels
};

struct SuggestionSet {
    cv::Point origin;
    cv::Mat_<int> labels;  // 0 outside every kept component, else its id (1..n)
    std::vector<Suggestion> items;
    uint8_t threshold = 128;
    bool above = true;
    int median = 1;
    int minArea = 0;
    std::string source;
    [[nodiscard]] bool empty() const { return items.empty(); }
    // The id under a label pixel, or 0.
    [[nodiscard]] int idAt(cv::Point labelPx) const;
    // 255 where a component in `ids` is.
    [[nodiscard]] MaskResult maskFor(const std::vector<int>& ids) const;
};

// Connected components of the thresholded prediction inside `bound` (cut to
// kMaxAssistWindow around its center): median blur with an odd kernel
// `median` (1 = none), threshold, a 3 x 3 elliptical opening, then
// 8-connected components of at least `minArea` pixels. With `labeled`, pixels
// already labeled ink or background are left out before the components are
// found, so a suggestion never covers (and accepting never overwrites) what
// the annotator decided.
[[nodiscard]] SuggestionSet suggestComponents(const PredictionGuide& guide,
                                              const vc::RenderCanvas& canvas,
                                              const cv::Rect& bound,
                                              uint8_t threshold,
                                              bool above,
                                              int median,
                                              int minArea,
                                              const TileRaster* labeled = nullptr);
// Clears `mask` where the raster is already supervised (ink or background).
void dropSupervised(const TileRaster& raster, MaskResult& mask);

enum class Cleanup : uint8_t { RemoveSpecks, FillHoles, Grow, Shrink, BackgroundRing };

struct CleanupResult {
    cv::Point origin;
    cv::Mat_<uint8_t> mask;
    PaintOp op;
    [[nodiscard]] uint64_t count() const;
};

// One cleanup over `bound` (cut to kMaxCleanupWindow around its center) with
// size `n` px. With R the label values and I = R & ink, K the ellipse of size
// 2n + 1:
//   RemoveSpecks: 8-connected components of I under n px lose ink (stay supervised).
//   FillHoles: 4-connected holes of I under n px that do not touch the bound get ink.
//   Grow: dilate(I, K) minus I gets ink.
//   Shrink: I minus erode(I, K) loses ink (stays supervised).
//   BackgroundRing: dilate(I, K) minus I, where unlabeled, becomes background.
[[nodiscard]] CleanupResult cleanup(const TileRaster& snapshot, const cv::Rect& bound, Cleanup kind, int n);

[[nodiscard]] const char* cleanupName(Cleanup kind);

}  // namespace vc3d::ink_labels
