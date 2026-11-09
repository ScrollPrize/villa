#include "ink_labels/InkLabelAssist.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vc3d::ink_labels {

namespace {

// Index of the prediction pixel under grid coordinate `g` along one axis of
// `n` grid steps and `pixels` prediction pixels, flipped if asked.
int guideIndex(double g, int gridSteps, int pixels, bool flip)
{
    const double u = g * double(pixels) / double(std::max(1, gridSteps));
    int i = std::isfinite(u) ? int(std::floor(u)) : 0;
    i = std::clamp(i, 0, pixels - 1);
    return flip ? pixels - 1 - i : i;
}

cv::Mat_<uint8_t> thresholded(const cv::Mat_<uint8_t>& values, uint8_t threshold, bool above)
{
    cv::Mat_<uint8_t> out;
    cv::compare(values, cv::Scalar(threshold), out, above ? cv::CMP_GE : cv::CMP_LE);
    return out;
}

cv::Rect canvasRect(cv::Size size)
{
    return {0, 0, size.width, size.height};
}

cv::Point centerOf(const cv::Rect& r)
{
    return {r.x + r.width / 2, r.y + r.height / 2};
}

}  // namespace

uint8_t predictionAt(const PredictionGuide& guide, const vc::RenderCanvas& canvas, cv::Point2d labelPx)
{
    if (guide.scalar.empty())
        return 0;
    const cv::Vec2d g = canvas.canvasToGrid({labelPx.x, labelPx.y});
    const int i = guideIndex(g[0], canvas.grid.width, guide.scalar.cols, guide.hflip);
    const int j = guideIndex(g[1], canvas.grid.height, guide.scalar.rows, guide.vflip);
    return guide.scalar(j, i);
}

cv::Mat_<uint8_t> sampleGuide(const PredictionGuide& guide, const vc::RenderCanvas& canvas, const cv::Rect& labelRect)
{
    cv::Mat_<uint8_t> out(labelRect.size(), uint8_t(0));
    if (guide.scalar.empty() || labelRect.empty())
        return out;
    // canvasToGrid is separable, so one index per column and per row.
    std::vector<int> cols(std::size_t(labelRect.width)), rows(std::size_t(labelRect.height));
    for (int x = 0; x < labelRect.width; ++x) {
        const double g = canvas.canvasToGrid({labelRect.x + x + 0.5, 0.0})[0];
        cols[std::size_t(x)] = guideIndex(g, canvas.grid.width, guide.scalar.cols, guide.hflip);
    }
    for (int y = 0; y < labelRect.height; ++y) {
        const double g = canvas.canvasToGrid({0.0, labelRect.y + y + 0.5})[1];
        rows[std::size_t(y)] = guideIndex(g, canvas.grid.height, guide.scalar.rows, guide.vflip);
    }
    for (int y = 0; y < labelRect.height; ++y) {
        const uint8_t* src = guide.scalar.ptr(rows[std::size_t(y)]);
        uint8_t* dst = out.ptr(y);
        for (int x = 0; x < labelRect.width; ++x)
            dst[x] = src[cols[std::size_t(x)]];
    }
    return out;
}

cv::Rect assistBound(const cv::Rect& bound, cv::Size canvas, cv::Point center, int maxSize)
{
    cv::Rect r = bound & canvasRect(canvas);
    if (r.empty())
        return {};
    if (r.width > maxSize) {
        r.x = std::clamp(center.x - maxSize / 2, r.x, r.x + r.width - maxSize);
        r.width = maxSize;
    }
    if (r.height > maxSize) {
        r.y = std::clamp(center.y - maxSize / 2, r.y, r.y + r.height - maxSize);
        r.height = maxSize;
    }
    return r;
}

uint64_t MaskResult::count() const
{
    return mask.empty() ? 0 : uint64_t(cv::countNonZero(mask));
}

std::optional<MaskResult> floodRegion(const TileRaster& snapshot,
                                      const PredictionGuide* guide,
                                      GuideKind kind,
                                      const vc::RenderCanvas& canvas,
                                      cv::Point seed,
                                      const cv::Rect& bound,
                                      const FloodOptions& options)
{
    const cv::Rect full = canvasRect(snapshot.size());
    if (!full.contains(seed))
        return std::nullopt;
    const cv::Rect b = assistBound(bound.empty() ? full : bound, snapshot.size(), seed, kMaxAssistWindow);
    if (!b.contains(seed))
        return std::nullopt;
    const cv::Point local = seed - b.tl();

    cv::Mat_<uint8_t> values;
    snapshot.readRect(b, values);
    cv::Mat_<uint8_t> pass;
    if (kind == GuideKind::Prediction) {
        if (!guide || guide->empty())
            return std::nullopt;
        pass = thresholded(sampleGuide(*guide, canvas, b), options.threshold, options.above);
    } else {
        cv::compare(values, cv::Scalar(values(local)), pass, cv::CMP_EQ);
    }
    if (options.unlabeledOnly) {
        cv::Mat_<uint8_t> labeled;
        cv::bitwise_and(values, cv::Scalar(kInk | kSupervision), labeled);
        cv::Mat_<uint8_t> unlabeled;
        cv::compare(labeled, cv::Scalar(0), unlabeled, cv::CMP_EQ);
        cv::bitwise_and(pass, unlabeled, pass);
    }
    if (pass(local) == 0)
        return std::nullopt;

    cv::Mat_<uint8_t> fill = cv::Mat_<uint8_t>::zeros(b.height + 2, b.width + 2);
    const int connectivity = options.connectivity == 8 ? 8 : 4;
    cv::floodFill(pass, fill, local, cv::Scalar(255), nullptr, cv::Scalar(0), cv::Scalar(0),
                  connectivity | cv::FLOODFILL_MASK_ONLY | (255 << 8));
    MaskResult result;
    result.origin = b.tl();
    result.mask = fill(cv::Rect(1, 1, b.width, b.height)).clone();
    // Touching a side of the bound that is not a side of the canvas.
    const auto anyOn = [&result](const cv::Rect& line) { return cv::countNonZero(result.mask(line)) > 0; };
    result.reachedBound = (b.x > 0 && anyOn({0, 0, 1, b.height})) || (b.y > 0 && anyOn({0, 0, b.width, 1}))
                          || (b.br().x < full.width && anyOn({b.width - 1, 0, 1, b.height}))
                          || (b.br().y < full.height && anyOn({0, b.height - 1, b.width, 1}));
    return result;
}

void dropSupervised(const TileRaster& raster, MaskResult& mask)
{
    if (mask.mask.empty())
        return;
    cv::Mat_<uint8_t> values;
    raster.readRect(cv::Rect(mask.origin, mask.mask.size()), values);
    cv::Mat_<uint8_t> supervised;
    cv::bitwise_and(values, cv::Scalar(kSupervision), supervised);
    mask.mask.setTo(0, supervised);
}

int SuggestionSet::idAt(cv::Point labelPx) const
{
    const cv::Point local = labelPx - origin;
    if (labels.empty() || local.x < 0 || local.y < 0 || local.x >= labels.cols || local.y >= labels.rows)
        return 0;
    return labels(local);
}

MaskResult SuggestionSet::maskFor(const std::vector<int>& ids) const
{
    MaskResult result;
    result.origin = origin;
    result.mask = cv::Mat_<uint8_t>::zeros(labels.size());
    if (labels.empty() || ids.empty())
        return result;
    std::vector<uint8_t> chosen(items.size() + 1, 0);
    for (const int id : ids)
        if (id > 0 && std::size_t(id) < chosen.size())
            chosen[std::size_t(id)] = 255;
    for (int y = 0; y < labels.rows; ++y) {
        const int* src = labels[y];
        uint8_t* dst = result.mask.ptr(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int id = src[x];
            if (id > 0 && std::size_t(id) < chosen.size())
                dst[x] = chosen[std::size_t(id)];
        }
    }
    return result;
}

SuggestionSet suggestComponents(const PredictionGuide& guide,
                                const vc::RenderCanvas& canvas,
                                const cv::Rect& bound,
                                uint8_t threshold,
                                bool above,
                                int median,
                                int minArea,
                                const TileRaster* labeled)
{
    SuggestionSet set;
    set.threshold = threshold;
    set.above = above;
    set.median = median <= 1 ? 1 : (median >= 5 ? 5 : 3);
    set.minArea = std::max(0, minArea);
    set.source = guide.source;
    const cv::Rect full = canvasRect(canvas.size);
    const cv::Rect clipped = (bound.empty() ? full : bound) & full;
    const cv::Rect b = assistBound(clipped, canvas.size, centerOf(clipped), kMaxAssistWindow);
    set.origin = b.tl();
    if (b.empty() || guide.empty())
        return set;

    cv::Mat_<uint8_t> values = sampleGuide(guide, canvas, b);
    if (set.median > 1)
        cv::medianBlur(values, values, set.median);
    cv::Mat_<uint8_t> mask = thresholded(values, threshold, above);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    if (labeled) {
        MaskResult open{b.tl(), mask, false};
        dropSupervised(*labeled, open);
    }

    cv::Mat_<int> labels;
    cv::Mat stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    std::vector<int> remap(std::size_t(std::max(n, 1)), 0);
    int next = 1;
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < std::max(1, set.minArea))
            continue;
        remap[std::size_t(i)] = next;
        set.items.push_back({next,
                             area,
                             cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT) + b.x,
                                      stats.at<int>(i, cv::CC_STAT_TOP) + b.y,
                                      stats.at<int>(i, cv::CC_STAT_WIDTH),
                                      stats.at<int>(i, cv::CC_STAT_HEIGHT))});
        ++next;
    }
    for (int y = 0; y < labels.rows; ++y) {
        int* row = labels[y];
        for (int x = 0; x < labels.cols; ++x)
            row[x] = remap[std::size_t(row[x])];
    }
    set.labels = labels;
    return set;
}

uint64_t CleanupResult::count() const
{
    return mask.empty() ? 0 : uint64_t(cv::countNonZero(mask));
}

CleanupResult cleanup(const TileRaster& snapshot, const cv::Rect& bound, Cleanup kind, int n)
{
    CleanupResult result;
    const cv::Rect full = canvasRect(snapshot.size());
    const cv::Rect clipped = (bound.empty() ? full : bound) & full;
    const cv::Rect b = assistBound(clipped, snapshot.size(), centerOf(clipped), kMaxCleanupWindow);
    result.origin = b.tl();
    result.mask = cv::Mat_<uint8_t>::zeros(b.size());
    if (b.empty())
        return result;
    n = std::max(1, n);

    cv::Mat_<uint8_t> values;
    snapshot.readRect(b, values);
    cv::Mat_<uint8_t> ink;
    cv::bitwise_and(values, cv::Scalar(kInk), ink);
    cv::compare(ink, cv::Scalar(0), ink, cv::CMP_NE);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * n + 1, 2 * n + 1});

    const auto smallComponents = [&](const cv::Mat_<uint8_t>& image, int connectivity, bool skipBorder) {
        cv::Mat_<int> labels;
        cv::Mat stats, centroids;
        const int count = cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, CV_32S);
        std::vector<uint8_t> keep(std::size_t(std::max(count, 1)), 0);
        for (int i = 1; i < count; ++i) {
            const int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            const bool touches = x == 0 || y == 0 || x + w == image.cols || y + h == image.rows;
            if (stats.at<int>(i, cv::CC_STAT_AREA) < n && !(skipBorder && touches))
                keep[std::size_t(i)] = 255;
        }
        cv::Mat_<uint8_t> out(image.size());
        for (int y = 0; y < image.rows; ++y) {
            const int* src = labels[y];
            uint8_t* dst = out.ptr(y);
            for (int x = 0; x < image.cols; ++x)
                dst[x] = keep[std::size_t(src[x])];
        }
        return out;
    };

    switch (kind) {
        case Cleanup::RemoveSpecks:
            result.mask = smallComponents(ink, 8, false);
            result.op = {0, kInk};
            break;
        case Cleanup::FillHoles: {
            cv::Mat_<uint8_t> background;
            cv::bitwise_not(ink, background);
            result.mask = smallComponents(background, 4, true);
            result.op = {uint8_t(kInk | kSupervision), 0};
            break;
        }
        case Cleanup::Grow: {
            cv::Mat_<uint8_t> grown;
            cv::dilate(ink, grown, kernel);
            cv::Mat_<uint8_t> notInk;
            cv::bitwise_not(ink, notInk);
            cv::bitwise_and(grown, notInk, result.mask);
            result.op = {uint8_t(kInk | kSupervision), 0};
            break;
        }
        case Cleanup::Shrink: {
            cv::Mat_<uint8_t> eroded;
            cv::erode(ink, eroded, kernel);
            cv::Mat_<uint8_t> notEroded;
            cv::bitwise_not(eroded, notEroded);
            cv::bitwise_and(ink, notEroded, result.mask);
            result.op = {0, kInk};
            break;
        }
        case Cleanup::BackgroundRing: {
            cv::Mat_<uint8_t> grown;
            cv::dilate(ink, grown, kernel);
            cv::Mat_<uint8_t> notInk;
            cv::bitwise_not(ink, notInk);
            cv::Mat_<uint8_t> labeled;
            cv::bitwise_and(values, cv::Scalar(kInk | kSupervision), labeled);
            cv::Mat_<uint8_t> unlabeled;
            cv::compare(labeled, cv::Scalar(0), unlabeled, cv::CMP_EQ);
            cv::bitwise_and(grown, notInk, result.mask);
            cv::bitwise_and(result.mask, unlabeled, result.mask);
            result.op = {kSupervision, 0};
            break;
        }
    }
    return result;
}

const char* cleanupName(Cleanup kind)
{
    switch (kind) {
        case Cleanup::RemoveSpecks:
            return "remove specks";
        case Cleanup::FillHoles:
            return "fill holes";
        case Cleanup::Grow:
            return "grow";
        case Cleanup::Shrink:
            return "shrink";
        case Cleanup::BackgroundRing:
            return "background ring";
    }
    return "cleanup";
}

}  // namespace vc3d::ink_labels
