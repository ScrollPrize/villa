#include "FiberMapKollesis.hpp"

#include <algorithm>
#include <cmath>

namespace vc3d::fiber_map::kollesis
{

namespace
{

// The seed for the unit sheet length: a low percentile of the gaps rather
// than the minimum, so one spurious short gap (two seams a grouping gap
// apart that are really one) does not make every real sheet read as many.
constexpr double kSeedPercentile = 0.25;
// The refinement stops once the sheet counts stop changing; this many rounds
// without settling, or any gap reading more than kMaxSheetsPerGap sheets (a
// unit that is a fraction of a real sheet, from a spurious short gap), means
// the estimate is unusable and no statistics are reported.
constexpr int kMaxRefineIterations = 32;
constexpr double kMaxSheetsPerGap = 100.0;

}  // namespace

double Seam::xVx() const
{
    if (left && right) {
        return 0.5 * (left->xVx + right->xVx);
    }
    if (left) {
        return left->xVx;
    }
    if (right) {
        return right->xVx;
    }
    return 0.0;
}

Bound fitBound(const std::vector<Termination>& points)
{
    Bound bound;
    bound.count = static_cast<int>(points.size());
    if (points.empty()) {
        return bound;
    }
    double sum = 0.0;
    for (const Termination& point : points) {
        sum += point.xSceneVx;
    }
    bound.xVx = sum / static_cast<double>(points.size());
    return bound;
}

SheetStatistics estimateSheetStatistics(const std::vector<double>& gapsVx)
{
    SheetStatistics stats;
    stats.sheetCounts.assign(gapsVx.size(), 1);
    std::vector<double> positive;
    for (const double gap : gapsVx) {
        if (std::isfinite(gap) && gap > 0.0) {
            positive.push_back(gap);
        }
    }
    if (positive.empty()) {
        return stats;
    }
    std::sort(positive.begin(), positive.end());
    const auto seedIndex = static_cast<std::size_t>(
        std::floor(kSeedPercentile * static_cast<double>(positive.size() - 1)));
    double unit = positive[seedIndex];
    // Assign each gap its sheet count at the current unit, then re-estimate
    // the unit as the per-sheet mean, until the counts stop changing: the
    // unit is then the mean at those counts and the counts are the rounding
    // at that unit, so the predictions a gap gets agree with the unit shown.
    bool settled = false;
    std::vector<int> next(gapsVx.size(), 1);
    for (int iteration = 0; iteration < kMaxRefineIterations && unit > 0.0; ++iteration) {
        double totalLength = 0.0;
        long long totalSheets = 0;
        for (std::size_t i = 0; i < gapsVx.size(); ++i) {
            const double gap = gapsVx[i];
            next[i] = 1;
            if (!std::isfinite(gap) || gap <= 0.0) {
                continue;
            }
            const double ratio = gap / unit;
            if (!(ratio <= kMaxSheetsPerGap)) {
                // Unusable: a fraction of a sheet is passing for the unit.
                return SheetStatistics{std::vector<int>(gapsVx.size(), 1), 0.0, 0.0};
            }
            next[i] = std::max(1, static_cast<int>(std::lround(ratio)));
            totalLength += gap;
            totalSheets += next[i];
        }
        if (totalSheets <= 0) {
            break;
        }
        settled = next == stats.sheetCounts;
        stats.sheetCounts = next;
        unit = totalLength / static_cast<double>(totalSheets);
        if (settled) {
            break;
        }
    }
    if (!settled) {
        return SheetStatistics{std::vector<int>(gapsVx.size(), 1), 0.0, 0.0};
    }
    stats.unitLengthVx = unit;
    // Spread: the sample deviation of each gap's per-sheet length.
    std::vector<double> perSheet;
    for (std::size_t i = 0; i < gapsVx.size(); ++i) {
        const double gap = gapsVx[i];
        if (std::isfinite(gap) && gap > 0.0) {
            perSheet.push_back(gap / static_cast<double>(stats.sheetCounts[i]));
        }
    }
    if (perSheet.size() >= 2) {
        double mean = 0.0;
        for (const double value : perSheet) {
            mean += value;
        }
        mean /= static_cast<double>(perSheet.size());
        double variance = 0.0;
        for (const double value : perSheet) {
            variance += (value - mean) * (value - mean);
        }
        variance /= static_cast<double>(perSheet.size() - 1);
        stats.spreadVx = std::sqrt(variance);
    }
    return stats;
}

Model buildModel(std::vector<Termination> terminations, const Params& params)
{
    Model model;
    terminations.erase(
        std::remove_if(terminations.begin(), terminations.end(),
                       [](const Termination& t) {
                           return !std::isfinite(t.xMapVx) || !std::isfinite(t.xSceneVx);
                       }),
        terminations.end());
    if (terminations.empty()) {
        return model;
    }
    std::sort(terminations.begin(), terminations.end(),
              [](const Termination& a, const Termination& b) {
                  if (a.xMapVx != b.xMapVx) {
                      return a.xMapVx < b.xMapVx;
                  }
                  if (a.fiberId != b.fiberId) {
                      return a.fiberId < b.fiberId;
                  }
                  return a.controlIndex < b.controlIndex;
              });

    // Groups: a run of terminations where each is within the gap of the
    // previous one along the map.
    const double gapVx = params.windingWidthVx > 0.0
                             ? params.gapWindings * params.windingWidthVx
                             : 0.0;
    std::vector<std::vector<Termination>> groups;
    for (const Termination& t : terminations) {
        if (groups.empty() || t.xMapVx - groups.back().back().xMapVx > gapVx) {
            groups.emplace_back();
        }
        groups.back().push_back(t);
    }

    model.seams.reserve(groups.size());
    for (std::vector<Termination>& group : groups) {
        std::vector<Termination> lefts;
        std::vector<Termination> rights;
        for (const Termination& t : group) {
            (t.left ? lefts : rights).push_back(t);
        }
        Seam seam;
        if (!lefts.empty()) {
            seam.left = fitBound(lefts);
        }
        if (!rights.empty()) {
            seam.right = fitBound(rights);
        }
        seam.members = std::move(group);
        model.seams.push_back(std::move(seam));
    }
    std::stable_sort(model.seams.begin(), model.seams.end(),
                     [](const Seam& a, const Seam& b) { return a.xVx() < b.xVx(); });
    if (model.seams.size() >= 2) {
        model.sheetLengthVx.reserve(model.seams.size() - 1);
        for (std::size_t i = 0; i + 1 < model.seams.size(); ++i) {
            model.sheetLengthVx.push_back(model.seams[i + 1].xVx() - model.seams[i].xVx());
        }
    }

    model.statistics = estimateSheetStatistics(model.sheetLengthVx);
    if (!model.statistics.valid()) {
        return model;
    }
    const SheetStatistics& stats = model.statistics;
    if (params.predictInterior) {
        for (std::size_t i = 0; i < model.sheetLengthVx.size(); ++i) {
            const int count = stats.sheetCounts[i];
            if (count < 2) {
                continue;
            }
            // Spaced evenly over the measured gap: its ends are known, so
            // the unit length only decides how many sheets fit.
            const double start = model.seams[i].xVx();
            const double perSheet = model.sheetLengthVx[i] / static_cast<double>(count);
            for (int step = 1; step < count; ++step) {
                PredictedSeam seam;
                seam.xVx = start + perSheet * static_cast<double>(step);
                seam.spreadVx = stats.spreadVx;
                seam.extrapolated = false;
                seam.gapIndex = static_cast<int>(i);
                seam.step = step;
                model.predicted.push_back(seam);
            }
        }
    }
    const double last = model.seams.back().xVx();
    for (int step = 1; step <= params.extrapolateCount; ++step) {
        PredictedSeam seam;
        seam.xVx = last + stats.unitLengthVx * static_cast<double>(step);
        // Each sheet stepped adds its own deviation: the spread grows with
        // the square root of the count.
        seam.spreadVx = stats.spreadVx * std::sqrt(static_cast<double>(step));
        seam.extrapolated = true;
        seam.gapIndex = -1;
        seam.step = step;
        model.predicted.push_back(seam);
    }
    return model;
}

}  // namespace vc3d::fiber_map::kollesis
