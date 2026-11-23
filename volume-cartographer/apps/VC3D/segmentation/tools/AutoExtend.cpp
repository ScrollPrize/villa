#include "AutoExtend.hpp"

#include "segmentation/SurfaceGridEdit.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>

namespace segmentation
{

const char* extendSideName(ManualAddTool::ExtendSide side)
{
    switch (side) {
    case ManualAddTool::ExtendSide::Up: return "up";
    case ManualAddTool::ExtendSide::Down: return "down";
    case ManualAddTool::ExtendSide::Left: return "left";
    case ManualAddTool::ExtendSide::Right: return "right";
    }
    return "right";
}

std::optional<ManualAddTool::ExtendSide> extendSideFromName(const std::string& name)
{
    for (auto side : {ManualAddTool::ExtendSide::Up, ManualAddTool::ExtendSide::Down,
                      ManualAddTool::ExtendSide::Left, ManualAddTool::ExtendSide::Right}) {
        if (name == extendSideName(side)) {
            return side;
        }
    }
    return std::nullopt;
}

std::optional<AutoExtendResult> autoExtendSurface(QuadSurface& surface,
                                                  const ManualAddTool::SheetNormalFn& normal,
                                                  const AutoExtendOptions& options,
                                                  std::string* status)
{
    auto fail = [&](const std::string& message) -> std::optional<AutoExtendResult> {
        if (status) {
            *status = message;
        }
        return std::nullopt;
    };
    if (options.cells < 1 || options.rounds < 1 || options.sides.empty()) {
        return fail("Nothing to extend: cells, rounds and sides must all be set.");
    }
    if (!surface.rawPointsPtr() || surface.rawPointsPtr()->empty()) {
        return fail("The surface has no points.");
    }

    ManualAddTool::Config config;
    config.sheetWalk = options.sheetWalk;
    config.interpolationMode = ManualAddTool::InterpolationMode::ThinPlateSpline;
    config.maxPreviewSpan = std::max(config.maxPreviewSpan, options.cells);
    const int threads = options.threads > 0
        ? options.threads
        : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    AutoExtendResult result;
    for (int round = 1; round <= options.rounds; ++round) {
        for (const auto side : options.sides) {
            AutoExtendStep step;
            step.side = side;
            step.round = round;

            const auto padding = ManualAddTool::paddingForExtension(*surface.rawPointsPtr(), side, options.cells);
            if (padding.any() &&
                !resizeSurfaceGrid(surface, padding.top, padding.bottom, padding.left, padding.right)) {
                return fail("Could not make room past the edge.");
            }
            ManualAddTool tool;
            std::string why;
            if (!tool.begin(*surface.rawPointsPtr(), config) ||
                !tool.beginExtension(side, options.cells, std::nullopt, &why)) {
                if (padding.any()) {
                    resizeSurfaceGrid(surface, -padding.top, -padding.bottom, -padding.left, -padding.right);
                }
                step.skipped = why.empty() ? "no edge" : why;
                result.steps.push_back(step);
                continue;
            }
            const auto walk = tool.sheetWalk();
            const auto& fill = tool.fillVertices();
            if (!walk || fill.size() != walk->origins.size() * static_cast<std::size_t>(walk->cells)) {
                return fail("The extension and its walk disagree in size.");
            }

            // Advance and redistribute one coherent edge at a time.
            std::vector<cv::Vec3f> guess(fill.size());
            std::atomic<int> missing{0};
            const ManualAddTool::SheetNormalFn counted = [&](const cv::Vec3f& world) {
                auto n = normal ? normal(world) : std::nullopt;
                if (!n || cv::norm(*n) <= 0.5) {
                    missing.fetch_add(1, std::memory_order_relaxed);
                }
                return n;
            };
            try {
                ManualAddTool::walkSheetExtension(*walk, fill, counted, guess, static_cast<unsigned>(threads));
            } catch (...) {
                // No guessed vertices have been published. Undo this step's
                // padding as well before returning the error to the CLI.
                if (padding.any()) {
                    resizeSurfaceGrid(surface, -padding.top, -padding.bottom, -padding.left, -padding.right);
                }
                try {
                    throw;
                } catch (const std::exception& e) {
                    return fail(std::string("Sheet following failed: ") + e.what());
                } catch (...) {
                    return fail("Sheet following failed: unknown error");
                }
            }

            auto* points = surface.rawPointsPtr();
            std::vector<cv::Point> cells;
            cells.reserve(fill.size());
            for (std::size_t i = 0; i < fill.size(); ++i) {
                (*points)(fill[i].row, fill[i].col) = guess[i];
                cells.emplace_back(fill[i].col, fill[i].row);
            }
            markMaskCellsValid(surface, cells);
            surface.invalidateCache();

            step.lines = static_cast<int>(walk->origins.size());
            step.added = static_cast<int>(fill.size());
            step.stepsWithoutNormal = missing.load();
            result.added += step.added;
            result.steps.push_back(step);
        }
    }
    // saveAsNew writes only loaded channels. Resizing loads them already,
    // but extensions within existing empty borders must preserve them too.
    for (const auto& name : surface.channelNames()) {
        surface.channel(name, SURF_CHANNEL_NORESIZE);
    }
    return result;
}

} // namespace segmentation
