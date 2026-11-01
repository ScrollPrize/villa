#pragma once

#include "ManualAddTool.hpp"

#include <optional>
#include <string>
#include <vector>

class QuadSurface;

namespace segmentation
{

// Grows a surface past its edges without a user: each round extends every
// row or column of each chosen side by `cells` vertices with Manual Add's
// edge extension, walks every new line along the sheet normals (as
// "Extensions follow the sheet" does in the GUI) and keeps the result.
// Rounds start from the edge the previous round left. The grid is padded as
// needed and every channel stays aligned; cells under a mask are marked valid
// so the new vertices survive a reload.
struct AutoExtendOptions
{
    std::vector<ManualAddTool::ExtendSide> sides{ManualAddTool::ExtendSide::Up,
                                                 ManualAddTool::ExtendSide::Down,
                                                 ManualAddTool::ExtendSide::Left,
                                                 ManualAddTool::ExtendSide::Right};
    int cells{10};
    int rounds{8};
    // Worker threads for the walk; 0 uses the hardware concurrency.
    int threads{0};
};

struct AutoExtendStep
{
    ManualAddTool::ExtendSide side{ManualAddTool::ExtendSide::Right};
    int round{0};
    int lines{0};
    int added{0};
    // Walk steps with no usable normal; those steps continue straight.
    int stepsWithoutNormal{0};
    // Set when the side had no edge to extend this round.
    std::string skipped;
};

struct AutoExtendResult
{
    std::vector<AutoExtendStep> steps;
    int added{0};
};

std::optional<AutoExtendResult> autoExtendSurface(QuadSurface& surface,
                                                  const ManualAddTool::SheetNormalFn& normal,
                                                  const AutoExtendOptions& options,
                                                  std::string* status = nullptr);

const char* extendSideName(ManualAddTool::ExtendSide side);
std::optional<ManualAddTool::ExtendSide> extendSideFromName(const std::string& name);

} // namespace segmentation
