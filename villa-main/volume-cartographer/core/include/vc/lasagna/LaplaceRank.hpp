#pragma once

#include "vc/lasagna/MaxflowGraph.hpp"

#include <cstddef>
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

namespace vc::lasagna {

enum class LaplaceRankSide {
    SideA,
    SideB
};

struct LaplaceRankRoles {
    LaplaceRankSide solveSide = LaplaceRankSide::SideA;
    LaplaceRankSide targetSide = LaplaceRankSide::SideB;
    size_t solveCount = 0;
    size_t targetCount = 0;
};

struct LaplaceRankEvaluation {
    double lambda = 0.0;
    std::string status;
    bool solved = false;
    double solveSecondsTotal = 0.0;
    double maxAbsValue = 0.0;
    std::vector<double> values;
};

using LaplaceRankEvaluator = std::function<LaplaceRankEvaluation(double lambda)>;
using LaplaceRankProgressCallback = std::function<void(const nlohmann::json& event)>;

[[nodiscard]] const char* toString(LaplaceRankSide side) noexcept;

[[nodiscard]] LaplaceRankRoles selectLaplaceRankRoles(
    size_t sideACount,
    size_t sideBCount);

[[nodiscard]] bool laplaceRankAccepted(
    const LaplaceRankEvaluation& evaluation,
    double confidenceFloor);

[[nodiscard]] LaplaceRankEvaluation selectAdaptiveLaplaceRankLambda(
    double startLambda,
    double confidenceFloor,
    const LaplaceRankEvaluator& evaluator);

[[nodiscard]] nlohmann::json rankSnapPairsJson(
    const nlohmann::json& request,
    const LaplaceRankProgressCallback& progressCallback = {});

} // namespace vc::lasagna
