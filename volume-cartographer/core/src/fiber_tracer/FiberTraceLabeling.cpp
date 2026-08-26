#include "vc/fiber_tracer/FiberTraceLabeling.hpp"

#include "vc/core/io/PolylineObj.hpp"

#include <Highs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kBinaryTolerance = 1.0e-6;

std::size_t labelIndex(FiberTracePieceLabel label)
{
    return static_cast<std::size_t>(label);
}

bool isBroken(FiberTracePieceLabel label)
{
    return label == FiberTracePieceLabel::Broken;
}

bool isVertical(FiberTracePieceLabel label)
{
    return label == FiberTracePieceLabel::VEven ||
        label == FiberTracePieceLabel::VOdd;
}

bool isOdd(FiberTracePieceLabel label)
{
    return label == FiberTracePieceLabel::HOdd ||
        label == FiberTracePieceLabel::VOdd;
}

FiberTracePieceLabel makeLabel(bool vertical, bool odd)
{
    if (vertical)
        return odd ? FiberTracePieceLabel::VOdd : FiberTracePieceLabel::VEven;
    return odd ? FiberTracePieceLabel::HOdd : FiberTracePieceLabel::HEven;
}

void requireOk(HighsStatus status, const char* operation)
{
    if (status != HighsStatus::kOk)
        throw std::runtime_error(std::string("HiGHS failed to ") + operation);
}

HighsInt checkedHighsInt(std::size_t value, const char* description)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<HighsInt>::max()))
        throw std::overflow_error(std::string(description) + " exceeds HiGHS index range");
    return static_cast<HighsInt>(value);
}

bool binaryValue(const std::vector<double>& values, std::size_t index)
{
    if (index >= values.size() || !std::isfinite(values[index]))
        throw std::runtime_error("HiGHS returned an invalid labeling solution");
    if (std::abs(values[index]) <= kBinaryTolerance)
        return false;
    if (std::abs(values[index] - 1.0) <= kBinaryTolerance)
        return true;
    throw std::runtime_error("HiGHS returned a non-binary labeling solution");
}

struct RowBuilder {
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<HighsInt> starts{0};
    std::vector<HighsInt> indices;
    std::vector<double> values;

    void add(
        double rowLower,
        double rowUpper,
        std::initializer_list<std::pair<std::size_t, double>> entries)
    {
        lower.push_back(rowLower);
        upper.push_back(rowUpper);
        for (const auto& [index, value] : entries) {
            indices.push_back(checkedHighsInt(index, "MILP column index"));
            values.push_back(value);
        }
        starts.push_back(checkedHighsInt(indices.size(), "MILP nonzero count"));
    }
};

void canonicalizeLabels(
    std::vector<FiberTracePieceLabel>& labels,
    const FiberTraceConstraintReport& constraints,
    const std::vector<std::size_t>& degree)
{
    const std::size_t count = labels.size();
    std::vector<std::vector<std::size_t>> adjacency(count);
    for (const auto& constraint : constraints.constraints) {
        if (isBroken(labels[constraint.pieceA]) ||
            isBroken(labels[constraint.pieceB])) {
            continue;
        }
        adjacency[constraint.pieceA].push_back(constraint.pieceB);
        adjacency[constraint.pieceB].push_back(constraint.pieceA);
    }
    for (auto& neighbors : adjacency)
        std::sort(neighbors.begin(), neighbors.end());

    std::vector<unsigned char> visited(count, 0);
    for (std::size_t seed = 0; seed < count; ++seed) {
        if (degree[seed] == 0) {
            labels[seed] = FiberTracePieceLabel::Broken;
            visited[seed] = 1;
            continue;
        }
        if (visited[seed] || isBroken(labels[seed]))
            continue;
        const bool flipVertical = isVertical(labels[seed]);
        const bool flipOdd = isOdd(labels[seed]);
        std::queue<std::size_t> pending;
        pending.push(seed);
        visited[seed] = 1;
        while (!pending.empty()) {
            const std::size_t piece = pending.front();
            pending.pop();
            labels[piece] = makeLabel(
                isVertical(labels[piece]) != flipVertical,
                isOdd(labels[piece]) != flipOdd);
            for (const std::size_t neighbor : adjacency[piece]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
    }
}

std::string outputStem(const std::filesystem::path& outputBase)
{
    const std::string stem = outputBase.has_extension()
        ? outputBase.stem().string()
        : outputBase.filename().string();
    if (stem.empty())
        throw std::invalid_argument("label OBJ output basename is empty");
    return stem;
}

}  // namespace

FiberTraceLabelingReport solveFiberTraceLabels(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingConfig& config)
{
    if (!std::isfinite(config.brokenCostPerConstraint) ||
        config.brokenCostPerConstraint < 0.0) {
        throw std::invalid_argument(
            "Fiber trace broken cost per constraint must be finite and nonnegative");
    }
    if (!std::isfinite(config.relativeMipGap) || config.relativeMipGap < 0.0) {
        throw std::invalid_argument(
            "Fiber trace labeling relative MIP gap must be finite and nonnegative");
    }

    const std::size_t pieceCount = constraints.pieces.size();
    const std::size_t edgeCount = constraints.constraints.size();
    FiberTraceLabelingReport report;
    report.labels.assign(pieceCount, FiberTracePieceLabel::Broken);
    if (pieceCount == 0) {
        report.modelStatus = "Empty";
        return report;
    }

    std::vector<std::size_t> degree(pieceCount, 0);
    for (const auto& constraint : constraints.constraints) {
        if (constraint.pieceA >= pieceCount || constraint.pieceB >= pieceCount ||
            constraint.pieceA == constraint.pieceB) {
            throw std::invalid_argument(
                "Fiber trace constraint references an invalid piece pair");
        }
        if (!std::isfinite(constraint.parallelScore) ||
            constraint.parallelScore < 0.0 || constraint.parallelScore > 1.0 ||
            !std::isfinite(constraint.windingDistance) ||
            constraint.windingDistance < 0.0 ||
            constraint.windingDistance >= 1.5) {
            throw std::invalid_argument(
                "Fiber trace constraint contains invalid optimization scores");
        }
        ++degree[constraint.pieceA];
        ++degree[constraint.pieceB];
    }

    const std::size_t activeBase = 0;
    const std::size_t verticalBase = pieceCount;
    const std::size_t oddBase = 2 * pieceCount;
    const std::size_t pairBase = 3 * pieceCount;
    const std::size_t verticalDifferenceBase = pairBase + edgeCount;
    const std::size_t oddDifferenceBase = verticalDifferenceBase + edgeCount;
    const std::size_t variableCount = oddDifferenceBase + edgeCount;

    HighsModel model;
    auto& lp = model.lp_;
    lp.num_col_ = checkedHighsInt(variableCount, "MILP variable count");
    lp.col_cost_.assign(variableCount, 0.0);
    lp.col_lower_.assign(variableCount, 0.0);
    lp.col_upper_.assign(variableCount, 1.0);
    lp.integrality_.assign(variableCount, HighsVarType::kContinuous);
    std::fill_n(
        lp.integrality_.begin(), 3 * pieceCount, HighsVarType::kInteger);
    lp.offset_ = 0.0;
    lp.sense_ = ObjSense::kMinimize;

    RowBuilder rows;
    rows.lower.reserve(2 * pieceCount + 7 * edgeCount);
    rows.upper.reserve(2 * pieceCount + 7 * edgeCount);
    rows.starts.reserve(2 * pieceCount + 7 * edgeCount + 1);
    rows.indices.reserve(4 * pieceCount + 28 * edgeCount);
    rows.values.reserve(4 * pieceCount + 28 * edgeCount);

    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const double penalty = config.brokenCostPerConstraint *
            static_cast<double>(degree[piece]);
        lp.offset_ += penalty;
        lp.col_cost_[activeBase + piece] = -penalty;
        rows.add(-kHighsInf, 0.0, {
            {verticalBase + piece, 1.0}, {activeBase + piece, -1.0}});
        rows.add(-kHighsInf, 0.0, {
            {oddBase + piece, 1.0}, {activeBase + piece, -1.0}});
    }

    for (std::size_t edge = 0; edge < edgeCount; ++edge) {
        const auto& constraint = constraints.constraints[edge];
        const std::size_t a = constraint.pieceA;
        const std::size_t b = constraint.pieceB;
        const std::size_t pair = pairBase + edge;
        const std::size_t verticalDifference = verticalDifferenceBase + edge;
        const std::size_t oddDifference = oddDifferenceBase + edge;

        const double orientationSame = 1.0 - constraint.parallelScore;
        const double orientationDifferent = constraint.parallelScore;
        const double windingSame = constraint.windingDistance;
        const double windingDifferent =
            std::abs(1.0 - constraint.windingDistance);
        lp.col_cost_[pair] =
            std::min(orientationSame, orientationDifferent) +
            std::min(windingSame, windingDifferent);

        rows.add(-kHighsInf, 0.0,
            {{pair, 1.0}, {activeBase + a, -1.0}});
        rows.add(-kHighsInf, 0.0,
            {{pair, 1.0}, {activeBase + b, -1.0}});
        rows.add(-1.0, kHighsInf,
            {{pair, 1.0}, {activeBase + a, -1.0}, {activeBase + b, -1.0}});

        const auto addGatedRelation = [&](std::size_t relation,
                                           std::size_t valueBase,
                                           double sameCost,
                                           double differentCost) {
            lp.col_cost_[relation] = std::abs(differentCost - sameCost);
            if (differentCost == sameCost) {
                lp.col_upper_[relation] = 0.0;
            } else if (differentCost > sameCost) {
                rows.add(-1.0, kHighsInf, {
                    {relation, 1.0}, {valueBase + a, -1.0},
                    {valueBase + b, 1.0}, {pair, -1.0}});
                rows.add(-1.0, kHighsInf, {
                    {relation, 1.0}, {valueBase + a, 1.0},
                    {valueBase + b, -1.0}, {pair, -1.0}});
            } else {
                rows.add(0.0, kHighsInf, {
                    {relation, 1.0}, {valueBase + a, 1.0},
                    {valueBase + b, 1.0}, {pair, -1.0}});
                rows.add(-2.0, kHighsInf, {
                    {relation, 1.0}, {valueBase + a, -1.0},
                    {valueBase + b, -1.0}, {pair, -1.0}});
            }
        };
        addGatedRelation(
            verticalDifference,
            verticalBase,
            orientationSame,
            orientationDifferent);
        addGatedRelation(
            oddDifference,
            oddBase,
            windingSame,
            windingDifferent);
    }

    lp.num_row_ = checkedHighsInt(rows.lower.size(), "MILP row count");
    lp.row_lower_ = std::move(rows.lower);
    lp.row_upper_ = std::move(rows.upper);
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_ = std::move(rows.starts);
    lp.a_matrix_.index_ = std::move(rows.indices);
    lp.a_matrix_.value_ = std::move(rows.values);
    const std::size_t rowCount = static_cast<std::size_t>(lp.num_row_);

    Highs highs;
    requireOk(highs.setOptionValue("output_flag", false), "disable solver output");
    requireOk(highs.setOptionValue("random_seed", HighsInt{0}), "set random seed");
    requireOk(
        highs.setOptionValue("mip_rel_gap", config.relativeMipGap),
        "set relative MIP gap");
    requireOk(highs.setOptionValue("mip_abs_gap", 1.0e-6), "set absolute MIP gap");
    const std::size_t requestedThreads = config.parallelThreads == 0
        ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
        : config.parallelThreads;
    requireOk(
        highs.setOptionValue("threads", checkedHighsInt(requestedThreads, "thread count")),
        "set solver thread count");
    requireOk(highs.passModel(std::move(model)), "load labeling model");
    const auto solveStarted = std::chrono::steady_clock::now();
    requireOk(highs.run(), "solve labeling model");
    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solveStarted).count();
    const auto status = highs.getModelStatus();
    report.modelStatus = highs.modelStatusToString(status);
    if (status != HighsModelStatus::kOptimal) {
        throw std::runtime_error(
            "HiGHS fiber trace labeling did not reach an optimal solution: " +
            report.modelStatus);
    }

    const auto& solution = highs.getSolution().col_value;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const bool active = binaryValue(solution, activeBase + piece);
        const bool vertical = binaryValue(solution, verticalBase + piece);
        const bool odd = binaryValue(solution, oddBase + piece);
        if (!active && (vertical || odd))
            throw std::runtime_error("HiGHS returned a non-canonical broken label");
        report.labels[piece] = active
            ? makeLabel(vertical, odd)
            : FiberTracePieceLabel::Broken;
    }
    canonicalizeLabels(report.labels, constraints, degree);

    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const auto label = report.labels[piece];
        ++report.labelCounts[labelIndex(label)];
        if (isBroken(label)) {
            report.brokenCost += config.brokenCostPerConstraint *
                static_cast<double>(degree[piece]);
        }
    }
    for (const auto& constraint : constraints.constraints) {
        const auto a = report.labels[constraint.pieceA];
        const auto b = report.labels[constraint.pieceB];
        if (isBroken(a) || isBroken(b))
            continue;
        report.orientationCost += isVertical(a) == isVertical(b)
            ? 1.0 - constraint.parallelScore
            : constraint.parallelScore;
        report.windingCost += isOdd(a) == isOdd(b)
            ? constraint.windingDistance
            : std::abs(1.0 - constraint.windingDistance);
    }
    report.objective =
        report.brokenCost + report.orientationCost + report.windingCost;
    const double solverObjective = highs.getObjectiveValue();
    const double objectiveTolerance =
        1.0e-6 * std::max(1.0, std::abs(solverObjective));
    if (!std::isfinite(report.objective) ||
        std::abs(report.objective - solverObjective) > objectiveTolerance) {
        throw std::runtime_error(
            "HiGHS labeling objective does not match decoded labels");
    }

    report.variables = variableCount;
    report.integerVariables = 3 * pieceCount;
    report.rows = rowCount;
    report.mipNodes = highs.getInfo().mip_node_count;
    report.mipGap = highs.getInfo().mip_gap;
    return report;
}

FiberTraceLabelObjPaths fiberTraceLabelObjPaths(
    const std::filesystem::path& outputBase)
{
    const auto directory = outputBase.parent_path();
    const std::string stem = outputStem(outputBase);
    return {
        directory / (stem + "_h_even.obj"),
        directory / (stem + "_h_odd.obj"),
        directory / (stem + "_v_even.obj"),
        directory / (stem + "_v_odd.obj"),
        directory / (stem + "_broken.obj"),
    };
}

FiberTraceLabelObjReport writeFiberTraceLabelObjs(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingReport& labeling,
    const std::filesystem::path& outputBase)
{
    if (labeling.labels.size() != constraints.pieces.size())
        throw std::invalid_argument("Fiber trace label count does not match pieces");
    FiberTraceLabelObjReport result;
    result.paths = fiberTraceLabelObjPaths(outputBase);
    std::array<std::vector<vc::core::io::NamedPolyline>, 5> lines;
    for (std::size_t pieceIndex = 0;
         pieceIndex < constraints.pieces.size();
         ++pieceIndex) {
        const auto& piece = constraints.pieces[pieceIndex];
        const std::size_t group = labelIndex(labeling.labels[pieceIndex]);
        lines[group].push_back({
            "piece_" + std::to_string(pieceIndex) + "_trace_" +
                std::to_string(piece.traceIndex) + "_part_" +
                std::to_string(piece.pieceIndex),
            piece.samplePointsBaseXYZ,
        });
    }
    const auto directory = result.paths.hEven.parent_path();
    if (!directory.empty())
        std::filesystem::create_directories(directory);
    const std::array<std::filesystem::path, 5> paths{
        result.paths.hEven,
        result.paths.hOdd,
        result.paths.vEven,
        result.paths.vOdd,
        result.paths.broken,
    };
    const std::array<const char*, 5> comments{
        "VC3D crop-trace H/even pieces",
        "VC3D crop-trace H/odd pieces",
        "VC3D crop-trace V/even pieces",
        "VC3D crop-trace V/odd pieces",
        "VC3D broken crop-trace pieces",
    };
    for (std::size_t group = 0; group < lines.size(); ++group) {
        vc::core::io::writePolylinesObj(lines[group], paths[group], comments[group]);
        result.pieceCounts[group] = lines[group].size();
    }
    return result;
}

}  // namespace vc::fiber_tracer
