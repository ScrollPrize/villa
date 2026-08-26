#include "vc/fiber_tracer/FiberTraceLabeling.hpp"

#include "vc/core/io/PolylineObj.hpp"

#include <Highs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
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

struct NeighborEdge {
    std::size_t neighbor = 0;
    std::size_t edge = 0;
};

struct ConstraintTriangle {
    std::array<std::size_t, 3> pieces{};
    std::array<std::size_t, 3> edges{};
};

std::vector<std::vector<NeighborEdge>> buildAdjacency(
    std::size_t pieceCount,
    std::span<const FiberTraceConstraint> constraints)
{
    std::vector<std::vector<NeighborEdge>> adjacency(pieceCount);
    for (std::size_t edge = 0; edge < constraints.size(); ++edge) {
        const auto& constraint = constraints[edge];
        adjacency[constraint.pieceA].push_back({constraint.pieceB, edge});
        adjacency[constraint.pieceB].push_back({constraint.pieceA, edge});
    }
    for (auto& neighbors : adjacency) {
        std::sort(
            neighbors.begin(), neighbors.end(),
            [](const NeighborEdge& left, const NeighborEdge& right) {
                return std::tie(left.neighbor, left.edge) <
                    std::tie(right.neighbor, right.edge);
            });
        for (std::size_t index = 1; index < neighbors.size(); ++index) {
            if (neighbors[index - 1].neighbor == neighbors[index].neighbor) {
                throw std::invalid_argument(
                    "Fiber trace labeling graph contains duplicate piece pairs");
            }
        }
    }
    return adjacency;
}

std::vector<std::size_t> componentRoots(
    const std::vector<std::vector<NeighborEdge>>& adjacency)
{
    std::vector<std::size_t> roots;
    std::vector<unsigned char> visited(adjacency.size(), 0);
    std::queue<std::size_t> pending;
    for (std::size_t seed = 0; seed < adjacency.size(); ++seed) {
        if (visited[seed])
            continue;
        roots.push_back(seed);
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty()) {
            const std::size_t piece = pending.front();
            pending.pop();
            for (const auto& neighbor : adjacency[piece]) {
                if (!visited[neighbor.neighbor]) {
                    visited[neighbor.neighbor] = 1;
                    pending.push(neighbor.neighbor);
                }
            }
        }
    }
    return roots;
}

std::vector<ConstraintTriangle> enumerateTriangles(
    const std::vector<std::vector<NeighborEdge>>& adjacency)
{
    std::vector<ConstraintTriangle> triangles;
    for (std::size_t a = 0; a < adjacency.size(); ++a) {
        const auto& neighborsA = adjacency[a];
        for (const auto& ab : neighborsA) {
            const std::size_t b = ab.neighbor;
            if (b <= a)
                continue;
            const auto& neighborsB = adjacency[b];
            auto ac = std::upper_bound(
                neighborsA.begin(), neighborsA.end(), b,
                [](std::size_t value, const NeighborEdge& entry) {
                    return value < entry.neighbor;
                });
            auto bc = std::upper_bound(
                neighborsB.begin(), neighborsB.end(), b,
                [](std::size_t value, const NeighborEdge& entry) {
                    return value < entry.neighbor;
                });
            while (ac != neighborsA.end() && bc != neighborsB.end()) {
                if (ac->neighbor < bc->neighbor) {
                    ++ac;
                } else if (bc->neighbor < ac->neighbor) {
                    ++bc;
                } else {
                    triangles.push_back({
                        {a, b, ac->neighbor},
                        {ab.edge, ac->edge, bc->edge},
                    });
                    ++ac;
                    ++bc;
                }
            }
        }
    }
    return triangles;
}

void canonicalizeLabels(
    std::vector<FiberTracePieceLabel>& labels,
    std::span<const FiberTraceConstraint> constraints,
    const std::vector<std::size_t>& degree)
{
    const std::size_t count = labels.size();
    std::vector<std::vector<std::size_t>> adjacency(count);
    for (const auto& constraint : constraints) {
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
    if (config.lpSolver != "choose" && config.lpSolver != "simplex" &&
        config.lpSolver != "hipo" && config.lpSolver != "ipm") {
        throw std::invalid_argument(
            "Fiber trace LP solver must be choose, simplex, hipo, or ipm");
    }
    if (!config.relaxIntegrality &&
        (config.lpParallel || config.lpSolver != "choose")) {
        throw std::invalid_argument(
            "Fiber trace LP solver options require relaxation mode");
    }
    if (config.exactPerpendicularMilp && !config.hvOnly) {
        throw std::invalid_argument(
            "Exact perpendicular MILP requires H/V-only labeling");
    }
    if (config.exactPerpendicularMilp && config.relaxIntegrality) {
        throw std::invalid_argument(
            "Exact perpendicular MILP and LP relaxation are mutually exclusive");
    }

    const std::size_t pieceCount = constraints.pieces.size();
    FiberTraceLabelingReport report;
    report.hvOnly = config.hvOnly;
    report.exactPerpendicularMilp = config.exactPerpendicularMilp;
    report.continuousPieceValues =
        config.relaxIntegrality || config.exactPerpendicularMilp;
    if (!report.continuousPieceValues)
        report.labels.assign(pieceCount, FiberTracePieceLabel::Broken);
    if (pieceCount == 0) {
        report.modelStatus = "Empty";
        return report;
    }

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
            (!config.hvOnly && constraint.windingDistance >= 1.5)) {
            throw std::invalid_argument(
                "Fiber trace constraint contains invalid optimization scores");
        }
    }

    std::vector<FiberTraceConstraint> filteredConstraints;
    std::span<const FiberTraceConstraint> labelingConstraints =
        constraints.constraints;
    if (config.excludeParallelSeparateWinding) {
        filteredConstraints.reserve(constraints.constraints.size());
        for (const auto& constraint : constraints.constraints) {
            const bool exclude = !constraint.hardContinuity &&
                constraint.parallelScore > 0.5 &&
                constraint.windingDistance >= 0.5;
            if (exclude) {
                ++report.excludedParallelSeparateWinding;
            } else {
                filteredConstraints.push_back(constraint);
            }
        }
        labelingConstraints = filteredConstraints;
    }
    const std::size_t edgeCount = labelingConstraints.size();
    report.retainedConstraints = edgeCount;
    std::vector<std::size_t> degree(pieceCount, 0);
    for (const auto& constraint : labelingConstraints) {
        ++degree[constraint.pieceA];
        ++degree[constraint.pieceB];
    }
    const auto adjacency = buildAdjacency(pieceCount, labelingConstraints);
    const auto gaugeRoots = componentRoots(adjacency);
    const auto triangles = config.relaxIntegrality
        ? enumerateTriangles(adjacency)
        : std::vector<ConstraintTriangle>{};

    std::vector<std::size_t> perpendicularSignIndex(
        edgeCount, std::numeric_limits<std::size_t>::max());
    std::size_t perpendicularSignCount = 0;
    if (config.exactPerpendicularMilp) {
        for (std::size_t edge = 0; edge < edgeCount; ++edge) {
            if (labelingConstraints[edge].parallelScore <= 0.5)
                perpendicularSignIndex[edge] = perpendicularSignCount++;
        }
    }
    report.perpendicularBranchVariables = perpendicularSignCount;

    const std::size_t activeBase = 0;
    const std::size_t verticalBase = pieceCount;
    const std::size_t oddBase = 2 * pieceCount;
    const std::size_t pairBase = (config.hvOnly ? 2 : 3) * pieceCount;
    const std::size_t verticalDifferenceBase = pairBase + edgeCount;
    const std::size_t oddDifferenceBase = verticalDifferenceBase + edgeCount;
    const std::size_t perpendicularSignBase = oddDifferenceBase +
        (config.hvOnly ? 0 : edgeCount);
    const std::size_t variableCount =
        perpendicularSignBase + perpendicularSignCount;

    HighsModel model;
    auto& lp = model.lp_;
    lp.num_col_ = checkedHighsInt(variableCount, "MILP variable count");
    lp.col_cost_.assign(variableCount, 0.0);
    lp.col_lower_.assign(variableCount, 0.0);
    lp.col_upper_.assign(variableCount, 1.0);
    lp.integrality_.assign(variableCount, HighsVarType::kContinuous);
    if (config.exactPerpendicularMilp) {
        std::fill_n(
            lp.integrality_.begin(), pieceCount, HighsVarType::kInteger);
        std::fill(
            lp.integrality_.begin() +
                static_cast<std::ptrdiff_t>(perpendicularSignBase),
            lp.integrality_.end(),
            HighsVarType::kInteger);
    } else if (!config.relaxIntegrality) {
        std::fill_n(
            lp.integrality_.begin(),
            (config.hvOnly ? 2 : 3) * pieceCount,
            HighsVarType::kInteger);
    }
    lp.offset_ = 0.0;
    lp.sense_ = ObjSense::kMinimize;
    for (const std::size_t root : gaugeRoots) {
        lp.col_upper_[verticalBase + root] = 0.0;
        if (!config.hvOnly)
            lp.col_upper_[oddBase + root] = 0.0;
    }

    RowBuilder rows;
    const std::size_t expectedRows = config.hvOnly
        ? pieceCount + 8 * edgeCount + 4 * triangles.size()
        : 2 * pieceCount + 13 * edgeCount + 8 * triangles.size();
    const std::size_t exactRows = 2 * perpendicularSignCount;
    rows.lower.reserve(expectedRows + exactRows);
    rows.upper.reserve(expectedRows + exactRows);
    rows.starts.reserve(expectedRows + exactRows + 1);
    rows.indices.reserve(
        4 * pieceCount + 46 * edgeCount + 48 * triangles.size() +
        8 * perpendicularSignCount);
    rows.values.reserve(
        4 * pieceCount + 46 * edgeCount + 48 * triangles.size() +
        8 * perpendicularSignCount);

    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const double penalty = config.brokenCostPerConstraint *
            static_cast<double>(degree[piece]);
        lp.offset_ += penalty;
        lp.col_cost_[activeBase + piece] = -penalty;
        rows.add(-kHighsInf, 0.0, {
            {verticalBase + piece, 1.0}, {activeBase + piece, -1.0}});
        if (!config.hvOnly) {
            rows.add(-kHighsInf, 0.0, {
                {oddBase + piece, 1.0}, {activeBase + piece, -1.0}});
        }
    }

    for (std::size_t edge = 0; edge < edgeCount; ++edge) {
        const auto& constraint = labelingConstraints[edge];
        const std::size_t a = constraint.pieceA;
        const std::size_t b = constraint.pieceB;
        const std::size_t pair = pairBase + edge;
        const std::size_t verticalDifference = verticalDifferenceBase + edge;

        const double orientationSame = 1.0 - constraint.parallelScore;
        const double orientationDifferent = constraint.parallelScore;
        const double windingSame = constraint.windingDistance;
        const double windingDifferent =
            std::abs(1.0 - constraint.windingDistance);
        lp.col_cost_[pair] = orientationSame +
            (config.hvOnly ? 0.0 : windingSame);

        rows.add(-kHighsInf, 0.0,
            {{pair, 1.0}, {activeBase + a, -1.0}});
        rows.add(-kHighsInf, 0.0,
            {{pair, 1.0}, {activeBase + b, -1.0}});
        rows.add(-1.0, kHighsInf,
            {{pair, 1.0}, {activeBase + a, -1.0}, {activeBase + b, -1.0}});

        const auto addGatedDifference = [&](std::size_t difference,
                                             std::size_t valueBase,
                                             double sameCost,
                                             double differentCost) {
            lp.col_cost_[difference] = differentCost - sameCost;
            rows.add(-kHighsInf, 0.0, {
                {difference, 1.0}, {pair, -1.0}});
            rows.add(-1.0, kHighsInf, {
                {difference, 1.0}, {valueBase + a, -1.0},
                {valueBase + b, 1.0}, {pair, -1.0}});
            rows.add(-1.0, kHighsInf, {
                {difference, 1.0}, {valueBase + a, 1.0},
                {valueBase + b, -1.0}, {pair, -1.0}});
            rows.add(-kHighsInf, 0.0, {
                {difference, 1.0}, {valueBase + a, -1.0},
                {valueBase + b, -1.0}});
            rows.add(-kHighsInf, 2.0, {
                {difference, 1.0}, {valueBase + a, 1.0},
                {valueBase + b, 1.0}});
        };
        addGatedDifference(
            verticalDifference,
            verticalBase,
            orientationSame,
            orientationDifferent);
        if (perpendicularSignIndex[edge] !=
            std::numeric_limits<std::size_t>::max()) {
            const std::size_t sign = perpendicularSignBase +
                perpendicularSignIndex[edge];
            rows.add(-kHighsInf, 3.0, {
                {verticalDifference, 1.0},
                {verticalBase + a, -1.0},
                {verticalBase + b, 1.0},
                {sign, 2.0},
                {pair, 1.0},
            });
            rows.add(-kHighsInf, 1.0, {
                {verticalDifference, 1.0},
                {verticalBase + a, 1.0},
                {verticalBase + b, -1.0},
                {sign, -2.0},
                {pair, 1.0},
            });
        }
        if (!config.hvOnly) {
            addGatedDifference(
                oddDifferenceBase + edge,
                oddBase,
                windingSame,
                windingDifferent);
        }
    }

    const auto addTriangleCuts = [&](const ConstraintTriangle& triangle,
                                      std::size_t differenceBase) {
        const std::array<std::size_t, 3> differences{
            differenceBase + triangle.edges[0],
            differenceBase + triangle.edges[1],
            differenceBase + triangle.edges[2],
        };
        for (std::size_t edge = 0; edge < 3; ++edge) {
            rows.add(-kHighsInf, 3.0, {
                {differences[edge], 1.0},
                {differences[(edge + 1) % 3], -1.0},
                {differences[(edge + 2) % 3], -1.0},
                {activeBase + triangle.pieces[0], 1.0},
                {activeBase + triangle.pieces[1], 1.0},
                {activeBase + triangle.pieces[2], 1.0},
            });
        }
        rows.add(-kHighsInf, 2.0, {
            {differences[0], 1.0},
            {differences[1], 1.0},
            {differences[2], 1.0},
        });
    };
    for (const auto& triangle : triangles) {
        addTriangleCuts(triangle, verticalDifferenceBase);
        if (!config.hvOnly)
            addTriangleCuts(triangle, oddDifferenceBase);
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
    if (config.relaxIntegrality) {
        requireOk(
            highs.setOptionValue("parallel", config.lpParallel ? "on" : "choose"),
            "set LP parallel mode");
        requireOk(
            highs.setOptionValue("solver", config.lpSolver),
            "set LP solver");
    }
    requireOk(highs.setOptionValue("output_flag", false), "disable solver output");
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
    report.activeValues.resize(pieceCount);
    report.verticalValues.resize(pieceCount);
    report.oddValues.resize(pieceCount);
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const auto readRelaxed = [&](std::size_t index) {
            if (index >= solution.size() || !std::isfinite(solution[index]) ||
                solution[index] < -kBinaryTolerance ||
                solution[index] > 1.0 + kBinaryTolerance) {
                throw std::runtime_error(
                    "HiGHS returned an invalid relaxed labeling solution");
            }
            return std::clamp(solution[index], 0.0, 1.0);
        };
        report.activeValues[piece] = readRelaxed(activeBase + piece);
        report.verticalValues[piece] = readRelaxed(verticalBase + piece);
        report.oddValues[piece] = config.hvOnly
            ? 0.0
            : readRelaxed(oddBase + piece);
        if (config.exactPerpendicularMilp) {
            report.activeValues[piece] = binaryValue(
                solution, activeBase + piece) ? 1.0 : 0.0;
            continue;
        }
        if (config.relaxIntegrality)
            continue;
        const bool active = binaryValue(solution, activeBase + piece);
        const bool vertical = binaryValue(solution, verticalBase + piece);
        const bool odd = config.hvOnly
            ? false
            : binaryValue(solution, oddBase + piece);
        if (!active && (vertical || odd))
            throw std::runtime_error("HiGHS returned a non-canonical broken label");
        report.labels[piece] = active
            ? makeLabel(vertical, odd)
            : FiberTracePieceLabel::Broken;
    }
    if (!report.continuousPieceValues)
        canonicalizeLabels(report.labels, labelingConstraints, degree);

    if (report.continuousPieceValues) {
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            report.brokenCost += config.brokenCostPerConstraint *
                static_cast<double>(degree[piece]) *
                (1.0 - report.activeValues[piece]);
        }
        for (std::size_t edge = 0; edge < edgeCount; ++edge) {
            const auto& constraint = labelingConstraints[edge];
            const double pair = config.exactPerpendicularMilp
                ? report.activeValues[constraint.pieceA] *
                    report.activeValues[constraint.pieceB]
                : solution[pairBase + edge];
            const double orientationSame = 1.0 - constraint.parallelScore;
            const double orientationDifferent = constraint.parallelScore;
            const double difference = config.exactPerpendicularMilp
                ? pair * std::abs(
                    report.verticalValues[constraint.pieceA] -
                    report.verticalValues[constraint.pieceB])
                : solution[verticalDifferenceBase + edge];
            report.orientationCost += orientationSame * pair +
                (orientationDifferent - orientationSame) * difference;
            if (!config.hvOnly) {
                const double windingSame = constraint.windingDistance;
                const double windingDifferent =
                    std::abs(1.0 - constraint.windingDistance);
                report.windingCost += windingSame * pair +
                    (windingDifferent - windingSame) *
                        solution[oddDifferenceBase + edge];
            }
        }
    } else {
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            const auto label = report.labels[piece];
            ++report.labelCounts[labelIndex(label)];
            if (isBroken(label)) {
                report.brokenCost += config.brokenCostPerConstraint *
                    static_cast<double>(degree[piece]);
            }
        }
        for (const auto& constraint : labelingConstraints) {
            const auto a = report.labels[constraint.pieceA];
            const auto b = report.labels[constraint.pieceB];
            if (isBroken(a) || isBroken(b))
                continue;
            report.orientationCost += isVertical(a) == isVertical(b)
                ? 1.0 - constraint.parallelScore
                : constraint.parallelScore;
            if (!config.hvOnly) {
                report.windingCost += isOdd(a) == isOdd(b)
                    ? constraint.windingDistance
                    : std::abs(1.0 - constraint.windingDistance);
            }
        }
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
    report.integerVariables = config.exactPerpendicularMilp
        ? pieceCount + perpendicularSignCount
        : (config.relaxIntegrality
              ? 0
              : (config.hvOnly ? 2 : 3) * pieceCount);
    report.rows = rowCount;
    report.gaugeRoots = gaugeRoots.size();
    report.triangles = triangles.size();
    report.triangleRows = (config.hvOnly ? 4 : 8) * triangles.size();
    report.mipNodes = highs.getInfo().mip_node_count;
    report.mipGap = highs.getInfo().mip_gap;
    return report;
}

FiberDirectionLabelComparisonReport compareFiberDirectionLabels(
    const FiberTraceConstraintReport& constraints,
    std::span<const FiberDirectionGroup> traceDirections,
    const FiberTraceLabelingReport& labeling)
{
    const std::size_t pieceCount = constraints.pieces.size();
    if (labeling.continuousPieceValues || labeling.labels.size() != pieceCount) {
        throw std::invalid_argument(
            "Fiber direction comparison requires one discrete label per piece");
    }
    if (traceDirections.size() != constraints.inputTraces) {
        throw std::invalid_argument(
            "Fiber direction comparison trace count does not match constraints");
    }
    for (const auto direction : traceDirections) {
        if (direction == FiberDirectionGroup::Mixed) {
            throw std::invalid_argument(
                "Fiber direction comparison does not accept mixed traces");
        }
    }

    FiberDirectionLabelComparisonReport result;
    std::vector<unsigned char> represented(traceDirections.size(), 0);
    std::vector<unsigned char> active(pieceCount, 0);
    std::vector<std::vector<std::size_t>> adjacency(pieceCount);
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        if (labelIndex(labeling.labels[piece]) >= 5) {
            throw std::invalid_argument(
                "Fiber direction comparison received an invalid piece label");
        }
        const auto& descriptor = constraints.pieces[piece];
        if (descriptor.traceIndex >= traceDirections.size()) {
            throw std::invalid_argument(
                "Fiber direction comparison piece references an invalid trace");
        }
        represented[descriptor.traceIndex] = 1;
        active[piece] = isBroken(labeling.labels[piece]) ? 0 : 1;
        if (active[piece]) {
            if (isVertical(labeling.labels[piece]))
                ++result.rawV;
            else
                ++result.rawH;
        } else {
            ++result.rawBroken;
        }
    }
    result.representedTraces = static_cast<std::size_t>(std::count(
        represented.begin(), represented.end(), static_cast<unsigned char>(1)));

    for (const auto& constraint : constraints.constraints) {
        if (constraint.pieceA >= pieceCount ||
            constraint.pieceB >= pieceCount ||
            constraint.pieceA == constraint.pieceB) {
            throw std::invalid_argument(
                "Fiber direction comparison constraint has invalid endpoints");
        }
        if (!active[constraint.pieceA] || !active[constraint.pieceB])
            continue;
        adjacency[constraint.pieceA].push_back(constraint.pieceB);
        adjacency[constraint.pieceB].push_back(constraint.pieceA);
    }
    for (auto& neighbors : adjacency)
        std::sort(neighbors.begin(), neighbors.end());

    constexpr std::size_t noComponent =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> component(pieceCount, noComponent);
    std::vector<std::vector<std::size_t>> componentPieces;
    std::queue<std::size_t> pending;
    for (std::size_t seed = 0; seed < pieceCount; ++seed) {
        if (!active[seed] || component[seed] != noComponent)
            continue;
        const std::size_t componentIndex = componentPieces.size();
        componentPieces.emplace_back();
        component[seed] = componentIndex;
        pending.push(seed);
        while (!pending.empty()) {
            const std::size_t piece = pending.front();
            pending.pop();
            componentPieces.back().push_back(piece);
            for (const std::size_t neighbor : adjacency[piece]) {
                if (component[neighbor] == noComponent) {
                    component[neighbor] = componentIndex;
                    pending.push(neighbor);
                }
            }
        }
    }
    result.activeComponents = componentPieces.size();

    std::vector<unsigned char> flipped(componentPieces.size(), 0);
    for (std::size_t componentIndex = 0;
         componentIndex < componentPieces.size();
         ++componentIndex) {
        std::size_t identityErrors = 0;
        std::size_t flippedErrors = 0;
        for (const std::size_t piece : componentPieces[componentIndex]) {
            const auto initial =
                traceDirections[constraints.pieces[piece].traceIndex];
            const bool rawVertical = isVertical(labeling.labels[piece]);
            const bool initialVertical =
                initial == FiberDirectionGroup::Direction2;
            identityErrors += rawVertical != initialVertical ? 1 : 0;
            flippedErrors += rawVertical == initialVertical ? 1 : 0;
        }
        if (flippedErrors < identityErrors) {
            flipped[componentIndex] = 1;
            ++result.flippedComponents;
        }
    }

    std::vector<unsigned char> erroneousTrace(traceDirections.size(), 0);
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const auto& descriptor = constraints.pieces[piece];
        const auto initial = traceDirections[descriptor.traceIndex];
        const std::size_t rowIndex =
            initial == FiberDirectionGroup::Direction1 ? 0 : 1;
        auto& row = result.confusion[rowIndex];
        ++row.pieces;

        FiberDirectionLabelError error;
        error.pieceIndex = piece;
        error.filteredTraceIndex = descriptor.traceIndex;
        error.tracePieceIndex = descriptor.pieceIndex;
        error.beginArcBaseVoxels = descriptor.beginArcBaseVoxels;
        error.endArcBaseVoxels = descriptor.endArcBaseVoxels;
        error.initialDirection = initial;
        error.rawLabel = labeling.labels[piece];
        if (!active[piece]) {
            ++row.broken;
            ++row.errors;
            ++result.brokenErrors;
            erroneousTrace[descriptor.traceIndex] = 1;
            error.kind = FiberDirectionLabelErrorKind::Broken;
            result.errors.push_back(error);
            continue;
        }

        error.componentIndex = component[piece];
        error.componentFlipped = flipped[component[piece]] != 0;
        const bool alignedVertical =
            isVertical(labeling.labels[piece]) != error.componentFlipped;
        error.alignedDirection = alignedVertical
            ? FiberDirectionGroup::Direction2
            : FiberDirectionGroup::Direction1;
        if (alignedVertical)
            ++row.alignedDirection2;
        else
            ++row.alignedDirection1;
        if (error.alignedDirection != initial) {
            ++row.errors;
            ++result.orientationErrors;
            erroneousTrace[descriptor.traceIndex] = 1;
            error.kind = FiberDirectionLabelErrorKind::Orientation;
            result.errors.push_back(error);
        }
    }
    result.errorTraces = static_cast<std::size_t>(std::count(
        erroneousTrace.begin(), erroneousTrace.end(),
        static_cast<unsigned char>(1)));
    return result;
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

std::filesystem::path fiberTraceLabelRelaxationCsvPath(
    const std::filesystem::path& outputBase)
{
    const auto directory = outputBase.parent_path();
    return directory / (outputStem(outputBase) + "_values.csv");
}

std::filesystem::path writeFiberTraceLabelRelaxationCsv(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingReport& labeling,
    const std::filesystem::path& outputBase)
{
    const std::size_t count = constraints.pieces.size();
    if (labeling.activeValues.size() != count ||
        labeling.verticalValues.size() != count ||
        labeling.oddValues.size() != count) {
        throw std::invalid_argument(
            "Fiber trace relaxed label count does not match pieces");
    }
    const auto path = fiberTraceLabelRelaxationCsvPath(outputBase);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Could not open relaxed labeling CSV: " + path.string());
    output << "piece_id,trace_index,piece_index,active,vertical,odd\n"
           << std::setprecision(17);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& piece = constraints.pieces[index];
        output << index << ',' << piece.traceIndex << ',' << piece.pieceIndex
               << ',' << labeling.activeValues[index]
               << ',' << labeling.verticalValues[index]
               << ',' << labeling.oddValues[index] << '\n';
    }
    if (!output)
        throw std::runtime_error("Could not write relaxed labeling CSV: " + path.string());
    return path;
}

FiberTraceRelaxationObjReport writeFiberTraceLabelRelaxationObjs(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingReport& labeling,
    const std::filesystem::path& outputBase)
{
    const std::size_t count = constraints.pieces.size();
    if (count == 0 || labeling.activeValues.size() != count ||
        labeling.verticalValues.size() != count ||
        labeling.oddValues.size() != count) {
        throw std::invalid_argument(
            "Fiber trace relaxed label count does not match nonempty pieces");
    }
    FiberTraceRelaxationObjReport result;
    result.activeThreshold = std::accumulate(
        labeling.activeValues.begin(), labeling.activeValues.end(), 0.0) /
        static_cast<double>(count);
    FiberTraceLabelingReport classified;
    classified.labels.reserve(count);
    for (std::size_t piece = 0; piece < count; ++piece) {
        if (labeling.activeValues[piece] < result.activeThreshold) {
            classified.labels.push_back(FiberTracePieceLabel::Broken);
        } else {
            classified.labels.push_back(makeLabel(
                labeling.verticalValues[piece] >= 0.5,
                labeling.oddValues[piece] >= 0.5));
        }
    }
    result.objects = writeFiberTraceLabelObjs(
        constraints,
        classified,
        outputBase);
    return result;
}

}  // namespace vc::fiber_tracer
