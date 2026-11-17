#include "LineAnnotationPieceLock.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::line_annotation {
namespace {
using vc::objects::FiberPieceApprovalStates;

std::string spanOwnerLabel(size_t span) { return "span " + std::to_string(span + 1); }

std::string approvedBy(const PieceLockState& state, size_t span, FiberPieceAspect aspect)
{
    const auto& covering = state.coveringRow[size_t(aspect)];
    if (span < covering.size() && covering[span] && *covering[span] < state.states.live.size()) {
        const auto& row = state.states.live[*covering[span]];
        return " (approved by " + row.author + " at " + row.approvedAt + ")";
    }
    return {};
}

// The stored span index of the session span `span`, when both its controls
// are stored and adjacent there.
std::optional<size_t> storedSpanOf(const std::vector<LineControlPoint>& controls, const FiberPieceContent& stored, size_t span)
{
    if (span + 1 >= controls.size()) return std::nullopt;
    const int64_t a = controls[span].storeId, b = controls[span + 1].storeId;
    if (a < 0 || b < 0) return std::nullopt;
    const auto ia = stored.indexOf(a), ib = stored.indexOf(b);
    if (!ia || !ib || *ib != *ia + 1) return std::nullopt;
    return *ia;
}
} // namespace

FiberPieceContent pieceContentOfStoredFiber(const StoredFiber& fiber)
{
    vc::geometry::Fiber geometry;
    nlohmann::json annotations = nlohmann::json::array();
    for (const auto& control : fiber.controlPoints) {
        geometry.controlPoints.push_back({control[0], control[1], control[2]});
        geometry.controlPointIds.push_back(control.storeId);
        annotations.push_back(storedControlPointToJson(control));
    }
    for (const auto& point : fiber.linePoints) geometry.linePoints.push_back({point[0], point[1], point[2]});
    return vc::objects::fiberPieceContent(geometry, annotations);
}

bool isLockControl(const LineControlPoint& control)
{
    return std::find(control.tags.begin(), control.tags.end(), vc::objects::kFiberLockControlPointTag) != control.tags.end();
}

cv::Vec3d lineTangentAt(const std::vector<cv::Vec3d>& linePoints, int index)
{
    if (linePoints.size() < 2 || index < 0 || size_t(index) >= linePoints.size())
        throw std::out_of_range("line tangent: index outside the line");
    const size_t i = size_t(index);
    const cv::Vec3d chord = linePoints[std::min(i + 1, linePoints.size() - 1)] - linePoints[i == 0 ? 0 : i - 1];
    const double length = cv::norm(chord);
    if (!(length > 0.0)) throw std::runtime_error("line tangent: degenerate line");
    return chord / length;
}

void lockControl(LineControlPoint& control, const std::vector<cv::Vec3d>& linePoints)
{
    setControlPointTag(control.tags, vc::objects::kFiberLockControlPointTag, true);
    if (!control.direction && control.optimizedIndex >= 0 && size_t(control.optimizedIndex) < linePoints.size())
        control.direction = lineTangentAt(linePoints, control.optimizedIndex);
}

LockControlInsertion insertLockControl(const std::vector<LineControlPoint>& controls,
                                       const std::vector<cv::Vec3d>& linePoints, int lineIndex)
{
    if (lineIndex < 0 || size_t(lineIndex) >= linePoints.size())
        throw std::out_of_range("lock control: line index outside the line");
    for (const auto& control : controls)
        if (control.optimizedIndex == lineIndex || control.linePosition == double(lineIndex))
            throw std::invalid_argument("lock control: a control point already sits on this line point");
    auto collapse = collapseControlPointsAtClick(controls, {}, double(lineIndex), linePoints[size_t(lineIndex)]);
    LockControlInsertion result;
    result.index = collapse.replacementIndex;
    result.oldToNewIndices = std::move(collapse.oldToNewIndices);
    result.controls = std::move(collapse.controlPoints);
    auto& inserted = result.controls[result.index];
    inserted.optimizedIndex = lineIndex;
    inserted.linePosition = double(lineIndex);
    inserted.storeId = -1;
    inserted.identity = 0;
    inserted.displayNormal.reset();
    inserted.displayNormalSource = "unknown";
    inserted.direction = lineTangentAt(linePoints, lineIndex);
    setControlPointTag(inserted.tags, vc::objects::kFiberLockControlPointTag, true);
    return result;
}

bool PieceLockState::anyLocked() const
{
    return std::any_of(lockedControls.begin(), lockedControls.end(), [](bool v) { return v; }) ||
           std::any_of(lockedSpans.begin(), lockedSpans.end(), [](bool v) { return v; }) ||
           std::any_of(normalsLockedSpans.begin(), normalsLockedSpans.end(), [](bool v) { return v; });
}

std::optional<std::pair<size_t, size_t>> plannedControlRange(const PiecePlan& plan,
                                                             const std::vector<LineControlPoint>& controls)
{
    if (controls.size() < 2) return std::nullopt;
    const auto indexOfIdentity = [&](uint64_t identity) -> std::optional<size_t> {
        if (identity == 0) return std::nullopt;
        for (size_t i = 0; i < controls.size(); ++i) if (controls[i].identity == identity) return i;
        return std::nullopt;
    };
    std::optional<size_t> first = plan.a.fiberEnd ? std::optional<size_t>(0) : indexOfIdentity(plan.a.identity);
    std::optional<size_t> last = plan.b.fiberEnd ? std::optional<size_t>(controls.size() - 1) : indexOfIdentity(plan.b.identity);
    if (!first || !last) return std::nullopt;
    if (*first > *last) std::swap(*first, *last);
    if (*first == *last) return std::nullopt;
    return std::make_pair(*first, *last);
}

PieceLockState computePieceLockState(const std::vector<LineControlPoint>& controls,
                                     std::vector<FiberPieceApproval> rows,
                                     std::optional<FiberPieceContent> storedContent,
                                     const std::optional<PiecePlan>& plan,
                                     const PieceLockState* previous)
{
    PieceLockState state;
    state.plan = plan;
    if (previous) {
        state.pendingFirstEdit = previous->pendingFirstEdit;
        state.firstEditDone = previous->firstEditDone;
    }
    state.rows = std::move(rows);
    state.storedContent = std::move(storedContent);
    if (state.storedContent) state.states = vc::objects::fiberPieceApprovalStates(*state.storedContent, state.rows);
    const size_t n = controls.size(), spans = n > 1 ? n - 1 : 0;
    state.lockedControls.assign(n, false);
    state.lockedSpans.assign(spans, false);
    state.normalsLockedSpans.assign(spans, false);
    for (size_t aspect = 0; aspect < 2; ++aspect) {
        state.coveringRow[aspect].assign(spans, std::nullopt);
        state.coveringCurrent[aspect].assign(spans, false);
    }
    for (size_t i = 0; i < n; ++i) state.lockedControls[i] = isLockControl(controls[i]);
    if (plan) state.openedControls = plannedControlRange(*plan, controls);
    if (state.openedControls) {
        for (size_t s = 0; s < spans; ++s) {
            const bool outside = s < state.openedControls->first || s >= state.openedControls->second;
            state.lockedSpans[s] = outside;
            state.normalsLockedSpans[s] = outside;
        }
    }
    if (state.storedContent) {
        for (size_t s = 0; s < spans; ++s) {
            const auto stored = storedSpanOf(controls, *state.storedContent, s);
            if (!stored) continue;
            for (size_t aspect = 0; aspect < 2; ++aspect) {
                const auto& entries = state.states.spans[aspect];
                if (*stored >= entries.size() || !entries[*stored].row) continue;
                state.coveringRow[aspect][s] = entries[*stored].row;
                state.coveringCurrent[aspect][s] = entries[*stored].current;
                if (state.openedControls || !entries[*stored].current) continue;
                (aspect == size_t(FiberPieceAspect::Geometry) ? state.lockedSpans : state.normalsLockedSpans)[s] = true;
            }
        }
    }
    return state;
}

std::pair<bool, bool> pieceTailRetention(const PieceLockState& state, const std::vector<LineControlPoint>& controls)
{
    if (!state.openedControls || !state.plan) return {false, false};
    const auto [first, last] = *state.openedControls;
    const bool openLeft = first == 0 && state.plan->a.fiberEnd;
    const bool openRight = last + 1 == controls.size() && state.plan->b.fiberEnd;
    return {!openLeft, !openRight};
}

PieceApprovalBoxState pieceApprovalBoxState(const PieceLockState& state)
{
    PieceApprovalBoxState boxes;
    boxes.pieceOpen = state.pieceOpen();
    if (!boxes.pieceOpen) return boxes;
    const auto [first, last] = *state.openedControls;
    for (size_t aspect = 0; aspect < 2; ++aspect) {
        bool covered = last > first, current = true, anyStale = false;
        for (size_t s = first; s < last; ++s) {
            if (s >= state.coveringRow[aspect].size() || !state.coveringRow[aspect][s]) { covered = false; continue; }
            if (!state.coveringCurrent[aspect][s]) { current = false; anyStale = true; }
        }
        boxes.approved[aspect] = covered && current;
        boxes.stale[aspect] = anyStale;
    }
    return boxes;
}

std::optional<std::string> pieceLockRefusal(const PieceLockState& state, std::span<const size_t> dirtySpans,
                                            std::span<const size_t> touchedControls, FiberPieceAspect aspect)
{
    for (const size_t control : touchedControls)
        if (control < state.lockedControls.size() && state.lockedControls[control])
            return "control point " + std::to_string(control + 1) + " is a lock control point of an approval piece; it cannot be changed";
    const auto& locked = aspect == FiberPieceAspect::Normals ? state.normalsLockedSpans : state.lockedSpans;
    for (const size_t span : dirtySpans) {
        if (span >= locked.size() || !locked[span]) continue;
        if (state.pieceOpen()) return spanOwnerLabel(span) + " lies outside the open piece; open the piece that covers it to change it";
        return spanOwnerLabel(span) + " is approved" + approvedBy(state, span, aspect) + "; open that piece to change it";
    }
    return std::nullopt;
}

std::vector<std::pair<double, double>> dimmedLineRanges(const PieceLockState& state,
                                                        const std::vector<LineControlPoint>& controls,
                                                        size_t linePointCount)
{
    std::vector<std::pair<double, double>> ranges;
    if (!state.openedControls || linePointCount == 0 || state.openedControls->second >= controls.size()) return ranges;
    const double last = double(linePointCount - 1);
    const double first = controls[state.openedControls->first].linePosition;
    const double end = controls[state.openedControls->second].linePosition;
    if (first > 0.0) ranges.emplace_back(0.0, first);
    if (end < last) ranges.emplace_back(end, last);
    return ranges;
}

PieceDisplayRanges pieceDisplayRanges(const PieceLockState& state, const std::vector<LineControlPoint>& controls,
                                      size_t linePointCount)
{
    PieceDisplayRanges ranges;
    ranges.dimmed = dimmedLineRanges(state, controls, linePointCount);
    const auto& covering = state.coveringRow[size_t(FiberPieceAspect::Geometry)];
    const auto& current = state.coveringCurrent[size_t(FiberPieceAspect::Geometry)];
    for (size_t s = 0; s + 1 < controls.size() && s < covering.size(); ++s) {
        if (!covering[s]) continue;
        auto& target = current[s] ? ranges.approved : ranges.stale;
        const double a = controls[s].linePosition, b = controls[s + 1].linePosition;
        if (!target.empty() && target.back().second == a) target.back().second = b;
        else target.emplace_back(a, b);
    }
    return ranges;
}

PieceSaveCheck pieceSaveCheck(const PieceLockState& state, const std::vector<LineControlPoint>& controls,
                              const std::vector<int>& storedIndexForSessionIndex)
{
    PieceSaveCheck check;
    check.pieceOpen = state.pieceOpen();
    if (check.pieceOpen) {
        const auto [first, last] = *state.openedControls;
        if (first >= storedIndexForSessionIndex.size() || last >= storedIndexForSessionIndex.size() ||
            storedIndexForSessionIndex[first] < 0 || storedIndexForSessionIndex[last] < 0)
            throw std::invalid_argument("piece lock: opened controls are not in the save snapshot");
        check.openedStored = {size_t(storedIndexForSessionIndex[first]), size_t(storedIndexForSessionIndex[last])};
        if (check.openedStored->first > check.openedStored->second) std::swap(check.openedStored->first, check.openedStored->second);
        const auto [leftLocked, rightLocked] = pieceTailRetention(state, controls);
        check.leftLocked = leftLocked;
        check.rightLocked = rightLocked;
        check.baseline = state.storedContent;
    } else {
        for (size_t s = 0; s < state.lockedSpans.size(); ++s) {
            if (!state.lockedSpans[s] && !state.normalsLockedSpans[s]) continue;
            const int64_t a = controls[s].storeId, b = controls[s + 1].storeId;
            if (a >= 0 && b >= 0) check.lockedSpanIds.emplace_back(a, b);
        }
    }
    return check;
}

std::optional<std::string> pieceSaveViolation(const FiberPieceContent& saved, const PieceSaveCheck& check,
                                              std::span<const FiberPieceApproval> rows)
{
    if (check.pieceOpen) {
        if (!check.baseline || !check.openedStored) return "the open piece has no baseline to check its surroundings against";
        const auto& base = *check.baseline;
        const auto [first, last] = *check.openedStored;
        if (last >= saved.controlIds.size()) return "the open piece's controls are not in the saved fiber";
        const auto sameControl = [&](size_t savedIndex, size_t baseIndex) {
            return baseIndex < base.controlIds.size() && saved.controlIds[savedIndex] == base.controlIds[baseIndex] &&
                   saved.controlPoints[savedIndex] == base.controlPoints[baseIndex] &&
                   vc::objects::fiberControlHash(saved, savedIndex, FiberPieceAspect::Geometry) ==
                       vc::objects::fiberControlHash(base, baseIndex, FiberPieceAspect::Geometry) &&
                   vc::objects::fiberControlHash(saved, savedIndex, FiberPieceAspect::Normals) ==
                       vc::objects::fiberControlHash(base, baseIndex, FiberPieceAspect::Normals);
        };
        if (check.leftLocked) {
            for (size_t i = 0; i < first; ++i)
                if (!sameControl(i, i)) return "control point " + std::to_string(i + 1) + " before the open piece changed";
            const int64_t lineEnd = saved.controlLineIndex[first];
            if (lineEnd < 0 || size_t(lineEnd) >= base.linePoints.size()) return "the line before the open piece changed its length";
            for (int64_t i = 0; i <= lineEnd; ++i)
                if (saved.linePoints[size_t(i)] != base.linePoints[size_t(i)]) return "the line before the open piece changed";
        }
        if (check.rightLocked) {
            const size_t savedCount = saved.controlIds.size(), baseCount = base.controlIds.size();
            const size_t after = savedCount - 1 - last;
            if (after > baseCount) return "control points after the open piece changed";
            for (size_t k = 1; k <= after; ++k)
                if (!sameControl(savedCount - k, baseCount - k)) return "control point " + std::to_string(savedCount - k + 1) + " after the open piece changed";
            const int64_t lineStart = saved.controlLineIndex[last];
            if (lineStart < 0) return "the line after the open piece changed its length";
            const size_t tail = saved.linePoints.size() - size_t(lineStart);
            if (tail > base.linePoints.size()) return "the line after the open piece changed its length";
            for (size_t k = 0; k < tail; ++k)
                if (saved.linePoints[saved.linePoints.size() - 1 - k] != base.linePoints[base.linePoints.size() - 1 - k])
                    return "the line after the open piece changed";
        }
        return std::nullopt;
    }
    if (check.lockedSpanIds.empty()) return std::nullopt;
    const auto states = vc::objects::fiberPieceApprovalStates(saved, rows);
    for (const auto& [a, b] : check.lockedSpanIds) {
        const auto ia = saved.indexOf(a), ib = saved.indexOf(b);
        const bool adjacent = ia && ib && (*ib == *ia + 1 || *ia == *ib + 1);
        const size_t span = adjacent ? std::min(*ia, *ib) : 0;
        bool current = adjacent;
        for (size_t aspect = 0; current && aspect < 2; ++aspect) {
            const auto& entry = states.spans[aspect];
            if (span < entry.size() && entry[span].row && !entry[span].current) current = false;
        }
        if (!current)
            return "approved span between control points " + std::to_string(a) + " and " + std::to_string(b) +
                   " changed outside its piece";
    }
    return std::nullopt;
}

std::optional<vc::objects::FiberPieceRange> openedPieceRange(const PieceLockState& state,
                                                             const std::vector<LineControlPoint>& controls,
                                                             const FiberPieceContent& content)
{
    if (!state.openedControls || !state.plan) return std::nullopt;
    const auto [first, last] = *state.openedControls;
    if (last >= controls.size()) return std::nullopt;
    const bool openLeft = first == 0 && state.plan->a.fiberEnd;
    const bool openRight = last + 1 == controls.size() && state.plan->b.fiberEnd;
    const int64_t firstId = controls[first].storeId, lastId = controls[last].storeId;
    if (openLeft) {
        if (lastId < 0 || controls[last - 1].storeId < 0) return std::nullopt;
        return vc::objects::fiberPieceRange(content, lastId, std::nullopt, controls[last - 1].storeId);
    }
    if (firstId < 0 || controls[first + 1].storeId < 0) return std::nullopt;
    if (openRight) return vc::objects::fiberPieceRange(content, firstId, std::nullopt, controls[first + 1].storeId);
    if (lastId < 0) return std::nullopt;
    return vc::objects::fiberPieceRange(content, firstId, lastId, controls[first + 1].storeId);
}

} // namespace vc3d::line_annotation
