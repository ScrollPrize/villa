#include "Fiber3DData.hpp"

#include "LineAnnotationPolylineIndices.hpp"
#include "vc/core/types/FiberCollection.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace vc3d::fibers3d
{
namespace
{
using Json = nlohmann::json;

void checkCancelled(const std::atomic_bool* cancelled)
{
    if (cancelled && cancelled->load())
        throw std::runtime_error("Fiber 3D query cancelled");
}

bool positive(double x)
{
    return std::isfinite(x) && x > 0.0;
}
bool finite(const Point& p)
{
    return std::all_of(p.begin(), p.end(), [](double x) { return std::isfinite(x); });
}

void validateBounds(const Bounds& bounds)
{
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(bounds.lower[i]) || !std::isfinite(bounds.upper[i]) || bounds.lower[i] > bounds.upper[i])
            throw std::runtime_error("Invalid 3D fiber region");
}

void validateOptions(const QueryOptions& options)
{
    if (!std::isfinite(options.minLengthMm) || options.minLengthMm < 0.0)
        throw std::runtime_error("Invalid minimum fiber length");
    if (options.byteBudget < 2 * sizeof(Point))
        throw std::runtime_error("Fiber 3D geometry budget is too small");
    if (options.bounds)
        validateBounds(*options.bounds);
}

char family(const std::string& value)
{
    return value == "H" ? 'H' : value == "V" ? 'V' : '?';
}

bool included(char family, const QueryOptions& options)
{
    return family == 'H' ? options.horizontal : family == 'V' ? options.vertical : options.unclassified;
}

vc::fibers::FamilyFilter familyFilter(const QueryOptions& options)
{
    using Filter = vc::fibers::FamilyFilter;
    if (options.horizontal && options.vertical)
        return options.unclassified ? Filter::All : Filter::Classified;
    if (options.horizontal)
        return options.unclassified ? Filter::HorizontalOrUnclassified : Filter::Horizontal;
    if (options.vertical)
        return options.unclassified ? Filter::VerticalOrUnclassified : Filter::Vertical;
    return options.unclassified ? Filter::Unclassified : Filter::None;
}

void limited(Scene& scene, const QString& message)
{
    scene.complete = false;
    if (!scene.message.contains(message)) {
        if (!scene.message.isEmpty())
            scene.message += QLatin1Char(' ');
        scene.message += message;
    }
}

Point scaled(const cv::Vec3d& point, double scale)
{
    return {point[0] * scale, point[1] * scale, point[2] * scale};
}

bool tagged(const Json& object, const char* tag)
{
    const auto tags = object.find("tags");
    if (tags == object.end() || !tags->is_array())
        return false;
    return std::any_of(tags->begin(), tags->end(), [tag](const Json& value) { return value.is_string() && value.get<std::string>() == tag; });
}

void styleRange(Fiber& fiber, int64_t begin, int64_t end, SpanStyle style)
{
    // [begin,end) refers to original SEGMENTS, i.e. the point range [begin,end].
    for (auto& run : fiber.runs) {
        const auto first = std::max(begin, run.firstPoint);
        const auto last = std::min(end, run.firstPoint + int64_t(run.points.size()) - 1);
        if (first >= last)
            continue;
        if (run.styles.empty())
            run.styles.assign(run.points.size() - 1, SpanStyle::Solid);
        for (auto i = first; i < last; ++i) {
            auto& target = run.styles[size_t(i - run.firstPoint)];
            // A known gap takes precedence over damage/inferred provenance.
            if (target != SpanStyle::Gap)
                target = style;
        }
    }
}

// Use the same ordered exact-membership contract as the annotation loader,
// but retain only one small AFV block at a time. Controls outside the visible
// ROI still delimit gap/damaged spans correctly inside that ROI.
std::optional<std::vector<size_t>> afvControlIndices(vc::fibers::FiberCollection& reader, int64_t id, const std::vector<cv::Vec3d>& controls, const std::atomic_bool* cancelled)
{
    std::vector<size_t> indices;
    int64_t nextBlock = 0, nextPoint = 0;
    while (indices.size() < controls.size()) {
        checkCancelled(cancelled);
        const auto blocks = reader.fiberBlocks(id, nextBlock, 1);
        if (blocks.empty())
            return std::nullopt;
        const auto& block = blocks.front();
        nextBlock = block.firstSegment + int64_t(block.points.size()) - 1;
        if (nextBlock < nextPoint)
            return std::nullopt;
        size_t offset = size_t(std::max<int64_t>(0, nextPoint - block.firstSegment));
        while (indices.size() < controls.size() && offset < block.points.size()) {
            checkCancelled(cancelled);
            std::vector<cv::Vec3d> remainder;
            remainder.reserve(block.points.size() - offset);
            for (size_t i = offset; i < block.points.size(); ++i) {
                const auto& p = block.points[i];
                remainder.emplace_back(p[0], p[1], p[2]);
            }
            const auto match = line_annotation::orderedControlPointLineIndices({controls[indices.size()]}, remainder);
            if (!match)
                break;
            const size_t index = size_t(block.firstSegment) + offset + match->front();
            indices.push_back(index);
            nextPoint = int64_t(index) + 1;
            offset += match->front() + 1;
        }
        // A block repeats the preceding block's last point. Never match it
        // twice to two distinct controls.
        nextPoint = std::max(nextPoint, nextBlock + 1);
    }
    return indices;
}

void annotateAutomated(Fiber& fiber, vc::fibers::FiberCollection& reader, int64_t pointCount, double scale, const std::atomic_bool* cancelled)
{
    const auto annotation = Json::parse(reader.annotation(int64_t(fiber.id)));
    if (!annotation.is_object())
        throw std::runtime_error("Invalid AFV annotation metadata");
    if (const auto tags = annotation.find("tags"); tags != annotation.end() && tags->is_array())
        for (const auto& tag : *tags)
            if (tag.is_string())
                fiber.tags.push_back(QString::fromStdString(tag.get<std::string>()));

    if (const auto ranges = annotation.find("inferred_gap_point_ranges"); ranges != annotation.end()) {
        if (!ranges->is_array())
            throw std::runtime_error("Invalid AFV inferred gap ranges");
        for (const auto& range : *ranges) {
            if (!range.is_array() || range.size() != 2 || !range[0].is_number_integer() || !range[1].is_number_integer())
                throw std::runtime_error("Invalid AFV inferred gap point range");
            const auto begin = range[0].get<int64_t>(), end = range[1].get<int64_t>();
            if (begin < 0 || end < begin || end >= pointCount)
                throw std::runtime_error("AFV inferred gap point range exceeds the fiber");
            styleRange(fiber, begin, end, SpanStyle::InferredGap);
        }
    }

    const auto cp = annotation.find("control_points");
    if (cp == annotation.end() || !cp->is_array())
        return;
    std::vector<cv::Vec3d> controls;
    std::vector<SpanStyle> spans;
    bool haveTaggedSpans = false;
    for (const auto& entry : *cp) {
        checkCancelled(cancelled);
        const auto& position = entry.is_object() ? entry.at("position") : entry;
        if (!position.is_array() || position.size() != 3)
            throw std::runtime_error("Invalid AFV control point");
        const cv::Vec3d point(position[0].get<double>(), position[1].get<double>(), position[2].get<double>());
        const auto displayPoint = scaled(point, scale);
        if (!finite(displayPoint))
            throw std::runtime_error("Non-finite AFV control point");
        controls.push_back(point);
        fiber.markers.push_back({displayPoint, MarkerKind::ControlPoint});
        if (entry.is_object()) {
            if (tagged(entry, "break"))
                fiber.markers.push_back({displayPoint, MarkerKind::Break});
            if (tagged(entry, "kollesis_termination"))
                fiber.markers.push_back({displayPoint, MarkerKind::KollesisTermination});
        }
        auto style = SpanStyle::Solid;
        if (entry.is_object() && entry.contains("segment_to_next")) {
            const auto& segment = entry.at("segment_to_next");
            if (tagged(segment, "gap"))
                style = SpanStyle::Gap;
            else if (tagged(segment, "damaged"))
                style = SpanStyle::Damaged;
        }
        spans.push_back(style);
        haveTaggedSpans |= style != SpanStyle::Solid;
    }
    if (!haveTaggedSpans)
        return;
    const auto indices = afvControlIndices(reader, int64_t(fiber.id), controls, cancelled);
    if (!indices)
        throw std::runtime_error("AFV tagged controls are not ordered members of the fiber");
    for (size_t i = 0; i + 1 < indices->size(); ++i)
        if (spans[i] != SpanStyle::Solid)
            styleRange(fiber, int64_t((*indices)[i]), int64_t((*indices)[i + 1]), spans[i]);
}

bool appendAnnotationRuns(Fiber& fiber, const std::vector<cv::Vec3d>& points, double scale, const QueryOptions& options, size_t& bytes, const std::atomic_bool* cancelled)
{
    Run run;
    auto flush = [&]() {
        if (run.points.size() >= 2)
            fiber.runs.push_back(std::move(run));
        run = {};
    };
    for (size_t i = 1; i < points.size(); ++i) {
        if ((i & 255) == 0)
            checkCancelled(cancelled);
        const auto a = scaled(points[i - 1], scale), b = scaled(points[i], scale);
        if (!finite(a) || !finite(b))
            throw std::runtime_error("Non-finite annotated fiber point");
        double lo, hi;
        if (options.bounds && !vc::fibers::clipSegment(a, b, {options.bounds->lower, options.bounds->upper}, lo, hi)) {
            flush();
            continue;
        }
        const size_t extra = (run.points.empty() ? 2 : 1) * sizeof(Point);
        if (extra > options.byteBudget - bytes) {
            flush();
            return false;
        }
        if (run.points.empty()) {
            run.firstPoint = int64_t(i - 1);
            run.points.push_back(a);
        }
        run.points.push_back(b);
        bytes += extra;
    }
    flush();
    return true;
}
}  // namespace

Scene loadAutomated(const AutomatedSource& source, const QueryOptions& options, const std::atomic_bool* cancelled)
{
    checkCancelled(cancelled);
    validateOptions(options);
    validateBounds(source.volumeBounds);
    if (!positive(source.nativeToViewer))
        throw std::runtime_error("Unknown AFV coordinate frame");
    if (options.minLengthMm > 0.0 && !positive(source.nativeVoxelMm))
        throw std::runtime_error("Cannot filter lengths without a verified AFV voxel size");
    vc::fibers::FiberCollection reader(std::filesystem::path(source.path.toStdString()), 32 * 1024 * 1024);
    const auto uuid = QString::fromStdString(Json::parse(reader.metadata("uuid")).get<std::string>());
    if (source.uuid.isEmpty() || source.uuid != uuid)
        throw std::runtime_error("AFV collection changed; reopen the collection before displaying it");
    Scene scene;
    scene.clipBounds = options.bounds;
    if (positive(source.nativeVoxelMm))
        scene.voxelSizeMm = source.nativeVoxelMm / source.nativeToViewer;
    const auto bounds = options.bounds.value_or(source.volumeBounds);
    vc::fibers::ViewRegion view;
    for (int axis = 0; axis < 3; ++axis) {
        view.box.lower[axis] = bounds.lower[axis] / source.nativeToViewer;
        view.box.upper[axis] = bounds.upper[axis] / source.nativeToViewer;
    }
    view.local = view.box;
    const auto prefix = uuid + QLatin1Char(':');
    int64_t selected = 0;
    if (options.selectedKey.startsWith(prefix))
        selected = options.selectedKey.mid(prefix.size()).toLongLong();
    const size_t probeLimit = options.maximumFibers && options.maximumFibers < SIZE_MAX ? options.maximumFibers + 1 : 0;
    auto page = reader.viewRegion(
        view,
        2,
        probeLimit,
        selected,
        options.byteBudget,
        cancelled,
        positive(source.nativeVoxelMm) ? options.minLengthMm / source.nativeVoxelMm : 0.0,
        familyFilter(options),
        16 * 1024 * 1024);
    if (page.candidateLimitReached) {
        limited(scene, QStringLiteral("Region contains too many candidate blocks; reduce the displayed region."));
        return scene;
    }
    if (!page.complete)
        limited(scene, QStringLiteral("Geometry budget reached; reduce the displayed region."));
    std::unordered_map<int64_t, size_t> index;
    for (auto& block : page.blocks) {
        checkCancelled(cancelled);
        auto found = index.find(block.fiberId);
        if (found == index.end()) {
            if (options.maximumFibers && scene.fibers.size() >= options.maximumFibers) {
                limited(scene, QStringLiteral("Fiber limit reached; only a stable subset is displayed."));
                continue;
            }
            const auto summary = reader.summary(block.fiberId);
            Fiber fiber;
            fiber.key = prefix + QString::number(block.fiberId);
            fiber.id = uint64_t(block.fiberId);
            fiber.source = Source::Automated;
            fiber.label = QString::fromStdString(summary.name);
            fiber.family = family(summary.family);
            if (positive(source.nativeVoxelMm))
                fiber.lengthMm = summary.length * source.nativeVoxelMm;
            found = index.emplace(block.fiberId, scene.fibers.size()).first;
            scene.fibers.push_back(std::move(fiber));
        }
        Run run;
        run.firstPoint = block.firstSegment;
        run.points = std::move(block.points);
        for (auto& point : run.points) {
            for (auto& x : point)
                x *= source.nativeToViewer;
            if (!finite(point))
                throw std::runtime_error("Non-finite AFV point");
        }
        scene.fibers[found->second].runs.push_back(std::move(run));
    }
    for (auto& fiber : scene.fibers) {
        checkCancelled(cancelled);
        annotateAutomated(fiber, reader, reader.summary(int64_t(fiber.id)).pointCount, source.nativeToViewer, cancelled);
    }
    checkCancelled(cancelled);
    return scene;
}

Scene loadAnnotations(
    const LineAnnotationController::FiberMapSnapshot& snapshot,
    const std::vector<LineAnnotationController::FiberSummary>& summaries,
    const QueryOptions& options,
    const std::atomic_bool* cancelled)
{
    checkCancelled(cancelled);
    validateOptions(options);
    Scene scene;
    scene.clipBounds = options.bounds;
    // An AFV-only project has no native geometry or native frame to resolve.
    if (std::none_of(snapshot.fibers.begin(), snapshot.fibers.end(), [](const auto& fiber) { return !fiber.readOnly; }))
        return scene;
    if (!annotation::hasCoordinateFrame(snapshot.frame))
        throw std::runtime_error("Unknown annotation coordinate frame");
    const double scale = 1.0 / snapshot.frame.factor;
    const bool physical = snapshot.voxelSizeUm && positive(*snapshot.voxelSizeUm);
    if (options.minLengthMm > 0.0 && !physical)
        throw std::runtime_error("Cannot filter lengths without a verified annotation voxel size");
    if (physical)
        scene.voxelSizeMm = *snapshot.voxelSizeUm * snapshot.frame.factor / 1000.0;
    std::unordered_map<uint64_t, const LineAnnotationController::FiberSummary*> summaryById;
    for (const auto& summary : summaries)
        summaryById.emplace(summary.id, &summary);
    std::vector<const LineAnnotationController::FiberMapFiber*> order;
    for (const auto& fiber : snapshot.fibers)
        order.push_back(&fiber);
    std::stable_sort(order.begin(), order.end(), [&](const auto* a, const auto* b) {
        const auto aKey = QStringLiteral("annotation:%1").arg(a->id);
        const auto bKey = QStringLiteral("annotation:%1").arg(b->id);
        if ((aKey == options.selectedKey) != (bKey == options.selectedKey))
            return aKey == options.selectedKey;
        return aKey < bKey;
    });
    size_t bytes = 0;
    for (const auto* source : order) {
        checkCancelled(cancelled);
        if (source->readOnly || !included(source->hvTag, options) || source->linePoints.size() < 2)
            continue;
        Fiber fiber;
        // Runtime IDs are unique among loaded sources, including identical
        // filenames from different source roots. A reload invalidates the
        // controller's generation and selection along with these IDs.
        fiber.key = QStringLiteral("annotation:%1").arg(source->id);
        fiber.fileName = QString::fromStdString(source->fileName);
        fiber.id = source->id;
        fiber.label = source->label;
        fiber.family = source->hvTag;
        if (const auto summary = summaryById.find(source->id); summary != summaryById.end())
            for (const auto& tag : summary->second->tags)
                fiber.tags.push_back(QString::fromStdString(tag));
        if (physical) {
            double length = 0.0;
            for (size_t i = 1; i < source->linePoints.size(); ++i) {
                if ((i & 255) == 0)
                    checkCancelled(cancelled);
                const auto delta = source->linePoints[i] - source->linePoints[i - 1];
                length += cv::norm(delta);
            }
            fiber.lengthMm = length * *snapshot.voxelSizeUm / 1000.0;
            if (fiber.lengthMm < options.minLengthMm)
                continue;
        }
        const bool complete = appendAnnotationRuns(fiber, source->linePoints, scale, options, bytes, cancelled);
        if (!fiber.runs.empty()) {
            if (options.maximumFibers && scene.fibers.size() >= options.maximumFibers) {
                limited(scene, QStringLiteral("Fiber limit reached; only a stable subset is displayed."));
                break;
            }
            const auto indices = line_annotation::orderedControlPointLineIndices(source->controlPoints, source->linePoints);
            if (!indices)
                throw std::runtime_error("Annotated controls are not ordered members of the fiber");
            for (size_t i = 0; i + 1 < indices->size(); ++i) {
                if (source->gapSegments.size() + 1 == indices->size() && source->gapSegments[i])
                    styleRange(fiber, int64_t((*indices)[i]), int64_t((*indices)[i + 1]), SpanStyle::Gap);
                else if (source->damagedSegments.size() + 1 == indices->size() && source->damagedSegments[i])
                    styleRange(fiber, int64_t((*indices)[i]), int64_t((*indices)[i + 1]), SpanStyle::Damaged);
            }
            for (size_t i = 0; i < source->controlPoints.size(); ++i) {
                const auto point = scaled(source->controlPoints[i], scale);
                fiber.markers.push_back({point, MarkerKind::ControlPoint});
                if (source->breaks.size() == source->controlPoints.size() && source->breaks[i])
                    fiber.markers.push_back({point, MarkerKind::Break});
                if (source->kollesisTerminations.size() == source->controlPoints.size() && source->kollesisTerminations[i])
                    fiber.markers.push_back({point, MarkerKind::KollesisTermination});
            }
            scene.fibers.push_back(std::move(fiber));
        }
        if (!complete) {
            limited(scene, QStringLiteral("Geometry budget reached; reduce the displayed region."));
            break;
        }
    }
    checkCancelled(cancelled);
    return scene;
}
}  // namespace vc3d::fibers3d
