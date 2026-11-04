#include "Fiber3DData.hpp"
#include "vc/fiber_tracer/FiberDisplay.hpp"
#include "LineAnnotationPolylineIndices.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "vc/core/types/FiberView.hpp"
#include <vc_object_store/store/GeometryViews.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

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

QString storeFiberKey(const vc::objects::AttachmentIdentity& attachment, const std::string& uuid)
{
    return QString::fromStdString(attachment.storeUuid) + QLatin1Char(':') + QString::fromStdString(uuid);
}

Scene loadStore(const StoreSource& source, const QueryOptions& options, const std::atomic_bool* cancelled)
{
    validateOptions(options);
    checkCancelled(cancelled);
    if (!source.session || source.volume.volume.empty() || !positive(source.voxelSizeMm))
        throw std::runtime_error("No object store or verified CT coordinate frame");
    Scene scene;
    scene.voxelSizeMm = source.voxelSizeMm;
    scene.clipBounds = options.bounds;
    if (!options.horizontal && !options.vertical && !options.unclassified)
        return scene;
    size_t bytes = 0;
    source.session->read(source.attachment, [&](const auto& db, const auto& txn, const auto& detached) {
        auto check = [&] {
            checkCancelled(cancelled);
            checkCancelled(&detached);
        };
        vc::objects::GeometrySnapshot snapshot(db, txn);
        vc::objects::ObjectSelection selection;
        selection.kind = "fiber";
        selection.volume = source.volume.volume;
        selection.targetScale = source.volume.scale;
        selection.orderBy = "length_desc";
        selection.limit = 128;
        if (options.minLengthMm > 0)
            selection.minLength = options.minLengthMm / source.voxelSizeMm;
        if (options.bounds) {
            const auto& b = *options.bounds;
            selection.bbox = {b.lower[0], b.lower[1], b.lower[2], b.upper[0], b.upper[1], b.upper[2]};
        }
        if (source.onlyUuid)
            selection.uuids = {*source.onlyUuid};
        const auto transforms = db.volumeTransforms(txn);
        bool exhausted = false;
        auto append = [&](const vc::objects::ObjectRecord& record) {
            check();
            if (!record.fiber || !included(family(record.fiber->family), options))
                return;
            if (options.maximumFibers && scene.fibers.size() >= options.maximumFibers) {
                limited(scene, QStringLiteral("Fiber limit reached; increase it in Settings or narrow the region."));
                exhausted = true;
                return;
            }
            const auto view = snapshot.viewTransformedAt(record.uuid, source.volume.volume, source.volume.scale, &transforms);
            const auto& canonical = std::get<vc::objects::FiberView>(view.canonical());
            const int64_t n = canonical.linePointCount() ? canonical.linePointCount() : canonical.controlPointCount();
            const auto nc = canonical.controlPointCount();
            if (n < 2)
                return;
            // Bound the temporary complete-fiber read independently of the scene.
            // A too-large object is reported, never silently decimated.
            constexpr uint64_t workBudget = 64 * 1024 * 1024;
            if (uint64_t(n) > workBudget / (4 * sizeof(Point)) || uint64_t(nc) > workBudget / 2048) {
                limited(scene, QStringLiteral("A fiber exceeds the per-object read budget."));
                return;
            }
            Fiber fiber;
            fiber.objectUuid = record.uuid;
            fiber.key = storeFiberKey(source.attachment, record.uuid);
            fiber.label = QString::fromStdString(record.name.empty() ? record.uuid : record.name);
            fiber.source = Source::Store;
            fiber.fileName = QString::fromStdString(record.uuid + ".json");
            fiber.family = family(record.fiber->family);
            fiber.lengthMm = (record.volumeLength ? *record.volumeLength
                                                  : db.geometryLengthInVolume(txn, record.uuid, source.volume.volume, source.volume.scale)) *
                             source.voxelSizeMm;
            for (const auto& tag : record.tags)
                fiber.tags.push_back(QString::fromStdString(tag));
            const char* array = canonical.linePointCount() ? "line_points" : "control_points";
            const auto points = view.points(array, 0, n);
            std::vector<cv::Vec3d> displayed;
            displayed.reserve(points.size());
            for (const auto& p : points)
                displayed.emplace_back(p[0], p[1], p[2]);
            const bool complete = appendAnnotationRuns(fiber, displayed, 1.0, options, bytes, cancelled);
            if (!fiber.runs.empty()) {
                const auto metadata = Json::parse(record.meta).value("source", Json::object());
                for (const auto& range : metadata.value("inferred_gap_point_ranges", Json::array())) {
                    if (!range.is_array() || range.size() != 2)
                        throw std::runtime_error("Invalid inferred gap range");
                    const auto a = range[0].get<int64_t>(), b = range[1].get<int64_t>();
                    if (a < 0 || b < a || b >= n)
                        throw std::runtime_error("Inferred gap outside fiber");
                    styleRange(fiber, a, b, SpanStyle::InferredGap);
                }
                const auto controls = canonical.controlPoints(0, nc);
                auto attrs = Json::parse(db.geometryPointAnnotations(txn, canonical.revision().uuid, "control_points"));
                const auto attrBytes = attrs.dump().size();
                const size_t overhead = sizeof(Fiber) + size_t(nc) * (3 * sizeof(Marker) + 2 * sizeof(std::optional<cv::Vec3d>)) +
                                        fiber.runs.size() * sizeof(Run) + size_t(n) * sizeof(SpanStyle) + attrBytes;
                if (overhead > options.byteBudget - bytes) {
                    limited(scene, QStringLiteral("Memory budget reached; increase it in Settings or narrow the region."));
                    exhausted = true;
                    return;
                }
                bytes += overhead;
                std::vector<cv::Vec3d> cp, line;
                for (const auto& p : controls)
                    cp.emplace_back(p[0], p[1], p[2]);
                // Match in canonical coordinates, before affine rounding.
                if (nc) {
                    const auto original = canonical.linePointCount() ? canonical.linePoints(0, n) : controls;
                    for (const auto& p : original)
                        line.emplace_back(p[0], p[1], p[2]);
                }
                const auto indices = line_annotation::orderedControlPointLineIndices(cp, line);
                for (size_t i = 0; i < controls.size(); ++i) {
                    check();
                    const auto p = view.mapping().apply(controls[i]);
                    const auto entry = i < attrs.size() && attrs[i].is_object() ? attrs[i] : Json::object();
                    fiber.markers.push_back({p, MarkerKind::ControlPoint});
                    if (tagged(entry, "break"))
                        fiber.markers.push_back({p, MarkerKind::Break});
                    if (tagged(entry, "kollesis_termination"))
                        fiber.markers.push_back({p, MarkerKind::KollesisTermination});
                    auto normal = vc::fiber_tracer::displayNormalFromJson(entry);
                    auto direction = vc::fiber_tracer::controlDirectionFromJson(entry);
                    if (normal) {
                        const auto v = view.mapping().mapNormal({(*normal)[0], (*normal)[1], (*normal)[2]});
                        normal = cv::Vec3d(v[0], v[1], v[2]);
                    }
                    if (direction) {
                        const auto v = view.mapping().mapDirection({(*direction)[0], (*direction)[1], (*direction)[2]});
                        direction = cv::Vec3d(v[0], v[1], v[2]);
                    }
                    fiber.displayNormals.push_back(normal);
                    fiber.controlDirections.push_back(direction);
                    const auto span = entry.value("segment_to_next", Json::object());
                    const auto style = tagged(span, "gap")       ? SpanStyle::Gap
                                       : tagged(span, "damaged") ? SpanStyle::Damaged
                                                                 : SpanStyle::Solid;
                    if (style != SpanStyle::Solid && i + 1 < controls.size()) {
                        if (!indices)
                            throw std::runtime_error("Tagged controls are not ordered members of the fiber");
                        styleRange(fiber, (*indices)[i], (*indices)[i + 1], style);
                    }
                }
                scene.fibers.push_back(std::move(fiber));
            }
            if (!complete) {
                limited(scene, QStringLiteral("Memory budget reached; increase it in Settings or narrow the region."));
                exhausted = true;
            }
        };
        // Priority does not bypass family, length, coordinate or ROI filters.
        const auto prefix = QString::fromStdString(source.attachment.storeUuid) + QLatin1Char(':');
        std::string selected;
        if (options.selectedKey.startsWith(prefix))
            selected = options.selectedKey.mid(prefix.size()).toStdString();
        if (!selected.empty() && (!source.onlyUuid || *source.onlyUuid == selected)) {
            auto first = selection;
            first.uuids = {selected};
            for (const auto& record : db.selectObjects(txn, first, std::vector<std::string>{"name", "tags", "fiber", "meta"}))
                append(record);
        }
        while (!exhausted) {
            check();
            const auto records = db.selectObjects(txn, selection, std::vector<std::string>{"name", "tags", "fiber", "meta"});
            for (const auto& record : records) {
                if (record.uuid != selected)
                    append(record);
                if (exhausted)
                    break;
            }
            if (records.size() < size_t(*selection.limit) || source.onlyUuid)
                break;
            selection.afterUuid = records.back().uuid;
            selection.afterLength = records.back().volumeLength;
        }
        check();
    });
    checkCancelled(cancelled);
    if (!source.session->current(source.attachment))
        throw std::runtime_error("Object store detached during fiber load");
    return scene;
}
Fiber generatedAnnotationFiber(const line_annotation::GeneratedViews& views)
{
    Fiber fiber;
    const auto gaps = line_annotation::generatedGapLineRanges(views.controlPoints);
    const auto damaged = line_annotation::generatedDamagedLineRanges(views.controlPoints);
    Run run;
    for (size_t i = 0; i < views.linePoints.size(); ++i) {
        const auto& p = views.linePoints[i];
        if (!line_annotation::finiteGeneratedPoint(p)) {
            if (run.points.size() > 1)
                fiber.runs.push_back(std::move(run));
            run = {};
            run.firstPoint = int64_t(i + 1);
        } else {
            if (!run.points.empty() && (!gaps.empty() || !damaged.empty())) {
                const auto style = line_annotation::generatedLineSegmentInGap(double(i - 1), double(i), gaps)      ? SpanStyle::Gap
                                   : line_annotation::generatedLineSegmentInGap(double(i - 1), double(i), damaged) ? SpanStyle::Damaged
                                                                                                                   : SpanStyle::Solid;
                run.styles.push_back(style);
            }
            run.points.push_back({p[0], p[1], p[2]});
        }
    }
    if (run.points.size() > 1)
        fiber.runs.push_back(std::move(run));
    for (const auto& cp : views.controlPoints) {
        if (!line_annotation::finiteGeneratedPoint(cp.point))
            continue;
        const Point point{cp.point[0], cp.point[1], cp.point[2]};
        fiber.markers.push_back({point, MarkerKind::ControlPoint});
        if (cp.isBreak)
            fiber.markers.push_back({point, MarkerKind::Break});
        if (cp.isKollesisTermination)
            fiber.markers.push_back({point, MarkerKind::KollesisTermination});
    }
    return fiber;
}
}  // namespace vc3d::fibers3d
