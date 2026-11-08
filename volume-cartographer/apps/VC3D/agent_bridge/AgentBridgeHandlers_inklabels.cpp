#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QDir>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QPointer>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "CState.hpp"
#include "CWindow.hpp"
#include "ViewerManager.hpp"
#include "ink_labels/InkLabelController.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/VolumePkg.hpp"

using namespace vc3d::ink_labels;
namespace fs = std::filesystem;

namespace {

constexpr int kSaveTimeoutMs = 300000;
constexpr int kImportTimeoutMs = 300000;
constexpr int kMaxMaskSide = 8192;
constexpr int kUndoListed = 10;
// Suggestions listed in one reply (the wire limit is 1 MiB).
constexpr int kMaxSuggestionsListed = 2000;

QString qstr(const std::string& s)
{
    return QString::fromStdString(s);
}

QString qpath(const fs::path& p)
{
    return QString::fromStdString(p.string());
}

[[noreturn]] void fail(int code, const QString& message, QJsonObject data = {})
{
    throw AgentBridgeError{code, message, std::move(data)};
}

[[noreturn]] void failKind(int code, const QString& message, const QString& kind, QJsonObject data = {})
{
    data["kind"] = kind;
    throw AgentBridgeError{code, message, std::move(data)};
}

QString className(PaintClass cls)
{
    switch (cls) {
        case PaintClass::Ink:
            return QStringLiteral("ink");
        case PaintClass::Background:
            return QStringLiteral("background");
        case PaintClass::Validation:
            return QStringLiteral("validation");
    }
    return {};
}

QString actorName(Actor actor)
{
    return actor == Actor::Agent ? QStringLiteral("agent") : QStringLiteral("user");
}

QJsonArray undoList(const History* history)
{
    QJsonArray list;
    if (!history)
        return list;
    const auto& entries = history->undoEntries();
    int listed = 0;
    for (auto it = entries.rbegin(); it != entries.rend() && listed < kUndoListed; ++it, ++listed) {
        QJsonObject e;
        e["serial"] = double(it->serial);
        e["label"] = qstr(it->label);
        e["actor"] = actorName(it->actor);
        list.append(e);
    }
    return list;
}

QJsonArray filesJson(const std::vector<std::pair<LabelKind, fs::path>>& files, const std::map<LabelKind, std::string>& sha)
{
    QJsonArray out;
    for (const auto& [kind, path] : files) {
        QJsonObject f;
        f["kind"] = QLatin1String(kindName(kind));
        f["path"] = qpath(path);
        const auto it = sha.find(kind);
        f["sha256"] = it == sha.end() ? QString() : qstr(it->second);
        out.append(f);
    }
    return out;
}

QJsonObject countsJson(const LabelCounts& c)
{
    QJsonObject o;
    o["ink"] = double(c.ink);
    o["background"] = double(c.background);
    o["validation"] = double(c.validation);
    o["unlabeled"] = double(c.unlabeled);
    return o;
}

QJsonObject rectJson(const cv::Rect& rect)
{
    QJsonObject o;
    o["x"] = rect.x;
    o["y"] = rect.y;
    o["width"] = rect.width;
    o["height"] = rect.height;
    return o;
}

QJsonObject assistJson(const InkLabelController& labels)
{
    const InkLabelController::AssistSettings& a = labels.assistSettings();
    QJsonObject settings;
    settings["guide"] = a.guide == GuideKind::Prediction ? QStringLiteral("prediction") : QStringLiteral("labels");
    settings["threshold"] = int(a.threshold);
    settings["above"] = a.above;
    settings["connectivity"] = a.connectivity;
    settings["unlabeledOnly"] = a.unlabeledOnly;
    settings["median"] = a.median;
    settings["minArea"] = a.minArea;
    settings["cleanupSize"] = a.cleanupSize;
    QJsonObject o;
    o["running"] = labels.assistRunning();
    o["suggestions"] = labels.suggestions() ? int(labels.suggestions()->items.size()) : 0;
    o["picked"] = int(labels.selectedSuggestions().size());
    o["settings"] = settings;
    return o;
}

// The prediction selected in the Ink Detection dock, which fill and
// suggestions read, and its pixel size in label pixels.
QJsonValue predictionJson(const InkLabelController& labels, const InkDetectionOverlayController* ink)
{
    if (!ink || ink->selectedPath().empty())
        return QJsonValue::Null;
    QJsonObject o;
    o["path"] = qpath(ink->selectedPath());
    o["singleChannel"] = ink->selectedIsSingleChannel();
    o["hflip"] = ink->horizontalFlip();
    o["vflip"] = ink->verticalFlip();
    const cv::Mat_<uint8_t>& scalar = ink->selectedScalar();
    const InkLabelController::SessionInfo* s = labels.session();
    if (!scalar.empty()) {
        o["width"] = scalar.cols;
        o["height"] = scalar.rows;
        if (s) {
            // The overlay stretches the prediction over the whole grid.
            o["labelPxPerPixelX"] = double(s->canvas.size.width) / scalar.cols;
            o["labelPxPerPixelY"] = double(s->canvas.size.height) / scalar.rows;
        }
    }
    return o;
}

QJsonObject stateJson(const InkLabelController& labels, const InkDetectionOverlayController* ink)
{
    QJsonObject r;
    const InkLabelController::SessionInfo* s = labels.session();
    r["open"] = s != nullptr;
    r["labelMode"] = labels.labelModeEnabled();
    r["class"] = className(labels.paintClass());
    r["erase"] = labels.erase();
    r["backgroundKeepsInk"] = labels.backgroundKeepsInk();
    r["tool"] = labels.toolName();
    r["radius"] = labels.radius();
    if (!s) {
        r["readOnly"] = false;
        r["locked"] = false;
        r["lockReason"] = QJsonValue::Null;
        r["segment"] = QJsonValue::Null;
        r["store"] = QJsonValue::Null;
        r["canvas"] = QJsonValue::Null;
        r["dirty"] = false;
        r["stateSerial"] = 0;
        r["undo"] = QJsonArray();
        r["redoDepth"] = 0;
        r["historyBytes"] = 0;
        r["counts"] = QJsonValue::Null;
        r["lastSave"] = QJsonValue::Null;
        r["autosave"] = QJsonValue::Null;
        r["versions"] = QJsonArray();
        r["assist"] = assistJson(labels);
        r["prediction"] = predictionJson(labels, ink);
        return r;
    }
    r["readOnly"] = s->readOnly;
    r["locked"] = s->locked;
    r["lockReason"] = s->lockReason.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(s->lockReason);
    r["segment"] = qstr(s->segment);
    r["store"] = qpath(s->paths.dir);
    r["version"] = s->paths.version;
    QJsonArray versions;
    for (const int v : labels.versions())
        versions.append(v);
    r["versions"] = versions;
    r["assist"] = assistJson(labels);
    r["prediction"] = predictionJson(labels, ink);
    QJsonObject canvas;
    canvas["width"] = s->canvas.size.width;
    canvas["height"] = s->canvas.size.height;
    canvas["gridCols"] = s->canvas.grid.width;
    canvas["gridRows"] = s->canvas.grid.height;
    canvas["scaleX"] = double(s->canvas.gridScale[0]);
    canvas["scaleY"] = double(s->canvas.gridScale[1]);
    canvas["renderScale"] = s->canvas.renderScale;
    // Label px = center + (renderScale / scale) * (grid - gridCols / 2).
    canvas["centerX"] = 0.5 * s->canvas.size.width;
    canvas["centerY"] = 0.5 * s->canvas.size.height;
    r["canvas"] = canvas;
    r["dirty"] = s->dirty;
    r["stateSerial"] = double(s->stateSerial);
    const History* history = labels.history();
    r["undo"] = undoList(history);
    r["undoDepth"] = history ? double(history->undoDepth()) : 0.0;
    r["redoDepth"] = history ? double(history->redoDepth()) : 0.0;
    r["historyBytes"] = history ? double(history->bytes()) : 0.0;
    r["counts"] = s->locked ? QJsonValue(QJsonValue::Null) : QJsonValue(countsJson(LabelCounts::from(s->counts)));
    if (const auto& last = labels.lastSave(); last) {
        QJsonObject save;
        save["atUtc"] = last->atUtc.isValid() ? last->atUtc.toUTC().toString(Qt::ISODate) : QString();
        save["files"] = filesJson(last->files, last->sha256);
        r["lastSave"] = save;
    } else {
        r["lastSave"] = QJsonValue::Null;
    }
    if (labels.lastAutosave().isValid()) {
        QJsonObject autosave;
        autosave["atUtc"] = labels.lastAutosave().toUTC().toString(Qt::ISODate);
        r["autosave"] = autosave;
    } else {
        r["autosave"] = QJsonValue::Null;
    }
    return r;
}

// Maps a failed open to its bridge error.
[[noreturn]] void failOpen(const InkLabelController::OpenOutcome& out)
{
    using E = InkLabelController::OpenError;
    switch (out.error) {
        case E::NoProject:
            fail(-32000, out.message);
        case E::NoSegment:
            failKind(-32007, out.message, QStringLiteral("segment"));
        case E::InvalidScale:
            failKind(-32009, out.message, QStringLiteral("invalid_scale"));
        case E::LayerLocked: {
            QJsonObject data;
            data["expected"] = out.expected;
            data["got"] = out.got;
            failKind(-32009, out.message, QStringLiteral("label_layer_locked"), data);
        }
        case E::AutosavePending: {
            QJsonObject data;
            data["param"] = QStringLiteral("recover");
            data["autosaveAtUtc"] = out.autosaveAtUtc;
            failKind(-32602, out.message, QStringLiteral("autosave_pending"), data);
        }
        case E::SaveInFlight:
            failKind(-32004, out.message, QStringLiteral("ink_label_save"));
        case E::LockedByOther:
            failKind(-32009, out.message, QStringLiteral("locked_by_other"));
        case E::Busy:
            failKind(-32009, out.message, QStringLiteral("open_pending"));
        case E::NoVersion:
            throwParamError(QStringLiteral("version"), out.message);
        case E::ReadFailed:
        case E::None:
            break;
    }
    QJsonObject data;
    data["detail"] = out.message;
    fail(-32005, QStringLiteral("Cannot open the ink labels"), data);
}

// The open, writable session, or the error that says why there is none.
const InkLabelController::SessionInfo& writableSession(const InkLabelController& labels)
{
    const InkLabelController::SessionInfo* session = labels.session();
    if (!session)
        failKind(-32007, QStringLiteral("No ink label session is open; call ink_labels.open first"),
                 QStringLiteral("ink_label_session"));
    if (session->locked)
        failKind(-32009, session->lockReason, QStringLiteral("label_layer_locked"));
    if (session->readOnly)
        failKind(-32009, session->lockReason, QStringLiteral("read_only"));
    if (labels.strokeOpen())
        failKind(-32009, QStringLiteral("A brush stroke is in progress in the viewer"), QStringLiteral("stroke_open"));
    return *session;
}

// A writable session where a fill, suggestion pass or cleanup can start.
void requireAssist(const InkLabelController& labels)
{
    writableSession(labels);
    if (labels.assistRunning())
        failKind(-32004, QStringLiteral("A fill, suggestion or cleanup from the dock is still running"),
                 QStringLiteral("ink_label_assist"));
}

std::optional<cv::Rect> boundParam(const QJsonObject& p)
{
    if (!p.contains("bound"))
        return std::nullopt;
    const QJsonObject b = p.value("bound").toObject();
    const cv::Rect rect(b.value("x").toInt(), b.value("y").toInt(), b.value("width").toInt(), b.value("height").toInt());
    if (rect.empty())
        throwParamError(QStringLiteral("bound"), QStringLiteral("bound must have a positive width and height"));
    return rect;
}

// A label pixel from a label-space {x, y} or, through `viewer`, a
// volume-space {x, y, z} point, round-tripped within 2 voxels as
// canvas.click does.
cv::Point2d labelPoint(const InkLabelController& labels, VolumeViewerBase* viewer, const QJsonValue& value, const QString& param)
{
    const QJsonObject o = value.toObject();
    cv::Point2d label;
    if (!viewer) {
        label = {jsonRequireFinite(o.value("x"), "x"), jsonRequireFinite(o.value("y"), "y")};
    } else {
        if (!o.contains("z"))
            throwParamError(param, QStringLiteral("%1 needs x, y and z in volume space").arg(param));
        const cv::Vec3f vol = jsonToVec3(value, "points");
        const QPointF scene = viewer->volumeToScene(vol);
        const cv::Vec3f back = viewer->sceneToVolume(scene);
        const double dist = cv::norm(back - vol);
        const auto mapped = labels.sceneToLabel(viewer, scene);
        if (!std::isfinite(dist) || dist > 2.0 || !mapped) {
            QJsonObject data;
            data["point"] = vec3ToJson(vol);
            data["detail"] = QStringLiteral("point is not on the labeled segment (round-trip %1 voxels)").arg(dist, 0, 'f', 3);
            fail(-32003, QStringLiteral("Invalid coordinates"), data);
        }
        label = *mapped;
    }
    // The rasterizer works in fixed point; such points cannot be painted.
    if (std::abs(label.x) > kMaxShapeCoordinate || std::abs(label.y) > kMaxShapeCoordinate)
        throwParamError(param, QStringLiteral("%1 must lie within %2 label px of the canvas origin").arg(param).arg(kMaxShapeCoordinate, 0, 'f', 0));
    return label;
}

// An integer parameter that must be one of a few values.
int checkedChoice(const QJsonObject& o, const char* name, std::initializer_list<int> allowed)
{
    const int value = o.value(QLatin1String(name)).toInt();
    if (std::find(allowed.begin(), allowed.end(), value) == allowed.end()) {
        QStringList names;
        for (const int v : allowed)
            names << QString::number(v);
        throwParamError(QString::fromLatin1(name),
                        QStringLiteral("%1 must be one of %2").arg(QLatin1String(name), names.join(QStringLiteral(", "))));
    }
    return value;
}

QJsonObject editResult(const InkLabelController& labels, uint64_t changed)
{
    QJsonObject result;
    result["changedPixels"] = double(changed);
    const InkLabelController::SessionInfo* after = labels.session();
    result["stateSerial"] = double(after ? after->stateSerial : 0);
    result["undoDepth"] = labels.history() ? double(labels.history()->undoDepth()) : 0.0;
    return result;
}

}  // namespace


InkLabelController* AgentBridgeServer::inkLabelController() const
{
    InkLabelController* labels = _window ? _window->_inkLabelController.get() : nullptr;
    if (!labels)
        fail(-32010, QStringLiteral("Ink labels are not available"));
    return labels;
}


const InkDetectionOverlayController* AgentBridgeServer::inkDetectionOverlay() const
{
    return _window && _window->_viewerManager ? _window->_viewerManager->inkDetectionOverlay() : nullptr;
}


void AgentBridgeServer::requireInkLabelProject() const
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg())
        fail(-32000, QStringLiteral("No volume package loaded"));
}


QJsonObject AgentBridgeServer::handleInkLabelsState(const QJsonValue&)
{
    requireInkLabelProject();
    return stateJson(*inkLabelController(), inkDetectionOverlay());
}


QJsonObject AgentBridgeServer::handleInkLabelsOpen(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();

    InkLabelController::OpenOptions options;
    const QString recover = p.value("recover").toString(QStringLiteral("fail"));
    options.recover = recover == QLatin1String("restore")   ? RecoverPolicy::Restore
                      : recover == QLatin1String("discard") ? RecoverPolicy::Discard
                                                            : RecoverPolicy::Fail;
    options.readOnlyIfLocked = p.value("readOnlyIfLocked").toBool(true);
    if (p.contains("version"))
        options.version = p.value("version").toInt(1);
    if (p.value("startNewVersion").toBool(false) && p.contains("version"))
        throwParamError(QStringLiteral("startNewVersion"), QStringLiteral("give version or startNewVersion, not both"));

    const InkLabelController::OpenOutcome out =
        p.value("startNewVersion").toBool(false) ? labels->startNewVersion() : labels->openForActiveSegment(options);
    if (!out.ok)
        failOpen(out);
    QJsonObject result = stateJson(*labels, inkDetectionOverlay());
    result["restoredAutosave"] = out.restoredAutosave;
    result["warnings"] = QJsonArray::fromStringList(out.warnings);
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsPaint(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const InkLabelController::SessionInfo* session = labels->session();
    if (!session)
        failKind(-32007, QStringLiteral("No ink label session is open; call ink_labels.open first"),
                 QStringLiteral("ink_label_session"));

    // Exactly one shape, validated before anything is touched.
    const int shapes = int(p.contains("stroke")) + int(p.contains("polygon")) + int(p.contains("mask"));
    if (shapes != 1)
        throwParamError(QStringLiteral("stroke"), QStringLiteral("give exactly one of stroke, polygon or mask"));
    const bool values = p.contains("mask") && p.value("mask").toObject().value("format").toString() == QLatin1String("values");
    if (!values && !p.contains("class"))
        throwParamError(QStringLiteral("class"), QStringLiteral("class is required unless mask.format is \"values\""));

    if (p.contains("expect")) {
        const QJsonObject expect = p.value("expect").toObject();
        const bool segmentOk = !expect.contains("segment") || expect.value("segment").toString() == qstr(session->segment);
        const bool serialOk = !expect.contains("serial") || uint64_t(expect.value("serial").toDouble()) == session->stateSerial;
        if (!segmentOk || !serialOk) {
            QJsonObject current;
            current["segment"] = qstr(session->segment);
            current["serial"] = double(session->stateSerial);
            QJsonObject data;
            data["current"] = current;
            data["kind"] = QStringLiteral("expect_mismatch");
            throwParamError(QStringLiteral("expect"), QStringLiteral("The label state changed since the caller looked"), data);
        }
    }
    writableSession(*labels);

    const QString clsName = p.value("class").toString();
    const PaintClass cls = clsName == QLatin1String("background")   ? PaintClass::Background
                           : clsName == QLatin1String("validation") ? PaintClass::Validation
                                                                    : PaintClass::Ink;
    const bool erase = p.value("erase").toBool(false);
    const PaintOp op = paintOp(cls, erase, p.value("backgroundKeepsInk").toBool(false));
    const bool volumeSpace = p.value("space").toString(QStringLiteral("label")) == QLatin1String("volume");
    const cv::Size canvas = session->canvas.size;
    const QString what = (erase ? QStringLiteral("erase ") : QString()) + clsName;

    VolumeViewerBase* viewer = nullptr;
    if (volumeSpace && !p.contains("mask")) {
        viewer = resolveViewer(QJsonValue());
        if (!labels->viewerShowsSession(viewer))
            failKind(-32009,
                     QStringLiteral("The segmentation viewer does not show the labeled segment on its flattened view; "
                                    "segments.activate the segment, or paint in label space"),
                     QStringLiteral("viewer"));
    }

    std::vector<Shape> list;
    uint64_t changed = 0;
    if (p.contains("stroke")) {
        const QJsonObject stroke = p.value("stroke").toObject();
        const QJsonArray points = stroke.value("points").toArray();
        if (points.isEmpty())
            throwParamError(QStringLiteral("stroke.points"), QStringLiteral("stroke.points must not be empty"));
        const double radius = stroke.value("radius").toDouble();
        std::vector<cv::Point2d> pts;
        for (const QJsonValue& v : points)
            pts.push_back(labelPoint(*labels, viewer, v, QStringLiteral("stroke.points")));
        if (pts.size() == 1)
            list.push_back({Shape::Kind::Disc, {pts.front()}, radius});
        for (std::size_t i = 1; i < pts.size(); ++i)
            list.push_back({Shape::Kind::Capsule, {pts[i - 1], pts[i]}, radius});
    } else if (p.contains("polygon")) {
        const QJsonArray points = p.value("polygon").toObject().value("points").toArray();
        if (points.size() < 3)
            throwParamError(QStringLiteral("polygon.points"), QStringLiteral("polygon.points needs at least 3 points"));
        Shape shape{Shape::Kind::Polygon, {}, 0.0};
        for (const QJsonValue& v : points)
            shape.points.push_back(labelPoint(*labels, viewer, v, QStringLiteral("polygon.points")));
        list.push_back(std::move(shape));
    }

    bool clipped = false;
    if (!list.empty()) {
        const CanvasOverlap overlap = canvasOverlap(list, canvas);
        if (overlap == CanvasOverlap::Outside)
            fail(-32003, QStringLiteral("The shape lies entirely off the label canvas (%1 x %2 px)").arg(canvas.width).arg(canvas.height));
        clipped = overlap == CanvasOverlap::Clipped;
        changed = labels->applyShapes(list, op, Actor::Agent,
                                      QStringLiteral("agent %1 %2").arg(what, list.front().kind == Shape::Kind::Polygon
                                                                                  ? QStringLiteral("polygon")
                                                                                  : QStringLiteral("stroke")));
    } else {
        // Mask: an absolute single-channel 8-bit image in label space.
        if (volumeSpace)
            throwParamError(QStringLiteral("space"), QStringLiteral("mask paints are in label space only"));
        const QJsonObject mask = p.value("mask").toObject();
        const QString path = mask.value("path").toString();
        if (!QDir::isAbsolutePath(path))
            throwParamError(QStringLiteral("mask.path"), QStringLiteral("mask.path must be an absolute path"));
        const QString lower = path.toLower();
        if (!lower.endsWith(QLatin1String(".png")) && !lower.endsWith(QLatin1String(".tif")) && !lower.endsWith(QLatin1String(".tiff")))
            throwParamError(QStringLiteral("mask.path"), QStringLiteral("mask.path must be a .png or .tif image"));
        const cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
        if (image.empty())
            throwParamError(QStringLiteral("mask.path"), QStringLiteral("cannot read %1").arg(path));
        if (image.type() != CV_8UC1)
            throwParamError(QStringLiteral("mask.path"), QStringLiteral("mask must be a single 8-bit channel"));
        if (image.cols > kMaxMaskSide || image.rows > kMaxMaskSide)
            throwParamError(QStringLiteral("mask.path"), QStringLiteral("mask is larger than %1 x %1 px").arg(kMaxMaskSide));
        double maxValue = 0;
        cv::minMaxLoc(image, nullptr, &maxValue);
        if (values && maxValue > kValueMask)
            throwParamError(QStringLiteral("mask.path"),
                            QStringLiteral("a values mask holds label values 0 to 7 (bit 1 ink, 2 supervised, 4 validation), "
                                           "as ink_labels.read writes them; it has %1")
                                .arg(maxValue));
        const QJsonObject origin = mask.value("origin").toObject();
        const cv::Point at(origin.value("x").toInt(), origin.value("y").toInt());
        const cv::Rect area(at, image.size());
        const cv::Rect inside = area & cv::Rect(0, 0, canvas.width, canvas.height);
        if (inside.empty())
            fail(-32003, QStringLiteral("The mask lies entirely off the label canvas (%1 x %2 px)").arg(canvas.width).arg(canvas.height));
        clipped = inside != area;
        changed = values ? labels->applyValues(at, cv::Mat_<uint8_t>(image), Actor::Agent, QStringLiteral("agent values"))
                         : labels->applyMask(at, cv::Mat_<uint8_t>(image), op, Actor::Agent, QStringLiteral("agent %1 mask").arg(what));
    }

    QJsonObject result = editResult(*labels, changed);
    result["clipped"] = clipped;
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsRead(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    if (!labels->session() || !labels->raster())
        failKind(-32007, QStringLiteral("No readable ink label session is open"), QStringLiteral("ink_label_session"));

    const QJsonObject r = p.value("rect").toObject();
    const cv::Rect rect(r.value("x").toInt(), r.value("y").toInt(), r.value("width").toInt(), r.value("height").toInt());
    const QString path = p.value("path").toString();
    if (!QDir::isAbsolutePath(path))
        throwParamError(QStringLiteral("path"), QStringLiteral("path must be absolute"));
    const QString lower = path.toLower();
    if (!lower.endsWith(QLatin1String(".png")) && !lower.endsWith(QLatin1String(".tif")) && !lower.endsWith(QLatin1String(".tiff")))
        throwParamError(QStringLiteral("path"), QStringLiteral("path must end in .png or .tif"));
    if (!fs::is_directory(fs::path(path.toStdString()).parent_path()))
        throwParamError(QStringLiteral("path"), QStringLiteral("the folder of path does not exist"));

    const QString formatName = p.value("format").toString(QStringLiteral("values"));
    const auto format = formatName == QLatin1String("ink")           ? InkLabelController::ReadFormat::Ink
                        : formatName == QLatin1String("supervision") ? InkLabelController::ReadFormat::Supervision
                        : formatName == QLatin1String("validation")  ? InkLabelController::ReadFormat::Validation
                                                                     : InkLabelController::ReadFormat::Values;
    const InkLabelController::ReadOutcome out = labels->readRect(rect, format, fs::path(path.toStdString()));
    if (!out.ok) {
        if (out.rect.empty())
            throwParamError(QStringLiteral("rect"), out.error);
        QJsonObject data;
        data["detail"] = out.error;
        fail(-32005, QStringLiteral("Cannot write the label image"), data);
    }
    QJsonObject result;
    result["path"] = path;
    result["rect"] = rectJson(out.rect);
    result["format"] = formatName;
    result["counts"] = countsJson(out.counts);
    result["stateSerial"] = double(labels->session()->stateSerial);
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsHistory(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    writableSession(*labels);

    const bool undo = p.value("action").toString() == QLatin1String("undo");
    const int steps = p.value("steps").toInt(1);
    const bool force = p.value("force").toBool(false);
    int applied = 0;
    bool blocked = false;
    for (int i = 0; i < steps; ++i) {
        const History* history = labels->history();
        if (undo) {
            const HistoryEntry* top = history ? history->top() : nullptr;
            if (!top)
                break;
            if (!force && top->actor != Actor::Agent) {
                blocked = true;
                break;
            }
            if (!labels->undo(Actor::Agent, force))
                break;
        } else if (!labels->redo(Actor::Agent)) {
            break;
        }
        ++applied;
    }
    if (blocked && applied == 0) {
        QJsonObject data;
        data["serial"] = double(labels->history()->top()->serial);
        data["label"] = qstr(labels->history()->top()->label);
        failKind(-32009, QStringLiteral("The last label edit is the user's; pass force to undo it"), QStringLiteral("other_actor_entry"),
                 data);
    }
    QJsonObject result;
    result["applied"] = applied;
    result["stoppedAtOtherActor"] = blocked;
    result["stateSerial"] = double(labels->session()->stateSerial);
    result["undo"] = undoList(labels->history());
    result["redoDepth"] = double(labels->history()->redoDepth());
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsSave(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const InkLabelController::SessionInfo& session = writableSession(*labels);
    if (labels->trainerSaveInFlight())
        failKind(-32004, QStringLiteral("A save of these labels is already running"), QStringLiteral("ink_label_save"));
    const bool asNewVersion = p.value("asNewVersion").toBool(false);
    if (asNewVersion && p.value("overwriteExternalEdits").toBool(false))
        throwParamError(QStringLiteral("asNewVersion"), QStringLiteral("a new version writes new files; overwriteExternalEdits does not apply"));
    if (!session.dirty && !asNewVersion) {
        QJsonObject result;
        result["saved"] = false;
        result["stateSerial"] = double(session.stateSerial);
        result["files"] = QJsonArray();
        return result;
    }

    QFuture<SaveOutcome> future;
    if (asNewVersion) {
        future = labels->saveAsNewVersion();
    } else {
        InkLabelController::SaveOptions options;
        options.overwriteExternalEdits = p.value("overwriteExternalEdits").toBool(false);
        future = labels->save(options);
    }
    if (future.isFinished()) {
        const SaveOutcome outcome = future.result();
        if (!outcome.ok && outcome.externallyEdited.isEmpty()) {
            QJsonObject data;
            data["detail"] = outcome.error;
            fail(-32005, QStringLiteral("ink_labels.save failed"), data);
        }
    }
    const int token = beginDeferred(kSaveTimeoutMs, QStringLiteral("ink label save"));
    QPointer<AgentBridgeServer> self(this);
    QPointer<InkLabelController> labelsGuard(labels);
    auto* watcher = new QFutureWatcher<SaveOutcome>(this);
    connect(watcher, &QFutureWatcherBase::finished, this, [self, labelsGuard, watcher, token]() {
        watcher->deleteLater();
        if (!self)
            return;
        const SaveOutcome outcome = watcher->result();
        if (outcome.ok) {
            QJsonObject result;
            result["saved"] = true;
            result["files"] = filesJson(outcome.files, outcome.sha256);
            result["stateSerial"] = double(outcome.stateSerial);
            result["ms"] = outcome.ms;
            const InkLabelController::SessionInfo* now = labelsGuard ? labelsGuard->session() : nullptr;
            result["version"] = now ? now->paths.version : 0;
            self->completeDeferredResult(token, result);
            return;
        }
        QJsonObject data;
        if (!outcome.externallyEdited.isEmpty()) {
            data["kind"] = QStringLiteral("external_edit");
            data["param"] = QStringLiteral("overwriteExternalEdits");
            data["files"] = QJsonArray::fromStringList(outcome.externallyEdited);
            self->completeDeferredError(token, -32602,
                                        QStringLiteral("Label files changed outside VC3D; pass overwriteExternalEdits to replace them"),
                                        data);
            return;
        }
        data["detail"] = outcome.error;
        self->completeDeferredError(token, -32005, QStringLiteral("ink_labels.save failed"), data);
    });
    watcher->setFuture(future);
    throw AgentBridgeDeferred{};
}


QJsonObject AgentBridgeServer::handleInkLabelsVersions(const QJsonValue&)
{
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const InkLabelController::SessionInfo* session = labels->session();
    if (!session)
        failKind(-32007, QStringLiteral("No ink label session is open; call ink_labels.open first"),
                 QStringLiteral("ink_label_session"));
    QJsonArray versions;
    for (const int v : labels->versions())
        versions.append(v);
    QJsonObject result;
    result["versions"] = versions;
    result["open"] = session->paths.version;
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsDatasetEntry(const QJsonValue&)
{
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const InkLabelController::SessionInfo* session = labels->session();
    if (!session || session->locked)
        failKind(-32007, QStringLiteral("No ink label session with a usable canvas is open"), QStringLiteral("ink_label_session"));
    QJsonObject result;
    result["entry"] = labels->datasetEntry();
    result["store"] = qpath(session->paths.dir);
    result["version"] = session->paths.version;
    result["saved"] = labels->lastSave().has_value();
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsSetToolState(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();

    if (p.contains("class")) {
        const QString name = p.value("class").toString();
        labels->setPaintClass(name == QLatin1String("background")   ? PaintClass::Background
                              : name == QLatin1String("validation") ? PaintClass::Validation
                                                                    : PaintClass::Ink);
    }
    if (p.contains("erase"))
        labels->setErase(p.value("erase").toBool());
    if (p.contains("backgroundKeepsInk"))
        labels->setBackgroundKeepsInk(p.value("backgroundKeepsInk").toBool());
    if (p.contains("tool")) {
        const QString name = p.value("tool").toString();
        labels->setTool(name == QLatin1String("polygon") ? InkLabelController::Tool::Polygon
                        : name == QLatin1String("fill")  ? InkLabelController::Tool::Fill
                                                         : InkLabelController::Tool::Brush);
    }
    if (p.contains("radius"))
        labels->setRadius(p.value("radius").toDouble());
    if (p.contains("display")) {
        const QJsonObject d = p.value("display").toObject();
        labels->setDisplay(d.value("show").toBool(labels->showLabels()), d.value("ink").toBool(labels->showInk()),
                           d.value("supervision").toBool(labels->showSupervision()),
                           d.value("validation").toBool(labels->showValidation()), d.value("opacity").toDouble(labels->opacity()));
    }
    if (p.contains("assist")) {
        const QJsonObject a = p.value("assist").toObject();
        InkLabelController::AssistSettings s = labels->assistSettings();
        if (a.contains("guide"))
            s.guide = a.value("guide").toString() == QLatin1String("labels") ? GuideKind::Labels : GuideKind::Prediction;
        if (a.contains("threshold"))
            s.threshold = uint8_t(a.value("threshold").toInt());
        if (a.contains("above"))
            s.above = a.value("above").toBool();
        if (a.contains("connectivity"))
            s.connectivity = checkedChoice(a, "connectivity", {4, 8});
        if (a.contains("unlabeledOnly"))
            s.unlabeledOnly = a.value("unlabeledOnly").toBool();
        if (a.contains("median"))
            s.median = checkedChoice(a, "median", {1, 3, 5});
        if (a.contains("minArea"))
            s.minArea = a.value("minArea").toInt();
        if (a.contains("cleanupSize"))
            s.cleanupSize = a.value("cleanupSize").toInt();
        labels->setAssistSettings(s);
    }
    if (p.contains("labelMode")) {
        const bool on = p.value("labelMode").toBool();
        if (on && !labels->labelModeEnabled()) {
            // Never a dialog: the session must be open for the active segment
            // first (ink_labels.open decides about autosaves).
            if (!labels->session())
                failKind(-32007, QStringLiteral("No ink label session is open; call ink_labels.open first"),
                         QStringLiteral("ink_label_session"));
            InkLabelController::OpenOptions reuse;
            reuse.recover = RecoverPolicy::Fail;
            const InkLabelController::OpenOutcome out = labels->openForActiveSegment(reuse);
            if (!out.ok)
                failOpen(out);
            labels->setLabelModeEnabled(true);
            if (!labels->labelModeEnabled())
                failKind(-32009,
                         QStringLiteral("Label mode was refused: turn off segmentation editing and approval painting first"),
                         QStringLiteral("label_mode_refused"));
        } else if (!on) {
            labels->setLabelModeEnabled(false);
        }
    }
    return stateJson(*labels, inkDetectionOverlay());
}


QJsonObject AgentBridgeServer::handleInkLabelsClose(const QJsonValue&)
{
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const InkLabelController::SessionInfo* session = labels->session();
    QJsonObject result;
    result["closed"] = session != nullptr;
    result["keptAsAutosave"] = session && session->dirty && !session->readOnly && !session->locked;
    labels->closeSession();
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsFill(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    requireAssist(*labels);
    const bool volumeSpace = p.value("space").toString(QStringLiteral("label")) == QLatin1String("volume");
    VolumeViewerBase* viewer = nullptr;
    if (volumeSpace) {
        viewer = resolveViewer(QJsonValue());
        if (!labels->viewerShowsSession(viewer))
            failKind(-32009, QStringLiteral("The segmentation viewer does not show the labeled segment on its flattened view"),
                     QStringLiteral("viewer"));
    }
    const cv::Point2d point = labelPoint(*labels, viewer, p.value("point"), QStringLiteral("point"));
    const std::optional<cv::Rect> bound = boundParam(p);

    InkLabelController::AssistSettings s = labels->assistSettings();
    if (p.contains("guide"))
        s.guide = p.value("guide").toString() == QLatin1String("labels") ? GuideKind::Labels : GuideKind::Prediction;
    if (p.contains("threshold"))
        s.threshold = uint8_t(p.value("threshold").toInt());
    if (p.contains("above"))
        s.above = p.value("above").toBool();
    if (p.contains("connectivity"))
        s.connectivity = checkedChoice(p, "connectivity", {4, 8});
    if (p.contains("unlabeledOnly"))
        s.unlabeledOnly = p.value("unlabeledOnly").toBool();
    labels->setAssistSettings(s);
    if (s.guide == GuideKind::Prediction && !labels->predictionGuide())
        failKind(-32009, QStringLiteral("Select a single-channel ink prediction in the Ink Detection dock to fill from it"),
                 QStringLiteral("no_prediction"));

    const QString clsName = p.value("class").toString(QStringLiteral("ink"));
    const PaintClass cls = clsName == QLatin1String("background")   ? PaintClass::Background
                           : clsName == QLatin1String("validation") ? PaintClass::Validation
                                                                    : PaintClass::Ink;
    const PaintOp op = paintOp(cls, p.value("erase").toBool(false), p.value("backgroundKeepsInk").toBool(false));
    const InkLabelController::AssistReport report = labels->runAssistNow([&] { return labels->fillAt(point, bound, op); });
    if (!report.error.isEmpty()) {
        QJsonObject data;
        data["detail"] = report.error;
        fail(-32005, QStringLiteral("ink_labels.fill failed"), data);
    }
    QJsonObject result = editResult(*labels, report.changedPixels);
    result["reachedBound"] = report.reachedBound;
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsSuggest(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    requireAssist(*labels);
    InkLabelController::AssistSettings s = labels->assistSettings();
    if (p.contains("threshold"))
        s.threshold = uint8_t(p.value("threshold").toInt());
    if (p.contains("above"))
        s.above = p.value("above").toBool();
    if (p.contains("median"))
        s.median = checkedChoice(p, "median", {1, 3, 5});
    if (p.contains("minArea"))
        s.minArea = p.value("minArea").toInt();
    labels->setAssistSettings(s);
    if (!labels->predictionGuide())
        failKind(-32009, QStringLiteral("Select a single-channel ink prediction in the Ink Detection dock to get suggestions"),
                 QStringLiteral("no_prediction"));
    const std::optional<cv::Rect> bound = boundParam(p);
    const InkLabelController::AssistReport report = labels->runAssistNow([&] { return labels->suggest(bound); });
    if (!report.error.isEmpty() || !labels->suggestions()) {
        QJsonObject data;
        data["detail"] = report.error;
        fail(-32005, QStringLiteral("ink_labels.suggest failed"), data);
    }
    const SuggestionSet& set = *labels->suggestions();
    QJsonArray items;
    for (const Suggestion& item : set.items) {
        if (items.size() >= kMaxSuggestionsListed)
            break;
        QJsonObject o;
        o["id"] = item.id;
        o["area"] = item.area;
        o["bbox"] = rectJson(item.bbox);
        items.append(o);
    }
    QJsonObject result;
    result["count"] = int(set.items.size());
    result["suggestions"] = items;
    result["truncated"] = int(set.items.size()) > kMaxSuggestionsListed;
    result["window"] = rectJson(cv::Rect(set.origin, set.labels.size()));
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsAcceptSuggestions(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    writableSession(*labels);
    const SuggestionSet* set = labels->suggestions();
    if (!set)
        failKind(-32007, QStringLiteral("No suggestions are shown; call ink_labels.suggest first"), QStringLiteral("suggestions"));
    if (p.contains("ids") && p.value("all").toBool(false))
        throwParamError(QStringLiteral("ids"), QStringLiteral("give ids or all, not both"));
    if (p.value("all").toBool(false)) {
        labels->pickAllSuggestions();
    } else if (p.contains("ids")) {
        std::set<int> wanted;
        for (const QJsonValue& v : p.value("ids").toArray()) {
            const int id = v.toInt();
            if (id < 1 || id > int(set->items.size()))
                throwParamError(QStringLiteral("ids"), QStringLiteral("no suggestion has id %1 (ids are 1 to %2)").arg(id).arg(set->items.size()));
            wanted.insert(id);
        }
        for (const int id : std::set<int>(labels->selectedSuggestions()))
            labels->setSuggestionSelected(id, false);
        for (const int id : wanted)
            labels->setSuggestionSelected(id, true);
    }
    const int picked = int(labels->selectedSuggestions().size());
    if (picked == 0)
        throwParamError(QStringLiteral("ids"), QStringLiteral("nothing is picked; give ids or all"));
    const uint64_t changed = labels->acceptSuggestions(Actor::Agent);
    QJsonObject result = editResult(*labels, changed);
    result["accepted"] = picked;
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsDiscardSuggestions(const QJsonValue&)
{
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    const bool had = labels->suggestions() != nullptr || labels->assistRunning();
    labels->cancelAssist();
    QJsonObject result;
    result["discarded"] = had;
    return result;
}


QJsonObject AgentBridgeServer::handleInkLabelsCleanup(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    requireAssist(*labels);
    const QString name = p.value("kind").toString();
    const Cleanup kind = name == QLatin1String("fill_holes")        ? Cleanup::FillHoles
                         : name == QLatin1String("grow")            ? Cleanup::Grow
                         : name == QLatin1String("shrink")          ? Cleanup::Shrink
                         : name == QLatin1String("background_ring") ? Cleanup::BackgroundRing
                                                                    : Cleanup::RemoveSpecks;
    InkLabelController::AssistSettings s = labels->assistSettings();
    if (p.contains("size"))
        s.cleanupSize = p.value("size").toInt();
    labels->setAssistSettings(s);
    const std::optional<cv::Rect> bound = boundParam(p);
    const InkLabelController::AssistReport report = labels->runAssistNow([&] { return labels->runCleanup(kind, bound); });
    if (!report.error.isEmpty()) {
        QJsonObject data;
        data["detail"] = report.error;
        fail(-32005, QStringLiteral("ink_labels.cleanup failed"), data);
    }
    return editResult(*labels, report.changedPixels);
}


QJsonObject AgentBridgeServer::handleInkLabelsImport(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    requireInkLabelProject();
    InkLabelController* labels = inkLabelController();
    requireAssist(*labels);
    const QString path = p.value("path").toString();
    if (!QDir::isAbsolutePath(path))
        throwParamError(QStringLiteral("path"), QStringLiteral("path must be an absolute path"));
    if (!fs::exists(fs::path(path.toStdString())))
        throwParamError(QStringLiteral("path"), QStringLiteral("%1 does not exist").arg(path));

    ImportOptions options;
    options.kind = kindForFileName(path.toStdString());
    if (p.contains("label")) {
        const QString name = p.value("label").toString();
        options.kind = name == QLatin1String("supervision")  ? LabelKind::Supervision
                       : name == QLatin1String("validation") ? LabelKind::Validation
                                                             : LabelKind::Ink;
    }
    options.allowResample = p.value("allowResample").toBool(false);
    if (p.contains("legacyMaskAs"))
        options.legacyMaskAs = p.value("legacyMaskAs").toString() == QLatin1String("ignore") ? LegacyMaskAs::Ignore
                                                                                             : LegacyMaskAs::Supervision;
    if (!labels->importFile(path.toStdString(), options, Actor::Agent)) {
        QJsonObject data;
        data["detail"] = labels->lastAssistReport().error;
        fail(-32005, QStringLiteral("ink_labels.import could not start"), data);
    }

    const int token = beginDeferred(kImportTimeoutMs, QStringLiteral("ink label import"));
    QPointer<AgentBridgeServer> self(this);
    QPointer<InkLabelController> guard(labels);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(labels, &InkLabelController::importFinished, this,
                          [self, guard, connection, token, path](const QString& donePath, const ImportOutcome& outcome,
                                                                const ImportOptions& used) {
                              if (donePath != path)
                                  return;
                              QObject::disconnect(*connection);
                              if (!self || !guard)
                                  return;
                              InkLabelController* labels = guard.data();
                              if (outcome.ok) {
                                  QJsonObject result = editResult(*labels, outcome.setPixels);
                                  result["label"] = QLatin1String(kindName(used.kind));
                                  result["format"] = QLatin1String(formatName(outcome.format));
                                  result["setPixels"] = double(outcome.setPixels);
                                  result["resampledFrom"] = outcome.resampledFrom;
                                  result["warnings"] = QJsonArray::fromStringList(outcome.warnings);
                                  self->completeDeferredResult(token, result);
                                  return;
                              }
                              QJsonObject data;
                              if (outcome.needsDecision) {
                                  const bool legacy = outcome.format == ImportFormat::LegacyPngPair;
                                  data["kind"] = legacy ? QStringLiteral("legacy_mask") : QStringLiteral("resample_needed");
                                  data["param"] = legacy ? QStringLiteral("legacyMaskAs") : QStringLiteral("allowResample");
                                  self->completeDeferredError(token, -32602, outcome.error, data);
                                  return;
                              }
                              data["detail"] = outcome.error;
                              self->completeDeferredError(token, -32005, QStringLiteral("ink_labels.import failed"), data);
                          });
    throw AgentBridgeDeferred{};
}
