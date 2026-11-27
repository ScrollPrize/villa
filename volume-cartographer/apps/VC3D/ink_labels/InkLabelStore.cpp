#include "ink_labels/InkLabelStore.hpp"

#include "FileHash.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "ink_labels/InkLabelImage.hpp"
#include <vc_reader/image_write.hpp>

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QFile>
#include <QJsonDocument>
#include <QSaveFile>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstring>
#include <system_error>
#include <thread>

namespace fs = std::filesystem;

namespace vc3d::ink_labels {

namespace {

constexpr int kTiffTile = 1024;

uint32_t floatBits(float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

float bitsFloat(uint32_t bits)
{
    float v = 0.f;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

QString hexBits(uint32_t bits)
{
    return QStringLiteral("0x%1").arg(bits, 8, 16, QLatin1Char('0'));
}

std::optional<uint32_t> parseHexBits(const QJsonValue& value)
{
    if (!value.isString())
        return std::nullopt;
    bool ok = false;
    QString text = value.toString();
    if (text.startsWith(QLatin1String("0x")) || text.startsWith(QLatin1String("0X")))
        text = text.mid(2);
    const uint32_t bits = text.toUInt(&ok, 16);
    return ok ? std::optional<uint32_t>(bits) : std::nullopt;
}

QString str(const std::string& s)
{
    return QString::fromStdString(s);
}

QString pathString(const fs::path& p)
{
    return QString::fromStdString(p.string());
}

std::string versionSuffix(int version)
{
    return version > 1 ? "_v" + std::to_string(version) : std::string{};
}

fs::path tempPathFor(const fs::path& finalPath)
{
    static std::atomic<uint64_t> counter{0};
    return fs::path(finalPath.string() + ".tmp-" + std::to_string(QCoreApplication::applicationPid()) + "-"
                    + std::to_string(counter.fetch_add(1)));
}

std::string utcNow()
{
    return QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();
}

QJsonObject canvasToJson(const CanvasKey& key)
{
    QJsonObject o;
    o["width"] = key.width;
    o["height"] = key.height;
    o["grid_cols"] = key.gridCols;
    o["grid_rows"] = key.gridRows;
    o["scale"] = QJsonArray{double(key.scaleX()), double(key.scaleY())};
    o["scale_f32_hex"] = QJsonArray{hexBits(key.scaleXBits), hexBits(key.scaleYBits)};
    o["render_scale"] = key.renderScale;
    return o;
}

std::optional<CanvasKey> canvasFromJson(const QJsonObject& o)
{
    CanvasKey key;
    key.width = o.value("width").toInt(-1);
    key.height = o.value("height").toInt(-1);
    key.gridCols = o.value("grid_cols").toInt(-1);
    key.gridRows = o.value("grid_rows").toInt(-1);
    key.renderScale = o.value("render_scale").toDouble(1.0);
    if (key.width <= 0 || key.height <= 0 || key.gridCols <= 0 || key.gridRows <= 0)
        return std::nullopt;
    const QJsonArray hex = o.value("scale_f32_hex").toArray();
    const QJsonArray scale = o.value("scale").toArray();
    if (hex.size() == 2) {
        const auto x = parseHexBits(hex[0]);
        const auto y = parseHexBits(hex[1]);
        if (!x || !y)
            return std::nullopt;
        key.scaleXBits = *x;
        key.scaleYBits = *y;
    } else if (scale.size() == 2) {
        key.scaleXBits = floatBits(float(scale[0].toDouble()));
        key.scaleYBits = floatBits(float(scale[1].toDouble()));
    } else {
        return std::nullopt;
    }
    return key;
}

// Fills `block` (a view into the TIFF tile buffer) with 255 where the
// snapshot has `bit`, else 0.
void fillBlock(const TileRaster& raster, uint8_t bit, const cv::Rect& blockRect, cv::Mat_<uint8_t>& block)
{
    const int tx0 = blockRect.x / kTileSize, tx1 = (blockRect.br().x - 1) / kTileSize;
    const int ty0 = blockRect.y / kTileSize, ty1 = (blockRect.br().y - 1) / kTileSize;
    for (int ty = ty0; ty <= ty1; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx) {
            const int index = raster.tileIndex(tx, ty);
            const cv::Rect part = blockRect & raster.tileRect(index);
            if (part.empty())
                continue;
            cv::Mat_<uint8_t> dst = block(cv::Rect(part.x - blockRect.x, part.y - blockRect.y, part.width, part.height));
            const TileState& t = raster.tile(index);
            if (!t.dense) {
                dst.setTo((t.uniform & bit) ? 255 : 0);
                continue;
            }
            for (int y = 0; y < part.height; ++y) {
                const uint8_t* src = t.dense->px.data() + std::size_t((part.y - ty * kTileSize + y) * kTileSize)
                                     + std::size_t(part.x - tx * kTileSize);
                uint8_t* out = dst.ptr(y);
                for (int x = 0; x < part.width; ++x)
                    out[x] = (src[x] & bit) ? 255 : 0;
            }
        }
}

bool writeLabelTiff(const TileRaster& raster, uint8_t bit, const fs::path& path, std::string* error)
{
    try {
        const cv::Size size = raster.size();
        vcr::sources::TiffOptions options;
        options.compression = "lzw";
        options.tile = std::array<int, 2>{kTiffTile, kTiffTile};
        vcr::sources::TiffWriter writer(path, "|u1", size.height, size.width, 1, options);
        cv::Mat_<uint8_t> buffer(kTiffTile, kTiffTile);
        for (int by = 0; by < size.height; by += kTiffTile)
            for (int bx = 0; bx < size.width; bx += kTiffTile) {
                const cv::Rect blockRect(bx, by, std::min(kTiffTile, size.width - bx), std::min(kTiffTile, size.height - by));
                cv::Mat_<uint8_t> block = buffer(cv::Rect(0, 0, blockRect.width, blockRect.height));
                fillBlock(raster, bit, blockRect, block);
                const cv::Mat packed = block.isContinuous() ? cv::Mat(block) : block.clone();
                writer.writeTile(by, bx, packed.data, block.rows, block.cols);
            }
        writer.finish();
        return true;
    } catch (const std::exception& e) {
        if (error)
            *error = e.what();
        return false;
    }
}

void removeQuietly(const fs::path& path)
{
    std::error_code ec;
    fs::remove(path, ec);
}

// Accumulates label files into a raster, block by block. A tile is cloned
// only when a block sets a bit in it, and collapses back to Uniform as soon
// as every in-canvas pixel of it has been seen in the current file.
class RasterBuilder {
public:
    explicit RasterBuilder(TileRaster& raster) : _raster(raster), _work(std::size_t(raster.tileCount())) {}

    void beginFile() { _covered.assign(std::size_t(_raster.tileCount()), 0); }

    void addBlock(uint8_t bit, cv::Point topLeft, const cv::Mat& sample0)
    {
        const cv::Rect blockRect(topLeft, sample0.size());
        const int tx0 = blockRect.x / kTileSize, tx1 = (blockRect.br().x - 1) / kTileSize;
        const int ty0 = blockRect.y / kTileSize, ty1 = (blockRect.br().y - 1) / kTileSize;
        const bool wide = sample0.depth() == CV_16U;
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx) {
                const int index = _raster.tileIndex(tx, ty);
                const cv::Rect tileRect = _raster.tileRect(index);
                const cv::Rect part = blockRect & tileRect;
                if (part.empty())
                    continue;
                const cv::Mat src = sample0(cv::Rect(part.x - blockRect.x, part.y - blockRect.y, part.width, part.height));
                if (cv::countNonZero(src) > 0) {
                    Tile& tile = workTile(index);
                    for (int y = 0; y < part.height; ++y) {
                        uint8_t* dst = tile.px.data() + std::size_t((part.y - ty * kTileSize + y) * kTileSize)
                                       + std::size_t(part.x - tx * kTileSize);
                        if (wide) {
                            const uint16_t* row = src.ptr<uint16_t>(y);
                            for (int x = 0; x < part.width; ++x)
                                if (row[x] > 0)
                                    dst[x] |= bit;
                        } else {
                            const uint8_t* row = src.ptr<uint8_t>(y);
                            for (int x = 0; x < part.width; ++x)
                                if (row[x] > 0)
                                    dst[x] |= bit;
                        }
                    }
                }
                _covered[std::size_t(index)] += part.area();
                if (_covered[std::size_t(index)] >= tileRect.area())
                    flush(index);
            }
    }

    void endFile()
    {
        for (int i = 0; i < _raster.tileCount(); ++i)
            flush(i);
    }

private:
    Tile& workTile(int index)
    {
        auto& slot = _work[std::size_t(index)];
        if (slot)
            return *slot;
        slot = std::make_shared<Tile>();
        const TileState& current = _raster.tile(index);
        const cv::Rect bounds = _raster.tileRect(index);
        const cv::Rect inCanvas(bounds.x % kTileSize, bounds.y % kTileSize, bounds.width, bounds.height);
        if (current.dense) {
            *slot = *current.dense;
        } else if (current.uniform != 0) {
            for (int y = inCanvas.y; y < inCanvas.br().y; ++y)
                std::fill_n(slot->px.data() + std::size_t(y * kTileSize + inCanvas.x), inCanvas.width, current.uniform);
        }
        return *slot;
    }

    void flush(int index)
    {
        auto& slot = _work[std::size_t(index)];
        if (!slot)
            return;
        const cv::Rect bounds = _raster.tileRect(index);
        const cv::Rect inCanvas(bounds.x % kTileSize, bounds.y % kTileSize, bounds.width, bounds.height);
        finalizeTile(*slot, inCanvas);
        _raster.setTile(index, publishTile(std::move(slot), inCanvas));
        slot.reset();
    }

    TileRaster& _raster;
    std::vector<std::shared_ptr<Tile>> _work;
    std::vector<int> _covered;
};

}  // namespace

// ---------------------------------------------------------------------------
// Names and paths
// ---------------------------------------------------------------------------

const char* kindName(LabelKind kind)
{
    switch (kind) {
        case LabelKind::Ink:
            return "inklabels";
        case LabelKind::Supervision:
            return "supervision_mask";
        case LabelKind::Validation:
            return "validation_mask";
    }
    return "inklabels";
}

std::optional<LabelKind> kindFromName(std::string_view name)
{
    for (const LabelKind kind : kAllKinds)
        if (name == kindName(kind))
            return kind;
    return std::nullopt;
}

uint8_t bitFor(LabelKind kind)
{
    switch (kind) {
        case LabelKind::Ink:
            return kInk;
        case LabelKind::Supervision:
            return kSupervision;
        case LabelKind::Validation:
            return kValidation;
    }
    return 0;
}

float CanvasKey::scaleX() const
{
    return bitsFloat(scaleXBits);
}

float CanvasKey::scaleY() const
{
    return bitsFloat(scaleYBits);
}

QString CanvasKey::describe() const
{
    QString scale = QString::number(double(scaleX()), 'g', 6);
    if (scaleXBits != scaleYBits)
        scale += QStringLiteral(" x ") + QString::number(double(scaleY()), 'g', 6);
    return QStringLiteral("grid %1 x %2 at scale %3 (%4 x %5 px)").arg(gridCols).arg(gridRows).arg(scale).arg(width).arg(height);
}

CanvasKey canvasKeyFor(const vc::RenderCanvas& canvas)
{
    return {canvas.size.width,
            canvas.size.height,
            canvas.grid.width,
            canvas.grid.height,
            floatBits(canvas.gridScale[0]),
            floatBits(canvas.gridScale[1]),
            canvas.renderScale};
}

fs::path StorePaths::file(LabelKind kind, Target target) const
{
    std::string name = prefix + "_" + kindName(kind) + versionSuffix(version) + ".tif";
    if (target == Target::Autosave)
        name += ".autosave";
    return dir / name;
}

fs::path StorePaths::sidecar(Target target) const
{
    std::string name = prefix + "_inklabels" + versionSuffix(version) + ".json";
    if (target == Target::Autosave)
        name += ".autosave";
    return dir / name;
}

fs::path StorePaths::saving(Target target) const
{
    return fs::path(sidecar(target).string() + ".saving");
}

fs::path StorePaths::previous(LabelKind kind) const
{
    return fs::path(file(kind, Target::Trainer).string() + ".prev");
}

fs::path StorePaths::lock() const
{
    return dir / (prefix + "_inklabels.lock");
}

StorePaths storePathsFor(const fs::path& projectDir, const std::string& segmentName)
{
    return {projectDir / "ink_labels" / segmentName, segmentName, 1};
}

// ---------------------------------------------------------------------------
// Sidecar
// ---------------------------------------------------------------------------

fs::path labelRootFor(const fs::path& projectFile, const fs::path& cacheRoot, const fs::path& dataRoot)
{
    const fs::path projectDir = projectFile.parent_path();
    if (cacheRoot.empty())
        return projectDir;
    std::error_code ec;
    const fs::path dir = fs::weakly_canonical(projectDir, ec).lexically_normal();
    const fs::path cache = fs::weakly_canonical(cacheRoot, ec).lexically_normal();
    auto d = dir.begin();
    for (auto c = cache.begin(); c != cache.end(); ++c, ++d) {
        if (c->empty())
            continue;  // a trailing separator
        if (d == dir.end() || *d != *c)
            return projectDir;
    }
    std::string name = projectFile.filename().string();
    for (const char* suffix : {".json", ".volpkg"})
        if (name.size() > std::strlen(suffix) && name.compare(name.size() - std::strlen(suffix), std::string::npos, suffix) == 0)
            name.resize(name.size() - std::strlen(suffix));
    return dataRoot / name;
}

namespace {

std::optional<QJsonObject> readJsonObject(const fs::path& path, QString* error)
{
    QFile file(pathString(path));
    if (!file.open(QIODevice::ReadOnly)) {
        if (error)
            *error = QStringLiteral("cannot read %1").arg(pathString(path));
        return std::nullopt;
    }
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        if (error)
            *error = QStringLiteral("%1 is not valid JSON: %2").arg(pathString(path), parseError.errorString());
        return std::nullopt;
    }
    return doc.object();
}

bool writeJsonObject(const fs::path& path, const QJsonObject& root, QString* error)
{
    QSaveFile file(pathString(path));
    if (!file.open(QIODevice::WriteOnly)) {
        if (error)
            *error = QStringLiteral("cannot write %1: %2").arg(pathString(path), file.errorString());
        return false;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    if (!file.commit()) {
        if (error)
            *error = QStringLiteral("cannot write %1: %2").arg(pathString(path), file.errorString());
        return false;
    }
    return true;
}

std::optional<Sidecar> sidecarFromJson(const QJsonObject& root, const fs::path& path, QString* error)
{
    const auto fail = [error](const QString& message) -> std::optional<Sidecar> {
        if (error)
            *error = message;
        return std::nullopt;
    };
    if (root.value("schema").toString() != QLatin1String(kSidecarSchema))
        return fail(QStringLiteral("%1 has schema \"%2\", expected \"%3\"")
                        .arg(pathString(path), root.value("schema").toString(), QLatin1String(kSidecarSchema)));
    const auto canvas = canvasFromJson(root.value("canvas").toObject());
    if (!canvas)
        return fail(QStringLiteral("%1 has no valid canvas").arg(pathString(path)));

    Sidecar s;
    s.canvas = *canvas;
    const QJsonObject segment = root.value("segment").toObject();
    s.segmentName = segment.value("name").toString().toStdString();
    s.segmentUuid = segment.value("uuid").toString().toStdString();
    s.segmentPath = segment.value("path").toString().toStdString();
    s.geometrySha256 = root.value("geometry_sha256").toString().toStdString();
    s.savedAtUtc = root.value("saved_at_utc").toString().toStdString();
    s.savedBy = root.value("saved_by").toString().toStdString();
    s.stateSerial = uint64_t(std::max(0.0, root.value("state_serial").toDouble(0)));
    s.writeSeq = uint64_t(std::max(0.0, root.value("write_seq").toDouble(0)));
    s.autosave = root.value("autosave").toBool(false);
    s.edits = root.value("edits").toObject();
    s.provenance = root.value("provenance").toArray();
    if (root.value("prediction_shown").isString())
        s.predictionShown = root.value("prediction_shown").toString().toStdString();
    for (const QJsonValue& value : root.value("grid_anchors").toArray()) {
        const QJsonArray a = value.toArray();
        if (a.size() != 5)
            continue;
        GridAnchor anchor;
        anchor.row = a[0].toInt(-1);
        anchor.col = a[1].toInt(-1);
        anchor.point = cv::Vec3f(float(a[2].toDouble()), float(a[3].toDouble()), float(a[4].toDouble()));
        if (anchor.row >= 0 && anchor.col >= 0)
            s.gridAnchors.push_back(anchor);
    }
    const QJsonObject files = root.value("files").toObject();
    for (const LabelKind kind : kAllKinds) {
        const QJsonValue entry = files.value(QLatin1String(kindName(kind)));
        if (!entry.isObject())
            continue;
        const QJsonObject o = entry.toObject();
        s.fileNames[kind] = o.value("name").toString().toStdString();
        s.sha256[kind] = o.value("sha256").toString().toStdString();
    }
    return s;
}

QJsonObject sidecarToJson(const Sidecar& s)
{
    QJsonObject root;
    root["schema"] = QLatin1String(kSidecarSchema);
    QJsonObject segment;
    segment["name"] = str(s.segmentName);
    segment["uuid"] = str(s.segmentUuid);
    segment["path"] = str(s.segmentPath);
    root["segment"] = segment;
    root["canvas"] = canvasToJson(s.canvas);
    root["geometry_sha256"] = str(s.geometrySha256);
    QJsonObject files;
    for (const LabelKind kind : kAllKinds) {
        const auto sha = s.sha256.find(kind);
        if (sha == s.sha256.end()) {
            files[QLatin1String(kindName(kind))] = QJsonValue::Null;
            continue;
        }
        QJsonObject entry;
        const auto name = s.fileNames.find(kind);
        entry["name"] = name == s.fileNames.end() ? QString() : str(name->second);
        entry["sha256"] = str(sha->second);
        files[QLatin1String(kindName(kind))] = entry;
    }
    root["files"] = files;
    root["state_serial"] = double(s.stateSerial);
    root["write_seq"] = double(s.writeSeq);
    root["saved_at_utc"] = str(s.savedAtUtc);
    root["saved_by"] = str(s.savedBy);
    root["edits"] = s.edits;
    root["prediction_shown"] = s.predictionShown ? QJsonValue(str(*s.predictionShown)) : QJsonValue(QJsonValue::Null);
    QJsonArray provenance = s.provenance;
    while (provenance.size() > kMaxProvenanceEntries)
        provenance.removeFirst();
    root["provenance"] = provenance;
    if (!s.gridAnchors.empty()) {
        QJsonArray anchors;
        for (const GridAnchor& a : s.gridAnchors)
            anchors.append(QJsonArray{a.row, a.col, double(a.point[0]), double(a.point[1]), double(a.point[2])});
        root["grid_anchors"] = anchors;
    }
    if (s.autosave)
        root["autosave"] = true;
    return root;
}

}  // namespace

std::optional<Sidecar> readSidecar(const fs::path& path, QString* error)
{
    const auto root = readJsonObject(path, error);
    if (!root)
        return std::nullopt;
    return sidecarFromJson(*root, path, error);
}

bool writeSidecar(const fs::path& path, const Sidecar& s, QString* error)
{
    return writeJsonObject(path, sidecarToJson(s), error);
}

std::string geometrySha256(const QuadSurface& surface)
{
    QCryptographicHash hash(QCryptographicHash::Sha256);
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    const cv::Vec2f scale = surface.scale();
    const int32_t dims[2] = {points ? points->cols : 0, points ? points->rows : 0};
    const uint32_t bits[2] = {floatBits(scale[0]), floatBits(scale[1])};
    hash.addData(QByteArrayView(reinterpret_cast<const char*>(dims), sizeof dims));
    hash.addData(QByteArrayView(reinterpret_cast<const char*>(bits), sizeof bits));
    if (points) {
        for (int r = 0; r < points->rows; ++r)
            hash.addData(QByteArrayView(reinterpret_cast<const char*>(points->ptr(r)),
                                        qsizetype(sizeof(cv::Vec3f)) * points->cols));
    }
    return hash.result().toHex().toStdString();
}

std::string fileSha256(const fs::path& path)
{
    return vc3d::fileSha256Hex(pathString(path)).toStdString();
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

SaveOutcome saveLabels(const SaveRequest& request)
{
    const auto started = std::chrono::steady_clock::now();
    SaveOutcome outcome;
    outcome.stateSerial = request.sidecar.stateSerial;
    outcome.target = request.target;
    const StorePaths& paths = request.paths;

    std::error_code ec;
    fs::create_directories(paths.dir, ec);
    if (ec) {
        outcome.error = QStringLiteral("cannot create %1: %2").arg(pathString(paths.dir), QString::fromStdString(ec.message()));
        return outcome;
    }

    std::vector<LabelKind> kinds{LabelKind::Ink, LabelKind::Supervision};
    if (request.writeValidation)
        kinds.push_back(LabelKind::Validation);

    // 1. Temp files, one writer per kind, all reading the immutable snapshot.
    struct Pending {
        LabelKind kind;
        fs::path temp;
        std::string error;
        bool ok = false;
        std::string sha;
    };
    std::vector<Pending> pending;
    for (const LabelKind kind : kinds)
        pending.push_back({kind, tempPathFor(paths.file(kind, Target::Trainer)), {}, false, {}});
    {
        std::vector<std::thread> writers;
        for (auto& p : pending)
            writers.emplace_back([&p, &request] {
                p.ok = writeLabelTiff(request.snapshot, bitFor(p.kind), p.temp, &p.error);
                if (p.ok)
                    p.sha = fileSha256(p.temp);
            });
        for (auto& w : writers)
            w.join();
    }
    const auto discardTemps = [&pending] {
        for (const auto& p : pending)
            removeQuietly(p.temp);
    };
    for (const auto& p : pending)
        if (!p.ok || p.sha.empty()) {
            discardTemps();
            outcome.error = QStringLiteral("cannot write %1: %2")
                                .arg(pathString(paths.file(p.kind)), QString::fromStdString(p.error.empty() ? "hash failed" : p.error));
            return outcome;
        }

    // 2. Trainer target: refuse to replace files changed outside VC3D.
    if (request.target == Target::Trainer && !request.overwriteExternalEdits) {
        for (const auto& p : pending) {
            const fs::path finalPath = paths.file(p.kind);
            if (!fs::exists(finalPath, ec))
                continue;
            const auto expected = request.expectedSha256.find(p.kind);
            const std::string current = fileSha256(finalPath);
            if (expected == request.expectedSha256.end() || expected->second != current)
                outcome.externallyEdited.append(QString::fromStdString(finalPath.filename().string()));
        }
        if (!outcome.externallyEdited.isEmpty()) {
            discardTemps();
            outcome.error = QStringLiteral("label files changed outside VC3D: %1").arg(outcome.externallyEdited.join(QStringLiteral(", ")));
            return outcome;
        }
    }

    // 3. The generation about to go in place: the sidecar with the new
    //    files' hashes, and the hash of every file it replaces (none when the
    //    file is new). Written to .saving before any file is replaced and
    //    removed only after the sidecar is written, so readers and the next
    //    open can tell a complete generation from an interrupted one.
    std::vector<Pending*> ordered;
    for (const LabelKind kind : {LabelKind::Validation, LabelKind::Supervision, LabelKind::Ink})
        for (auto& p : pending)
            if (p.kind == kind)
                ordered.push_back(&p);
    Sidecar sidecar = request.sidecar;
    sidecar.autosave = request.target == Target::Autosave;
    sidecar.fileNames.clear();
    sidecar.sha256.clear();
    for (const Pending* p : ordered) {
        sidecar.fileNames[p->kind] = paths.file(p->kind, request.target).filename().string();
        sidecar.sha256[p->kind] = p->sha;
    }
    sidecar.savedAtUtc = utcNow();
    // A checkpoint without validation drops an older checkpoint's validation
    // file; that removal is part of the generation too.
    const bool dropsAutosaveValidation = request.target == Target::Autosave && !request.writeValidation;
    QJsonObject replaces;
    const auto recordReplaced = [&](LabelKind kind) {
        const fs::path finalPath = paths.file(kind, request.target);
        QJsonObject entry;
        entry["sha256"] = fs::exists(finalPath, ec) ? QJsonValue(str(fileSha256(finalPath))) : QJsonValue(QJsonValue::Null);
        replaces[QLatin1String(kindName(kind))] = entry;
    };
    for (const Pending* p : ordered)
        recordReplaced(p->kind);
    if (dropsAutosaveValidation)
        recordReplaced(LabelKind::Validation);
    QJsonObject marker;
    marker["schema"] = QLatin1String(kSavingSchema);
    marker["sidecar"] = sidecarToJson(sidecar);
    marker["replaces"] = replaces;
    const fs::path savingPath = paths.saving(request.target);
    {
        QString markerError;
        if (!writeJsonObject(savingPath, marker, &markerError)) {
            discardTemps();
            outcome.error = markerError;
            return outcome;
        }
    }

    // 4. Rename in a fixed order, ink last, so a reader that sees the new
    //    ink file also sees the masks that go with it. Each replaced file
    //    keeps a way back: the trainer target's previous file is copied to
    //    .prev (the old .prev moved aside first), a checkpoint's is linked
    //    aside. A failure puts every replaced file back.
    struct Step {
        fs::path finalPath;
        fs::path temp;          // empty: the step removes the file
        bool had = false;       // finalPath existed before
        fs::path prevAside;     // trainer: the old .prev, moved aside
        bool prevCopied = false;
        fs::path aside;         // checkpoint: the replaced file
        bool applied = false;
    };
    std::vector<Step> steps;
    for (Pending* p : ordered)
        steps.push_back({paths.file(p->kind, request.target), p->temp});
    if (dropsAutosaveValidation)
        steps.push_back({paths.file(LabelKind::Validation, Target::Autosave), {}});
    const auto previousOf = [](const fs::path& finalPath) { return fs::path(finalPath.string() + ".prev"); };

    const auto rollBack = [&]() -> bool {
        bool ok = true;
        const auto check = [&ok](const std::error_code& e) { ok = ok && !e; };
        for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
            const Step& step = *it;
            const fs::path prev = previousOf(step.finalPath);
            std::error_code rec;
            if (step.applied) {
                if (!step.had) {
                    fs::remove(step.finalPath, rec);
                    check(rec);
                } else if (step.prevCopied) {
                    // .prev is a copy of the replaced file.
                    fs::rename(prev, step.finalPath, rec);
                    check(rec);
                } else if (!step.aside.empty()) {
                    fs::rename(step.aside, step.finalPath, rec);
                    check(rec);
                }
            } else {
                if (step.prevCopied)
                    fs::remove(prev, rec);
                if (!step.aside.empty())
                    fs::remove(step.aside, rec);
            }
            if (!step.prevAside.empty()) {
                fs::rename(step.prevAside, prev, rec);
                check(rec);
            }
        }
        for (const auto& p : pending)
            removeQuietly(p.temp);
        if (ok)
            removeQuietly(savingPath);
        return ok;
    };
    const auto fail = [&](const QString& error) {
        outcome.error = error;
        if (!rollBack())
            outcome.error += QStringLiteral("; the previous files could not all be put back, and %1 records the interrupted save")
                                 .arg(pathString(savingPath.filename()));
        return outcome;
    };

    int renamed = 0;
    for (Step& step : steps) {
        step.had = fs::exists(step.finalPath, ec);
        const fs::path prev = previousOf(step.finalPath);
        if (step.had && request.target == Target::Trainer) {
            if (fs::exists(prev, ec)) {
                step.prevAside = tempPathFor(prev);
                fs::rename(prev, step.prevAside, ec);
                if (ec) {
                    step.prevAside.clear();
                    return fail(QStringLiteral("cannot keep the previous %1: %2")
                                    .arg(pathString(prev.filename()), QString::fromStdString(ec.message())));
                }
            }
            fs::copy_file(step.finalPath, prev, fs::copy_options::overwrite_existing, ec);
            if (ec)
                return fail(QStringLiteral("cannot keep the previous %1: %2")
                                .arg(pathString(step.finalPath.filename()), QString::fromStdString(ec.message())));
            step.prevCopied = true;
        } else if (step.had) {
            step.aside = tempPathFor(step.finalPath);
            fs::create_hard_link(step.finalPath, step.aside, ec);
            if (ec) {
                ec.clear();
                fs::copy_file(step.finalPath, step.aside, ec);
            }
            if (ec) {
                step.aside.clear();
                return fail(QStringLiteral("cannot keep the previous %1: %2")
                                .arg(pathString(step.finalPath.filename()), QString::fromStdString(ec.message())));
            }
        }
        if (step.temp.empty()) {
            fs::remove(step.finalPath, ec);
        } else {
            fs::rename(step.temp, step.finalPath, ec);
        }
        if (ec)
            return fail(QStringLiteral("cannot replace %1: %2").arg(pathString(step.finalPath), QString::fromStdString(ec.message())));
        step.applied = true;
        if (++renamed == request.crashAfterRenamesForTesting) {
            outcome.error = QStringLiteral("stopped for testing");
            return outcome;
        }
    }

    // 5. The sidecar: its hashes describe the files now in place.
    QString sidecarError;
    if (!writeSidecar(paths.sidecar(request.target), sidecar, &sidecarError))
        return fail(sidecarError);
    removeQuietly(savingPath);
    for (const Step& step : steps) {
        if (!step.prevAside.empty())
            removeQuietly(step.prevAside);
        if (!step.aside.empty())
            removeQuietly(step.aside);
    }
    for (const Pending* p : ordered) {
        outcome.files.emplace_back(p->kind, paths.file(p->kind, request.target));
        outcome.sha256[p->kind] = p->sha;
    }

    // 6. A trainer save supersedes the checkpoint, whatever its state serial:
    //    after an undo the saved state can have a smaller serial than a
    //    checkpoint written before the undo.
    if (request.target == Target::Trainer)
        removeTarget(paths, Target::Autosave);

    outcome.ok = true;
    outcome.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return outcome;
}

// ---------------------------------------------------------------------------
// Interrupted saves
// ---------------------------------------------------------------------------

namespace {

struct SavingRecord {
    Sidecar sidecar;
    // Per kind the save touches: the hash after it (empty: the save removes
    // the file) and before it (empty: the file did not exist).
    std::map<LabelKind, std::pair<std::string, std::string>> touched;
};

std::optional<SavingRecord> readSavingRecord(const StorePaths& paths, Target target, QString* error)
{
    const fs::path path = paths.saving(target);
    const auto root = readJsonObject(path, error);
    if (!root)
        return std::nullopt;
    if (root->value("schema").toString() != QLatin1String(kSavingSchema)) {
        if (error)
            *error = QStringLiteral("%1 has schema \"%2\", expected \"%3\"")
                         .arg(pathString(path), root->value("schema").toString(), QLatin1String(kSavingSchema));
        return std::nullopt;
    }
    const auto sidecar = sidecarFromJson(root->value("sidecar").toObject(), path, error);
    if (!sidecar)
        return std::nullopt;
    SavingRecord record;
    record.sidecar = *sidecar;
    const QJsonObject replaces = root->value("replaces").toObject();
    for (const LabelKind kind : kAllKinds) {
        const QJsonValue entry = replaces.value(QLatin1String(kindName(kind)));
        if (!entry.isObject())
            continue;
        const QJsonValue before = entry.toObject().value("sha256");
        const auto after = sidecar->sha256.find(kind);
        record.touched[kind] = {after == sidecar->sha256.end() ? std::string() : after->second,
                                before.isString() ? before.toString().toStdString() : std::string()};
    }
    if (record.touched.empty()) {
        if (error)
            *error = QStringLiteral("%1 lists no files").arg(pathString(path));
        return std::nullopt;
    }
    return record;
}

// The file's hash, empty when it does not exist, and a value no hash takes
// when it exists but cannot be read.
std::string currentSha(const fs::path& path)
{
    std::error_code ec;
    if (!fs::exists(path, ec))
        return {};
    const std::string sha = fileSha256(path);
    return sha.empty() ? std::string("unreadable") : sha;
}

SaveGeneration classify(const StorePaths& paths, Target target, const SavingRecord& record)
{
    bool allNew = true, allOld = true;
    for (const auto& [kind, shas] : record.touched) {
        const std::string current = currentSha(paths.file(kind, target));
        allNew = allNew && current == shas.first;
        allOld = allOld && current == shas.second;
    }
    return allNew ? SaveGeneration::New : allOld ? SaveGeneration::Previous : SaveGeneration::Mixed;
}

}  // namespace

SaveGeneration saveGeneration(const StorePaths& paths, Target target)
{
    std::error_code ec;
    if (!fs::exists(paths.saving(target), ec))
        return SaveGeneration::Settled;
    const auto record = readSavingRecord(paths, target, nullptr);
    return record ? classify(paths, target, *record) : SaveGeneration::Mixed;
}

RepairResult repairInterruptedSave(const StorePaths& paths, Target target, QString* message)
{
    const auto say = [message](const QString& text) {
        if (message)
            *message = text;
    };
    std::error_code ec;
    const fs::path savingPath = paths.saving(target);
    if (!fs::exists(savingPath, ec))
        return RepairResult::NothingToRepair;
    QString error;
    const auto record = readSavingRecord(paths, target, &error);
    if (!record) {
        say(QStringLiteral("A save of the labels in %1 was interrupted and its record cannot be read (%2); the labels are "
                           "not read.")
                .arg(pathString(paths.dir), error));
        return RepairResult::Failed;
    }
    switch (classify(paths, target, *record)) {
        case SaveGeneration::New:
            if (!writeSidecar(paths.sidecar(target), record->sidecar, &error)) {
                say(error);
                return RepairResult::Failed;
            }
            removeQuietly(savingPath);
            say(QStringLiteral("A save of the labels in %1 was interrupted after its files were written; it was completed.")
                    .arg(pathString(paths.dir)));
            return RepairResult::Finished;
        case SaveGeneration::Previous:
        case SaveGeneration::Settled:
            removeQuietly(savingPath);
            say(QStringLiteral("A save of the labels in %1 was interrupted before it replaced any file; the previous save is "
                               "unchanged.")
                    .arg(pathString(paths.dir)));
            return RepairResult::RolledBack;
        case SaveGeneration::Mixed:
            break;
    }
    if (target != Target::Trainer) {
        say(QStringLiteral("An interrupted checkpoint left files of two different states in %1.").arg(pathString(paths.dir)));
        return RepairResult::Failed;
    }
    // Every replaced file must come back from its .prev before any is moved.
    std::vector<LabelKind> restore, remove;
    for (const auto& [kind, shas] : record->touched) {
        const fs::path file = paths.file(kind, Target::Trainer);
        const std::string current = currentSha(file);
        if (current == shas.second)
            continue;
        if (current == shas.first && shas.second.empty()) {
            remove.push_back(kind);
        } else if (current == shas.first && currentSha(paths.previous(kind)) == shas.second) {
            restore.push_back(kind);
        } else {
            say(QStringLiteral("An interrupted save left files of two different saves in %1, and %2 cannot be put back: it "
                               "changed since, or its .prev copy is not the previous save. The labels are not read.")
                    .arg(pathString(paths.dir), pathString(file.filename())));
            return RepairResult::Failed;
        }
    }
    for (const LabelKind kind : restore) {
        fs::rename(paths.previous(kind), paths.file(kind, Target::Trainer), ec);
        if (ec) {
            say(QStringLiteral("Cannot put back %1: %2")
                    .arg(pathString(paths.file(kind, Target::Trainer)), QString::fromStdString(ec.message())));
            return RepairResult::Failed;
        }
    }
    for (const LabelKind kind : remove)
        fs::remove(paths.file(kind, Target::Trainer), ec);
    if (classify(paths, target, *record) != SaveGeneration::Previous) {
        say(QStringLiteral("An interrupted save in %1 could not be undone; the labels are not read.").arg(pathString(paths.dir)));
        return RepairResult::Failed;
    }
    removeQuietly(savingPath);
    say(QStringLiteral("A save of the labels in %1 was interrupted part way; the previous save was put back.")
            .arg(pathString(paths.dir)));
    return RepairResult::RolledBack;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

LoadOutcome loadLabels(const StorePaths& paths, const CanvasKey& expected, Target target)
{
    LoadOutcome outcome;
    std::error_code ec;
    const SaveGeneration generation = saveGeneration(paths, target);
    if (generation == SaveGeneration::Mixed) {
        outcome.interruptedSave = true;
        outcome.error = QStringLiteral("an interrupted save left files of two different saves in %1; they are not read. "
                                       "Opening them in VC3D with write access puts the previous save back.")
                            .arg(pathString(paths.dir));
        return outcome;
    }
    const fs::path sidecarPath = paths.sidecar(target);
    if (generation == SaveGeneration::New) {
        // The files are the save's; its sidecar is the one in the record.
        if (auto record = readSavingRecord(paths, target, nullptr))
            outcome.sidecar = record->sidecar;
    } else if (fs::exists(sidecarPath, ec)) {
        QString error;
        outcome.sidecar = readSidecar(sidecarPath, &error);
        if (!outcome.sidecar)
            outcome.warnings.append(error);
    }
    if (outcome.sidecar && outcome.sidecar->canvas != expected) {
        outcome.canvasMismatch = true;
        outcome.error = QStringLiteral("Labels were saved for %1. The segment now has %2.")
                            .arg(outcome.sidecar->canvas.describe(), expected.describe());
        return outcome;
    }

    outcome.raster = TileRaster(cv::Size(expected.width, expected.height));
    RasterBuilder builder(outcome.raster);
    QStringList mismatched;
    for (const LabelKind kind : kAllKinds) {
        const fs::path file = paths.file(kind, target);
        if (!fs::exists(file, ec))
            continue;
        outcome.anyFile = true;
        if (kind == LabelKind::Validation)
            outcome.validationPresent = true;

        std::array<uint64_t, 256> histogram{};
        uint64_t wideNonBinary = 0;
        bool multiSample = false;
        QString shapeError;
        builder.beginFile();
        std::string readError;
        const bool ok = forEachLabelImageBlock(
            file,
            [&](const LabelImageInfo& info, cv::Point topLeft, const cv::Mat& block) {
                if (int(info.width) != expected.width || int(info.height) != expected.height) {
                    if (int(info.width) == expected.height && int(info.height) == expected.width)
                        shapeError = QStringLiteral("%1 is %2 x %3 px: the %4 x %5 px canvas on its side, which usually "
                                                    "means a rotated render.")
                                         .arg(pathString(file.filename()))
                                         .arg(info.width)
                                         .arg(info.height)
                                         .arg(expected.width)
                                         .arg(expected.height);
                    else
                        shapeError = QStringLiteral("%1 is %2 x %3 px but the canvas is %4 x %5 px.")
                                         .arg(pathString(file.filename()))
                                         .arg(info.width)
                                         .arg(info.height)
                                         .arg(expected.width)
                                         .arg(expected.height);
                    return false;
                }
                cv::Mat sample0 = block;
                if (block.channels() > 1) {
                    multiSample = true;
                    cv::extractChannel(block, sample0, 0);
                }
                if (sample0.depth() == CV_8U) {
                    for (int y = 0; y < sample0.rows; ++y) {
                        const uint8_t* row = sample0.ptr<uint8_t>(y);
                        for (int x = 0; x < sample0.cols; ++x)
                            ++histogram[row[x]];
                    }
                } else {
                    for (int y = 0; y < sample0.rows; ++y) {
                        const uint16_t* row = sample0.ptr<uint16_t>(y);
                        for (int x = 0; x < sample0.cols; ++x)
                            if (row[x] != 0 && row[x] != 65535)
                                ++wideNonBinary;
                    }
                }
                builder.addBlock(bitFor(kind), topLeft, sample0);
                return true;
            },
            &readError);
        builder.endFile();
        if (!shapeError.isEmpty()) {
            outcome.error = shapeError;
            return outcome;
        }
        if (!ok) {
            outcome.error = QString::fromStdString(readError);
            return outcome;
        }
        if (multiSample)
            outcome.warnings.append(QStringLiteral("%1 has more than one sample per pixel; only the first is used.")
                                        .arg(pathString(file.filename())));
        std::vector<std::pair<uint64_t, int>> other;
        for (int v = 1; v < 255; ++v)
            if (histogram[std::size_t(v)] > 0)
                other.emplace_back(histogram[std::size_t(v)], v);
        if (!other.empty()) {
            std::sort(other.begin(), other.end(), std::greater<>());
            QStringList common;
            for (std::size_t i = 0; i < std::min<std::size_t>(4, other.size()); ++i)
                common.append(QStringLiteral("%1 (%2 px)").arg(other[i].second).arg(other[i].first));
            outcome.warnings.append(QStringLiteral("%1 holds values other than 0 and 255; any value above 0 counts as labeled. "
                                                   "Most common: %2.")
                                        .arg(pathString(file.filename()), common.join(QStringLiteral(", "))));
        }
        if (wideNonBinary > 0)
            outcome.warnings.append(QStringLiteral("%1 holds 16-bit values other than 0 and 65535 (%2 px); any value above 0 "
                                                   "counts as labeled.")
                                        .arg(pathString(file.filename()))
                                        .arg(wideNonBinary));

        const std::string sha = fileSha256(file);
        outcome.sha256[kind] = sha;
        if (outcome.sidecar) {
            const auto recorded = outcome.sidecar->sha256.find(kind);
            if (recorded == outcome.sidecar->sha256.end() || recorded->second != sha)
                mismatched.append(pathString(file.filename()));
        }
    }
    if (outcome.sidecar) {
        for (const auto& [kind, sha] : outcome.sidecar->sha256)
            if (!outcome.sha256.count(kind))
                mismatched.append(pathString(paths.file(kind, target).filename()) + QStringLiteral(" (missing)"));
    }
    if (!mismatched.isEmpty())
        outcome.warnings.append(QStringLiteral("Changed outside VC3D or written by an interrupted save: %1; the previous "
                                               "generation is in .prev.")
                                    .arg(mismatched.join(QStringLiteral(", "))));
    outcome.ok = true;
    return outcome;
}

bool targetExists(const StorePaths& paths, Target target)
{
    std::error_code ec;
    if (fs::exists(paths.sidecar(target), ec) || fs::exists(paths.saving(target), ec))
        return true;
    for (const LabelKind kind : kAllKinds)
        if (fs::exists(paths.file(kind, target), ec))
            return true;
    return false;
}

bool moveTargetAside(const StorePaths& paths, Target target)
{
    bool ok = true;
    std::error_code ec;
    std::vector<fs::path> files{paths.sidecar(target), paths.saving(target)};
    for (const LabelKind kind : kAllKinds)
        files.push_back(paths.file(kind, target));
    for (const fs::path& file : files) {
        if (!fs::exists(file, ec))
            continue;
        fs::path aside = file;
        aside += ".unusable";
        fs::remove(aside, ec);
        fs::rename(file, aside, ec);
        ok = ok && !ec;
    }
    return ok;
}

bool removeTarget(const StorePaths& paths, Target target)
{
    bool ok = true;
    std::error_code ec;
    for (const LabelKind kind : kAllKinds) {
        fs::remove(paths.file(kind, target), ec);
        ok = ok && !ec;
    }
    fs::remove(paths.sidecar(target), ec);
    ok = ok && !ec;
    fs::remove(paths.saving(target), ec);
    return ok && !ec;
}

}  // namespace vc3d::ink_labels

// ---------------------------------------------------------------------------
// Grid reorders
// ---------------------------------------------------------------------------

namespace vc3d::ink_labels {

namespace {

constexpr int kAnchorLattice = 7;
// Voxels. A flip copies the points; a quarter turn resamples them at whole
// grid positions, which can move them by float rounding only.
constexpr float kAnchorTolerance = 0.05f;
constexpr int kMinReorderMatches = 3;

bool validVertex(const cv::Vec3f& p)
{
    return p[0] != -1.f && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Where the saved vertex (row, col) is in the current grid.
cv::Point currentOf(const GridReorder& r, int row, int col, cv::Size grid)
{
    int a = r.swap ? col : row;
    int b = r.swap ? row : col;
    if (r.flipRows)
        a = grid.height - 1 - a;
    if (r.flipCols)
        b = grid.width - 1 - b;
    return {b, a};
}

}  // namespace

std::vector<GridAnchor> gridAnchors(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<GridAnchor> anchors;
    if (points.empty())
        return anchors;
    // The first valid vertex of each block of a 7 x 7 split of the grid.
    for (int by = 0; by < kAnchorLattice; ++by) {
        const int r0 = by * points.rows / kAnchorLattice, r1 = (by + 1) * points.rows / kAnchorLattice;
        for (int bx = 0; bx < kAnchorLattice; ++bx) {
            const int c0 = bx * points.cols / kAnchorLattice, c1 = (bx + 1) * points.cols / kAnchorLattice;
            bool found = false;
            for (int r = r0; r < r1 && !found; ++r)
                for (int c = c0; c < c1 && !found; ++c)
                    if (validVertex(points(r, c))) {
                        anchors.push_back({r, c, points(r, c)});
                        found = true;
                    }
        }
    }
    return anchors;
}

ReorderResult detectReorder(const std::vector<GridAnchor>& anchors, const cv::Mat_<cv::Vec3f>& points)
{
    ReorderResult result;
    if (anchors.empty() || points.empty())
        return result;
    const cv::Size grid = points.size();
    struct Score {
        int comparable = 0, matched = 0;
    };
    const auto score = [&](const GridReorder& r) {
        Score s;
        for (const GridAnchor& a : anchors) {
            if (!validVertex(a.point))
                continue;
            if (r.swap ? (a.col >= grid.height || a.row >= grid.width) : (a.row >= grid.height || a.col >= grid.width))
                continue;
            const cv::Point at = currentOf(r, a.row, a.col, grid);
            const cv::Vec3f& p = points(at.y, at.x);
            if (!validVertex(p))
                continue;
            ++s.comparable;
            if (cv::norm(p - a.point) <= kAnchorTolerance)
                ++s.matched;
        }
        return s;
    };
    const Score same = score(GridReorder{});
    Score best;
    GridReorder bestReorder;
    for (int swap = 0; swap < (grid.width == grid.height ? 2 : 1); ++swap)
        for (int code = 0; code < 4; ++code) {
            const GridReorder r{swap != 0, (code & 1) != 0, (code & 2) != 0};
            if (r.identity())
                continue;
            const Score s = score(r);
            if (s.matched > best.matched) {
                best = s;
                bestReorder = r;
            }
        }
    if (best.matched >= kMinReorderMatches && best.matched > same.matched && 2 * best.matched >= best.comparable) {
        result.match = ReorderMatch::Reordered;
        result.reorder = bestReorder;
    } else if (same.comparable > 0) {
        result.match = ReorderMatch::Same;
    }
    return result;
}

RemapOutcome remapLabels(const TileRaster& raster, const vc::RenderCanvas& canvas, const GridReorder& reorder)
{
    RemapOutcome out;
    const cv::Size size = raster.size();
    if (size != canvas.size) {
        out.error = QStringLiteral("the labels are not at the segment's canvas size");
        return out;
    }
    if (reorder.swap && (size.width != size.height || canvas.gridScale[0] != canvas.gridScale[1])) {
        out.error = QStringLiteral("a turned grid needs a square canvas with equal scales");
        return out;
    }
    // The saved label pixel that shows the grid position a current pixel
    // centre shows. The map is affine; read it off three pixels.
    const auto savedCentre = [&](double x, double y) {
        const cv::Vec2d g = canvas.canvasToGrid({x, y});
        double row = g[1], col = g[0];
        if (reorder.flipRows)
            row = canvas.grid.height - 1 - row;
        if (reorder.flipCols)
            col = canvas.grid.width - 1 - col;
        if (reorder.swap)
            std::swap(row, col);
        return canvas.gridToCanvas({col, row});
    };
    const cv::Vec2d o = savedCentre(0.5, 0.5);
    const cv::Vec2d lx = savedCentre(1.5, 0.5) - o;
    const cv::Vec2d ly = savedCentre(0.5, 1.5) - o;
    const auto whole = [](double v, int* out) {
        const double r = std::round(v);
        if (std::abs(v - r) > 1e-3)
            return false;
        *out = int(r);
        return true;
    };
    int bx = 0, by = 0, lxx = 0, lxy = 0, lyx = 0, lyy = 0;
    if (!whole(o[0] - 0.5, &bx) || !whole(o[1] - 0.5, &by) || !whole(lx[0], &lxx) || !whole(lx[1], &lxy)
        || !whole(ly[0], &lyx) || !whole(ly[1], &lyy)) {
        out.error = QStringLiteral("the label pixel pitch does not divide the grid step, so labels cannot move exactly");
        return out;
    }
    // saved = (bx + lxx * x + lyx * y, by + lxy * x + lyy * y)
    const auto savedPixel = [&](int x, int y) { return cv::Point(bx + lxx * x + lyx * y, by + lxy * x + lyy * y); };

    const auto labeled = [](const std::array<uint64_t, 8>& c) {
        uint64_t n = 0;
        for (int v = 1; v < 8; ++v)
            n += c[std::size_t(v)];
        return n;
    };
    TileRaster moved(size);
    cv::Mat_<uint8_t> src, flipped, dst;
    for (int index = 0; index < moved.tileCount(); ++index) {
        const cv::Rect rect = moved.tileRect(index);
        const cv::Point a = savedPixel(rect.x, rect.y);
        const cv::Point b = savedPixel(rect.br().x - 1, rect.br().y - 1);
        const cv::Rect from(std::min(a.x, b.x), std::min(a.y, b.y), std::abs(b.x - a.x) + 1, std::abs(b.y - a.y) + 1);
        if ((from & cv::Rect(0, 0, size.width, size.height)).empty())
            continue;
        raster.readRect(from, src);
        double lo = 0, hi = 0;
        cv::minMaxLoc(src, &lo, &hi);
        if (lo == hi) {
            if (hi != 0)
                moved.setTile(index, TileState{nullptr, uint8_t(hi)});
            continue;
        }
        if (!reorder.swap) {
            // Current x follows saved x (sign lxx), current y saved y.
            const int code = lxx < 0 && lyy < 0 ? -1 : (lxx < 0 ? 1 : (lyy < 0 ? 0 : 2));
            if (code == 2)
                dst = src;
            else
                cv::flip(src, dst, code);
        } else {
            // Current x follows saved y (sign lxy), current y saved x (lyx).
            const int code = lxy < 0 && lyx < 0 ? -1 : (lyx < 0 ? 1 : (lxy < 0 ? 0 : 2));
            if (code == 2)
                flipped = src;
            else
                cv::flip(src, flipped, code);
            cv::transpose(flipped, dst);
        }
        auto tile = std::make_shared<Tile>();
        for (int y = 0; y < rect.height; ++y)
            std::copy(dst.ptr(y), dst.ptr(y) + rect.width, tile->px.data() + std::size_t(y * kTileSize));
        const cv::Rect inCanvas(0, 0, rect.width, rect.height);
        finalizeTile(*tile, inCanvas);
        moved.setTile(index, publishTile(std::move(tile), inCanvas));
    }
    const uint64_t before = labeled(raster.counts());
    const uint64_t after = labeled(moved.counts());
    out.droppedLabeled = before > after ? before - after : 0;
    out.raster = std::move(moved);
    out.ok = true;
    return out;
}

}  // namespace vc3d::ink_labels
