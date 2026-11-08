#include "ink_labels/InkLabelImport.hpp"

#include <vc_reader/zarr.hpp>
#include <vc_reader/array.hpp>
#include "ink_labels/InkLabelImage.hpp"
#include <vc_reader/image_write.hpp>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <set>
#include <system_error>

namespace fs = std::filesystem;

namespace vc3d::ink_labels {

namespace {

constexpr int kBlock = 1024;

QString qpath(const fs::path& p)
{
    return QString::fromStdString(p.string());
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

bool endsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Sets `bit` where a block is above 0 and clears it where it is 0.
struct BitWriter {
    EditTxn& txn;
    uint8_t bit;
    uint64_t setPixels = 0;
    uint64_t nonBinary = 0;

    void block(cv::Point topLeft, const cv::Mat& sample0)
    {
        cv::Mat on;
        cv::compare(sample0, cv::Scalar(0), on, cv::CMP_GT);
        cv::Mat off;
        cv::bitwise_not(on, off);
        const int full = sample0.depth() == CV_16U ? 65535 : 255;
        cv::Mat partial = (sample0 > 0) & (sample0 < full);
        nonBinary += uint64_t(cv::countNonZero(partial));
        setPixels += uint64_t(cv::countNonZero(on));
        txn.applyMask(topLeft, cv::Mat_<uint8_t>(on), PaintOp{bit, 0});
        txn.applyMask(topLeft, cv::Mat_<uint8_t>(off), PaintOp{0, bit});
    }
};

// How a source of `source` px lands on the canvas: directly, resampled from
// a lower render scale, or not at all.
struct Placement {
    bool ok = false;
    bool needsDecision = false;
    double renderScale = 1.0;
    QString error;
};

Placement placementFor(cv::Size source, const vc::RenderCanvas& canvas, bool allowResample, const QString& name)
{
    Placement p;
    const cv::Size target = canvas.size;
    if (source == target) {
        p.ok = true;
        return p;
    }
    if (source == cv::Size(target.height, target.width)) {
        p.error = QStringLiteral("%1 is %2 x %3 px: the %4 x %5 px canvas on its side, which usually means a rotated render.")
                      .arg(name)
                      .arg(source.width)
                      .arg(source.height)
                      .arg(target.width)
                      .arg(target.height);
        return p;
    }
    for (const double rs : {0.5, 0.25}) {
        if (source != vc::renderCanvasSize(canvas.grid, canvas.gridScale, rs))
            continue;
        if (allowResample) {
            p.ok = true;
            p.renderScale = rs;
        } else {
            p.needsDecision = true;
            p.renderScale = rs;
            p.error = QStringLiteral("%1 is %2 x %3 px, the canvas rendered at scale %4. Import it resampled (nearest neighbor) to "
                                     "%5 x %6 px?")
                          .arg(name)
                          .arg(source.width)
                          .arg(source.height)
                          .arg(rs)
                          .arg(target.width)
                          .arg(target.height);
        }
        return p;
    }
    p.error = QStringLiteral("%1 is %2 x %3 px but the canvas is %4 x %5 px. To carry labels across a re-render use "
                             "vesuvius.tifxyz_label_transfer.")
                  .arg(name)
                  .arg(source.width)
                  .arg(source.height)
                  .arg(target.width)
                  .arg(target.height);
    return p;
}

// Applies a whole in-memory source, directly or resampled, in canvas blocks.
void applySource(BitWriter& writer, const cv::Mat& source, const vc::RenderCanvas& canvas, double rs)
{
    const cv::Size target = canvas.size;
    if (rs == 1.0) {
        for (int y = 0; y < target.height; y += kBlock)
            for (int x = 0; x < target.width; x += kBlock) {
                const cv::Rect r(x, y, std::min(kBlock, target.width - x), std::min(kBlock, target.height - y));
                writer.block(r.tl(), source(r));
            }
        return;
    }
    // Target pixel (x, y) takes the source pixel under its center:
    // floor(Ws/2 + rs * (x + 0.5 - W/2)), clamped, and likewise in y.
    const auto index = [](int i, int targetSize, int sourceSize, double scale) {
        const double s = 0.5 * sourceSize + scale * (i + 0.5 - 0.5 * targetSize);
        return std::clamp(int(std::floor(s)), 0, sourceSize - 1);
    };
    std::vector<int> cols(std::size_t(target.width)), rows(std::size_t(target.height));
    for (int x = 0; x < target.width; ++x)
        cols[std::size_t(x)] = index(x, target.width, source.cols, rs);
    for (int y = 0; y < target.height; ++y)
        rows[std::size_t(y)] = index(y, target.height, source.rows, rs);
    for (int y0 = 0; y0 < target.height; y0 += kBlock)
        for (int x0 = 0; x0 < target.width; x0 += kBlock) {
            const cv::Rect r(x0, y0, std::min(kBlock, target.width - x0), std::min(kBlock, target.height - y0));
            cv::Mat block(r.size(), source.type());
            for (int y = 0; y < r.height; ++y) {
                const int sy = rows[std::size_t(r.y + y)];
                if (source.depth() == CV_16U) {
                    const uint16_t* src = source.ptr<uint16_t>(sy);
                    uint16_t* dst = block.ptr<uint16_t>(y);
                    for (int x = 0; x < r.width; ++x)
                        dst[x] = src[cols[std::size_t(r.x + x)]];
                } else {
                    const uint8_t* src = source.ptr<uint8_t>(sy);
                    uint8_t* dst = block.ptr<uint8_t>(y);
                    for (int x = 0; x < r.width; ++x)
                        dst[x] = src[cols[std::size_t(r.x + x)]];
                }
            }
            writer.block(r.tl(), block);
        }
}

void finish(ImportOutcome& out, const BitWriter& writer, const QString& name)
{
    out.setPixels = writer.setPixels;
    out.nonBinary = writer.nonBinary;
    if (writer.nonBinary > 0)
        out.warnings.append(QStringLiteral("%1 holds %2 px with values other than 0 and full scale; any value above 0 counts as "
                                           "labeled.")
                                .arg(name)
                                .arg(writer.nonBinary));
    out.ok = true;
}

void importTiff(const fs::path& path, const vc::RenderCanvas& canvas, const ImportOptions& options, EditTxn& into, ImportOutcome& out)
{
    const QString name = qpath(path.filename());
    BitWriter writer{into, bitFor(options.kind)};
    Placement placement;
    bool decided = false;
    bool multiSample = false;
    cv::Mat whole;
    std::string error;
    const bool ok = forEachLabelImageBlock(
        path,
        [&](const LabelImageInfo& info, cv::Point topLeft, const cv::Mat& block) {
            if (!decided) {
                decided = true;
                out.sourceShape = cv::Size(int(info.width), int(info.height));
                placement = placementFor(out.sourceShape, canvas, options.allowResample, name);
                if (!placement.ok)
                    return false;
                if (placement.renderScale != 1.0)
                    whole.create(out.sourceShape, CV_MAKETYPE(info.bitsPerSample == 16 ? CV_16U : CV_8U, 1));
            }
            cv::Mat sample0 = block;
            if (block.channels() > 1) {
                multiSample = true;
                cv::extractChannel(block, sample0, 0);
            }
            if (placement.renderScale == 1.0)
                writer.block(topLeft, sample0);
            else
                sample0.copyTo(whole(cv::Rect(topLeft, sample0.size())));
            return true;
        },
        &error);
    if (decided && !placement.ok) {
        out.error = placement.error;
        out.needsDecision = placement.needsDecision;
        out.resampledFrom = placement.renderScale;
        return;
    }
    if (!ok) {
        out.error = QStringLiteral("cannot read %1: %2").arg(name, QString::fromStdString(error));
        return;
    }
    if (placement.renderScale != 1.0)
        applySource(writer, whole, canvas, placement.renderScale);
    out.resampledFrom = placement.renderScale;
    if (multiSample)
        out.warnings.append(QStringLiteral("%1 has more than one sample per pixel; only the first is used.").arg(name));
    finish(out, writer, name);
}

std::optional<cv::Mat> readPng(const fs::path& path, ImportOutcome& out)
{
    const QString name = qpath(path.filename());
    std::error_code ec;
    const auto bytes = fs::file_size(path, ec);
    if (ec) {
        out.error = QStringLiteral("cannot read %1").arg(name);
        return std::nullopt;
    }
    if (bytes > kMaxPngBytes) {
        out.error = QStringLiteral("%1 is larger than 1 GiB; save it as a tiled TIFF to import it.").arg(name);
        return std::nullopt;
    }
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        out.error = QStringLiteral("cannot decode %1").arg(name);
        return std::nullopt;
    }
    if (image.depth() != CV_8U && image.depth() != CV_16U) {
        out.error = QStringLiteral("%1 is not an 8 or 16 bit image").arg(name);
        return std::nullopt;
    }
    if (image.channels() > 1) {
        // Channel 0 of the OpenCV decode, as create_label_zarrs (cv2) keeps.
        cv::Mat first;
        cv::extractChannel(image, first, 0);
        out.warnings.append(QStringLiteral("%1 has more than one channel; only the first is used.").arg(name));
        image = first;
    }
    return image;
}

bool importPng(const fs::path& path, const vc::RenderCanvas& canvas, const ImportOptions& options, EditTxn& into,
               ImportOutcome& out, LabelKind kind)
{
    const QString name = qpath(path.filename());
    const auto image = readPng(path, out);
    if (!image)
        return false;
    out.sourceShape = image->size();
    const Placement placement = placementFor(image->size(), canvas, options.allowResample, name);
    if (!placement.ok) {
        out.error = placement.error;
        out.needsDecision = placement.needsDecision;
        out.resampledFrom = placement.renderScale;
        return false;
    }
    BitWriter writer{into, bitFor(kind)};
    applySource(writer, *image, canvas, placement.renderScale);
    out.resampledFrom = placement.renderScale;
    finish(out, writer, name);
    return true;
}

// A zarr v2 array (.zarray) or a v3 node whose zarr.json says it is an array
// (a v3 multiscale group has a zarr.json too).
bool hasZarrArrayMetadata(const fs::path& dir)
{
    return vcr::zarr::isLocalZarrArray(dir);
}

// The array itself, or level 0 of a multiscale group. A level's own chunk
// folders can be named "0" too, so the folder's own metadata wins.
fs::path zarrArrayPath(const fs::path& path)
{
    if (!hasZarrArrayMetadata(path) && hasZarrArrayMetadata(path / "0"))
        return path / "0";
    return path;
}

void importZarr(const fs::path& path, const vc::RenderCanvas& canvas, const ImportOptions& options, EditTxn& into, ImportOutcome& out)
{
    const QString name = qpath(path.filename());
    std::optional<vcr::zarr::OpenedArray> dataset;
    try {
        dataset.emplace(vcr::zarr::openArray(zarrArrayPath(path).string(), {}));
        if (dataset->meta.dtype != "|u1" && dataset->meta.dtype != "<u2")
            throw std::invalid_argument("expected uint8 or uint16 labels");
    } catch (const std::exception& e) {
        out.error = QStringLiteral("cannot open %1 (%2). Only uint8 and uint16 zarr arrays can be imported; convert other types, "
                                   "such as bool, with numpy (array.astype('uint8') * 255) or export a TIFF.")
                        .arg(name, QString::fromUtf8(e.what()));
        return;
    }
    const auto& shape = dataset->meta.shape;
    if (shape.size() != 2 && shape.size() != 3) {
        out.error = QStringLiteral("%1 has %2 dimensions; expected a 2D image or a (65, H, W) label zarr.").arg(name).arg(shape.size());
        return;
    }
    const bool planes = shape.size() == 3;
    const size_t height = shape[shape.size() - 2], width = shape[shape.size() - 1];
    out.sourceShape = cv::Size(int(width), int(height));
    out.format = planes ? ImportFormat::LabelZarr65 : ImportFormat::Zarr2D;
    const Placement placement = placementFor(out.sourceShape, canvas, options.allowResample, name);
    if (!placement.ok) {
        out.error = placement.error;
        out.needsDecision = placement.needsDecision;
        out.resampledFrom = placement.renderScale;
        return;
    }
    const int type = dataset->meta.dtype == "<u2" ? CV_16UC1 : CV_8UC1;
    const size_t plane = planes ? shape[0] / 2 : 0;
    // Row bands of at most 64 MiB.
    const size_t rowBytes = width * dataset->meta.itemsize();
    const size_t bandRows = std::max<size_t>(1, std::min<size_t>(kBlock, (size_t(64) << 20) / std::max<size_t>(1, rowBytes)));
    BitWriter writer{into, bitFor(options.kind)};
    cv::Mat whole;
    if (placement.renderScale != 1.0)
        whole.create(out.sourceShape, type);
    for (size_t y = 0; y < height; y += bandRows) {
        const size_t rows = std::min(bandRows, height - y);
        cv::Mat band(int(rows), int(width), type);
        try {
            const vcr::Index start = planes ? vcr::Index{int64_t(plane), int64_t(y), 0} : vcr::Index{int64_t(y), 0};
            const vcr::Index size = planes ? vcr::Index{1, int64_t(rows), int64_t(width)} : vcr::Index{int64_t(rows), int64_t(width)};
            dataset->array->read_many({start}, {size}, {band.data});
        } catch (const std::exception& e) {
            out.error = QStringLiteral("cannot read rows %1 to %2 of %3: %4").arg(y).arg(y + rows).arg(name).arg(QString::fromUtf8(e.what()));
            return;
        }
        if (placement.renderScale != 1.0) {
            band.copyTo(whole(cv::Rect(0, int(y), int(width), int(rows))));
            continue;
        }
        for (int x = 0; x < int(width); x += kBlock) {
            const cv::Rect r(x, 0, std::min(kBlock, int(width) - x), int(rows));
            writer.block(cv::Point(x, int(y)), band(r));
        }
    }
    if (placement.renderScale != 1.0)
        applySource(writer, whole, canvas, placement.renderScale);
    out.resampledFrom = placement.renderScale;
    finish(out, writer, name);
    if (planes && shape[0] != 65)
        out.warnings.append(
            QStringLiteral("%1 has %2 planes; plane %3 was read (create_label_zarrs writes 65, label in 32).").arg(name).arg(shape[0]).arg(plane));
}

fs::path legacyMaskFor(const fs::path& inkPng)
{
    const std::string stem = inkPng.stem().string();
    const std::string base = stem.substr(0, stem.size() - std::string("_inklabels").size());
    return inkPng.parent_path() / (base + "_mask.png");
}

}  // namespace

const char* formatName(ImportFormat format)
{
    switch (format) {
        case ImportFormat::Unknown:
            return "unknown";
        case ImportFormat::Tiff:
            return "tiff";
        case ImportFormat::Png:
            return "png";
        case ImportFormat::LabelZarr65:
            return "label_zarr";
        case ImportFormat::Zarr2D:
            return "zarr_2d";
        case ImportFormat::LegacyPngPair:
            return "legacy_png_pair";
    }
    return "unknown";
}

ImportFormat detectFormat(const fs::path& path)
{
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        const fs::path array = zarrArrayPath(path);
        if (!hasZarrArrayMetadata(array))
            return ImportFormat::Unknown;
        try {
            const auto dataset = vcr::zarr::openArray(array.string(), {});
            const size_t dims = dataset.meta.shape.size();
            return dims == 3 ? ImportFormat::LabelZarr65 : dims == 2 ? ImportFormat::Zarr2D : ImportFormat::Unknown;
        } catch (const std::exception&) {
            // Still a zarr: importLabel reports why it cannot be read.
            return ImportFormat::Zarr2D;
        }
    }
    const std::string name = lower(path.filename().string());
    if (endsWith(name, ".tif") || endsWith(name, ".tiff"))
        return ImportFormat::Tiff;
    if (endsWith(name, ".png")) {
        if (endsWith(name, "_inklabels.png") && fs::exists(legacyMaskFor(path), ec))
            return ImportFormat::LegacyPngPair;
        return ImportFormat::Png;
    }
    return ImportFormat::Unknown;
}

LabelKind kindForFileName(const fs::path& path)
{
    const std::string name = lower(path.filename().string());
    if (name.find("supervision_mask") != std::string::npos)
        return LabelKind::Supervision;
    if (name.find("validation_mask") != std::string::npos)
        return LabelKind::Validation;
    return LabelKind::Ink;
}

ImportOutcome importLabel(const fs::path& path, const vc::RenderCanvas& canvas, const ImportOptions& options, EditTxn& into)
{
    ImportOutcome out;
    out.format = detectFormat(path);
    switch (out.format) {
        case ImportFormat::Unknown:
            out.error = QStringLiteral("%1 is not a TIFF, PNG or zarr label image").arg(qpath(path.filename()));
            return out;
        case ImportFormat::Tiff:
            importTiff(path, canvas, options, into, out);
            return out;
        case ImportFormat::Png:
            importPng(path, canvas, options, into, out, options.kind);
            return out;
        case ImportFormat::LabelZarr65:
        case ImportFormat::Zarr2D:
            importZarr(path, canvas, options, into, out);
            return out;
        case ImportFormat::LegacyPngPair: {
            const fs::path mask = legacyMaskFor(path);
            if (options.legacyMaskAs == LegacyMaskAs::Ask) {
                out.needsDecision = true;
                out.error = QStringLiteral("%1 has a legacy %2 (the valid surface). Use it as the supervision mask?")
                                .arg(qpath(path.filename()), qpath(mask.filename()));
                return out;
            }
            if (!importPng(path, canvas, options, into, out, LabelKind::Ink))
                return out;
            if (options.legacyMaskAs == LegacyMaskAs::Supervision) {
                ImportOutcome maskOut;
                if (!importPng(mask, canvas, options, into, maskOut, LabelKind::Supervision)) {
                    out.ok = false;
                    out.error = maskOut.error;
                    out.needsDecision = maskOut.needsDecision;
                    return out;
                }
                out.warnings.append(maskOut.warnings);
                out.warnings.append(QStringLiteral("%1 imported as the supervision mask (%2 px).")
                                        .arg(qpath(mask.filename()))
                                        .arg(maskOut.setPixels));
            }
            return out;
        }
    }
    return out;
}

std::vector<int> listVersions(const fs::path& dir, std::string_view prefix)
{
    std::set<int> versions;
    std::error_code ec;
    if (!fs::is_directory(dir, ec))
        return {};
    const std::string base(prefix);
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        for (const char* kind : {"_inklabels", "_supervision_mask", "_validation_mask"}) {
            const std::string head = base + kind;
            if (name.compare(0, head.size(), head) != 0)
                continue;
            std::string rest = name.substr(head.size());
            std::string ext;
            for (const char* e : {".tif", ".json"})
                if (endsWith(rest, e)) {
                    ext = e;
                    break;
                }
            // .prev, .autosave and .tmp-* do not end in .tif or .json.
            if (ext.empty())
                continue;
            rest = rest.substr(0, rest.size() - ext.size());
            if (ext == ".json" && std::string(kind) != "_inklabels")
                continue;
            if (rest.empty()) {
                versions.insert(1);
            } else if (rest.size() > 2 && rest[0] == '_' && rest[1] == 'v'
                       && std::all_of(rest.begin() + 2, rest.end(), [](unsigned char c) { return std::isdigit(c); })) {
                int number = 0;
                const char* first = rest.data() + 2;
                const char* last = rest.data() + rest.size();
                const auto [end, ec] = std::from_chars(first, last, number);
                if (ec == std::errc() && end == last && number >= 2)
                    versions.insert(number);
            }
        }
    }
    return {versions.begin(), versions.end()};
}

QJsonObject datasetEntry(const StorePaths& paths, const fs::path& projectDir, const std::vector<int>& versions)
{
    const QString seg = QString::fromStdString(paths.prefix);
    QJsonObject entry;
    entry["segments_path"] = qpath(projectDir / "ink_labels");
    entry["segment_names"] = QJsonArray{seg};
    QJsonObject volumes;
    volumes[seg] = qpath(paths.dir / (paths.prefix + ".zarr"));
    entry["surface_volume_paths"] = volumes;
    // The trainer picks the newest version per kind unless told otherwise;
    // unversioned files are its "v1".
    if (versions.size() > 1 || paths.version > 1)
        entry["label_version"] = QStringLiteral("v%1").arg(paths.version);
    return entry;
}

}  // namespace vc3d::ink_labels
