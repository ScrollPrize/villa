#include "InkDetectionOverlayController.hpp"

#include "../CState.hpp"
#include "../VolumeViewerCmaps.hpp"
#include "../volume_viewers/VolumeViewerBase.hpp"
#include "OpenDataSegmentCache.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace {
constexpr qreal kInkDetectionZ = 18.0;

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool isJpegPath(const std::filesystem::path& path)
{
    const std::string ext = lowerCopy(path.extension().string());
    return ext == ".jpg" || ext == ".jpeg";
}

std::string detectionGroupKey(const vc3d::opendata::OpenDataInkDetectionEntry& entry)
{
    if (!entry.segmentLongId.empty()) {
        return "long:" + entry.segmentLongId;
    }
    if (!entry.sampleId.empty() || !entry.segmentId.empty()) {
        return "segment:" + entry.sampleId + "/" + entry.segmentId;
    }
    return "path:" + entry.localPath.parent_path().parent_path().string();
}

std::vector<vc3d::opendata::OpenDataInkDetectionEntry> inkDetectionsForAttachedSegmentPath(
    const std::filesystem::path& segmentPath)
{
    std::vector<vc3d::opendata::OpenDataInkDetectionEntry> out =
        vc3d::opendata::cachedInkDetectionsForSegmentDirectory(segmentPath);

    std::error_code ec;
    if (!std::filesystem::is_directory(segmentPath, ec)) {
        return out;
    }

    for (const auto& child : std::filesystem::directory_iterator(segmentPath, ec)) {
        if (ec) {
            break;
        }
        if (!child.is_directory(ec) || ec) {
            ec.clear();
            continue;
        }
        auto childEntries = vc3d::opendata::cachedInkDetectionsForSegmentDirectory(child.path());
        out.insert(out.end(),
                   std::make_move_iterator(childEntries.begin()),
                   std::make_move_iterator(childEntries.end()));
    }
    return out;
}

cv::Mat_<uint8_t> toU8Scalar(const cv::Mat& src)
{
    if (src.empty()) {
        return {};
    }

    cv::Mat scalar;
    if (src.channels() == 1) {
        scalar = src;
    } else if (src.channels() == 3) {
        cv::cvtColor(src, scalar, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, scalar, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }

    if (scalar.depth() == CV_8U) {
        return scalar.clone();
    }

    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(scalar, &minValue, &maxValue);
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue <= minValue) {
        cv::Mat out(scalar.rows, scalar.cols, CV_8UC1, cv::Scalar(0));
        return out;
    }

    cv::Mat out;
    scalar.convertTo(out, CV_8UC1, 255.0 / (maxValue - minValue), -minValue * 255.0 / (maxValue - minValue));
    return out;
}

bool imageLooksSingleChannel(const cv::Mat& src)
{
    if (src.empty()) {
        return false;
    }
    if (src.channels() == 1) {
        return true;
    }
    if (src.channels() != 3 && src.channels() != 4) {
        return false;
    }
    if (src.depth() != CV_8U) {
        return false;
    }

    for (int y = 0; y < src.rows; ++y) {
        if (src.channels() == 3) {
            const auto* row = src.ptr<cv::Vec3b>(y);
            for (int x = 0; x < src.cols; ++x) {
                if (row[x][0] != row[x][1] || row[x][1] != row[x][2]) {
                    return false;
                }
            }
        } else {
            const auto* row = src.ptr<cv::Vec4b>(y);
            for (int x = 0; x < src.cols; ++x) {
                if (row[x][0] != row[x][1] || row[x][1] != row[x][2]) {
                    return false;
                }
            }
        }
    }
    return true;
}

QImage colorMatToImage(const cv::Mat& src)
{
    if (src.empty() || src.depth() != CV_8U || (src.channels() != 3 && src.channels() != 4)) {
        return {};
    }

    QImage image(src.cols, src.rows, QImage::Format_ARGB32);
    for (int y = 0; y < src.rows; ++y) {
        auto* out = reinterpret_cast<QRgb*>(image.scanLine(y));
        if (src.channels() == 3) {
            const auto* in = src.ptr<cv::Vec3b>(y);
            for (int x = 0; x < src.cols; ++x) {
                const int b = in[x][0];
                const int g = in[x][1];
                const int r = in[x][2];
                const int a = (r == 0 && g == 0 && b == 0) ? 0 : 255;
                out[x] = qRgba(r, g, b, a);
            }
        } else {
            const auto* in = src.ptr<cv::Vec4b>(y);
            for (int x = 0; x < src.cols; ++x) {
                const int b = in[x][0];
                const int g = in[x][1];
                const int r = in[x][2];
                const int a = (r == 0 && g == 0 && b == 0) ? 0 : in[x][3];
                out[x] = qRgba(r, g, b, a);
            }
        }
    }
    return image;
}

} // namespace

InkDetectionOverlayController::InkDetectionOverlayController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase("ink_detection_overlay", parent)
    , _state(state)
{
    if (_state) {
        connect(_state, &CState::vpkgChanged, this, &InkDetectionOverlayController::refreshAvailableDetections);
        connect(_state, &CState::surfacesLoaded, this, &InkDetectionOverlayController::refreshAvailableDetections);
        connect(_state, &CState::surfaceChanged, this, [this](const std::string& name, std::shared_ptr<Surface>, bool isEditUpdate) {
            if (isEditUpdate && name == "segmentation" && activeSegmentKey() == _selectedSegmentKey) {
                return;
            }
            if (name.empty() || name == "segmentation") {
                refreshOptionsForActiveSurface();
            }
        });
    }
    refreshAvailableDetections();
}

void InkDetectionOverlayController::refreshAvailableDetections()
{
    std::vector<Option> next;
    std::set<std::filesystem::path> seen;
    std::set<std::string> seenSourceUrls;

    if (_state && _state->vpkg()) {
        std::vector<vc3d::opendata::OpenDataInkDetectionEntry> entries;
        for (const auto& segmentPath : _state->vpkg()->availableSegmentPaths()) {
            auto pathEntries = inkDetectionsForAttachedSegmentPath(segmentPath);
            entries.insert(entries.end(),
                           std::make_move_iterator(pathEntries.begin()),
                           std::make_move_iterator(pathEntries.end()));
        }

        std::set<std::string> groupsWithJpeg;
        for (const auto& entry : entries) {
            if (isJpegPath(entry.localPath)) {
                groupsWithJpeg.insert(detectionGroupKey(entry));
            }
        }

        for (const auto& entry : entries) {
            if (!isJpegPath(entry.localPath) &&
                groupsWithJpeg.find(detectionGroupKey(entry)) != groupsWithJpeg.end()) {
                continue;
            }
            if (!entry.sourceUrl.empty() && !seenSourceUrls.insert(entry.sourceUrl).second) {
                continue;
            }
            std::error_code ec;
            auto canonical = std::filesystem::weakly_canonical(entry.localPath, ec);
            if (ec) {
                canonical = entry.localPath.lexically_normal();
            }
            if (!seen.insert(canonical).second) {
                continue;
            }

            Option option;
            option.label = QString::fromStdString(entry.label);
            option.sampleId = entry.sampleId;
            option.segmentId = entry.segmentId;
            option.segmentLongId = entry.segmentLongId;
            option.localPath = canonical;
            const cv::Mat probe = cv::imread(canonical.string(), cv::IMREAD_UNCHANGED);
            option.singleChannel = imageLooksSingleChannel(probe);
            next.push_back(std::move(option));
        }
    }

    std::sort(next.begin(), next.end(), [](const Option& a, const Option& b) {
        if (a.segmentId != b.segmentId) {
            return a.segmentId < b.segmentId;
        }
        return a.label < b.label;
    });

    _allOptions = std::move(next);
    refreshOptionsForActiveSurface();
}

void InkDetectionOverlayController::setSelectedPath(const std::filesystem::path& path)
{
    if (_selectedPath == path) {
        return;
    }
    _selectedPath = path;
    if (!_selectedSegmentKey.empty()) {
        _selectedPathBySegment[_selectedSegmentKey] = _selectedPath;
    }
    _selectedImage = {};
    loadSelectedImage();
    emit selectionChanged();
    refreshAll();
}

void InkDetectionOverlayController::clearSelection()
{
    setSelectedPath({});
}

void InkDetectionOverlayController::toggleVisibility()
{
    if (!hasLoadedSelection()) {
        return;
    }

    if (_opacity > 0) {
        _opacityBeforeToggle = _opacity;
        setOpacity(0);
    } else {
        setOpacity(_opacityBeforeToggle > 0 ? _opacityBeforeToggle : 70);
    }
}

void InkDetectionOverlayController::setOpacity(int opacity)
{
    const int clamped = std::clamp(opacity, 0, 100);
    if (_opacity == clamped) {
        return;
    }
    _opacity = clamped;
    if (_opacity > 0) {
        _opacityBeforeToggle = _opacity;
    }
    emit opacityChanged(_opacity);
    refreshAll();
}

bool InkDetectionOverlayController::hasLoadedSelection() const
{
    return !_selectedPath.empty() && !_selectedImage.image.isNull();
}

void InkDetectionOverlayController::setColormapId(const std::string& id)
{
    if (_colormapId == id) {
        return;
    }
    _colormapId = id;
    if (_selectedImage.singleChannel && !_selectedImage.scalar.empty()) {
        _selectedImage.image = renderSingleChannelImage(_selectedImage.scalar, _colormapId);
        _selectedImage.colormapId = _colormapId;
    }
    refreshAll();
}

bool InkDetectionOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    return viewer && !_selectedPath.empty() && !_selectedImage.image.isNull() && _opacity > 0;
}

void InkDetectionOverlayController::collectPrimitives(
    VolumeViewerBase* viewer,
    ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!isOverlayEnabledFor(viewer)) {
        return;
    }

    auto* surface = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    if (!surface || !imageMatchesSurface(*surface)) {
        return;
    }

    const cv::Vec2f surfScale = surface->scale();
    if (std::abs(surfScale[0]) < 1.0e-6f || std::abs(surfScale[1]) < 1.0e-6f) {
        return;
    }

    const cv::Vec3f center = surface->center();
    auto gridToScene = [&](int row, int col) -> QPointF {
        const float surfX = static_cast<float>(col) / surfScale[0] - center[0];
        const float surfY = static_cast<float>(row) / surfScale[1] - center[1];
        return viewer->surfaceCoordsToScene(surfX, surfY);
    };

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty() || _selectedImage.image.width() <= 0 ||
        _selectedImage.image.height() <= 0) {
        return;
    }

    const QPointF grid00 = gridToScene(0, 0);
    const QPointF grid01 = gridToScene(0, 1);
    const QPointF grid10 = gridToScene(1, 0);
    const QPointF colStep = grid01 - grid00;
    const QPointF rowStep = grid10 - grid00;
    const qreal gridScaleX = std::hypot(colStep.x(), colStep.y());
    const qreal gridScaleY = std::hypot(rowStep.x(), rowStep.y());
    if (gridScaleX < 1.0e-6 || gridScaleY < 1.0e-6) {
        return;
    }

    const qreal imageScaleX = gridScaleX *
        (static_cast<qreal>(points->cols) / static_cast<qreal>(_selectedImage.image.width()));
    const qreal imageScaleY = gridScaleY *
        (static_cast<qreal>(points->rows) / static_cast<qreal>(_selectedImage.image.height()));

    builder.addImage(_selectedImage.image,
                     grid00,
                     imageScaleX,
                     imageScaleY,
                     static_cast<qreal>(_opacity) / 100.0,
                     kInkDetectionZ);
}

void InkDetectionOverlayController::refreshOptionsForActiveSurface()
{
    if (!_selectedSegmentKey.empty()) {
        _selectedPathBySegment[_selectedSegmentKey] = _selectedPath;
    }

    const std::string nextSegmentKey = activeSegmentKey();
    std::vector<Option> next;
    for (const auto& option : _allOptions) {
        if (optionMatchesSegment(option, nextSegmentKey)) {
            next.push_back(option);
        }
    }

    std::filesystem::path nextSelectedPath;
    if (const auto savedIt = _selectedPathBySegment.find(nextSegmentKey);
        savedIt != _selectedPathBySegment.end()) {
        const auto& savedPath = savedIt->second;
        if (!savedPath.empty() &&
            std::any_of(next.begin(), next.end(), [&](const Option& option) {
                return option.localPath == savedPath;
            })) {
            nextSelectedPath = savedPath;
        }
    }

    const bool sameOptions = next.size() == _options.size() &&
        std::equal(next.begin(), next.end(), _options.begin(), [](const Option& a, const Option& b) {
            return a.localPath == b.localPath;
        });
    if (nextSegmentKey == _selectedSegmentKey && sameOptions && nextSelectedPath == _selectedPath) {
        return;
    }

    _selectedSegmentKey = nextSegmentKey;
    _options = std::move(next);
    _selectedPath = std::move(nextSelectedPath);

    _selectedImage = {};
    loadSelectedImage();
    emit availableDetectionsChanged();
    emit selectionChanged();
    refreshAll();
}

std::string InkDetectionOverlayController::activeSegmentKey() const
{
    if (!_state) {
        return {};
    }
    auto surface = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (!surface) {
        return {};
    }
    if (auto active = _state->activeSurface().lock();
        active.get() == surface.get() && !_state->activeSurfaceId().empty()) {
        return _state->activeSurfaceId();
    }
    return surface->id;
}

bool InkDetectionOverlayController::optionMatchesSegment(
    const Option& option,
    const std::string& segmentKey) const
{
    if (segmentKey.empty()) {
        return false;
    }
    return option.segmentId == segmentKey || option.segmentLongId == segmentKey;
}

void InkDetectionOverlayController::loadSelectedImage()
{
    if (_selectedPath.empty()) {
        return;
    }

    const cv::Mat src = cv::imread(_selectedPath.string(), cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        return;
    }

    LoadedImage loaded;
    loaded.path = _selectedPath;
    loaded.singleChannel = imageLooksSingleChannel(src);
    if (loaded.singleChannel) {
        loaded.scalar = toU8Scalar(src);
        loaded.image = renderSingleChannelImage(loaded.scalar, _colormapId);
        loaded.colormapId = _colormapId;
    } else {
        loaded.image = colorMatToImage(src);
    }

    _selectedImage = std::move(loaded);
}

QImage InkDetectionOverlayController::renderSingleChannelImage(
    const cv::Mat_<uint8_t>& scalar,
    const std::string& colormapId) const
{
    if (scalar.empty()) {
        return {};
    }

    QImage image(scalar.cols, scalar.rows, QImage::Format_ARGB32);
    if (colormapId.empty()) {
        for (int y = 0; y < scalar.rows; ++y) {
            auto* out = reinterpret_cast<QRgb*>(image.scanLine(y));
            const auto* in = scalar.ptr<uint8_t>(y);
            for (int x = 0; x < scalar.cols; ++x) {
                const int v = in[x];
                out[x] = qRgba(v, v, v, v);
            }
        }
        return image;
    }

    volume_viewer_cmaps::makeColors(
        scalar,
        volume_viewer_cmaps::resolve(colormapId),
        reinterpret_cast<uint32_t*>(image.bits()),
        image.bytesPerLine() / 4);

    for (int y = 0; y < scalar.rows; ++y) {
        auto* out = reinterpret_cast<QRgb*>(image.scanLine(y));
        const auto* in = scalar.ptr<uint8_t>(y);
        for (int x = 0; x < scalar.cols; ++x) {
            const QRgb color = out[x];
            out[x] = qRgba(qRed(color), qGreen(color), qBlue(color), in[x]);
        }
    }
    return image;
}

bool InkDetectionOverlayController::imageMatchesSurface(const QuadSurface& surface) const
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty() || _selectedImage.image.isNull()) {
        return false;
    }
    if (_selectedImage.image.width() <= 0 || _selectedImage.image.height() <= 0) {
        return false;
    }

    const auto optionIt = std::find_if(_options.begin(), _options.end(), [&](const Option& option) {
        return option.localPath == _selectedPath;
    });
    if (optionIt == _options.end()) {
        return false;
    }

    std::string segmentKey = surface.id;
    if (_state) {
        if (auto active = _state->activeSurface().lock();
            active.get() == &surface && !_state->activeSurfaceId().empty()) {
            segmentKey = _state->activeSurfaceId();
        }
        if (segmentKey.empty()) {
            segmentKey = _state->findSurfaceId(const_cast<QuadSurface*>(&surface));
        }
    }
    return optionMatchesSegment(*optionIt, segmentKey);
}
