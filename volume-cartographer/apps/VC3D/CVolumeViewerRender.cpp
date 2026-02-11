#include "CVolumeViewer.hpp"
#include "ViewerManager.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QDebug>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>


#include <QPainter>
#include <optional>

#include <opencv2/imgproc.hpp>

#define COLOR_FOCUS QColor(50, 255, 215)


void CVolumeViewer::renderVisible(bool force)
{
    if (isWindowMinimized()) {
        _dirtyWhileMinimized = true;
        return;
    }

    auto surf = _surf_weak.lock();
    if (surf && _surf_col) {
        auto currentSurface = _surf_col->surface(_surf_name);
        if (!currentSurface) {
            // Surface was cleared (e.g. during volume reload) without a change signal
            // reaching this viewer yet; drop the dangling pointer before rendering.
            _surf_weak.reset();
            surf.reset();
        }
    }

    if (!volume || !volume->zarrDataset() || !surf)
        return;

    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();

    if (!force && QRectF(curr_img_area).contains(bbox))
        return;


    curr_img_area = {static_cast<int>(bbox.left()),static_cast<int>(bbox.top()), static_cast<int>(bbox.width()), static_cast<int>(bbox.height())};

    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});

    QImage qimg = Mat2QImage(img);
    if (_overlayImageValid && !_overlayImage.isNull()) {
        qimg = qimg.convertToFormat(QImage::Format_RGBA8888);
        QPainter painter(&qimg);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
        painter.drawImage(0, 0, _overlayImage);
    }

    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);

    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
    }

    _center_marker->setParentItem(fBaseImageItem);

    fBaseImageItem->setOffset(curr_img_area.topLeft());
}


cv::Mat_<uint8_t> CVolumeViewer::render_composite(const cv::Rect &roi) {
    cv::Mat_<uint8_t> img;

    auto surf = _surf_weak.lock();
    if (!surf)
        return img;

    cv::Vec2f roi_c = {static_cast<float>(roi.x + roi.width / 2), static_cast<float>(roi.y + roi.height / 2)};
    cv::Vec3f ptr = surf->pointer();
    cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
    surf->move(ptr, diff / _scale);
    _ptr = ptr;
    _vis_center = roi_c;

    cv::Mat_<cv::Vec3f> base_coords;
    cv::Mat_<cv::Vec3f> normals;
    getCachedSurfaceCoords(base_coords, normals, roi, ptr, surf);

    cv::Mat_<cv::Vec3f> lightingNormals = getVolumeGradientNormals(normals, roi, surf);

    int z_start = _compositeSettings.reverseDirection ? -_compositeSettings.layersBehind : -_compositeSettings.layersFront;
    int z_end = _compositeSettings.reverseDirection ? _compositeSettings.layersFront : _compositeSettings.layersBehind;

    readCompositeFast(
        img,
        volume->zarrDataset(_ds_sd_idx),
        base_coords * _ds_scale,
        lightingNormals,
        _ds_scale,
        z_start, z_end,
        _compositeSettings.params,
        *cache
    );

    postprocessComposite(img, _compositeSettings);

    return img;
}

cv::Mat_<uint8_t> CVolumeViewer::renderCompositeForSurface(std::shared_ptr<QuadSurface> surface, cv::Size outputSize)
{
    if (!surface || !_compositeSettings.enabled || !volume) {
        return cv::Mat_<uint8_t>();
    }

    // Save current state
    float oldScale = _scale;
    cv::Vec2f oldVisCenter = _vis_center;
    auto oldSurf = _surf_weak.lock();
    float oldZOff = _z_off;
    cv::Vec3f oldPtr = _ptr;
    float oldDsScale = _ds_scale;
    int oldDsSdIdx = _ds_sd_idx;

    // Render at 1:1 with the surface's internal grid (raw points size)
    // Use surface's scale so that gen() computes sx = _scale/_scale = 1.0,
    // sampling 1:1 from the raw points grid
    cv::Size rawPointsSize = surface->rawPointsPtr()->size();
    float surfScale = surface->_scale[0];


    _surf_weak = surface;
    _scale = surfScale;  // Use surface's scale so gen() samples 1:1 from raw points
    _z_off = 0.0f;

    recalcScales();


    _ptr = surface->pointer();
    // Use raw points size for the ROI so we cover the whole surface
    cv::Rect roi(-rawPointsSize.width/2, -rawPointsSize.height/2,
                 rawPointsSize.width, rawPointsSize.height);

    _vis_center = cv::Vec2f(0, 0);

    cv::Mat_<uint8_t> result = render_composite(roi);

    // Resize to requested output size if different
    if (result.size() != outputSize) {
        cv::resize(result, result, outputSize, 0, 0, cv::INTER_LINEAR);
    }

    _surf_weak = oldSurf;
    _scale = oldScale;
    _vis_center = oldVisCenter;
    _z_off = oldZOff;
    _ptr = oldPtr;
    _ds_scale = oldDsScale;
    _ds_sd_idx = oldDsSdIdx;

    return result;
}


cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    auto surf = _surf_weak.lock();
    if (!surf)
        return cv::Mat();

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> baseGray;
    const int baseWindowLowInt = static_cast<int>(std::clamp(_baseWindowLow, 0.0f, 255.0f));
    const int baseWindowHighInt = static_cast<int>(
        std::clamp(_baseWindowHigh, static_cast<float>(baseWindowLowInt + 1), 255.0f));
    const float baseWindowSpan = std::max(1.0f, static_cast<float>(baseWindowHighInt - baseWindowLowInt));

    _overlayImageValid = false;
    _overlayImage = QImage();

    const bool useComposite = (_surf_name == "segmentation" && _compositeSettings.enabled &&
                               (_compositeSettings.layersFront > 0 || _compositeSettings.layersBehind > 0));

    cv::Mat baseColor;

    z5::Dataset* baseDataset = volume ? volume->zarrDataset(_ds_sd_idx) : nullptr;

    // Check if this is a plane surface that should use plane composite rendering
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
    const bool usePlaneComposite = (plane != nullptr && _compositeSettings.planeEnabled &&
                                    (_compositeSettings.planeLayersFront > 0 || _compositeSettings.planeLayersBehind > 0));

    if (useComposite) {
        baseGray = render_composite(roi);
    } else if (usePlaneComposite) {
        generateViewCoords(coords, roi, surf);
        baseGray = render_composite_plane(roi, coords, plane->normal(cv::Vec3f(0, 0, 0)));
    } else {
        generateViewCoords(coords, roi, surf);
        if (!baseDataset) {
            return cv::Mat();
        }
        readInterpolated3D(baseGray, baseDataset, coords * _ds_scale, cache, _useFastInterpolation);
    }

    if (baseGray.empty()) {
        return cv::Mat();
    }

    // Apply ISO cutoff - zero out values below threshold
    if (_compositeSettings.params.isoCutoff > 0) {
        cv::threshold(baseGray, baseGray, _compositeSettings.params.isoCutoff - 1, 0, cv::THRESH_TOZERO);
    }

    cv::Mat baseProcessed;

    // Apply stretching if enabled
    if (_stretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(baseGray, &minVal, &maxVal);
        const double range = std::max(1.0, maxVal - minVal);

        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(minVal);
        baseFloat /= static_cast<float>(range);
        baseFloat.convertTo(baseProcessed, CV_8U, 255.0f);
    } else {
        // Apply window/level transformation
        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(baseWindowLowInt);
        baseFloat /= baseWindowSpan;
        cv::max(baseFloat, 0.0f, baseFloat);
        cv::min(baseFloat, 1.0f, baseFloat);
        baseFloat.convertTo(baseProcessed, CV_8U, 255.0f);
    }

    // Apply colormap if specified
    if (!_baseColormapId.empty()) {
        const auto& spec = volume_viewer_cmaps::resolve(_baseColormapId);
        baseColor = volume_viewer_cmaps::makeColors(baseProcessed, spec);
    } else {
        // Convert to BGR
        if (baseProcessed.channels() == 1) {
            cv::cvtColor(baseProcessed, baseColor, cv::COLOR_GRAY2BGR);
        } else {
            baseColor = baseProcessed.clone();
        }
    }

    if (_overlayVolume && _overlayOpacity > 0.0f) {
        if (coords.empty()) {
            generateViewCoords(coords, roi, surf);
        }

        if (!coords.empty()) {
            int overlayIdx = 0;
            float overlayScale = 1.0f;
            if (_overlayVolume->numScales() > 0) {
                overlayIdx = std::min<int>(_ds_sd_idx, static_cast<int>(_overlayVolume->numScales()) - 1);
                overlayScale = std::pow(2.0f, -overlayIdx);
            }

            cv::Mat_<uint8_t> overlayValues;
            z5::Dataset* overlayDataset = _overlayVolume->zarrDataset(overlayIdx);
            readInterpolated3D(overlayValues, overlayDataset, coords * overlayScale, cache, /*nearest_neighbor=*/true);

            if (!overlayValues.empty()) {
                const int windowLow = static_cast<int>(std::clamp(_overlayWindowLow, 0.0f, 255.0f));
                const int windowHigh = static_cast<int>(std::clamp(_overlayWindowHigh, static_cast<float>(windowLow + 1), 255.0f));

                cv::Mat activeMask;
                cv::compare(overlayValues, windowLow, activeMask, cv::CmpTypes::CMP_GE);

                if (cv::countNonZero(activeMask) > 0) {
                    cv::Mat overlayScaled;
                    overlayValues.convertTo(overlayScaled, CV_32F);
                    overlayScaled -= static_cast<float>(windowLow);
                    overlayScaled.setTo(0.0f, overlayScaled < 0.0f);
                    const float windowSpan = std::max(1.0f, static_cast<float>(windowHigh - windowLow));
                    overlayScaled /= windowSpan;
                    cv::threshold(overlayScaled, overlayScaled, 1.0f, 1.0f, cv::THRESH_TRUNC);

                    cv::Mat overlayColorInput;
                    overlayScaled.convertTo(overlayColorInput, CV_8U, 255.0f);

                    const auto& spec = volume_viewer_cmaps::resolve(_overlayColormapId);
                    cv::Mat overlayColor = volume_viewer_cmaps::makeColors(overlayColorInput, spec);

                    if (!overlayColor.empty()) {
                        cv::Mat inactiveMask;
                        cv::bitwise_not(activeMask, inactiveMask);
                        overlayColor.setTo(cv::Scalar(0, 0, 0), inactiveMask);

                        cv::Mat overlayBGRA;
                        cv::cvtColor(overlayColor, overlayBGRA, cv::COLOR_BGR2BGRA);

                        std::vector<cv::Mat> channels;
                        cv::split(overlayBGRA, channels);
                        const uchar alphaValue = static_cast<uchar>(std::round(std::clamp(_overlayOpacity, 0.0f, 1.0f) * 255.0f));
                        channels[3].setTo(alphaValue, activeMask);
                        channels[3].setTo(0, inactiveMask);
                        cv::merge(channels, overlayBGRA);

                        cv::cvtColor(overlayBGRA, overlayBGRA, cv::COLOR_BGRA2RGBA);
                        QImage overlayImage(overlayBGRA.data, overlayBGRA.cols, overlayBGRA.rows, overlayBGRA.step, QImage::Format_RGBA8888);
                        _overlayImage = overlayImage.copy();
                        _overlayImageValid = true;
                    }
                }
            }
        }
    }

    // Surface overlap detection using SurfacePatchIndex (multi-surface with colors)
    // Only process in segmentation viewer - plane views don't need surface overlays
    if (_surf_name == "segmentation" && _surfaceOverlayEnabled && !_surfaceOverlays.empty() && _surf_col && !baseColor.empty()) {
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (patchIndex) {
            // Use subsampling for many surfaces (conservative 2x stride)
            const int stride = (_surfaceOverlays.size() > 50) ? 2 : 1;
            const int sampledRows = (coords.rows + stride - 1) / stride;
            const int sampledCols = (coords.cols + stride - 1) / stride;

            // Compute viewport bounding box for early culling (sparse sampling)
            cv::Vec3f viewMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            cv::Vec3f viewMax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
            for (int y = 0; y < coords.rows; y += 10) {
                for (int x = 0; x < coords.cols; x += 10) {
                    const cv::Vec3f& p = coords(y, x);
                    if (p[0] >= 0) {
                        for (int i = 0; i < 3; ++i) {
                            viewMin[i] = std::min(viewMin[i], p[i]);
                            viewMax[i] = std::max(viewMax[i], p[i]);
                        }
                    }
                }
            }
            // Expand by threshold for overlap detection
            viewMin -= cv::Vec3f(_surfaceOverlapThreshold, _surfaceOverlapThreshold, _surfaceOverlapThreshold);
            viewMax += cv::Vec3f(_surfaceOverlapThreshold, _surfaceOverlapThreshold, _surfaceOverlapThreshold);

            // Track overlay colors and counts at sampled resolution
            cv::Mat_<cv::Vec3f> overlayColorSum(sampledRows, sampledCols, cv::Vec3f(0, 0, 0));
            cv::Mat_<uint8_t> overlapCount(sampledRows, sampledCols, uint8_t(0));

            // Thread-local accumulators to avoid critical section
            const int num_threads = omp_get_max_threads();
            std::vector<cv::Mat_<cv::Vec3f>> thread_colorSum(num_threads);
            std::vector<cv::Mat_<uint8_t>> thread_count(num_threads);
            for (int t = 0; t < num_threads; ++t) {
                thread_colorSum[t] = cv::Mat_<cv::Vec3f>(sampledRows, sampledCols, cv::Vec3f(0, 0, 0));
                thread_count[t] = cv::Mat_<uint8_t>(sampledRows, sampledCols, uint8_t(0));
            }

            for (const auto& [overlayName, overlayColor] : _surfaceOverlays) {
                auto overlaySurf = _surf_col->surface(overlayName);
                auto overlayQuad = std::dynamic_pointer_cast<QuadSurface>(overlaySurf);
                if (!overlayQuad || overlaySurf == surf) {
                    continue;  // Skip self or invalid surfaces
                }

                // Bounding box culling: skip surfaces that can't intersect viewport
                // Use cached bbox - O(1) for surfaces loaded from disk
                const Rect3D surfBBox = overlayQuad->bbox();
                const cv::Vec3f& surfMin = surfBBox.low;
                const cv::Vec3f& surfMax = surfBBox.high;

                // Skip if bbox is invalid (uninitialized surface)
                if (surfMin[0] < 0) {
                    continue;
                }

                bool canIntersect = true;
                for (int i = 0; i < 3; ++i) {
                    if (surfMin[i] > viewMax[i] || surfMax[i] < viewMin[i]) {
                        canIntersect = false;
                        break;
                    }
                }
                if (!canIntersect) {
                    continue;
                }

                const cv::Vec3f colorVec(overlayColor[0], overlayColor[1], overlayColor[2]);

                // For each sampled base surface point, query distance to this overlay surface
                #pragma omp parallel for collapse(2)
                for (int sy = 0; sy < sampledRows; ++sy) {
                    for (int sx = 0; sx < sampledCols; ++sx) {
                        const int y = sy * stride;
                        const int x = sx * stride;
                        if (y >= coords.rows || x >= coords.cols) continue;

                        const cv::Vec3f& basePos = coords(y, x);
                        if (basePos[0] >= 0) {
                            auto result = patchIndex->locate(basePos, _surfaceOverlapThreshold, overlayQuad);
                            if (result && result->distance < _surfaceOverlapThreshold) {
                                const int tid = omp_get_thread_num();
                                thread_colorSum[tid](sy, sx) += colorVec;
                                thread_count[tid](sy, sx)++;
                            }
                        }
                    }
                }
            }

            // Merge thread-local accumulators
            for (int t = 0; t < num_threads; ++t) {
                overlayColorSum += thread_colorSum[t];
                for (int sy = 0; sy < sampledRows; ++sy) {
                    for (int sx = 0; sx < sampledCols; ++sx) {
                        overlapCount(sy, sx) += thread_count[t](sy, sx);
                    }
                }
            }

            // Blend averaged overlay colors where surfaces overlap
            const float blendFactor = 0.5f;
            for (int y = 0; y < baseColor.rows; ++y) {
                for (int x = 0; x < baseColor.cols; ++x) {
                    const int sy = y / stride;
                    const int sx = x / stride;
                    if (sy < sampledRows && sx < sampledCols && overlapCount(sy, sx) > 0) {
                        cv::Vec3b& pixel = baseColor.at<cv::Vec3b>(y, x);
                        cv::Vec3f avgColor = overlayColorSum(sy, sx) / static_cast<float>(overlapCount(sy, sx));
                        pixel = cv::Vec3b(
                            static_cast<uint8_t>(pixel[0] * (1.0f - blendFactor) + avgColor[0] * blendFactor),
                            static_cast<uint8_t>(pixel[1] * (1.0f - blendFactor) + avgColor[1] * blendFactor),
                            static_cast<uint8_t>(pixel[2] * (1.0f - blendFactor) + avgColor[2] * blendFactor)
                        );
                    }
                }
            }
        }
    }

    return baseColor;
}

void CVolumeViewer::setBaseColormap(const std::string& colormapId)
{
    if (_baseColormapId == colormapId) {
        return;
    }
    _baseColormapId = colormapId;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setStretchValues(bool enabled)
{
    if (_stretchValues == enabled) {
        return;
    }
    _stretchValues = enabled;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlayEnabled(bool enabled)
{
    if (_surfaceOverlayEnabled == enabled) {
        return;
    }
    _surfaceOverlayEnabled = enabled;
    if (volume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays)
{
    if (_surfaceOverlays == overlays) {
        return;
    }
    _surfaceOverlays = overlays;
    if (volume && _surfaceOverlayEnabled) {
        renderVisible(true);
    }
}

void CVolumeViewer::setSurfaceOverlapThreshold(float threshold)
{
    threshold = std::max(0.1f, threshold);
    if (std::abs(threshold - _surfaceOverlapThreshold) < 1e-6f) {
        return;
    }
    _surfaceOverlapThreshold = threshold;
    if (volume && _surfaceOverlayEnabled) {
        renderVisible(true);
    }
}

void CVolumeViewer::getCachedSurfaceCoords(
    cv::Mat_<cv::Vec3f>& base_coords, cv::Mat_<cv::Vec3f>& normals,
    const cv::Rect& roi, const cv::Vec3f& ptr, const std::shared_ptr<Surface>& surf)
{
    // Cache key: roi size, scale, ptr position, and surface instance
    // z_off is NOT part of the cache key — normals don't depend on z_off,
    // and we apply the z offset ourselves after retrieving cached values
    bool cacheValid = (!_cachedNormals.empty() &&
                       _cachedNormalsSize == roi.size() &&
                       std::abs(_cachedNormalsScale - _scale) < 1e-6f &&
                       cv::norm(_cachedNormalsPtr - ptr) < 1e-6f &&
                       _cachedNormalsSurf.lock() == surf);

    if (cacheValid) {
        if (std::abs(_z_off - _cachedNormalsZOff) < 1e-6f) {
            base_coords = _cachedBaseCoords;
        } else {
            const int h = _cachedBaseCoords.rows;
            const int w = _cachedBaseCoords.cols;
            if (_coordsWorkBuffer.rows != h || _coordsWorkBuffer.cols != w) {
                _coordsWorkBuffer.create(h, w);
            }
            const float z_delta = _z_off - _cachedNormalsZOff;
            #pragma omp parallel for collapse(2)
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    const cv::Vec3f& src = _cachedBaseCoords(j, i);
                    const cv::Vec3f& n = _cachedNormals(j, i);
                    if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                        _coordsWorkBuffer(j, i) = src + n * z_delta;
                    } else {
                        _coordsWorkBuffer(j, i) = src;
                    }
                }
            }
            base_coords = _coordsWorkBuffer;
        }
        normals = _cachedNormals;
    } else {
        surf->gen(&base_coords, &normals, roi.size(), ptr, _scale,
                  {static_cast<float>(-roi.width / 2), static_cast<float>(-roi.height / 2), _z_off});
        _cachedBaseCoords = base_coords;
        _cachedNormals = normals;
        _cachedNormalsSize = roi.size();
        _cachedNormalsScale = _scale;
        _cachedNormalsPtr = ptr;
        _cachedNormalsZOff = _z_off;
        _cachedNormalsSurf = surf;
    }
}

cv::Mat_<cv::Vec3f> CVolumeViewer::getVolumeGradientNormals(
    const cv::Mat_<cv::Vec3f>& meshNormals, const cv::Rect& roi,
    const std::shared_ptr<Surface>& surf)
{
    if (!_compositeSettings.useVolumeGradients || !_compositeSettings.params.lightingEnabled) {
        return meshNormals;
    }

    auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
    if (!quadSurf) {
        return meshNormals;
    }

    // Compute native gradients once per surface
    if (_cachedNativeVolumeGradients.empty() || _cachedGradientsSurf.lock() != surf) {
        const cv::Mat_<cv::Vec3f>* rawPts = quadSurf->rawPointsPtr();
        _cachedNativeVolumeGradients = computeVolumeGradientsNative(
            volume->zarrDataset(_ds_sd_idx), *rawPts, _ds_scale);
        _cachedGradientsSurf = surf;
    }

    // Warp native gradients to view coords using same transform as gen()
    const cv::Vec2f surfScale = quadSurf->scale();
    const cv::Vec3f center = quadSurf->center();
    const cv::Vec3f offset = {static_cast<float>(-roi.width / 2), static_cast<float>(-roi.height / 2), _z_off};
    const cv::Vec3f nominalOffset = offset / _scale + center;
    const cv::Vec3f ul = _ptr + cv::Vec3f(nominalOffset[0] * surfScale[0], nominalOffset[1] * surfScale[1], nominalOffset[2]);

    const double sx = static_cast<double>(surfScale[0]) / static_cast<double>(_scale);
    const double sy = static_cast<double>(surfScale[1]) / static_cast<double>(_scale);
    const double ox = static_cast<double>(ul[0]);
    const double oy = static_cast<double>(ul[1]);

    std::array<cv::Point2f, 3> srcf = {
        cv::Point2f(static_cast<float>(ox), static_cast<float>(oy)),
        cv::Point2f(static_cast<float>(ox + roi.width * sx), static_cast<float>(oy)),
        cv::Point2f(static_cast<float>(ox), static_cast<float>(oy + roi.height * sy))
    };
    std::array<cv::Point2f, 3> dstf = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(static_cast<float>(roi.width), 0.f),
        cv::Point2f(0.f, static_cast<float>(roi.height))
    };

    cv::Mat_<cv::Vec3f> lightingNormals;
    cv::Mat A = cv::getAffineTransform(srcf.data(), dstf.data());
    cv::warpAffine(_cachedNativeVolumeGradients, lightingNormals, A, roi.size(),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return lightingNormals;
}

void CVolumeViewer::generateViewCoords(
    cv::Mat_<cv::Vec3f>& coords, const cv::Rect& roi,
    const std::shared_ptr<Surface>& surf)
{
    if (dynamic_cast<PlaneSurface*>(surf.get())) {
        surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale,
                  {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});
    } else {
        cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
        _ptr = surf->pointer();
        cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
        surf->move(_ptr, diff / _scale);
        _vis_center = roi_c;
        surf->gen(&coords, nullptr, roi.size(), _ptr, _scale,
                  {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
    }
}

cv::Mat_<uint8_t> CVolumeViewer::render_composite_plane(const cv::Rect &roi, const cv::Mat_<cv::Vec3f> &coords, const cv::Vec3f &planeNormal)
{
    cv::Mat_<uint8_t> img;

    if (coords.empty() || !volume || !volume->zarrDataset(_ds_sd_idx)) {
        return img;
    }

    // Determine z range based on front and behind layers
    // For planes, "front" means along the positive normal direction
    int z_start = _compositeSettings.reverseDirection ? -_compositeSettings.planeLayersBehind : -_compositeSettings.planeLayersFront;
    int z_end = _compositeSettings.reverseDirection ? _compositeSettings.planeLayersFront : _compositeSettings.planeLayersBehind;

    // Build a constant-normal mat and use the standard composite path
    cv::Mat_<cv::Vec3f> normals(coords.size(), planeNormal);

    readCompositeFast(
        img,
        volume->zarrDataset(_ds_sd_idx),
        coords * _ds_scale,
        normals,
        _ds_scale,
        z_start, z_end,
        _compositeSettings.params,
        *cache
    );

    return img;
}