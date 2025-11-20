#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
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
#include "vc/core/util/SliceCache.hpp"

#include <omp.h>


#include <QPainter>
#include <optional>

#include <opencv2/imgproc.hpp>

#define COLOR_FOCUS QColor(50, 255, 215)


namespace
{
    constexpr float kScaleQuantization = 1000.0f;
    constexpr float kZOffsetQuantization = 1000.0f;
    constexpr float kDsScaleQuantization = 1000.0f;

    int quantizeFloat(float value, float multiplier)
    {
        return static_cast<int>(std::lround(value * multiplier));
    }

    bool planeIdForSurface(const std::string& name, uint8_t& outId)
    {
        if (name == "seg xz") {
            outId = 0;
            return true;
        }
        if (name == "seg yz") {
            outId = 1;
            return true;
        }
        return false;
    }


} // namespace

void CVolumeViewer::renderVisible(bool force)
{
    if (_surf && _surf_col) {
        Surface* currentSurface = _surf_col->surface(_surf_name);
        if (!currentSurface) {
            // Surface was cleared (e.g. during volume reload) without a change signal
            // reaching this viewer yet; drop the dangling pointer before rendering.
            _surf = nullptr;
        }
    }

    if (!volume || !volume->zarrDataset() || !_surf)
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

    // Composite rendering for segmentation view
    cv::Mat_<float> accumulator;
    int count = 0;

    // Alpha composition state for each pixel
    cv::Mat_<float> alpha_accumulator;
    cv::Mat_<float> value_accumulator;

    // Alpha composition parameters using the new settings
    const float alpha_min = _composite_alpha_min / 255.0f;
    const float alpha_max = _composite_alpha_max / 255.0f;
    const float alpha_opacity = _composite_material / 255.0f;
    const float alpha_cutoff = _composite_alpha_threshold / 10000.0f;

    // Determine the z range based on front and behind layers
    int z_start = _composite_reverse_direction ? -_composite_layers_behind : -_composite_layers_front;
    int z_end = _composite_reverse_direction ? _composite_layers_front : _composite_layers_behind;

    for (int z = z_start; z <= z_end; z++) {
        cv::Mat_<cv::Vec3f> slice_coords;
        cv::Mat_<uint8_t> slice_img;

        cv::Vec2f roi_c = {static_cast<float>(roi.x+roi.width/2), static_cast<float>(roi.y + roi.height/2)};
        _ptr = _surf->pointer();
        cv::Vec3f diff = {roi_c[0],roi_c[1],0};
        _surf->move(_ptr, diff/_scale);
        _vis_center = roi_c;
        float z_step = z * _ds_scale;  // Scale the step to maintain consistent physical distance
        _surf->gen(&slice_coords, nullptr, roi.size(), _ptr, _scale, {static_cast<float>(-roi.width/2), static_cast<float>(-roi.height/2), _z_off + z_step});

        readInterpolated3D(slice_img, volume->zarrDataset(_ds_sd_idx), slice_coords*_ds_scale, cache, _useFastInterpolation);

        // Convert to float for accumulation
        cv::Mat_<float> slice_float;
        slice_img.convertTo(slice_float, CV_32F);

        if (_composite_method == "alpha") {
            // Alpha composition algorithm
            if (alpha_accumulator.empty()) {
                alpha_accumulator = cv::Mat_<float>::zeros(slice_float.size());
                value_accumulator = cv::Mat_<float>::zeros(slice_float.size());
            }

            // Process each pixel
            for (int y = 0; y < slice_float.rows; y++) {
                for (int x = 0; x < slice_float.cols; x++) {
                    float pixel_value = slice_float(y, x);

                    // Normalize pixel value
                    float normalized_value = (pixel_value / 255.0f - alpha_min) / (alpha_max - alpha_min);
                    normalized_value = std::max(0.0f, std::min(1.0f, normalized_value)); // Clamp to [0,1]

                    // Skip empty areas (speed through)
                    if (normalized_value == 0.0f) {
                        continue;
                    }

                    float current_alpha = alpha_accumulator(y, x);

                    // Check alpha cutoff for early termination
                    if (current_alpha >= alpha_cutoff) {
                        continue;
                    }

                    // Calculate weight
                    float weight = (1.0f - current_alpha) * std::min(normalized_value * alpha_opacity, 1.0f);

                    // Accumulate
                    value_accumulator(y, x) += weight * normalized_value;
                    alpha_accumulator(y, x) += weight;
                }
            }
        } else {
            // Original composite methods
            if (accumulator.empty()) {
                accumulator = slice_float;
                if (_composite_method == "min") {
                    accumulator.setTo(255.0); // Initialize to max value for min operation
                    accumulator = cv::min(accumulator, slice_float);
                }
            } else {
                if (_composite_method == "max") {
                    accumulator = cv::max(accumulator, slice_float);
                } else if (_composite_method == "mean") {
                    accumulator += slice_float;
                    count++;
                } else if (_composite_method == "min") {
                    accumulator = cv::min(accumulator, slice_float);
                }
            }
        }
    }

    // Finalize alpha composition result
    if (_composite_method == "alpha") {
        accumulator = cv::Mat_<float>::zeros(value_accumulator.size());
        for (int y = 0; y < value_accumulator.rows; y++) {
            for (int x = 0; x < value_accumulator.cols; x++) {
                float final_value = value_accumulator(y, x) * 255.0f;
                accumulator(y, x) = std::max(0.0f, std::min(255.0f, final_value)); // Clamp to [0,255]
            }
        }
    }

    // Convert back to uint8
    if (_composite_method == "mean" && count > 0) {
        accumulator /= count;
    }
    accumulator.convertTo(img, CV_8U);
    return img;
}

cv::Mat_<uint8_t> CVolumeViewer::renderCompositeForSurface(QuadSurface* surface, cv::Size outputSize)
{
    if (!surface || !_composite_enabled || !volume) {
        return cv::Mat_<uint8_t>();
    }

    // Save current state
    float oldScale = _scale;
    cv::Vec2f oldVisCenter = _vis_center;
    Surface* oldSurf = _surf;
    float oldZOff = _z_off;
    cv::Vec3f oldPtr = _ptr;
    float oldDsScale = _ds_scale;
    int oldDsSdIdx = _ds_sd_idx;

    // Set up for surface rendering at 1:1 scale
    _surf = surface;
    _scale = 1.0f;
    _z_off = 0.0f;

    recalcScales();
    _ptr = _surf->pointer();
    cv::Rect roi(-outputSize.width/2, -outputSize.height/2,
                 outputSize.width, outputSize.height);

    _vis_center = cv::Vec2f(0, 0);

    cv::Mat_<uint8_t> result = render_composite(roi);

    _surf = oldSurf;
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
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> baseGray;
    const int baseWindowLowInt = static_cast<int>(std::clamp(_baseWindowLow, 0.0f, 255.0f));
    const int baseWindowHighInt = static_cast<int>(
        std::clamp(_baseWindowHigh, static_cast<float>(baseWindowLowInt + 1), 255.0f));
    const float baseWindowSpan = std::max(1.0f, static_cast<float>(baseWindowHighInt - baseWindowLowInt));

    _overlayImageValid = false;
    _overlayImage = QImage();

    const QRect roiRect(roi.x, roi.y, roi.width, roi.height);

    const bool useComposite = (_surf_name == "segmentation" && _composite_enabled &&
                               (_composite_layers_front > 0 || _composite_layers_behind > 0));

    cv::Mat baseColor;
    bool usedCache = false;
    AxisAlignedSliceCacheKey cacheKey{};
    bool cacheKeyValid = false;

    z5::Dataset* baseDataset = volume ? volume->zarrDataset(_ds_sd_idx) : nullptr;

    if (useComposite) {
        baseGray = render_composite(roi);
    } else {
        if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
            _surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});

            uint8_t planeId = 0;
            if (plane->axisAlignedRotationKey() >= 0 && cache && baseDataset &&
                planeIdForSurface(_surf_name, planeId)) {
                cacheKey.planeId = planeId;
                cacheKey.rotationKey = static_cast<uint16_t>(plane->axisAlignedRotationKey());
                const cv::Vec3f origin = plane->origin();
                cacheKey.originX = static_cast<int>(std::lround(origin[0]));
                cacheKey.originY = static_cast<int>(std::lround(origin[1]));
                cacheKey.originZ = static_cast<int>(std::lround(origin[2]));
                cacheKey.roiX = roi.x;
                cacheKey.roiY = roi.y;
                cacheKey.roiWidth = roi.width;
                cacheKey.roiHeight = roi.height;
                cacheKey.scaleMilli = quantizeFloat(_scale, kScaleQuantization);
                cacheKey.dsScaleMilli = quantizeFloat(_ds_scale, kDsScaleQuantization);
                cacheKey.zOffsetMilli = quantizeFloat(_z_off, kZOffsetQuantization);
                cacheKey.dsIndex = _ds_sd_idx;
                cacheKey.datasetPtr = reinterpret_cast<uintptr_t>(baseDataset);
                cacheKey.fastInterpolation = _useFastInterpolation ? 1 : 0;
                cacheKey.baseWindowLow = static_cast<uint8_t>(baseWindowLowInt);
                cacheKey.baseWindowHigh = static_cast<uint8_t>(baseWindowHighInt);
                cacheKeyValid = true;

                if (auto cached = axisAlignedSliceCache().get(cacheKey)) {
                    baseColor = *cached;
                    usedCache = !baseColor.empty();
                }
            }
        } else {
            cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
            _ptr = _surf->pointer();
            cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
            _surf->move(_ptr, diff / _scale);
            _vis_center = roi_c;
            _surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
        }

        if (!usedCache) {
            if (!baseDataset) {
                return cv::Mat();
            }
            readInterpolated3D(baseGray, baseDataset, coords * _ds_scale, cache, _useFastInterpolation);
        }
    }

    if (!usedCache && baseGray.empty()) {
        return cv::Mat();
    }

    if (!usedCache) {
        cv::Mat baseFloat;
        baseGray.convertTo(baseFloat, CV_32F);
        baseFloat -= static_cast<float>(baseWindowLowInt);
        baseFloat /= baseWindowSpan;
        cv::max(baseFloat, 0.0f, baseFloat);
        cv::min(baseFloat, 1.0f, baseFloat);
        baseFloat.convertTo(baseGray, CV_8U, 255.0f);

        if (baseGray.channels() == 1) {
            cv::cvtColor(baseGray, baseColor, cv::COLOR_GRAY2BGR);
        } else {
            baseColor = baseGray.clone();
        }

        if (cacheKeyValid && !baseColor.empty()) {
            axisAlignedSliceCache().put(cacheKey, baseColor);
        }
    }

    if (_overlayVolume && _overlayOpacity > 0.0f) {
        if (coords.empty()) {
            if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
                _surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0, 0, 0), _scale, {static_cast<float>(roi.x), static_cast<float>(roi.y), _z_off});
            } else {
                cv::Vec2f roi_c = {roi.x + roi.width / 2.0f, roi.y + roi.height / 2.0f};
                _ptr = _surf->pointer();
                cv::Vec3f diff = {roi_c[0], roi_c[1], 0};
                _surf->move(_ptr, diff / _scale);
                _vis_center = roi_c;
                _surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width / 2.0f, -roi.height / 2.0f, _z_off});
            }
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

    return baseColor;
}