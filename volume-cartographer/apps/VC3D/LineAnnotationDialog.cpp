#include "LineAnnotationDialog.hpp"

#include "FiberNameDisplay.hpp"
#include "FiberSliceGeometry.hpp"
#include "Keybinds.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QAbstractItemView>
#include <QBrush>
#include <QCloseEvent>
#include <QComboBox>
#include <QEvent>
#include <QFont>
#include <QGraphicsPathItem>
#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPainterPath>
#include <QPen>
#include <QProgressBar>
#include <QPushButton>
#include <QRect>
#include <QResizeEvent>
#include <QSettings>
#include <QShortcut>
#include <QSizePolicy>
#include <QSplitter>
#include <QSpinBox>
#include <QVariant>
#include <QTimer>
#include <QVariantAnimation>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {

// Half-width (in line-point indices) of the window used to least-squares-fit the side-view
// plane orientation around the cursor. The fit is interpolated between adjacent window centers
// so the orientation tracks the cursor continuously (see updateSidePlaneSurface).
constexpr int kSideFitHalfWindow = 20;
constexpr float kCurrentCutRotationStepRadians = 3.14159265358979323846f / 36.0f;
constexpr bool kGeneratedLineAnnotationOverlaysEnabled = true;
constexpr char kGeneratedDynamicCurrentCutOverlayKey[] = "line-z-slice-current";
constexpr float kNominalGeneratedRowWidth = 900.0f;
constexpr float kNominalGeneratedRowHeight = 260.0f;
constexpr double kSpanMetricHighlightThresholdDegrees = 45.0;
constexpr double kNormalOffsetEpsilon = 1.0e-6;

bool normalOffsetActive(double offsetVx)
{
    return std::abs(offsetVx) > kNormalOffsetEpsilon;
}

std::string staticStripOverlayKey(const std::string& surfaceName)
{
    return surfaceName + "_static";
}

std::string dynamicStripOverlayKey(const std::string& surfaceName)
{
    return surfaceName + "_dynamic";
}

CChunkedVolumeViewer::CameraState generatedPaneCamera(CChunkedVolumeViewer* viewer,
                                                      const CChunkedVolumeViewer::CameraState& fallback)
{
    CChunkedVolumeViewer::CameraState camera = fallback;
    camera.surfacePtrX = 0.0f;
    camera.surfacePtrY = 0.0f;
    camera.zOffset = 0.0f;
    camera.zOffsetWorldDir = {0, 0, 0};

    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad) {
        return camera;
    }

    const cv::Size size = quad->size();
    if (size.width <= 0 || size.height <= 0) {
        return camera;
    }

    constexpr float kPadding = 0.85f;
    const float scaleX = kNominalGeneratedRowWidth / static_cast<float>(std::max(1, size.width));
    const float scaleY = kNominalGeneratedRowHeight / static_cast<float>(std::max(1, size.height));
    camera.scale = std::clamp(std::min(scaleX, scaleY) * kPadding, 0.5f, 16.0f);
    return camera;
}

std::optional<cv::Vec2f> generatedStripSurfaceCenter(CChunkedVolumeViewer* viewer,
                                                     double linePosition)
{
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad || !std::isfinite(linePosition)) {
        return std::nullopt;
    }
    const auto* points = quad->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return std::nullopt;
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return cv::Vec2f{surfaceX, surfaceY};
}

std::optional<float> generatedStripScaleForLinePositionRange(
    CChunkedVolumeViewer* viewer,
    const std::optional<std::pair<double, double>>& range)
{
    if (!range) {
        return std::nullopt;
    }
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad || !std::isfinite(range->first) || !std::isfinite(range->second)) {
        return std::nullopt;
    }
    const cv::Vec2f scale = quad->scale();
    const double lineSpan = std::abs(range->second - range->first);
    if (!std::isfinite(lineSpan) || lineSpan <= 1.0e-6 || scale[0] == 0.0f) {
        return std::nullopt;
    }
    const double surfaceSpan = lineSpan / std::abs(static_cast<double>(scale[0]));
    if (!std::isfinite(surfaceSpan) || surfaceSpan <= 1.0e-6) {
        return std::nullopt;
    }
    constexpr double kViewportFill = 0.82;
    const double focusedScale =
        (static_cast<double>(kNominalGeneratedRowWidth) * kViewportFill) / surfaceSpan;
    if (!std::isfinite(focusedScale)) {
        return std::nullopt;
    }
    return static_cast<float>(std::clamp(focusedScale, 0.5, 64.0));
}

bool finitePoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

bool shouldShowSpanAlignmentMetric(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    return metric.pending ||
           !metric.error.empty() ||
           (metric.available && std::isfinite(metric.maxErrorDegrees));
}

bool shouldHighlightSpanAlignmentMetric(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    return metric.available &&
           std::isfinite(metric.maxErrorDegrees) &&
           metric.maxErrorDegrees > kSpanMetricHighlightThresholdDegrees;
}

QString spanAlignmentMetricText(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    if (metric.pending) {
        return QStringLiteral("...");
    }
    if (!metric.error.empty()) {
        return QStringLiteral("err");
    }
    if (!metric.available || !std::isfinite(metric.maxErrorDegrees)) {
        return {};
    }
    return QStringLiteral("%1%2")
        .arg(QString::number(std::llround(metric.maxErrorDegrees)))
        .arg(QChar(0x00b0));
}

QString spanAlignmentMetricToolTip(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    if (metric.pending) {
        return QObject::tr("Sampling Lasagna normals.");
    }
    if (!metric.error.empty()) {
        return QString::fromStdString(metric.error);
    }
    if (metric.available && std::isfinite(metric.maxErrorDegrees)) {
        return QObject::tr("Max normal-alignment error: %1 degrees")
            .arg(QString::number(metric.maxErrorDegrees, 'f', 1));
    }
    return {};
}

void installComboEventFilter(QComboBox* combo, QObject* filter)
{
    if (!combo || !filter) {
        return;
    }
    combo->installEventFilter(filter);
    if (auto* popupView = combo->view()) {
        popupView->installEventFilter(filter);
    }
}

QVariantList splitterSizesToVariantList(const QList<int>& sizes)
{
    QVariantList values;
    values.reserve(sizes.size());
    for (const int size : sizes) {
        values.push_back(size);
    }
    return values;
}

QList<int> splitterSizesFromVariant(const QVariant& value)
{
    QList<int> sizes;
    const QVariantList values = value.toList();
    sizes.reserve(values.size());
    for (const QVariant& entry : values) {
        bool ok = false;
        const int size = entry.toInt(&ok);
        if (!ok || size < 0) {
            return {};
        }
        sizes.push_back(size);
    }
    return sizes;
}

bool finiteZoom(float zoom)
{
    return std::isfinite(zoom) && zoom > 0.0f;
}

std::optional<float> zoomFromVariant(const QVariant& value)
{
    bool ok = false;
    const float zoom = value.toFloat(&ok);
    if (!ok || !finiteZoom(zoom)) {
        return std::nullopt;
    }
    return zoom;
}

QVariantList zoomsToVariantList(const std::vector<float>& zooms)
{
    QVariantList values;
    values.reserve(static_cast<int>(zooms.size()));
    for (const float zoom : zooms) {
        if (finiteZoom(zoom)) {
            values.push_back(zoom);
        }
    }
    return values;
}

std::vector<float> zoomsFromVariant(const QVariant& value)
{
    std::vector<float> zooms;
    const QVariantList values = value.toList();
    zooms.reserve(static_cast<size_t>(values.size()));
    for (const QVariant& entry : values) {
        if (const auto zoom = zoomFromVariant(entry)) {
            zooms.push_back(*zoom);
        }
    }
    return zooms;
}

cv::Vec3f normalizedOrNan(const cv::Vec3f& vector)
{
    const float n = cv::norm(vector);
    if (!finitePoint(vector) || n <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return vector * (1.0f / n);
}

} // namespace

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager,
                                           VolumeSelectorFactory volumeSelectorFactory,
                                           QWidget* parent)
    : QMainWindow(parent)
    , _viewerManager(viewerManager)
{
    setWindowTitle(tr("Line Annotation"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 700);

    auto* content = new QWidget(this);
    setCentralWidget(content);

    _layout = new QVBoxLayout(content);
    _layout->setContentsMargins(0, 0, 0, 0);
    _layout->setSpacing(0);

    auto* buttonRow = new QWidget(content);
    buttonRow->installEventFilter(this);
    auto* buttonLayout = new QHBoxLayout(buttonRow);
    buttonLayout->setContentsMargins(6, 6, 6, 6);
    buttonLayout->setSpacing(6);
    _initialDirectionCombo = new QComboBox(buttonRow);
    _initialDirectionCombo->addItem(tr("sideways"), static_cast<int>(InitialDirectionMode::Sideways));
    _initialDirectionCombo->addItem(tr("z (in/out)"), static_cast<int>(InitialDirectionMode::ZInOut));
    _initialDirectionCombo->setCurrentIndex(1);
    installComboEventFilter(_initialDirectionCombo, this);
    buttonLayout->addWidget(_initialDirectionCombo);
    _reoptimizationCombo = new QComboBox(buttonRow);
    _reoptimizationCombo->addItem(tr("auto-reopt"),
                                  static_cast<int>(ReoptimizationMode::AutoReoptimize));
    _reoptimizationCombo->addItem(tr("no optimization"),
                                  static_cast<int>(ReoptimizationMode::NoOptimization));
    _reoptimizationCombo->setCurrentIndex(0);
    installComboEventFilter(_reoptimizationCombo, this);
    buttonLayout->addWidget(_reoptimizationCombo);
    connect(_reoptimizationCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this](int) {
                emit reoptimizationModeChanged(reoptimizationMode());
            });
    _shiftScrollCombo = new QComboBox(buttonRow);
    _shiftScrollCombo->addItem(tr("along-line"),
                               static_cast<int>(ShiftScrollMode::AlongLine));
    _shiftScrollCombo->addItem(tr("straight"),
                               static_cast<int>(ShiftScrollMode::StraightNormal));
    _shiftScrollCombo->setCurrentIndex(0);
    installComboEventFilter(_shiftScrollCombo, this);
    buttonLayout->addWidget(_shiftScrollCombo);
    connect(_shiftScrollCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this](int) {
                handleShiftScrollModeChanged();
            });
    _initialCenterlineLengthSpin = new QSpinBox(buttonRow);
    _initialCenterlineLengthSpin->setObjectName(
        QStringLiteral("lineAnnotationInitialCenterlineLengthSpinBox"));
    _initialCenterlineLengthSpin->setRange(100, 1000000);
    _initialCenterlineLengthSpin->setSingleStep(100);
    _initialCenterlineLengthSpin->setPrefix(tr("Length "));
    _initialCenterlineLengthSpin->setSuffix(tr(" vx"));
    _initialCenterlineLengthSpin->setToolTip(
        tr("Total length of a newly generated centerline, split equally around the seed."));
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _initialCenterlineLengthSpin->setValue(
            settings.value(vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX,
                           vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT)
                .toInt());
    }
    _initialCenterlineLengthSpin->installEventFilter(this);
    buttonLayout->addWidget(_initialCenterlineLengthSpin);
    connect(_initialCenterlineLengthSpin,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            [](int value) {
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(
                    vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX, value);
            });
    auto* maxDistanceLabel = new QLabel(tr("Max CP dist"), buttonRow);
    maxDistanceLabel->installEventFilter(this);
    buttonLayout->addWidget(maxDistanceLabel);
    _maxControlPointDistanceSpin = new QSpinBox(buttonRow);
    _maxControlPointDistanceSpin->setObjectName(QStringLiteral("lineAnnotationMaxControlDistanceSpinBox"));
    _maxControlPointDistanceSpin->setRange(0, 1000000);
    _maxControlPointDistanceSpin->setValue(0);
    _maxControlPointDistanceSpin->setSuffix(tr(" vx"));
    _maxControlPointDistanceSpin->setSpecialValueText(tr("unlimited"));
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _maxControlPointDistanceSpin->setValue(
            settings.value(vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX,
                           vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX_DEFAULT)
                .toInt());
    }
    _maxControlPointDistanceSpin->installEventFilter(this);
    buttonLayout->addWidget(_maxControlPointDistanceSpin);
    connect(_maxControlPointDistanceSpin,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            [this](int value) {
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX,
                                  value);
                updateGeneratedDynamicOverlaysFast(false, false);
            });
    if (volumeSelectorFactory) {
        if (auto* volumeSelector = volumeSelectorFactory(buttonRow)) {
            volumeSelector->installEventFilter(this);
            buttonLayout->addWidget(volumeSelector);
        }
    }
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setText(tr("Z sens: %1").arg(_viewerManager ? _viewerManager->zScrollSensitivity() : 1.0, 0, 'f', 1));
    _sliceStepLabel->setToolTip(tr("Z-scroll sensitivity. Use Shift+G / Shift+H to adjust."));
    _sliceStepLabel->installEventFilter(this);
    buttonLayout->addWidget(_sliceStepLabel);
    if (_viewerManager) {
        connect(_viewerManager, &ViewerManager::zScrollSensitivityChanged, this, [this](double sensitivity) {
            if (_sliceStepLabel) {
                _sliceStepLabel->setText(tr("Z sens: %1").arg(sensitivity, 0, 'f', 1));
            }
        });
    }
    _optimizationStatusLabel = new QLabel(tr("not optimized"), buttonRow);
    _optimizationStatusLabel->installEventFilter(this);
    buttonLayout->addWidget(_optimizationStatusLabel);
    _sideStripIntersectionProgress = new QProgressBar(buttonRow);
    _sideStripIntersectionProgress->setObjectName(QStringLiteral("lineAnnotationSideStripIntersectionProgress"));
    _sideStripIntersectionProgress->setRange(0, 1);
    _sideStripIntersectionProgress->setValue(0);
    _sideStripIntersectionProgress->setTextVisible(true);
    _sideStripIntersectionProgress->setFormat(tr("strip intersections: 0"));
    _sideStripIntersectionProgress->setMinimumWidth(260);
    _sideStripIntersectionProgress->setVisible(false);
    _sideStripIntersectionProgress->installEventFilter(this);
    buttonLayout->addWidget(_sideStripIntersectionProgress);
    _fiberNameLabel = new QLabel(buttonRow);
    _fiberNameLabel->setObjectName(QStringLiteral("lineAnnotationFiberNameLabel"));
    _fiberNameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    _fiberNameLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _fiberNameLabel->setMinimumWidth(0);
    _fiberNameLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    _fiberNameLabel->installEventFilter(this);
    _showAsMeshButton = new QPushButton(tr("show as mesh"), buttonRow);
    _showAsMeshButton->setEnabled(false);
    _showAsMeshButton->installEventFilter(this);
    buttonLayout->addWidget(_showAsMeshButton);
    _fullOptimizationButton = new QPushButton(tr("reinit reopt"), buttonRow);
    _fullOptimizationButton->setEnabled(false);
    _fullOptimizationButton->installEventFilter(this);
    buttonLayout->addWidget(_fullOptimizationButton);
    _resetViewsButton = new QPushButton(tr("Reset views"), buttonRow);
    _resetViewsButton->setEnabled(false);
    _resetViewsButton->installEventFilter(this);
    buttonLayout->addWidget(_resetViewsButton);
    buttonLayout->addWidget(_fiberNameLabel, 1);
    _layout->addWidget(buttonRow, 0);
    connect(_showAsMeshButton, &QPushButton::clicked, this, [this]() {
        emit showAsMeshRequested();
    });
    connect(_fullOptimizationButton, &QPushButton::clicked, this, [this]() {
        emit fullOptimizationRequested();
    });
    connect(_resetViewsButton, &QPushButton::clicked, this, [this]() {
        resetGeneratedViews();
    });
    installGeneratedViewShortcuts();

    _mdiArea = new QMdiArea(content);
    _mdiArea->installEventFilter(this);
    _layout->addWidget(_mdiArea);

    restoreWindowGeometry();
    restoreGeneratedViewStateSettings();
}

void LineAnnotationDialog::showWithSavedGeometry()
{
    if (_workspaceEmbedded) {
        show();
        return;
    }
    if (_restoredWindowGeometry) {
        show();
    } else {
        showMaximized();
    }
}

LineAnnotationDialog::InitialDirectionMode LineAnnotationDialog::initialDirectionMode() const
{
    if (!_initialDirectionCombo) {
        return InitialDirectionMode::Sideways;
    }
    return static_cast<InitialDirectionMode>(_initialDirectionCombo->currentData().toInt());
}

LineAnnotationDialog::ReoptimizationMode LineAnnotationDialog::reoptimizationMode() const
{
    if (!_reoptimizationCombo) {
        return ReoptimizationMode::AutoReoptimize;
    }
    return static_cast<ReoptimizationMode>(_reoptimizationCombo->currentData().toInt());
}

int LineAnnotationDialog::initialCenterlineLengthVx() const
{
    return _initialCenterlineLengthSpin
        ? _initialCenterlineLengthSpin->value()
        : vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT;
}

LineAnnotationDialog::ShiftScrollMode LineAnnotationDialog::shiftScrollMode() const
{
    if (!_shiftScrollCombo) {
        return ShiftScrollMode::AlongLine;
    }
    return static_cast<ShiftScrollMode>(_shiftScrollCombo->currentData().toInt());
}

int LineAnnotationDialog::maxControlPointDistanceVx() const
{
    return _maxControlPointDistanceSpin ? _maxControlPointDistanceSpin->value() : 0;
}

void LineAnnotationDialog::setGeneratedControlPoints(
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.controlPoints = std::move(controlPoints);
    _generatedViews.spanAlignmentMetrics.clear();
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setGeneratedBranchLinePoints(
    std::vector<std::vector<cv::Vec3f>> branchLinePoints)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.branchLinePoints = std::move(branchLinePoints);
    _generatedViews.fiberIntersections.clear();
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setGeneratedBranchLinks(
    std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.branchLinks = std::move(branchLinks);
    _generatedViews.fiberIntersections.clear();
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setGeneratedBranchOverlayData(
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints,
    std::vector<std::vector<cv::Vec3f>> branchLinePoints,
    std::vector<GeneratedOverlay::BranchLinkMarker> branchLinks,
    bool requestSideStripIntersections)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.controlPoints = std::move(controlPoints);
    _generatedViews.branchLinePoints = std::move(branchLinePoints);
    _generatedViews.branchLinks = std::move(branchLinks);
    _generatedViews.fiberIntersections.clear();
    _generatedViews.spanAlignmentMetrics.clear();
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    rebuildGeneratedOverlays(requestSideStripIntersections);
}

void LineAnnotationDialog::setGeneratedFiberIntersectionMarkers(
    std::vector<GeneratedOverlay::FiberIntersectionMarker> markers)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.fiberIntersections = std::move(markers);
    rebuildGeneratedStaticStripOverlays();
    rebuildGeneratedDynamicOverlays();
}

void LineAnnotationDialog::setGeneratedSideStripIntersectionBusy(bool busy)
{
    if (_closing || !_sideStripIntersectionProgress) {
        return;
    }
    if (busy) {
        _sideStripIntersectionProgress->setVisible(true);
        _sideStripIntersectionProgress->setRange(0, 100);
        _sideStripIntersectionProgress->setValue(0);
        _sideStripIntersectionProgress->setFormat(tr("strip intersections: 0%"));
    } else {
        _sideStripIntersectionProgress->setRange(0, 100);
        _sideStripIntersectionProgress->setValue(100);
    }
}

void LineAnnotationDialog::setGeneratedSideStripIntersectionProgress(const QString& stage,
                                                                     size_t completed,
                                                                     size_t total)
{
    if (_closing || !_sideStripIntersectionProgress) {
        return;
    }
    (void)stage;
    _sideStripIntersectionProgress->setVisible(true);
    _sideStripIntersectionProgress->setRange(0, 100);
    int value = total > 0
        ? static_cast<int>(std::clamp((completed * 100) / total, size_t{0}, size_t{100}))
        : _sideStripIntersectionProgress->value();
    value = std::max(value, _sideStripIntersectionProgress->value());
    _sideStripIntersectionProgress->setValue(value);
    _sideStripIntersectionProgress->setFormat(
        tr("strip intersections: %1%").arg(value));
}

void LineAnnotationDialog::setGeneratedSideStripIntersectionResult(size_t markerCount)
{
    if (_closing || !_sideStripIntersectionProgress) {
        return;
    }
    _sideStripIntersectionProgress->setVisible(true);
    _sideStripIntersectionProgress->setRange(0, 100);
    _sideStripIntersectionProgress->setValue(100);
    _sideStripIntersectionProgress->setFormat(
        tr("strip intersections: %1").arg(markerCount));
}

void LineAnnotationDialog::setGeneratedSideStripIntersectionError()
{
    if (_closing || !_sideStripIntersectionProgress) {
        return;
    }
    _sideStripIntersectionProgress->setVisible(true);
    _sideStripIntersectionProgress->setRange(0, 100);
    _sideStripIntersectionProgress->setValue(100);
    _sideStripIntersectionProgress->setFormat(tr("strip intersections: error"));
}

void LineAnnotationDialog::setGeneratedPredSnapPoints(
    std::vector<GeneratedOverlay::PredSnapMarker> predSnapPoints)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.predSnapPoints = std::move(predSnapPoints);
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setGeneratedSpanAlignmentMetrics(
    std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    _generatedViews.spanAlignmentMetrics = std::move(spanAlignmentMetrics);
    updateGeneratedDynamicOverlaysFast(false, true);
}

void LineAnnotationDialog::setOptimizationBusy(bool busy)
{
    auto* content = centralWidget();
    if (!content) {
        return;
    }
    if (!_optimizationOverlay) {
        auto* overlay = new QWidget(content);
        overlay->setObjectName(QStringLiteral("lineAnnotationOptimizationOverlay"));
        overlay->setAttribute(Qt::WA_StyledBackground, true);
        overlay->setStyleSheet(QStringLiteral(
            "#lineAnnotationOptimizationOverlay { background-color: rgba(32, 32, 32, 120); }"
            "#lineAnnotationOptimizationOverlay QLabel { color: white; font-weight: 600; }"));
        auto* layout = new QVBoxLayout(overlay);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* label = new QLabel(tr("Optimizing..."), overlay);
        label->setAlignment(Qt::AlignCenter);
        layout->addStretch(1);
        layout->addWidget(label);
        layout->addStretch(1);
        overlay->hide();
        _optimizationOverlay = overlay;
    }
    updateOptimizationOverlayGeometry();
    _optimizationOverlay->setVisible(busy);
    if (busy) {
        _optimizationOverlay->raise();
    }
}

void LineAnnotationDialog::setOptimizationStatus(bool optimized)
{
    if (!_optimizationStatusLabel) {
        return;
    }
    _optimizationStatusLabel->setText(optimized ? tr("optimized") : tr("not optimized"));
}

void LineAnnotationDialog::setFiberDisplayName(const QString& name)
{
    _fiberDisplayName = name;
    updateFiberNameLabel();
}

void LineAnnotationDialog::setCloseAfterFinalizationAllowed(bool allowed)
{
    _closeAfterFinalizationAllowed = allowed;
}

void LineAnnotationDialog::setWorkspaceEmbedded(bool embedded)
{
    _workspaceEmbedded = embedded;
}

void LineAnnotationDialog::closeEvent(QCloseEvent* event)
{
    if (_closeAfterFinalizationAllowed) {
        _closing = true;
        clearGeneratedOverlayRefreshConnections();
        cancelControlPointPreviewAnimation();
        if (_lineUpdateTimer) {
            _lineUpdateTimer->stop();
        }
        saveGeneratedViewStateSettings();
        if (!_workspaceEmbedded) {
            saveWindowGeometry();
        }
        QMainWindow::closeEvent(event);
        return;
    }
    event->ignore();
    emit closeFinalizationRequested(event);
}

CChunkedVolumeViewer* LineAnnotationDialog::addPane(
    const std::string& surfaceName,
    const QString& title,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_mdiArea) {
        return nullptr;
    }

    auto* base = _viewerManager->createViewer(surfaceName,
                                             title,
                                             _mdiArea,
                                             ViewerManager::ViewerRole::Annotation);
    if (!base) {
        return nullptr;
    }

    auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
    if (!viewer) {
        return nullptr;
    }

    auto* subWindow = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
    if (subWindow) {
        subWindow->showMaximized();
        connect(subWindow, &QObject::destroyed, this, [this, surfaceName]() {
            if (!_suppressPaneClosed) {
                emit paneClosed(surfaceName);
            }
        });
    }

    viewer->applyCameraState(camera, false);
    bindPaneInteractions(surfaceName, viewer, true);
    _panes.push_back(Pane{surfaceName, viewer, subWindow});
    return viewer;
}

bool LineAnnotationDialog::setGeneratedRows(
    const std::vector<std::vector<std::pair<std::string, QString>>>& rows,
    const CChunkedVolumeViewer::CameraState& camera,
    const std::map<std::string, GeneratedOverlay>& overlays)
{
    if (!_viewerManager || !_layout) {
        return false;
    }

    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(false);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(false);
    }
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(false);
    }

    clearGeneratedOverlayRefreshConnections();
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    _panes.clear();
    _stripViewers.clear();
    clearFastGeneratedOverlayItemRefs();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;
    _hasGeneratedViews = false;
    _currentCutManualRotation = cv::Matx33f::eye();
    _currentCutManualRotationActive = false;
    _currentCutNormalOffsetVx = 0.0;
    _sideCutNormalOffsetVx = 0.0;
    _generatedControlIndex = {};
    _haveInitialCurrentCutCamera = false;
    _haveInitialSideCutCamera = false;
    _initialStripCameras.clear();

    for (const auto& row : rows) {
        if (row.empty()) {
            continue;
        }

        auto* rowWidget = new QWidget(this);
        rowWidget->installEventFilter(this);
        auto* rowLayout = new QHBoxLayout(rowWidget);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(0);
        _layout->addWidget(rowWidget, 1);

        for (const auto& [surfaceName, title] : row) {
            auto* base = _viewerManager->createViewerInWidget(
                surfaceName,
                rowWidget,
                ViewerManager::ViewerRole::Annotation);
            if (!base) {
                return false;
            }
            auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
            if (!viewer) {
                return false;
            }
            viewer->setObjectName(title);
            viewer->applyCameraState(generatedPaneCamera(viewer, camera), false);
            bindPaneInteractions(surfaceName, viewer, false);
            rowLayout->addWidget(viewer, 1);
            _panes.push_back(Pane{surfaceName, viewer, {}});
            if (auto overlay = overlays.find(surfaceName); overlay != overlays.end()) {
                setGeneratedOverlay(surfaceName, viewer, overlay->second);
            }
        }
    }
    const bool ok = !_panes.empty();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(ok);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(ok);
    }
    return ok;
}

void LineAnnotationDialog::bindPaneInteractions(const std::string& surfaceName,
                                                CChunkedVolumeViewer* viewer,
                                                bool seedPlacementEnabled)
{
    if (!viewer) {
        return;
    }

    viewer->setLineAnnotationPlacementPreviewEnabled(seedPlacementEnabled);
    viewer->installEventFilter(this);
    if (auto* view = viewer->graphicsView()) {
        view->installEventFilter(this);
        if (auto* viewport = view->viewport()) {
            viewport->installEventFilter(this);
        }
    }
    if (!seedPlacementEnabled) {
        return;
    }
    connect(viewer,
            &CChunkedVolumeViewer::sendLineAnnotationSeedRequested,
            this,
            [this, surfaceName](cv::Vec3f volumePoint, QPointF scenePoint) {
                emit lineSeedRequested(surfaceName, volumePoint, scenePoint);
            });
}

void LineAnnotationDialog::connectGeneratedOverlayRefresh(CChunkedVolumeViewer* viewer)
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    if (!viewer) {
        return;
    }
    _generatedOverlayRefreshConnections.push_back(
        viewer->connectOverlaysUpdated(this, [this]() {
            if (_closing || _generatedOverlayRefreshQueued) {
                return;
            }
            _generatedOverlayRefreshQueued = true;
            QTimer::singleShot(16, this, [this]() {
                if (_closing) {
                    return;
                }
                _generatedOverlayRefreshQueued = false;
                rebuildGeneratedOverlays();
            });
        }));
}

void LineAnnotationDialog::clearGeneratedOverlayRefreshConnections()
{
    for (const auto& connection : _generatedOverlayRefreshConnections) {
        QObject::disconnect(connection);
    }
    _generatedOverlayRefreshConnections.clear();
    _generatedOverlayRefreshQueued = false;
}

void LineAnnotationDialog::setGeneratedOverlay(const std::string& surfaceName,
                                               CChunkedVolumeViewer* viewer,
                                               const GeneratedOverlay& overlay)
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    if (!viewer) {
        return;
    }

    QPointer<CChunkedVolumeViewer> viewerPtr(viewer);
    const auto apply = [this, surfaceName, viewerPtr, overlay]() {
        if (!viewerPtr) {
            return;
        }
        applyGeneratedOverlay(surfaceName, viewerPtr, overlay);
    };
    viewer->renderVisible(true, "line annotation overlay");
    apply();
    viewer->connectOverlaysUpdated(this, apply);
}

bool LineAnnotationDialog::setGeneratedLineViews(
    const GeneratedViews& views,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_layout || views.linePoints.empty() ||
        views.lineUpVectors.size() != views.linePoints.size() ||
        !views.lineSideSlice || !views.currentCutSurface || !views.sideCutSurface) {
        return false;
    }

    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(false);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(false);
    }
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(false);
    }

    const bool replacingGeneratedViews = _hasGeneratedViews;
    const double previousCurrentLinePosition = _currentLinePosition;

    bool haveCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState currentCutCamera;
    if (_currentCutViewer) {
        currentCutCamera = _currentCutViewer->cameraState();
        haveCurrentCutCamera = true;
    }

    bool haveSideCutCamera = false;
    CChunkedVolumeViewer::CameraState sideCutCamera;
    if (_sideCutViewer) {
        sideCutCamera = _sideCutViewer->cameraState();
        haveSideCutCamera = true;
    }

    std::vector<CChunkedVolumeViewer::CameraState> stripCameras;
    stripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            stripCameras.push_back(viewer->cameraState());
        }
    }

    if (_generatedOuterSplitter) {
        _savedOuterSplitterSizes = _generatedOuterSplitter->sizes();
    }
    if (_generatedTopSplitter) {
        _savedTopSplitterSizes = _generatedTopSplitter->sizes();
    }
    if (_generatedStripSplitter) {
        _savedStripSplitterSizes = _generatedStripSplitter->sizes();
    }

    clearGeneratedOverlayRefreshConnections();
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    for (auto& container : _generatedContainers) {
        if (container) {
            _layout->removeWidget(container);
            delete container;
        }
    }
    _generatedContainers.clear();
    _generatedTopWidget = nullptr;
    _panes.clear();
    _stripViewers.clear();
    clearFastGeneratedOverlayItemRefs();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;

    _generatedViews = views;
    _linePointsd.clear();
    _linePointsd.reserve(_generatedViews.linePoints.size());
    for (const auto& p : _generatedViews.linePoints) {
        _linePointsd.emplace_back(p[0], p[1], p[2]);
    }
    _sideFitBracket[0] = SideFit{};
    _sideFitBracket[1] = SideFit{};
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    _hasGeneratedViews = true;
    _currentCutFollowsStripMouse = views.initialCurrentCutFollowsStripMouse;
    _currentCutNormalOffsetVx = 0.0;
    _sideCutNormalOffsetVx = 0.0;
    if (!replacingGeneratedViews) {
        _currentCutManualRotation = cv::Matx33f::eye();
        _currentCutManualRotationActive = false;
    }
    const double maxLinePosition = static_cast<double>(views.linePoints.size() - 1);
    _currentLinePosition = replacingGeneratedViews
        ? std::clamp(previousCurrentLinePosition, 0.0, maxLinePosition)
        : std::clamp(static_cast<double>(views.initialCenterIndex), 0.0, maxLinePosition);
    if (!updatePlaneSurface(views.currentCutSurface.get(), _currentLinePosition)) {
        return false;
    }
    if (!updateSidePlaneSurface(views.sideCutSurface.get(), _currentLinePosition)) {
        return false;
    }

    auto* outerSplitter = new QSplitter(Qt::Vertical, this);
    outerSplitter->setObjectName(QStringLiteral("LineAnnotationOuterSplitter"));
    outerSplitter->setChildrenCollapsible(false);
    outerSplitter->installEventFilter(this);
    _generatedContainers.push_back(outerSplitter);
    _generatedOuterSplitter = outerSplitter;
    _layout->addWidget(outerSplitter, 1);

    auto* topSplitter = new QSplitter(Qt::Horizontal, outerSplitter);
    topSplitter->setObjectName(QStringLiteral("LineAnnotationTopSplitter"));
    topSplitter->setChildrenCollapsible(false);
    topSplitter->installEventFilter(this);
    _generatedTopWidget = topSplitter;
    _generatedTopSplitter = topSplitter;
    outerSplitter->addWidget(topSplitter);

    auto* currentBase = _viewerManager->createViewerInWidget(
        views.currentCutName,
        topSplitter,
        ViewerManager::ViewerRole::Annotation);
    auto* currentViewer = currentBase
        ? qobject_cast<CChunkedVolumeViewer*>(currentBase->asQObject())
        : nullptr;
    if (!currentViewer) {
        return false;
    }
    currentViewer->setObjectName(tr("Current Line Cut"));
    auto currentApplyCamera = haveCurrentCutCamera
        ? currentCutCamera
        : generatedPaneCamera(currentViewer, camera);
    if (!haveCurrentCutCamera && _haveSavedCurrentCutZoom) {
        currentApplyCamera.scale = _savedCurrentCutZoom;
    }
    currentViewer->applyCameraState(currentApplyCamera, false);
    if (!haveCurrentCutCamera && finitePoint(_generatedViews.focusPoint)) {
        currentViewer->centerOnVolumePoint(_generatedViews.focusPoint, false);
    }
    currentViewer->setShiftScrollOverride(
        [this](int steps, QPointF, Qt::KeyboardModifiers modifiers) {
            (void)modifiers;
            if (shiftScrollMode() == ShiftScrollMode::StraightNormal) {
                return shiftCurrentCutPlaneNormalOffsetByScrollSteps(steps);
            }
            return shiftCurrentLinePositionByScrollSteps(steps);
        });
    bindPaneInteractions(views.currentCutName, currentViewer, false);
    connect(currentViewer,
            &CChunkedVolumeViewer::sendMousePressVolume,
            this,
            [this](cv::Vec3f volumePoint,
                   cv::Vec3f,
                   Qt::MouseButton button,
                   Qt::KeyboardModifiers modifiers,
                   QPointF) {
                if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                    emit generatedPredSnapPointRequested(_generatedViews.currentCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    if (!controlPointPlacementAllowedAt(_currentLinePosition)) {
                        return;
                    }
                    setCurrentCutFollowsStripMouse(true);
                    emit generatedControlPointRequested(_generatedViews.currentCutName,
                                                        volumePoint,
                                                        _currentLinePosition);
                }
            });
    topSplitter->addWidget(currentViewer);
    _currentCutViewer = currentViewer;
    _panes.push_back(Pane{views.currentCutName, currentViewer, {}});
    connectGeneratedOverlayRefresh(currentViewer);

    auto* sideBase = _viewerManager->createViewerInWidget(
        views.sideCutName,
        topSplitter,
        ViewerManager::ViewerRole::Annotation);
    auto* sideViewer = sideBase
        ? qobject_cast<CChunkedVolumeViewer*>(sideBase->asQObject())
        : nullptr;
    if (!sideViewer) {
        return false;
    }
    sideViewer->setObjectName(tr("Line Side Cut"));
    auto sideApplyCamera = haveSideCutCamera
        ? sideCutCamera
        : generatedPaneCamera(sideViewer, camera);
    if (!haveSideCutCamera && _haveSavedSideCutZoom) {
        sideApplyCamera.scale = _savedSideCutZoom;
    }
    sideViewer->applyCameraState(sideApplyCamera, false);
    if (!haveSideCutCamera && finitePoint(_generatedViews.focusPoint)) {
        sideViewer->centerOnVolumePoint(_generatedViews.focusPoint, false);
    }
    sideViewer->setProperty("vc_show_custom_normal_offset", true);
    sideViewer->setProperty("vc_custom_normal_offset_vx", _sideCutNormalOffsetVx);
    sideViewer->setShiftScrollOverride(
        [this](int steps, QPointF, Qt::KeyboardModifiers) {
            return shiftSideCutPlaneNormalOffsetByScrollSteps(steps);
        });
    bindPaneInteractions(views.sideCutName, sideViewer, false);
    connect(sideViewer,
            &CChunkedVolumeViewer::sendMousePressVolume,
            this,
            [this](cv::Vec3f volumePoint,
                   cv::Vec3f,
                   Qt::MouseButton button,
                   Qt::KeyboardModifiers modifiers,
                   QPointF) {
                if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                    emit generatedPredSnapPointRequested(_generatedViews.sideCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    if (!controlPointPlacementAllowedAt(_currentLinePosition)) {
                        return;
                    }
                    setCurrentCutFollowsStripMouse(true);
                    emit generatedControlPointRequested(_generatedViews.sideCutName,
                                                        volumePoint,
                                                        _currentLinePosition);
                }
            });
    topSplitter->addWidget(sideViewer);
    _sideCutViewer = sideViewer;
    _panes.push_back(Pane{views.sideCutName, sideViewer, {}});
    connectGeneratedOverlayRefresh(sideViewer);
    topSplitter->setStretchFactor(0, 1);
    topSplitter->setStretchFactor(1, 1);

    auto* stripSplitter = new QSplitter(Qt::Vertical, outerSplitter);
    stripSplitter->setObjectName(QStringLiteral("LineAnnotationStripSplitter"));
    stripSplitter->setChildrenCollapsible(false);
    stripSplitter->installEventFilter(this);
    _generatedStripSplitter = stripSplitter;
    outerSplitter->addWidget(stripSplitter);

    const std::pair<std::string, QString> stripSpecs[] = {
        {views.lineSurfaceName, views.lineSurfaceTitle},
        {views.lineSideSliceName, views.lineSideSliceTitle},
    };
    int stripIndex = 0;
    for (const auto& [surfaceName, title] : stripSpecs) {
        auto* base = _viewerManager->createViewerInWidget(
            surfaceName,
            stripSplitter,
            ViewerManager::ViewerRole::Annotation);
        auto* viewer = base ? qobject_cast<CChunkedVolumeViewer*>(base->asQObject()) : nullptr;
        if (!viewer) {
            return false;
        }
        viewer->setObjectName(title);
        const bool haveStripCamera = static_cast<size_t>(stripIndex) < stripCameras.size();
        auto stripCamera = haveStripCamera
            ? stripCameras[static_cast<size_t>(stripIndex)]
            : generatedPaneCamera(viewer, camera);
        if (!haveStripCamera) {
            if (const auto center =
                    generatedStripSurfaceCenter(viewer, _currentLinePosition)) {
                stripCamera.surfacePtrX = (*center)[0];
                stripCamera.surfacePtrY = (*center)[1];
            }
            if (const auto focusedScale = generatedStripScaleForLinePositionRange(
                    viewer,
                    views.initialStripLinePositionRange)) {
                stripCamera.scale = *focusedScale;
            }
            if (static_cast<size_t>(stripIndex) < _savedStripZooms.size()) {
                stripCamera.scale = _savedStripZooms[static_cast<size_t>(stripIndex)];
            }
        }
        viewer->applyCameraState(stripCamera, false);
        bindPaneInteractions(surfaceName, viewer, false);
        connect(viewer,
                &CChunkedVolumeViewer::sendMouseMoveVolume,
                this,
                [this, viewer](cv::Vec3f, Qt::MouseButtons, Qt::KeyboardModifiers, QPointF scenePoint) {
                    if (!_currentCutFollowsStripMouse) {
                        return;
                    }
                    const double position = linePositionFromStripScene(viewer, scenePoint);
                    if (std::isfinite(position)) {
                        requestCurrentLinePosition(position);
                    }
                });
        connect(viewer,
                &CChunkedVolumeViewer::sendMousePressVolume,
                this,
                [this, viewer, surfaceName](cv::Vec3f volumePoint,
                                            cv::Vec3f,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers modifiers,
                                            QPointF scenePoint) {
                    if (button != Qt::LeftButton ||
                        (modifiers != Qt::NoModifier && modifiers != Qt::ShiftModifier)) {
                        return;
                    }
                    const double position = linePositionFromStripScene(viewer, scenePoint);
                    if (std::isfinite(position)) {
                        setCurrentCutFollowsStripMouse(true);
                        setCurrentLinePosition(position);
                        if (modifiers == Qt::ShiftModifier) {
                            emit generatedPredSnapPointRequested(surfaceName, volumePoint);
                        } else {
                            if (!controlPointPlacementAllowedAt(position)) {
                                return;
                            }
                            emit generatedControlPointRequested(surfaceName, volumePoint, position);
                        }
                    }
                });
        stripSplitter->addWidget(viewer);
        _stripViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
        connectGeneratedOverlayRefresh(viewer);
        ++stripIndex;
    }

    stripSplitter->setStretchFactor(0, 1);
    stripSplitter->setStretchFactor(1, 1);
    outerSplitter->setStretchFactor(0, 2);
    outerSplitter->setStretchFactor(1, 1);

    // Restore user-adjusted splitter sizes across rebuilds (point placement re-runs this
    // function); fall back to the default 2:1 top/strip ratio on first build.
    if (_savedTopSplitterSizes.size() == topSplitter->count()) {
        topSplitter->setSizes(_savedTopSplitterSizes);
    }
    if (_savedStripSplitterSizes.size() == stripSplitter->count()) {
        stripSplitter->setSizes(_savedStripSplitterSizes);
    }
    if (_savedOuterSplitterSizes.size() == outerSplitter->count()) {
        outerSplitter->setSizes(_savedOuterSplitterSizes);
    } else {
        outerSplitter->setSizes({2, 1});
    }

    rebuildGeneratedOverlays();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(true);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(true);
    }
    captureInitialGeneratedViewState();
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(true);
    }
    return true;
}

LineAnnotationDialog::GeneratedControlPointContextResult
LineAnnotationDialog::showGeneratedControlPointContextMenu(
    const std::string& surfaceName,
    CChunkedVolumeViewer* viewer,
    const QPointF& scenePoint,
    const QPoint& globalPos,
    const vc3d::line_annotation::GeneratedLinkCandidateMenuState& linkCandidateState)
{
    if (!viewer || !_hasGeneratedViews || _generatedViews.controlPoints.empty() ||
        _generatedViews.linePoints.empty()) {
        return GeneratedControlPointContextResult::None;
    }

    double linePosition = std::numeric_limits<double>::quiet_NaN();
    if (viewer == _currentCutViewer || viewer == _sideCutViewer) {
        linePosition = _currentLinePosition;
    } else {
        for (size_t i = 0; i < _stripViewers.size(); ++i) {
            if (viewer == _stripViewers[i]) {
                linePosition = linePositionFromStripScene(viewer, scenePoint);
                break;
            }
        }
    }

    if (!vc3d::line_annotation::validGeneratedLinePosition(linePosition,
                                                           _generatedViews.linePoints.size())) {
        return GeneratedControlPointContextResult::None;
    }

    const bool stripViewer =
        std::any_of(_stripViewers.begin(),
                    _stripViewers.end(),
                    [viewer](const QPointer<CChunkedVolumeViewer>& candidate) {
                        return candidate == viewer;
                    });

    vc3d::line_annotation::GeneratedControlPointContextMenuOptions options;
    options.parent = this;
    options.surfaceName = surfaceName;
    options.viewer = viewer;
    options.scenePoint = scenePoint;
    options.globalPos = globalPos;
    options.controlPoints = _generatedViews.controlPoints;
    // On the cut viewers only plane-filtered X markers are drawn; restrict the
    // hit-test candidates the same way so the menu never targets an invisible
    // off-plane marker that happens to project near the click.
    PlaneSurface* cutPlane = nullptr;
    if (viewer == _currentCutViewer) {
        cutPlane = _generatedViews.currentCutSurface.get();
    } else if (viewer == _sideCutViewer) {
        cutPlane = _generatedViews.sideCutSurface.get();
    }
    if (!cutPlane) {
        options.fiberIntersections = _generatedViews.fiberIntersections;
    } else if (const std::optional<float> intersectionThreshold =
                   vc3d::line_annotation::generatedCrossSliceControlPointDistanceThreshold(
                       viewer)) {
        for (const auto& intersection : _generatedViews.fiberIntersections) {
            if (!finitePoint(intersection.point)) {
                continue;
            }
            const float distance = cutPlane->pointDist(intersection.point);
            if (std::isfinite(distance) && std::abs(distance) <= *intersectionThreshold) {
                options.fiberIntersections.push_back(intersection);
            }
        }
    }
    options.linePointCount = _generatedViews.linePoints.size();
    options.linePosition = linePosition;
    options.stripViewer = stripViewer;
    options.linkWithCandidateEnabled = linkCandidateState.enabled;
    options.linkWithCandidateLabel = linkCandidateState.label;
    options.branchLinkDirection = branchLinkDirectionForViewer(viewer, linePosition);
    options.deleteControlPoint = [this, surfaceName](double selectedLinePosition,
                                                     cv::Vec3f selectedPoint) {
        emit generatedControlPointDeleteRequested(surfaceName,
                                                  selectedLinePosition,
                                                  selectedPoint);
    };
    options.addBranch = [this, surfaceName](size_t controlPointIndex,
                                            cv::Vec3f linkedControlPoint,
                                            bool openAfterCreate,
                                            cv::Vec3f linkDirection) {
        emit generatedControlPointBranchRequested(surfaceName,
                                                  controlPointIndex,
                                                  linkedControlPoint,
                                                  openAfterCreate,
                                                  linkDirection);
    };
    options.openBranch = [this](uint64_t branchFiberId, int branchControlPointIndex) {
        emit generatedControlPointBranchOpenRequested(branchFiberId, branchControlPointIndex);
    };
    options.designateLinkCandidate = [this, surfaceName](size_t controlPointIndex,
                                                         cv::Vec3f volumePoint) {
        emit generatedControlPointLinkCandidateRequested(surfaceName,
                                                         controlPointIndex,
                                                         volumePoint);
    };
    options.linkWithCandidate = [this, surfaceName](size_t controlPointIndex,
                                                    cv::Vec3f volumePoint) {
        emit generatedControlPointLinkWithCandidateRequested(surfaceName,
                                                             controlPointIndex,
                                                             volumePoint);
    };
    options.openNearbyAnnotation = [this](uint64_t fiberId, cv::Vec3f volumePoint) {
        emit generatedNearbyAnnotationOpenRequested(fiberId, volumePoint);
    };
    options.unlinkBranch = [this, surfaceName](size_t controlPointIndex,
                                               uint64_t branchFiberId,
                                               int branchControlPointIndex) {
        emit generatedControlPointUnlinkRequested(surfaceName,
                                                  controlPointIndex,
                                                  branchFiberId,
                                                  branchControlPointIndex);
    };
    options.setBranchLinkPending = [this, surfaceName](size_t controlPointIndex,
                                                       uint64_t branchFiberId,
                                                       int branchControlPointIndex,
                                                       bool pending) {
        emit generatedControlPointLinkPendingChangeRequested(surfaceName,
                                                             controlPointIndex,
                                                             branchFiberId,
                                                             branchControlPointIndex,
                                                             pending);
    };
    return vc3d::line_annotation::showGeneratedControlPointContextMenu(options);
}

void LineAnnotationDialog::applyGeneratedOverlay(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    vc3d::line_annotation::applyGeneratedOverlay(viewer, surfaceName, overlay);
}

void LineAnnotationDialog::applyOverlayForViewer(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    vc3d::line_annotation::applyGeneratedOverlay(viewer, surfaceName, overlay);
}

void LineAnnotationDialog::clearControlPointContextPreview(const std::string& surfaceName,
                                                           CChunkedVolumeViewer* viewer)
{
    vc3d::line_annotation::clearGeneratedControlPointContextPreview(viewer, surfaceName);
}

double LineAnnotationDialog::linePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                                        const QPointF& scenePoint) const
{
    if (!viewer || !_hasGeneratedViews) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return vc3d::line_annotation::generatedLinePositionFromStripScene(viewer, scenePoint);
}

void LineAnnotationDialog::requestCurrentLinePosition(double position)
{
    // Hot path: called once per mouse-move event while following a strip viewer. Defer the
    // actual (potentially O(N)) plane + overlay rebuild to a single render-tick-cadence flush so
    // a fast cursor doesn't back up the event loop with one full update per move.
    _pendingLinePosition = position;
    _lineUpdatePending = true;
    if (!_lineUpdateTimer) {
        _lineUpdateTimer = new QTimer(this);
        _lineUpdateTimer->setSingleShot(true);
        _lineUpdateTimer->setInterval(16);  // ~one global render tick
        connect(_lineUpdateTimer, &QTimer::timeout, this, [this]() {
            if (_lineUpdatePending) {
                setCurrentLinePosition(_pendingLinePosition, false);
            }
        });
    }
    if (!_lineUpdateTimer->isActive()) {
        _lineUpdateTimer->start();
    }
}

void LineAnnotationDialog::setCurrentLinePosition(double position,
                                                  bool updateCurrentCutOverlay)
{
    // An immediate apply supersedes any coalesced mouse-follow update still pending in the timer,
    // so a discrete jump/click/scroll isn't clobbered by a stale flush a few ms later.
    _lineUpdatePending = false;
    if (_lineUpdateTimer) {
        _lineUpdateTimer->stop();
    }
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    position = std::clamp(position, 0.0, static_cast<double>(_generatedViews.linePoints.size() - 1));
    const bool currentChanged = std::abs(position - _currentLinePosition) >= 1.0e-3;
    if (!currentChanged) {
        return;
    }
    if (currentChanged && _generatedViews.currentCutSurface) {
        _currentCutNormalOffsetVx = 0.0;
        if (_currentCutViewer) {
            _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
        }
        _currentLinePosition = position;
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
        if (_currentCutViewer) {
            _currentCutViewer->markSurfaceGeometryChanged();
        }
    } else {
        _currentLinePosition = position;
    }
    if (_currentCutViewer) {
        // Non-force: let the global render tick coalesce a burst of cursor moves into one
        // render per tick instead of force-submitting synchronously on every mouse event.
        _currentCutViewer->renderVisible(false, "line annotation current cut");
    }
    if (_generatedViews.sideCutSurface) {
        (void)updateSidePlaneSurface(_generatedViews.sideCutSurface.get(), _currentLinePosition);
        (void)applyCutPlaneNormalOffset(_generatedViews.sideCutSurface.get(),
                                        _sideCutNormalOffsetVx);
    }
    if (_sideCutViewer) {
        // Moving the cursor along the strips moves along the line, so keep the current position
        // centered/visible in the side view. centerOnVolumePoint(false) pans and schedules the
        // (coalesced) render, so no separate renderVisible call is needed here.
        const cv::Vec3f sidePoint = interpolatedLinePoint(_currentLinePosition);
        if (finitePoint(sidePoint)) {
            _sideCutViewer->centerOnVolumePoint(sidePoint, false);
        } else {
            _sideCutViewer->renderVisible(false, "line annotation side cut");
        }
    }
    rebuildGeneratedDynamicOverlays(updateCurrentCutOverlay, false);
}

bool LineAnnotationDialog::shiftCurrentLinePositionByScrollSteps(int steps)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return true;
    }
    const int sliceStepSize = _viewerManager
        ? std::max(1, static_cast<int>(std::lround(_viewerManager->zScrollSensitivity())))
        : 1;
    const double position = vc3d::line_annotation::shiftedLinePosition(
        _currentLinePosition,
        steps,
        sliceStepSize,
        static_cast<int>(_generatedViews.linePoints.size()));
    _currentCutNormalOffsetVx = 0.0;
    if (_currentCutViewer) {
        _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
    }
    setCurrentLinePosition(position);
    return true;
}

void LineAnnotationDialog::cancelControlPointPreviewAnimation()
{
    if (!_controlPointPreviewAnimation) {
        return;
    }
    auto* animation = _controlPointPreviewAnimation.data();
    _controlPointPreviewAnimation = nullptr;
    animation->stop();
    animation->deleteLater();
}

void LineAnnotationDialog::jumpToPreviousControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto previous = vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        _currentLinePosition,
        positions);
    if (previous) {
        setCurrentLinePosition(*previous);
    }
}

void LineAnnotationDialog::jumpToNextControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto next = vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        _currentLinePosition,
        positions);
    if (next) {
        setCurrentLinePosition(*next);
    }
}

void LineAnnotationDialog::previewClosestControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const double originalPosition = _currentLinePosition;
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto closest = vc3d::line_annotation::closestGeneratedControlPointLinePosition(
        originalPosition,
        positions);
    if (!closest) {
        return;
    }
    setCurrentLinePosition(*closest);

    auto* animation = new QVariantAnimation(this);
    _controlPointPreviewAnimation = animation;
    constexpr int kClosestControlPointHoldMs = 500;
    constexpr int kClosestControlPointReturnMs = 2000;
    constexpr int kClosestControlPointTotalMs =
        kClosestControlPointHoldMs + kClosestControlPointReturnMs;
    animation->setDuration(kClosestControlPointTotalMs);
    animation->setStartValue(*closest);
    animation->setKeyValueAt(static_cast<double>(kClosestControlPointHoldMs) /
                                 static_cast<double>(kClosestControlPointTotalMs),
                             *closest);
    animation->setEndValue(originalPosition);
    connect(animation, &QVariantAnimation::valueChanged, this, [this, animation](const QVariant& value) {
        if (_controlPointPreviewAnimation == animation) {
            setCurrentLinePosition(value.toDouble());
        }
    });
    connect(animation, &QVariantAnimation::finished, this, [this, animation]() {
        if (_controlPointPreviewAnimation == animation) {
            _controlPointPreviewAnimation = nullptr;
            animation->deleteLater();
        }
    });
    QTimer::singleShot(30, animation, [animation]() {
        animation->start();
    });
}

bool LineAnnotationDialog::shiftCurrentCutPlaneNormalOffsetByScrollSteps(int steps)
{
    return shiftCutPlaneNormalOffsetByScrollSteps(_generatedViews.currentCutSurface.get(),
                                                  _currentCutViewer,
                                                  steps,
                                                  _currentCutNormalOffsetVx,
                                                  "line annotation current cut normal offset");
}

bool LineAnnotationDialog::shiftSideCutPlaneNormalOffsetByScrollSteps(int steps)
{
    return shiftCutPlaneNormalOffsetByScrollSteps(_generatedViews.sideCutSurface.get(),
                                                  _sideCutViewer,
                                                  steps,
                                                  _sideCutNormalOffsetVx,
                                                  "line annotation side cut normal offset");
}

bool LineAnnotationDialog::shiftCutPlaneNormalOffsetByScrollSteps(PlaneSurface* plane,
                                                                  CChunkedVolumeViewer* viewer,
                                                                  int steps,
                                                                  double& offsetVx,
                                                                  const char* renderReason)
{
    if (!_hasGeneratedViews || !plane || steps == 0) {
        return true;
    }
    const int sliceStepSize = _viewerManager
        ? std::max(1, static_cast<int>(std::lround(_viewerManager->zScrollSensitivity())))
        : 1;
    const cv::Vec3f origin = plane->origin();
    const cv::Vec3f normal = plane->normal({0.0f, 0.0f, 0.0f});
    const float normalNorm = cv::norm(normal);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2]) || normalNorm <= 1.0e-6f) {
        return true;
    }
    const cv::Vec3f shiftedOrigin =
        vc3d::line_annotation::shiftedPlaneOriginAlongNormal(origin,
                                                             normal,
                                                             steps,
                                                             sliceStepSize);
    if (!finitePoint(shiftedOrigin)) {
        return true;
    }
    plane->setFromNormalAndUp(shiftedOrigin, normal, plane->basisY());
    offsetVx += static_cast<double>(steps) *
                static_cast<double>(vc3d::line_annotation::shiftScrollLineStepSize(sliceStepSize));
    if (!normalOffsetActive(offsetVx)) {
        offsetVx = 0.0;
    }
    if (viewer) {
        viewer->setProperty("vc_custom_normal_offset_vx", offsetVx);
        viewer->markSurfaceGeometryChanged();
        viewer->renderVisible(true, renderReason);
    }
    rebuildGeneratedDynamicOverlays();
    return true;
}

bool LineAnnotationDialog::applyCutPlaneNormalOffset(PlaneSurface* plane, double offsetVx) const
{
    if (!plane || !normalOffsetActive(offsetVx)) {
        return true;
    }
    const cv::Vec3f normal = plane->normal({0.0f, 0.0f, 0.0f});
    const float normalNorm = cv::norm(normal);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) ||
        !std::isfinite(normal[2]) || normalNorm <= 1.0e-6f) {
        return false;
    }
    plane->setFromNormalAndUp(plane->origin() +
                                  normal * (static_cast<float>(offsetVx) / normalNorm),
                              normal,
                              plane->basisY());
    return true;
}

void LineAnnotationDialog::resetGeneratedCutNormalOffsets(bool forceRender)
{
    if (!_hasGeneratedViews) {
        return;
    }

    bool changed = false;
    const bool currentHadOffset = normalOffsetActive(_currentCutNormalOffsetVx);
    const bool sideHadOffset = normalOffsetActive(_sideCutNormalOffsetVx);
    _currentCutNormalOffsetVx = 0.0;
    _sideCutNormalOffsetVx = 0.0;
    if (_currentCutViewer) {
        _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
    }
    if (_sideCutViewer) {
        _sideCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
    }

    if (currentHadOffset && _generatedViews.currentCutSurface) {
        if (updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition)) {
            changed = true;
            if (_currentCutViewer) {
                _currentCutViewer->markSurfaceGeometryChanged();
                if (forceRender) {
                    _currentCutViewer->renderVisible(true, "line annotation current cut offset reset");
                }
            }
        }
    }
    if (sideHadOffset && _generatedViews.sideCutSurface) {
        if (updateSidePlaneSurface(_generatedViews.sideCutSurface.get(), _currentLinePosition)) {
            changed = true;
            if (_sideCutViewer) {
                _sideCutViewer->markSurfaceGeometryChanged();
                if (forceRender) {
                    _sideCutViewer->renderVisible(true, "line annotation side cut offset reset");
                }
            }
        }
    }
    if (changed) {
        rebuildGeneratedDynamicOverlays();
    }
}

void LineAnnotationDialog::handleShiftScrollModeChanged()
{
    if (shiftScrollMode() != ShiftScrollMode::AlongLine) {
        return;
    }

    const bool wasFollowing = _currentCutFollowsStripMouse;
    setCurrentCutFollowsStripMouse(true);
    if (!wasFollowing) {
        return;
    }

    const bool currentHadOffset = normalOffsetActive(_currentCutNormalOffsetVx);
    _currentCutNormalOffsetVx = 0.0;
    if (_currentCutViewer) {
        _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
    }
    if (!currentHadOffset || !_generatedViews.currentCutSurface) {
        return;
    }
    if (!updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition)) {
        return;
    }
    if (_currentCutViewer) {
        _currentCutViewer->markSurfaceGeometryChanged();
        _currentCutViewer->renderVisible(true, "line annotation current cut along-line mode");
    }
    rebuildGeneratedDynamicOverlays();
}

void LineAnnotationDialog::setCutFollowEnabled(bool enabled)
{
    // Programmatic twin of the private toggle.
    setCurrentCutFollowsStripMouse(enabled);
}

void LineAnnotationDialog::setCurrentCutFollowsStripMouse(bool follows)
{
    const bool wasFollowing = _currentCutFollowsStripMouse;
    _currentCutFollowsStripMouse = follows;
    if (follows && !wasFollowing) {
        resetGeneratedCutNormalOffsets(true);
    }
}

void LineAnnotationDialog::requestGeneratedSideStripIntersections()
{
    if (_closing || !_hasGeneratedViews || !_generatedViews.lineSideSlice) {
        return;
    }
    emit generatedSideStripIntersectionQueryRequested(_generatedViews.lineSideSliceName);
}

cv::Vec3f LineAnnotationDialog::branchLinkDirectionForViewer(CChunkedVolumeViewer* viewer,
                                                             double linePosition) const
{
    if (!_hasGeneratedViews || !viewer) {
        return normalizedOrNan({0.0f, 0.0f, 0.0f});
    }
    if (viewer == _sideCutViewer && _generatedViews.sideCutSurface) {
        return normalizedOrNan(
            _generatedViews.sideCutSurface->normal({0.0f, 0.0f, 0.0f}));
    }
    if (viewer == _currentCutViewer && _generatedViews.currentCutSurface) {
        return normalizedOrNan(
            _generatedViews.currentCutSurface->normal({0.0f, 0.0f, 0.0f}));
    }
    if (std::any_of(_stripViewers.begin(),
                    _stripViewers.end(),
                    [viewer](const QPointer<CChunkedVolumeViewer>& candidate) {
                        return candidate == viewer;
                    }) &&
        _generatedViews.sideCutSurface) {
        return normalizedOrNan(
            _generatedViews.sideCutSurface->normal({0.0f, 0.0f, 0.0f}));
    }
    return normalizedOrNan(interpolatedLineTangent(linePosition));
}

bool LineAnnotationDialog::controlPointPlacementAllowedAt(double linePosition) const
{
    return vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        linePosition,
        _generatedViews.controlPoints,
        static_cast<double>(maxControlPointDistanceVx()));
}

vc3d::line_annotation::GeneratedCurrentLineMarkerState
LineAnnotationDialog::currentLineMarkerState() const
{
    if (maxControlPointDistanceVx() <= 0) {
        return vc3d::line_annotation::GeneratedCurrentLineMarkerState::Neutral;
    }
    return vc3d::line_annotation::generatedLinePositionWithinAnyControlDistance(
               _currentLinePosition,
               _generatedViews.controlPoints,
               static_cast<double>(maxControlPointDistanceVx()))
        ? vc3d::line_annotation::GeneratedCurrentLineMarkerState::Allowed
        : vc3d::line_annotation::GeneratedCurrentLineMarkerState::Blocked;
}

void LineAnnotationDialog::installGeneratedViewShortcuts()
{
    const auto bindNavigationShortcut = [this](Qt::Key key, void (LineAnnotationDialog::*slot)()) {
        auto* shortcut = new QShortcut(QKeySequence(key), this);
        shortcut->setContext(Qt::WindowShortcut);
        connect(shortcut, &QShortcut::activated, this, slot);
    };

    const auto bindRotationShortcut =
        [this](Qt::Key key, vc3d::line_annotation::GeneratedCutRotationAxis axis, float radians) {
            auto* shortcut = new QShortcut(QKeySequence(key), this);
            shortcut->setContext(Qt::WindowShortcut);
            connect(shortcut, &QShortcut::activated, this, [this, axis, radians]() {
                (void)rotateCurrentCut(axis, radians);
            });
        };

    using vc3d::line_annotation::GeneratedCutRotationAxis;
    bindRotationShortcut(Qt::Key_W, GeneratedCutRotationAxis::Horizontal, kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_S, GeneratedCutRotationAxis::Horizontal, -kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_A, GeneratedCutRotationAxis::Vertical, -kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_D, GeneratedCutRotationAxis::Vertical, kCurrentCutRotationStepRadians);
    bindNavigationShortcut(Qt::Key_E, &LineAnnotationDialog::jumpToPreviousControlPoint);
    bindNavigationShortcut(Qt::Key_R, &LineAnnotationDialog::previewClosestControlPoint);
    bindNavigationShortcut(Qt::Key_T, &LineAnnotationDialog::jumpToNextControlPoint);

    auto* spaceShortcut = new QShortcut(QKeySequence(Qt::Key_Space), this);
    spaceShortcut->setContext(Qt::WindowShortcut);
    connect(spaceShortcut, &QShortcut::activated, this, [this]() {
        (void)toggleCurrentCutFollowFromKeyboard();
    });
}

cv::Vec3f LineAnnotationDialog::currentCutViewerCenterVolumePoint() const
{
    auto* viewer = _currentCutViewer.data();
    auto* view = viewer ? viewer->graphicsView() : nullptr;
    auto* viewport = view ? view->viewport() : nullptr;
    if (viewer && view && viewport && viewport->width() > 0 && viewport->height() > 0) {
        const QPointF sceneCenter = view->mapToScene(viewport->rect().center());
        const cv::Vec3f center = viewer->sceneToVolume(sceneCenter);
        if (finitePoint(center)) {
            return center;
        }
    }
    return interpolatedLinePoint(_currentLinePosition);
}

bool LineAnnotationDialog::rotateCurrentCut(vc3d::line_annotation::GeneratedCutRotationAxis axis,
                                            float radians)
{
    if (!_hasGeneratedViews || !_generatedViews.currentCutSurface || !_currentCutViewer) {
        return false;
    }
    const cv::Vec3f centerVolumePoint = currentCutViewerCenterVolumePoint();
    _currentCutManualRotation =
        vc3d::line_annotation::accumulatedGeneratedCutRotation(_currentCutManualRotation,
                                                               axis,
                                                               radians);
    _currentCutManualRotationActive = true;
    if (!updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition)) {
        return false;
    }
    _currentCutViewer->markSurfaceGeometryChanged();
    if (finitePoint(centerVolumePoint)) {
        _currentCutViewer->centerOnVolumePoint(centerVolumePoint, false);
    }
    _currentCutViewer->renderVisible(true, "line annotation current cut rotation");
    rebuildGeneratedDynamicOverlays();
    return true;
}

void LineAnnotationDialog::captureInitialGeneratedViewState()
{
    _initialCurrentLinePosition = _currentLinePosition;

    _haveInitialCurrentCutCamera = false;
    if (_currentCutViewer) {
        _initialCurrentCutCamera = _currentCutViewer->cameraState();
        _haveInitialCurrentCutCamera = true;
    }
    _haveInitialSideCutCamera = false;
    if (_sideCutViewer) {
        _initialSideCutCamera = _sideCutViewer->cameraState();
        _haveInitialSideCutCamera = true;
    }
    _initialStripCameras.clear();
    _initialStripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            _initialStripCameras.push_back(viewer->cameraState());
        }
    }
}

void LineAnnotationDialog::restoreInitialGeneratedViewerCameras()
{
    if (_currentCutViewer && _haveInitialCurrentCutCamera) {
        _currentCutViewer->applyCameraState(_initialCurrentCutCamera, false);
    }
    if (_sideCutViewer && _haveInitialSideCutCamera) {
        _sideCutViewer->applyCameraState(_initialSideCutCamera, false);
    }
    for (size_t i = 0; i < _stripViewers.size() && i < _initialStripCameras.size(); ++i) {
        if (_stripViewers[i]) {
            _stripViewers[i]->applyCameraState(_initialStripCameras[i], false);
        }
    }
}

void LineAnnotationDialog::resetGeneratedViews()
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }

    _currentLinePosition = std::clamp(_initialCurrentLinePosition,
                                      0.0,
                                      static_cast<double>(_generatedViews.linePoints.size() - 1));
    _currentCutManualRotation = cv::Matx33f::eye();
    _currentCutManualRotationActive = false;
    _currentCutNormalOffsetVx = 0.0;
    _sideCutNormalOffsetVx = 0.0;
    _currentCutFollowsStripMouse = true;

    if (_generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
        if (_currentCutViewer) {
            _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
            _currentCutViewer->markSurfaceGeometryChanged();
        }
    }
    if (_generatedViews.sideCutSurface) {
        (void)updateSidePlaneSurface(_generatedViews.sideCutSurface.get(), _currentLinePosition);
        if (_sideCutViewer) {
            _sideCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
        }
    }
    restoreInitialGeneratedViewerCameras();
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation reset current cut");
    }
    if (_sideCutViewer) {
        _sideCutViewer->renderVisible(true, "line annotation reset side cut");
    }
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            viewer->renderVisible(true, "line annotation reset strip view");
        }
    }
    rebuildGeneratedOverlays();
}

double LineAnnotationDialog::snappedControlPointPosition(double position) const
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return position;
    }
    std::vector<double> controlLinePositions;
    controlLinePositions.reserve(_generatedViews.controlPoints.size());
    for (const auto& control : _generatedViews.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
    }
    return vc3d::line_annotation::snappedControlPointLinePosition(position, controlLinePositions);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::staticStripOverlay() const
{
    return vc3d::line_annotation::makeGeneratedStaticStripOverlay(_generatedViews);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::zSliceOverlay(double linePosition,
                                                                           bool emphasized,
                                                                           CChunkedVolumeViewer* viewer,
                                                                           PlaneSurface* plane) const
{
    (void)emphasized;
    return vc3d::line_annotation::makeGeneratedCrossSliceControlOverlayForPlane(
        _generatedViews,
        linePosition,
        viewer,
        plane,
        &_generatedControlIndex);
}

void LineAnnotationDialog::rebuildGeneratedStaticStripOverlays()
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    if (_closing || !_hasGeneratedViews) {
        return;
    }

    for (size_t i = 0; i < _stripViewers.size(); ++i) {
        auto* viewer = _stripViewers[i].data();
        if (!viewer) {
            continue;
        }
        const std::string key = i == 0 ? _generatedViews.lineSurfaceName
                                       : _generatedViews.lineSideSliceName;
        GeneratedOverlay strip = staticStripOverlay();
        if (i == 1) {
            strip.fiberIntersections = _generatedViews.fiberIntersections;
        }
        applyOverlayForViewer(staticStripOverlayKey(key), viewer, strip);
    }

}

void LineAnnotationDialog::clearFastGeneratedOverlayItemRefs()
{
    _fastStripOverlayItems.clear();
    _fastCurrentCutOverlayItems = {};
}

void LineAnnotationDialog::updateGeneratedDynamicOverlaysFast(bool updateCurrentCutOverlay,
                                                              bool updateSpanLabels)
{
    if (!kGeneratedLineAnnotationOverlaysEnabled) {
        return;
    }
    if (_closing || !_hasGeneratedViews) {
        return;
    }

    const auto markerColorForState =
        [](vc3d::line_annotation::GeneratedCurrentLineMarkerState state) {
            switch (state) {
            case vc3d::line_annotation::GeneratedCurrentLineMarkerState::Allowed:
                return QColor(40, 220, 120, 245);
            case vc3d::line_annotation::GeneratedCurrentLineMarkerState::Blocked:
                return QColor(255, 70, 70, 245);
            case vc3d::line_annotation::GeneratedCurrentLineMarkerState::Neutral:
            default:
                return QColor(0, 245, 255, 245);
            }
        };

    const auto ensureStripItems =
        [this](size_t index,
               CChunkedVolumeViewer* viewer,
               const std::string& surfaceName) -> FastStripOverlayItems* {
        if (!viewer) {
            return nullptr;
        }
        if (index >= _fastStripOverlayItems.size()) {
            _fastStripOverlayItems.resize(index + 1);
        }
        auto& entry = _fastStripOverlayItems[index];
        const size_t spanLabelCount = _generatedViews.spanAlignmentMetrics.size();
        const bool recreate = entry.viewer != viewer ||
                              entry.surfaceName != surfaceName ||
                              !entry.currentLine ||
                              entry.spanLabels.size() != spanLabelCount;
        if (!recreate) {
            return &entry;
        }

        const std::string overlayKey = dynamicStripOverlayKey(surfaceName);
        viewer->clearOverlayGroup(overlayKey);
        entry = {};
        entry.viewer = viewer;
        entry.surfaceName = surfaceName;

        QPen currentPen(QColor(0, 245, 255, 245));
        currentPen.setWidthF(2.0);
        currentPen.setCapStyle(Qt::RoundCap);

        std::vector<QGraphicsItem*> items;
        items.reserve(1 + spanLabelCount * 2);
        entry.currentLine = new QGraphicsPathItem();
        entry.currentLine->setPen(currentPen);
        entry.currentLine->setBrush(Qt::NoBrush);
        entry.currentLine->setZValue(153.0);
        entry.currentLine->setVisible(false);
        items.push_back(entry.currentLine);
        entry.spanLabels.reserve(spanLabelCount);
        QFont labelFont;
        labelFont.setPointSize(9);
        labelFont.setBold(true);
        for (size_t labelIndex = 0; labelIndex < spanLabelCount; ++labelIndex) {
            auto* background = new QGraphicsRectItem();
            background->setPen(Qt::NoPen);
            background->setBrush(QColor(20, 20, 20, 155));
            background->setZValue(164.0);
            background->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            background->setVisible(false);

            auto* text = new QGraphicsSimpleTextItem();
            text->setFont(labelFont);
            text->setBrush(QBrush(QColor(255, 255, 255, 245)));
            text->setPen(Qt::NoPen);
            text->setZValue(165.0);
            text->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
            text->setVisible(false);

            entry.spanLabels.push_back({background, text});
            items.push_back(background);
            items.push_back(text);
        }
        viewer->setOverlayGroup(overlayKey, items);
        return &entry;
    };

    for (size_t i = 0; i < _stripViewers.size(); ++i) {
        auto* viewer = _stripViewers[i].data();
        if (!viewer) {
            continue;
        }
        const std::string key = i == 0 ? _generatedViews.lineSurfaceName
                                       : _generatedViews.lineSideSliceName;
        auto* entry = ensureStripItems(i, viewer, key);
        auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        if (!entry || !quad) {
            continue;
        }

        QPen currentPen(markerColorForState(currentLineMarkerState()));
        currentPen.setWidthF(2.0);
        currentPen.setCapStyle(Qt::RoundCap);
        entry->currentLine->setPen(currentPen);

        const QPointF currentScenePoint =
            vc3d::line_annotation::generatedStripLinePositionToScene(viewer,
                                                                      quad,
                                                                      _currentLinePosition);
        auto* view = viewer->graphicsView();
        auto* viewport = view ? view->viewport() : nullptr;
        if (std::isfinite(currentScenePoint.x()) &&
            std::isfinite(currentScenePoint.y()) &&
            view &&
            viewport &&
            viewport->width() > 0 &&
            viewport->height() > 0) {
            const QRect viewportRect = viewport->rect();
            const QPointF topScene =
                view->mapToScene(QPoint(viewportRect.center().x(), viewportRect.top()));
            const QPointF bottomScene =
                view->mapToScene(QPoint(viewportRect.center().x(), viewportRect.bottom()));
            QPainterPath path;
            path.moveTo(currentScenePoint.x(), topScene.y());
            path.lineTo(currentScenePoint.x(), bottomScene.y());
            entry->currentLine->setPath(path);
            entry->currentLine->setVisible(true);
        } else {
            entry->currentLine->setVisible(false);
        }

        if (updateSpanLabels) {
            const bool haveViewport = view &&
                                      viewport &&
                                      viewport->width() > 0 &&
                                      viewport->height() > 0;
            for (size_t labelIndex = 0; labelIndex < entry->spanLabels.size(); ++labelIndex) {
                auto& labelItems = entry->spanLabels[labelIndex];
                auto* background = labelItems.background;
                auto* text = labelItems.text;
                if (!background || !text) {
                    continue;
                }
                background->setVisible(false);
                text->setVisible(false);
                if (labelIndex >= _generatedViews.spanAlignmentMetrics.size() || !haveViewport) {
                    continue;
                }

                const auto& metric = _generatedViews.spanAlignmentMetrics[labelIndex];
                if (!shouldShowSpanAlignmentMetric(metric)) {
                    continue;
                }
                const auto centerLinePosition =
                    vc3d::line_annotation::generatedSpanAlignmentMetricCenterLinePosition(metric);
                if (!centerLinePosition) {
                    continue;
                }
                const QPointF centerScenePoint =
                    vc3d::line_annotation::generatedStripLinePositionToScene(
                        viewer,
                        quad,
                        *centerLinePosition);
                if (!std::isfinite(centerScenePoint.x()) ||
                    !std::isfinite(centerScenePoint.y())) {
                    continue;
                }

                const QRect viewportRect = viewport->rect();
                const int labelViewportY =
                    std::max(viewportRect.top(), viewportRect.bottom() - 18);
                const QPointF labelYScene =
                    view->mapToScene(QPoint(viewportRect.center().x(), labelViewportY));
                const QString label = spanAlignmentMetricText(metric);
                if (label.isEmpty()) {
                    continue;
                }
                const bool highlighted = shouldHighlightSpanAlignmentMetric(metric);
                const QColor textColor = highlighted
                    ? QColor(150, 0, 0, 245)
                    : QColor(255, 255, 255, 245);
                const QColor backgroundColor = highlighted
                    ? QColor(255, 232, 232, 235)
                    : QColor(20, 20, 20, 155);
                text->setText(label);
                text->setBrush(QBrush(textColor));
                text->setToolTip(spanAlignmentMetricToolTip(metric));
                const QRectF textRect = text->boundingRect();
                const QPointF textPos(centerScenePoint.x() - textRect.width() * 0.5,
                                      labelYScene.y() - textRect.height());
                text->setPos(textPos);

                background->setBrush(QBrush(backgroundColor));
                background->setRect(textRect.adjusted(-4.0, -2.0, 4.0, 2.0));
                background->setPos(textPos);
                background->setToolTip(text->toolTip());
                background->setVisible(true);
                text->setVisible(true);
            }
        }
    }

    if (_sideCutViewer && _generatedViews.sideCutSurface) {
        // Draw the full line on the side view by projecting each line point onto the plane and
        // connecting consecutive points (linear interpolation), in addition to the current-position
        // marker and nearby control points from the cross-slice overlay.
        GeneratedOverlay sideOverlay = zSliceOverlay(_currentLinePosition,
                                                     true,
                                                     _sideCutViewer,
                                                     _generatedViews.sideCutSurface.get());
        sideOverlay.linePoints = _generatedViews.linePoints;
        // Highlight the live cursor position on the line. The cross-slice overlay's emphasized
        // marker otherwise sits at the static focus/seed point; override it to the current
        // position so the highlight tracks the cursor as it moves along the line.
        const cv::Vec3f currentPoint = interpolatedLinePoint(_currentLinePosition);
        if (finitePoint(currentPoint)) {
            sideOverlay.pointMarker = currentPoint;
            sideOverlay.emphasizedPointMarker = true;
        }
        applyOverlayForViewer(dynamicStripOverlayKey(_generatedViews.sideCutName),
                              _sideCutViewer,
                              sideOverlay);
    }

    if (!updateCurrentCutOverlay || !_currentCutViewer) {
        return;
    }

    auto* viewer = _currentCutViewer.data();
    if (_fastCurrentCutOverlayItems.viewer != viewer ||
        !_fastCurrentCutOverlayItems.centerPoint ||
        !_fastCurrentCutOverlayItems.controlPoints ||
        !_fastCurrentCutOverlayItems.seedPoints ||
        !_fastCurrentCutOverlayItems.linkCandidatePoints ||
        !_fastCurrentCutOverlayItems.branchControlPoints ||
        !_fastCurrentCutOverlayItems.pendingBranchControlPoints ||
        !_fastCurrentCutOverlayItems.fiberIntersections ||
        !_fastCurrentCutOverlayItems.linkCandidateFiberIntersections ||
        !_fastCurrentCutOverlayItems.branchLinkFiberIntersections ||
        !_fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections ||
        !_fastCurrentCutOverlayItems.fiberIntersectionConnectors) {
        viewer->clearOverlayGroup(kGeneratedDynamicCurrentCutOverlayKey);
        _fastCurrentCutOverlayItems = {};
        _fastCurrentCutOverlayItems.viewer = viewer;

        QPen centerPen(QColor(0, 245, 255, 245));
        centerPen.setWidthF(1.5);
        QBrush centerBrush(QColor(0, 245, 255, 210));
        QPen controlPen(QColor(255, 230, 0, 220));
        controlPen.setWidthF(1.5);
        QBrush controlBrush(QColor(255, 230, 0, 170));

        _fastCurrentCutOverlayItems.centerPoint = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.centerPoint->setPen(centerPen);
        _fastCurrentCutOverlayItems.centerPoint->setBrush(centerBrush);
        _fastCurrentCutOverlayItems.centerPoint->setZValue(153.0);

        _fastCurrentCutOverlayItems.controlPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.controlPoints->setPen(controlPen);
        _fastCurrentCutOverlayItems.controlPoints->setBrush(controlBrush);
        _fastCurrentCutOverlayItems.controlPoints->setZValue(160.0);

        _fastCurrentCutOverlayItems.seedPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.seedPoints->setPen(controlPen);
        _fastCurrentCutOverlayItems.seedPoints->setBrush(controlBrush);
        _fastCurrentCutOverlayItems.seedPoints->setZValue(161.0);

        QPen linkCandidatePen(QColor(60, 235, 120, 245));
        linkCandidatePen.setWidthF(2.0);
        QBrush linkCandidateBrush(QColor(60, 235, 120, 175));
        _fastCurrentCutOverlayItems.linkCandidatePoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.linkCandidatePoints->setPen(linkCandidatePen);
        _fastCurrentCutOverlayItems.linkCandidatePoints->setBrush(linkCandidateBrush);
        _fastCurrentCutOverlayItems.linkCandidatePoints->setZValue(163.0);

        QPen branchControlPen(QColor(210, 95, 255, 245));
        branchControlPen.setWidthF(2.0);
        QBrush branchControlBrush(QColor(210, 95, 255, 175));
        _fastCurrentCutOverlayItems.branchControlPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.branchControlPoints->setPen(branchControlPen);
        _fastCurrentCutOverlayItems.branchControlPoints->setBrush(branchControlBrush);
        _fastCurrentCutOverlayItems.branchControlPoints->setZValue(162.0);

        QPen pendingBranchControlPen(QColor(80, 150, 255, 245));
        pendingBranchControlPen.setWidthF(2.0);
        QBrush pendingBranchControlBrush(QColor(80, 150, 255, 175));
        _fastCurrentCutOverlayItems.pendingBranchControlPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.pendingBranchControlPoints->setPen(pendingBranchControlPen);
        _fastCurrentCutOverlayItems.pendingBranchControlPoints->setBrush(
            pendingBranchControlBrush);
        _fastCurrentCutOverlayItems.pendingBranchControlPoints->setZValue(162.5);

        QPen fiberIntersectionPen(QColor(255, 245, 75, 245));
        fiberIntersectionPen.setWidthF(1.25);
        fiberIntersectionPen.setCapStyle(Qt::FlatCap);
        _fastCurrentCutOverlayItems.fiberIntersections = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.fiberIntersections->setPen(fiberIntersectionPen);
        _fastCurrentCutOverlayItems.fiberIntersections->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.fiberIntersections->setZValue(168.0);

        QPen linkCandidateFiberIntersectionPen(QColor(60, 235, 120, 245));
        linkCandidateFiberIntersectionPen.setWidthF(1.75);
        linkCandidateFiberIntersectionPen.setCapStyle(Qt::FlatCap);
        _fastCurrentCutOverlayItems.linkCandidateFiberIntersections = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.linkCandidateFiberIntersections->setPen(
            linkCandidateFiberIntersectionPen);
        _fastCurrentCutOverlayItems.linkCandidateFiberIntersections->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.linkCandidateFiberIntersections->setZValue(168.5);

        QPen branchLinkFiberIntersectionPen(QColor(210, 95, 255, 245));
        branchLinkFiberIntersectionPen.setWidthF(1.75);
        branchLinkFiberIntersectionPen.setCapStyle(Qt::FlatCap);
        _fastCurrentCutOverlayItems.branchLinkFiberIntersections = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.branchLinkFiberIntersections->setPen(
            branchLinkFiberIntersectionPen);
        _fastCurrentCutOverlayItems.branchLinkFiberIntersections->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.branchLinkFiberIntersections->setZValue(168.25);

        QPen pendingBranchLinkFiberIntersectionPen(QColor(80, 150, 255, 245));
        pendingBranchLinkFiberIntersectionPen.setWidthF(1.75);
        pendingBranchLinkFiberIntersectionPen.setCapStyle(Qt::FlatCap);
        _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections =
            new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections->setPen(
            pendingBranchLinkFiberIntersectionPen);
        _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections->setZValue(168.3);

        QPen fiberIntersectionConnectorPen(QColor(255, 60, 180, 225));
        fiberIntersectionConnectorPen.setWidthF(1.4);
        _fastCurrentCutOverlayItems.fiberIntersectionConnectors = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.fiberIntersectionConnectors->setPen(
            fiberIntersectionConnectorPen);
        _fastCurrentCutOverlayItems.fiberIntersectionConnectors->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.fiberIntersectionConnectors->setZValue(164.0);

        viewer->setOverlayGroup(kGeneratedDynamicCurrentCutOverlayKey,
                                {_fastCurrentCutOverlayItems.centerPoint,
                                 _fastCurrentCutOverlayItems.controlPoints,
                                 _fastCurrentCutOverlayItems.seedPoints,
                                 _fastCurrentCutOverlayItems.linkCandidatePoints,
                                 _fastCurrentCutOverlayItems.branchControlPoints,
                                 _fastCurrentCutOverlayItems.pendingBranchControlPoints,
                                 _fastCurrentCutOverlayItems.fiberIntersections,
                                 _fastCurrentCutOverlayItems.linkCandidateFiberIntersections,
                                 _fastCurrentCutOverlayItems.branchLinkFiberIntersections,
                                 _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections,
                                 _fastCurrentCutOverlayItems.fiberIntersectionConnectors});
    }

    QPointF centerScenePoint;
    if (normalOffsetActive(_currentCutNormalOffsetVx)) {
        centerScenePoint = viewer->volumeToScene(interpolatedLinePoint(_currentLinePosition));
    } else {
        centerScenePoint = viewer->surfaceCoordsToScene(0.0f, 0.0f);
    }
    QPainterPath centerPath;
    if (std::isfinite(centerScenePoint.x()) && std::isfinite(centerScenePoint.y())) {
        centerPath.addEllipse(centerScenePoint, 2.5, 2.5);
    }
    _fastCurrentCutOverlayItems.centerPoint->setPath(centerPath);

    QPainterPath controlPath;
    QPainterPath seedPath;
    QPainterPath linkCandidatePath;
    QPainterPath branchControlPath;
    QPainterPath pendingBranchControlPath;
    const double lineRadius =
        std::max(0.5, (_viewerManager ? _viewerManager->zScrollSensitivity() : 1.0) * 0.5);
    const double lower = _currentLinePosition - lineRadius;
    const double upper = _currentLinePosition + lineRadius;
    const auto& indices = _generatedControlIndex.sortedControlIndices;
    const auto positionForIndex = [this](size_t controlIndex) {
        return _generatedViews.controlPoints[controlIndex].linePosition;
    };
    const auto lowerIt = std::lower_bound(
        indices.begin(),
        indices.end(),
        lower,
        [&positionForIndex](size_t controlIndex, double value) {
            return positionForIndex(controlIndex) < value;
        });
    for (auto it = lowerIt; it != indices.end(); ++it) {
        const size_t controlIndex = *it;
        if (controlIndex >= _generatedViews.controlPoints.size()) {
            continue;
        }
        const auto& control = _generatedViews.controlPoints[controlIndex];
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        if (control.linePosition > upper) {
            break;
        }
        if (!finitePoint(control.point)) {
            continue;
        }
        const QPointF scenePoint = viewer->volumeToScene(control.point);
        if (!std::isfinite(scenePoint.x()) || !std::isfinite(scenePoint.y())) {
            continue;
        }
        if (control.isLinkCandidate) {
            linkCandidatePath.addEllipse(scenePoint, control.isSeed ? 11.0 : 10.0,
                                         control.isSeed ? 11.0 : 10.0);
        } else if (control.hasPendingLinks) {
            pendingBranchControlPath.addEllipse(scenePoint, 12.0, 12.0);
        } else if (control.hasBranches) {
            branchControlPath.addEllipse(scenePoint, 12.0, 12.0);
        } else if (control.isSeed) {
            seedPath.addEllipse(scenePoint, 11.0, 11.0);
        } else {
            controlPath.addEllipse(scenePoint, 10.0, 10.0);
        }
    }
    _fastCurrentCutOverlayItems.controlPoints->setPath(controlPath);
    _fastCurrentCutOverlayItems.seedPoints->setPath(seedPath);
    _fastCurrentCutOverlayItems.linkCandidatePoints->setPath(linkCandidatePath);
    _fastCurrentCutOverlayItems.branchControlPoints->setPath(branchControlPath);
    _fastCurrentCutOverlayItems.pendingBranchControlPoints->setPath(pendingBranchControlPath);

    QPainterPath fiberIntersectionPath;
    QPainterPath linkCandidateFiberIntersectionPath;
    QPainterPath branchLinkFiberIntersectionPath;
    QPainterPath pendingBranchLinkFiberIntersectionPath;
    QPainterPath fiberIntersectionConnectorPath;
    auto* currentCutPlane = _generatedViews.currentCutSurface.get();
    const std::optional<float> intersectionThreshold =
        (currentCutPlane && !_generatedViews.fiberIntersections.empty())
            ? vc3d::line_annotation::generatedCrossSliceControlPointDistanceThreshold(viewer)
            : std::nullopt;
    if (intersectionThreshold) {
        constexpr qreal kIntersectionArm = 7.5;
        for (const auto& intersection : _generatedViews.fiberIntersections) {
            if (!finitePoint(intersection.point)) {
                continue;
            }
            const float distance = currentCutPlane->pointDist(intersection.point);
            if (!std::isfinite(distance) || std::abs(distance) > *intersectionThreshold) {
                continue;
            }
            const QPointF scenePoint = viewer->volumeToScene(intersection.point);
            if (!std::isfinite(scenePoint.x()) || !std::isfinite(scenePoint.y())) {
                continue;
            }
            if (intersection.connectorStart && finitePoint(*intersection.connectorStart)) {
                const QPointF connectorScene =
                    viewer->volumeToScene(*intersection.connectorStart);
                if (std::isfinite(connectorScene.x()) && std::isfinite(connectorScene.y())) {
                    fiberIntersectionConnectorPath.moveTo(connectorScene);
                    fiberIntersectionConnectorPath.lineTo(scenePoint);
                }
            }
            QPainterPath& path = intersection.isLinkCandidateFiber
                ? linkCandidateFiberIntersectionPath
                : (intersection.projectedBranchLink
                       ? (intersection.pendingBranchLink
                              ? pendingBranchLinkFiberIntersectionPath
                              : branchLinkFiberIntersectionPath)
                       : fiberIntersectionPath);
            path.moveTo(scenePoint + QPointF(-kIntersectionArm, -kIntersectionArm));
            path.lineTo(scenePoint + QPointF(kIntersectionArm, kIntersectionArm));
            path.moveTo(scenePoint + QPointF(-kIntersectionArm, kIntersectionArm));
            path.lineTo(scenePoint + QPointF(kIntersectionArm, -kIntersectionArm));
        }
    }
    _fastCurrentCutOverlayItems.fiberIntersections->setPath(fiberIntersectionPath);
    _fastCurrentCutOverlayItems.linkCandidateFiberIntersections->setPath(
        linkCandidateFiberIntersectionPath);
    _fastCurrentCutOverlayItems.branchLinkFiberIntersections->setPath(
        branchLinkFiberIntersectionPath);
    _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections->setPath(
        pendingBranchLinkFiberIntersectionPath);
    _fastCurrentCutOverlayItems.fiberIntersectionConnectors->setPath(
        fiberIntersectionConnectorPath);
}

void LineAnnotationDialog::rebuildGeneratedDynamicOverlays(bool updateCurrentCutOverlay,
                                                           bool updateSpanLabels)
{
    updateGeneratedDynamicOverlaysFast(updateCurrentCutOverlay, updateSpanLabels);
}

void LineAnnotationDialog::rebuildGeneratedOverlays(bool requestSideStripIntersections)
{
    if (_closing) {
        return;
    }
    rebuildGeneratedStaticStripOverlays();
    rebuildGeneratedDynamicOverlays();
    if (requestSideStripIntersections) {
        requestGeneratedSideStripIntersections();
    }
}

cv::Vec3f LineAnnotationDialog::interpolatedLinePoint(double linePosition) const
{
    return vc3d::line_annotation::interpolatedGeneratedLinePoint(_generatedViews.linePoints,
                                                                 linePosition);
}

cv::Vec3f LineAnnotationDialog::interpolatedLineTangent(double linePosition) const
{
    if (_generatedViews.linePoints.size() < 2) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    const double maxPosition = static_cast<double>(_generatedViews.linePoints.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    // Smooth, continuous tangent: a central difference of the (piecewise-linear) line over a
    // window around the cursor, rather than the raw difference of the two bracketing integer
    // points. The raw adjacent-point difference is piecewise-constant in linePosition and snaps
    // the cut-plane orientation at every integer crossing (the main source of the cross-section
    // "jumpiness"); averaging over +/-kTangentHalfWindow line indices removes that stepping while
    // staying locally faithful to the line direction. interpolatedLinePoint() is continuous, so
    // the result varies continuously with the cursor.
    constexpr double kTangentHalfWindow = 4.0;
    const double lo = std::max(0.0, linePosition - kTangentHalfWindow);
    const double hi = std::min(maxPosition, linePosition + kTangentHalfWindow);
    cv::Vec3f tangent = interpolatedLinePoint(hi) - interpolatedLinePoint(lo);
    if (cv::norm(tangent) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return normalizedOrNan(tangent);
}

cv::Vec3f LineAnnotationDialog::interpolatedLineUp(double linePosition, const cv::Vec3f& tangent) const
{
    if (_generatedViews.lineUpVectors.empty()) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }

    linePosition = std::clamp(linePosition,
                              0.0,
                              static_cast<double>(_generatedViews.lineUpVectors.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(_generatedViews.lineUpVectors.size()) - 1);
    cv::Vec3f lowerUp = _generatedViews.lineUpVectors[static_cast<size_t>(lower)];
    cv::Vec3f upperUp = _generatedViews.lineUpVectors[static_cast<size_t>(upper)];
    if (!finitePoint(lowerUp) || !finitePoint(upperUp) ||
        cv::norm(lowerUp) <= 1.0e-6f ||
        cv::norm(upperUp) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    if (lowerUp.dot(upperUp) < 0.0f) {
        upperUp *= -1.0f;
    }

    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    cv::Vec3f up = lowerUp * (1.0f - t) + upperUp * t;
    up -= tangent * up.dot(tangent);
    if (cv::norm(up) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return normalizedOrNan(up);
}

bool LineAnnotationDialog::updatePlaneSurface(PlaneSurface* plane, double linePosition) const
{
    if (!plane) {
        return false;
    }
    const cv::Vec3f origin = interpolatedLinePoint(linePosition);
    const cv::Vec3f tangent = interpolatedLineTangent(linePosition);
    const cv::Vec3f upHint = interpolatedLineUp(linePosition, tangent);
    if (!finitePoint(origin) || !finitePoint(tangent) || !finitePoint(upHint) ||
        cv::norm(tangent) <= 1.0e-6f ||
        cv::norm(upHint) <= 1.0e-6f) {
        return false;
    }
    if (_currentCutManualRotationActive &&
        _generatedViews.currentCutSurface &&
        plane == _generatedViews.currentCutSurface.get()) {
        const auto frame =
            vc3d::line_annotation::generatedCutFrameWithManualRotation(tangent,
                                                                       upHint,
                                                                       _currentCutManualRotation);
        if (!vc3d::line_annotation::generatedCutFrameIsOrthonormal(frame)) {
            return false;
        }
        plane->setFromNormalAndUp(origin, frame.normal, frame.vertical);
    } else {
        plane->setFromNormalAndUp(origin, tangent, upHint);
    }
    return true;
}

bool LineAnnotationDialog::computeSideFit(int center, cv::Vec3f& normal, cv::Vec3f& upHint) const
{
    const auto& linePoints = _generatedViews.linePoints;
    const int count = static_cast<int>(linePoints.size());
    if (count < 3) {
        return false;
    }
    center = std::clamp(center, 0, count - 1);
    const int first = std::max(0, center - kSideFitHalfWindow);
    const int last = std::min(count - 1, center + kSideFitHalfWindow);

    vc3d::fiber_slice::ControlSpanSelection span;
    span.firstLineIndex = static_cast<size_t>(first);
    span.lastLineIndex = static_cast<size_t>(last);
    span.samples.reserve(static_cast<size_t>(last - first + 1));
    cv::Vec3d centroid{0.0, 0.0, 0.0};
    for (int i = first; i <= last; ++i) {
        const cv::Vec3f& p = linePoints[static_cast<size_t>(i)];
        if (!finitePoint(p)) {
            continue;
        }
        const cv::Vec3d pd(p[0], p[1], p[2]);
        span.samples.push_back(pd);
        centroid += pd;
    }
    if (span.samples.size() < 3) {
        return false;
    }
    span.centroid = centroid * (1.0 / static_cast<double>(span.samples.size()));
    span.valid = true;

    // fitLeastSquaresPlane reads linePoints[firstLineIndex/lastLineIndex] to derive the in-plane
    // direction hint, so it needs the full polyline as cv::Vec3d. _linePointsd is the cached
    // double-precision copy built once when the views were generated.
    const auto fit = vc3d::fiber_slice::fitLeastSquaresPlane(span, _linePointsd);
    if (!fit.valid) {
        return false;
    }
    normal = cv::Vec3f(static_cast<float>(fit.normal[0]),
                       static_cast<float>(fit.normal[1]),
                       static_cast<float>(fit.normal[2]));
    upHint = cv::Vec3f(static_cast<float>(fit.upHint[0]),
                       static_cast<float>(fit.upHint[1]),
                       static_cast<float>(fit.upHint[2]));
    if (!finitePoint(normal) || !finitePoint(upHint) ||
        cv::norm(normal) <= 1.0e-6f || cv::norm(upHint) <= 1.0e-6f) {
        return false;
    }
    return true;
}

bool LineAnnotationDialog::updateSidePlaneSurface(PlaneSurface* plane, double linePosition)
{
    if (!plane) {
        return false;
    }
    const int count = static_cast<int>(_generatedViews.linePoints.size());
    if (count < 3) {
        return false;
    }

    // Continuous side-view orientation: fit the best-fit plane at the two integer window centers
    // straddling the cursor and interpolate between them by the fractional position. Each fit
    // depends only on the static line geometry, so we cache the two bracketing fits and recompute
    // a center only when the straddle shifts (typically once per integer crossing). This keeps
    // the orientation smooth -- no snapping at discrete window centers -- while staying cheap.
    const double pos = std::clamp(linePosition, 0.0, static_cast<double>(count - 1));
    const int lowerCenter = static_cast<int>(std::floor(pos));
    const int upperCenter = std::min(lowerCenter + 1, count - 1);
    const int wantCenters[2] = {lowerCenter, upperCenter};

    SideFit next[2];
    for (int slot = 0; slot < 2; ++slot) {
        next[slot].center = wantCenters[slot];
        bool reused = false;
        for (const SideFit& cached : _sideFitBracket) {
            if (cached.valid && cached.center == wantCenters[slot]) {
                next[slot] = cached;
                reused = true;
                break;
            }
        }
        if (!reused) {
            next[slot].valid =
                computeSideFit(wantCenters[slot], next[slot].normal, next[slot].upHint);
        }
    }
    _sideFitBracket[0] = next[0];
    _sideFitBracket[1] = next[1];

    cv::Vec3f normal;
    cv::Vec3f upHint;
    if (next[0].valid && next[1].valid) {
        cv::Vec3f normal1 = next[1].normal;
        cv::Vec3f upHint1 = next[1].upHint;
        // PCA normals/up hints are sign-arbitrary; align the upper fit to the lower one before
        // blending so interpolation doesn't cancel them out near a sign flip.
        if (next[0].normal.dot(normal1) < 0.0f) {
            normal1 = -normal1;
        }
        if (next[0].upHint.dot(upHint1) < 0.0f) {
            upHint1 = -upHint1;
        }
        const float t = static_cast<float>(pos - static_cast<double>(lowerCenter));
        normal = next[0].normal * (1.0f - t) + normal1 * t;
        upHint = next[0].upHint * (1.0f - t) + upHint1 * t;
    } else if (next[0].valid || next[1].valid) {
        const SideFit& fit = next[0].valid ? next[0] : next[1];
        normal = fit.normal;
        upHint = fit.upHint;
    } else {
        return false;
    }

    normal = normalizedOrNan(normal);
    if (!finitePoint(normal)) {
        return false;
    }
    upHint -= normal * upHint.dot(normal);
    upHint = normalizedOrNan(upHint);
    if (!finitePoint(upHint)) {
        return false;
    }

    // Keep the plane centered on the cursor position rather than a window centroid; the origin
    // slides smoothly along the line while the interpolated orientation tracks it continuously.
    const cv::Vec3f origin = interpolatedLinePoint(pos);
    if (!finitePoint(origin)) {
        return false;
    }
    plane->setFromNormalAndUp(origin, normal, upHint);
    return true;
}

QPointF LineAnnotationDialog::stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                                       QuadSurface* surface,
                                                       double linePosition) const
{
    if (!viewer || !surface) {
        return {};
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return {};
    }
    const cv::Vec2f scale = surface->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return {};
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}

void LineAnnotationDialog::keyPressEvent(QKeyEvent* event)
{
    if (handleKeyPress(event)) {
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void LineAnnotationDialog::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    updateOptimizationOverlayGeometry();
    updateFiberNameLabel();
}

bool LineAnnotationDialog::toggleCurrentCutFollowFromKeyboard()
{
    if (_currentCutFollowsStripMouse) {
        setCurrentLinePosition(snappedControlPointPosition(_currentLinePosition));
        setCurrentCutFollowsStripMouse(false);
    } else {
        setCurrentCutFollowsStripMouse(true);
    }
    return true;
}

bool LineAnnotationDialog::handleKeyPress(QKeyEvent* event)
{
    if (!event) {
        return false;
    }
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
        (void)toggleCurrentCutFollowFromKeyboard();
        event->accept();
        return true;
    }
    if (_viewerManager &&
        event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            _viewerManager->setZScrollSensitivity(_viewerManager->zScrollSensitivity() - 0.1);
            event->accept();
            return true;
        }
        if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            _viewerManager->setZScrollSensitivity(_viewerManager->zScrollSensitivity() + 0.1);
            event->accept();
            return true;
        }
    }
    if (event->key() == Qt::Key_Escape) {
        close();
        event->accept();
        return true;
    }
    return false;
}

bool LineAnnotationDialog::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _fiberNameLabel && event->type() == QEvent::Resize) {
        updateFiberNameLabel();
    }
    if (event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (handleKeyPress(keyEvent)) {
            return true;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void LineAnnotationDialog::updateOptimizationOverlayGeometry()
{
    if (!_optimizationOverlay || !centralWidget()) {
        return;
    }
    _optimizationOverlay->setGeometry(centralWidget()->rect());
    if (_optimizationOverlay->isVisible()) {
        _optimizationOverlay->raise();
    }
}

void LineAnnotationDialog::updateFiberNameLabel()
{
    if (!_fiberNameLabel) {
        return;
    }

    const QString fullName = _fiberDisplayName.trimmed();
    if (fullName.isEmpty()) {
        _fiberNameLabel->clear();
        _fiberNameLabel->setToolTip(QString());
        return;
    }

    _fiberNameLabel->setText(vc3d::adaptFiberNameToWidth(
        fullName,
        _fiberNameLabel->fontMetrics(),
        _fiberNameLabel->contentsRect().width()));
    _fiberNameLabel->setToolTip(fullName);
}

void LineAnnotationDialog::restoreWindowGeometry()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QByteArray savedGeometry =
        settings.value(vc3d::settings::line_annotation::GEOMETRY).toByteArray();
    if (savedGeometry.isEmpty()) {
        return;
    }
    _restoredWindowGeometry = restoreGeometry(savedGeometry);
    if (!_restoredWindowGeometry) {
        settings.remove(vc3d::settings::line_annotation::GEOMETRY);
        settings.sync();
    }
}

void LineAnnotationDialog::saveWindowGeometry() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::line_annotation::GEOMETRY, saveGeometry());
    settings.sync();
}

void LineAnnotationDialog::restoreGeneratedViewStateSettings()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _savedOuterSplitterSizes =
        splitterSizesFromVariant(settings.value(
            vc3d::settings::line_annotation::OUTER_SPLITTER_SIZES));
    _savedTopSplitterSizes =
        splitterSizesFromVariant(settings.value(
            vc3d::settings::line_annotation::TOP_SPLITTER_SIZES));
    _savedStripSplitterSizes =
        splitterSizesFromVariant(settings.value(
            vc3d::settings::line_annotation::STRIP_SPLITTER_SIZES));

    if (const auto zoom =
            zoomFromVariant(settings.value(
                vc3d::settings::line_annotation::CURRENT_CUT_ZOOM))) {
        _savedCurrentCutZoom = *zoom;
        _haveSavedCurrentCutZoom = true;
    }
    if (const auto zoom =
            zoomFromVariant(settings.value(
                vc3d::settings::line_annotation::SIDE_CUT_ZOOM))) {
        _savedSideCutZoom = *zoom;
        _haveSavedSideCutZoom = true;
    }
    _savedStripZooms =
        zoomsFromVariant(settings.value(
            vc3d::settings::line_annotation::STRIP_ZOOMS));
}

void LineAnnotationDialog::saveGeneratedViewStateSettings()
{
    if (!_hasGeneratedViews) {
        return;
    }

    if (_generatedOuterSplitter) {
        _savedOuterSplitterSizes = _generatedOuterSplitter->sizes();
    }
    if (_generatedTopSplitter) {
        _savedTopSplitterSizes = _generatedTopSplitter->sizes();
    }
    if (_generatedStripSplitter) {
        _savedStripSplitterSizes = _generatedStripSplitter->sizes();
    }

    _haveSavedCurrentCutZoom = false;
    if (_currentCutViewer) {
        _savedCurrentCutZoom = _currentCutViewer->cameraState().scale;
        _haveSavedCurrentCutZoom = finiteZoom(_savedCurrentCutZoom);
    }
    _haveSavedSideCutZoom = false;
    if (_sideCutViewer) {
        _savedSideCutZoom = _sideCutViewer->cameraState().scale;
        _haveSavedSideCutZoom = finiteZoom(_savedSideCutZoom);
    }
    _savedStripZooms.clear();
    _savedStripZooms.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (!viewer) {
            continue;
        }
        const float zoom = viewer->cameraState().scale;
        if (finiteZoom(zoom)) {
            _savedStripZooms.push_back(zoom);
        }
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::line_annotation::OUTER_SPLITTER_SIZES,
                      splitterSizesToVariantList(_savedOuterSplitterSizes));
    settings.setValue(vc3d::settings::line_annotation::TOP_SPLITTER_SIZES,
                      splitterSizesToVariantList(_savedTopSplitterSizes));
    settings.setValue(vc3d::settings::line_annotation::STRIP_SPLITTER_SIZES,
                      splitterSizesToVariantList(_savedStripSplitterSizes));
    if (_haveSavedCurrentCutZoom) {
        settings.setValue(vc3d::settings::line_annotation::CURRENT_CUT_ZOOM,
                          _savedCurrentCutZoom);
    } else {
        settings.remove(vc3d::settings::line_annotation::CURRENT_CUT_ZOOM);
    }
    if (_haveSavedSideCutZoom) {
        settings.setValue(vc3d::settings::line_annotation::SIDE_CUT_ZOOM,
                          _savedSideCutZoom);
    } else {
        settings.remove(vc3d::settings::line_annotation::SIDE_CUT_ZOOM);
    }
    settings.setValue(vc3d::settings::line_annotation::STRIP_ZOOMS,
                      zoomsToVariantList(_savedStripZooms));
    settings.sync();
}
