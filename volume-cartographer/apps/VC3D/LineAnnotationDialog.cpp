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
#include <QAbstractSpinBox>
#include <QAction>
#include <QApplication>
#include <QBrush>
#include <QCloseEvent>
#include <QComboBox>
#include <QCursor>
#include <QEvent>
#include <QFont>
#include <QMenu>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QGraphicsPathItem>
#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <QInputDialog>
#include <QHideEvent>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPainterPath>
#include <QPen>
#include <QProgressBar>
#include <QRect>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSettings>
#include <QShortcut>
#include <QSizePolicy>
#include <QSplitter>
#include <QPushButton>
#include <QSpinBox>
#include <QWidgetAction>
#include <QVariant>
#include <QTimer>
#include <QToolButton>
#include <QVariantAnimation>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

// Half-width (in line-point indices) of the window used to least-squares-fit the side-view
// plane orientation around the cursor. The fit is interpolated between adjacent window centers
// so the orientation tracks the cursor continuously (see updateSidePlaneSurface).
constexpr float kCurrentCutRotationStepRadians = 3.14159265358979323846f / 36.0f;
constexpr bool kGeneratedLineAnnotationOverlaysEnabled = true;
constexpr char kGeneratedDynamicCurrentCutOverlayKey[] = "line-z-slice-current";
constexpr float kNominalGeneratedRowWidth = 900.0f;
constexpr float kNominalGeneratedRowHeight = 260.0f;
constexpr double kSpanMetricHighlightThresholdDegrees = 45.0;
constexpr double kNormalOffsetEpsilon = 1.0e-6;
// Slide range of the current-cut parallax ghost markers, in line-position units
// (one unit is one index step in the generated line points, not one voxel). A
// control point further away than this parks at the full offset.
constexpr double kGeneratedGhostSlideRangeLinePositions = 8.0;
// Full-offset distance of a ghost from its true landing spot, as a fraction of
// the visible scene width of the current cut viewer.
constexpr double kGeneratedGhostMaxOffsetViewportFraction = 0.35;
constexpr qreal kGeneratedGhostRadius = 8.0;
// Ghosts only show while the control point is within this multiple of the
// solid-marker window (lineRadius), so distant ones don't linger on screen.
constexpr double kGeneratedGhostVisibilityRadiusMultiplier = 10.0;
// Arrow-pan integrator tick (~60 Hz) and the dt cap that keeps a stalled event
// loop from teleporting the current position.
constexpr int kArrowPanTickMs = 16;
constexpr double kArrowPanMaximumStepSeconds = 0.1;
constexpr int kArrowPanSpeedIndicatorHideMs = 1200;

// Native Up/Down (and text editing) must keep working in the toolbar widgets,
// so the arrow handling stands down while one of them has the keyboard.
bool keyboardFocusIsTextEntry()
{
    QWidget* focus = QApplication::focusWidget();
    // QAbstractItemView covers a combo box's open popup, which takes focus and
    // navigates with Up/Down itself.
    return qobject_cast<QAbstractSpinBox*>(focus) != nullptr ||
           qobject_cast<QLineEdit*>(focus) != nullptr ||
           qobject_cast<QComboBox*>(focus) != nullptr ||
           qobject_cast<QAbstractItemView*>(focus) != nullptr;
}

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

    const cv::Size size = quad->gridSize();
    if (size.width <= 0 || size.height <= 0) {
        return camera;
    }

    const cv::Vec2d first = quad->gridToSurface({0.0, 0.0});
    const cv::Vec2d last = quad->gridToSurface(
        {static_cast<double>(std::max(0, size.width - 1)),
         static_cast<double>(std::max(0, size.height - 1))});
    const double extentX = std::abs(last[0] - first[0]);
    const double extentY = std::abs(last[1] - first[1]);
    if (!(extentX > 0.0) || !(extentY > 0.0)) {
        return camera;
    }
    constexpr double kPadding = 0.85;
    const double scaleX = static_cast<double>(kNominalGeneratedRowWidth) / extentX;
    const double scaleY = static_cast<double>(kNominalGeneratedRowHeight) / extentY;
    camera.scale = static_cast<float>(
        std::clamp(std::min(scaleX, scaleY) * kPadding, 0.01, 100000.0));
    return camera;
}

std::optional<cv::Vec2f> generatedStripSurfaceCenter(CChunkedVolumeViewer* viewer,
                                                     double linePosition,
                                                     const vc::lasagna::LineStripPositionMap* positionMap)
{
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad || !std::isfinite(linePosition)) {
        return std::nullopt;
    }
    const auto* points = quad->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const double gridColumn = positionMap && positionMap->valid()
        ? positionMap->originalPositionToStripGridColumn(linePosition)
        : linePosition;
    if (!std::isfinite(gridColumn)) {
        return std::nullopt;
    }
    const cv::Vec2d surfacePoint = quad->gridToSurface(
        {gridColumn, static_cast<double>(points->rows / 2)});
    return cv::Vec2f{static_cast<float>(surfacePoint[0]),
                     static_cast<float>(surfacePoint[1])};
}

std::optional<float> generatedStripScaleForLinePositionRange(
    CChunkedVolumeViewer* viewer,
    const std::optional<std::pair<double, double>>& range,
    const vc::lasagna::LineStripPositionMap* positionMap)
{
    if (!range) {
        return std::nullopt;
    }
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad || !std::isfinite(range->first) || !std::isfinite(range->second)) {
        return std::nullopt;
    }
    const double firstGrid = positionMap && positionMap->valid()
        ? positionMap->originalPositionToStripGridColumn(range->first)
        : range->first;
    const double secondGrid = positionMap && positionMap->valid()
        ? positionMap->originalPositionToStripGridColumn(range->second)
        : range->second;
    if (!std::isfinite(firstGrid) || !std::isfinite(secondGrid)) {
        return std::nullopt;
    }
    const double surfaceSpan = std::abs(
        quad->gridToSurface({secondGrid, 0.0})[0] -
        quad->gridToSurface({firstGrid, 0.0})[0]);
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
    if (metric.modeMarker == 'C' || metric.modeMarker == 'L' || metric.modeMarker == 'T')
        return true;
    using Kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind;
    if (metric.kind == Kind::NativeMeetingError) {
        return std::isfinite(metric.meetingErrorBaseVoxels);
    }
    if (metric.kind == Kind::NativeFailure) {
        return !metric.failureCode.empty();
    }
    return metric.pending ||
           !metric.error.empty() ||
           (metric.available && std::isfinite(metric.maxErrorDegrees));
}

bool shouldHighlightSpanAlignmentMetric(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    using Kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind;
    if (metric.kind == Kind::NativeFailure)
        return true;
    if (metric.kind == Kind::NativeMeetingError)
        return false;
    return metric.available &&
           std::isfinite(metric.maxErrorDegrees) &&
           metric.maxErrorDegrees > kSpanMetricHighlightThresholdDegrees;
}

QString spanAlignmentMetricText(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    using Kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind;
    QString value;
    if (metric.kind == Kind::NativeMeetingError) {
        if (std::isfinite(metric.meetingErrorBaseVoxels)) {
            value = QObject::tr("%1 vx").arg(
                QString::number(metric.meetingErrorBaseVoxels, 'f', 1));
        }
    } else if (metric.kind == Kind::Cspline) {
        value.clear();
    } else if (metric.available && std::isfinite(metric.maxErrorDegrees)) {
        value = QStringLiteral("%1%2")
            .arg(QString::number(std::llround(metric.maxErrorDegrees)))
            .arg(QChar(0x00b0));
    }
    QString firstLine(QChar(metric.modeMarker));
    if (!value.isEmpty())
        firstLine += QStringLiteral(" ") + value;
    if (!metric.message.empty())
        return firstLine + QStringLiteral("\n") + QString::fromStdString(metric.message);
    return firstLine;
}

QString spanAlignmentMetricToolTip(
    const vc3d::line_annotation::GeneratedSpanAlignmentMetric& metric)
{
    using Kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind;
    const QString status = metric.message.empty()
        ? QString{}
        : QObject::tr("Status: %1. ").arg(QString::fromStdString(metric.message));
    if (metric.kind == Kind::NativeMeetingError) {
        if (!std::isfinite(metric.meetingErrorBaseVoxels))
            return {};
        QString tooltip = QObject::tr("Trace meeting error: %1 base voxels")
            .arg(QString::number(metric.meetingErrorBaseVoxels, 'f', 2));
        if (std::isfinite(metric.meetingErrorRatio)) {
            tooltip += QObject::tr(" (%1% of selected trace length)")
                .arg(QString::number(metric.meetingErrorRatio * 100.0, 'f', 1));
        }
        if (!metric.meetingSource.empty()) {
            tooltip += QObject::tr(". Source: %1")
                .arg(QString::fromStdString(metric.meetingSource));
        }
        return status + tooltip;
    }
    if (metric.kind == Kind::NativeFailure) {
        QString tooltip = QObject::tr("Native fiber trace failed: %1")
            .arg(QString::fromStdString(metric.failureCode));
        if (!metric.failureDetail.empty()) {
            tooltip += QStringLiteral(". ") +
                QString::fromStdString(metric.failureDetail);
        }
        if (std::isfinite(metric.meetingErrorBaseVoxels)) {
            tooltip += QObject::tr(". Closest meeting: %1 base voxels")
                .arg(QString::number(metric.meetingErrorBaseVoxels, 'f', 2));
        }
        return status + tooltip;
    }
    if (metric.pending) {
        return status + QObject::tr("Sampling Lasagna normals.");
    }
    if (!metric.error.empty()) {
        return status + QString::fromStdString(metric.error);
    }
    if (metric.available && std::isfinite(metric.maxErrorDegrees)) {
        return status + QObject::tr("Max normal-alignment error: %1 degrees")
            .arg(QString::number(metric.maxErrorDegrees, 'f', 1));
    }
    return status;
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

constexpr int kOverviewBarHeightPx = 48;

// Schematic overview of the annotation shown above the cut views: a straight
// line with the control points, replacing the previous volume-rendered top
// strip (no rendering at all). Fixed height; the annotation compresses
// horizontally as it grows. Display-only except Ctrl+right-click on a control
// point, which invokes controlContextRequested with the dot's line position.
class LineAnnotationOverviewBar final : public QWidget
{
public:
    struct ControlDot {
        double linePosition = 0.0;
        QColor color;
        qreal radius = 4.5;
    };

    using QWidget::QWidget;

    std::function<void(double, QPoint)> controlContextRequested;

    void setLineData(size_t linePointCount, std::vector<ControlDot> dots)
    {
        _linePointCount = linePointCount;
        _dots = std::move(dots);
        update();
    }

    void setCurrentPosition(double position, const QColor& color)
    {
        _currentPosition = position;
        _currentColor = color;
        update();
    }

    std::optional<double> linePositionAtLocalX(qreal x) const
    {
        const qreal inner = innerWidth();
        if (_linePointCount < 2 || inner <= 0.0) {
            return std::nullopt;
        }
        const double t =
            std::clamp((x - kMarginPx) / static_cast<double>(inner), 0.0, 1.0);
        return t * static_cast<double>(_linePointCount - 1);
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), Qt::black);
        if (_linePointCount < 2) {
            return;
        }
        painter.setRenderHint(QPainter::Antialiasing, true);
        const qreal midY = height() * 0.5;
        painter.setPen(QPen(QColor(190, 190, 190), 2.0));
        painter.drawLine(QPointF(kMarginPx, midY),
                         QPointF(width() - kMarginPx, midY));

        if (std::isfinite(_currentPosition)) {
            painter.setPen(QPen(_currentColor, 2.0));
            const qreal x = xForLinePosition(_currentPosition);
            painter.drawLine(QPointF(x, 4.0), QPointF(x, height() - 4.0));
        }

        for (const ControlDot& dot : _dots) {
            if (!std::isfinite(dot.linePosition)) {
                continue;
            }
            const qreal x = xForLinePosition(dot.linePosition);
            painter.setPen(QPen(dot.color.darker(150), 1.0));
            painter.setBrush(dot.color);
            painter.drawEllipse(QPointF(x, midY), dot.radius, dot.radius);
        }
    }

    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::RightButton &&
            event->modifiers().testFlag(Qt::ControlModifier) &&
            controlContextRequested && _linePointCount >= 2) {
            constexpr qreal kHitRadiusPx = 8.0;
            const qreal x = event->position().x();
            const ControlDot* best = nullptr;
            qreal bestDistance = kHitRadiusPx;
            for (const ControlDot& dot : _dots) {
                if (!std::isfinite(dot.linePosition)) {
                    continue;
                }
                const qreal distance =
                    std::abs(xForLinePosition(dot.linePosition) - x);
                if (distance <= bestDistance) {
                    bestDistance = distance;
                    best = &dot;
                }
            }
            if (best) {
                controlContextRequested(best->linePosition,
                                        event->globalPosition().toPoint());
            }
        }
        event->accept();  // display-only otherwise
    }

private:
    static constexpr qreal kMarginPx = 8.0;

    qreal innerWidth() const { return width() - 2.0 * kMarginPx; }

    qreal xForLinePosition(double position) const
    {
        const double t = std::clamp(
            position / static_cast<double>(_linePointCount - 1), 0.0, 1.0);
        return kMarginPx + static_cast<qreal>(t) * innerWidth();
    }

    size_t _linePointCount = 0;
    std::vector<ControlDot> _dots;
    double _currentPosition = std::numeric_limits<double>::quiet_NaN();
    QColor _currentColor{0, 245, 255};
};

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

    // Popup menu on a toolbutton rather than a QMenuBar: the dialog can be
    // embedded as a workspace tab, where its own menu bar would stack under
    // the main window's.
    auto* annotationMenuButton = new QToolButton(buttonRow);
    annotationMenuButton->setObjectName(QStringLiteral("lineAnnotationMenuButton"));
    // Classic hamburger glyph; the arrow-style menu indicator would be
    // redundant next to it.
    annotationMenuButton->setText(QStringLiteral("☰"));
    annotationMenuButton->setToolTip(tr("Menu"));
    annotationMenuButton->setStyleSheet(
        QStringLiteral("QToolButton::menu-indicator { image: none; }"));
    annotationMenuButton->setPopupMode(QToolButton::InstantPopup);
    annotationMenuButton->installEventFilter(this);
    auto* annotationMenu = new QMenu(annotationMenuButton);
    annotationMenu->setToolTipsVisible(true);
    _autoReoptimizeAction = annotationMenu->addAction(tr("Auto-reoptimize"));
    _autoReoptimizeAction->setCheckable(true);
    _autoReoptimizeAction->setChecked(true);
    _autoReoptimizeAction->setToolTip(
        tr("Checked: re-optimize the line after every control-point edit.\n"
           "Unchecked: no optimization until \"Reinit reoptimization\" or close."));
    connect(_autoReoptimizeAction, &QAction::toggled, this, [this](bool checked) {
        emit reoptimizationModeChanged(checked ? ReoptimizationMode::AutoReoptimize
                                               : ReoptimizationMode::NoOptimization);
    });
    annotationMenu->addSeparator();
    _fullOptimizationAction = annotationMenu->addAction(tr("Reinit reoptimization"));
    _fullOptimizationAction->setEnabled(false);
    connect(_fullOptimizationAction, &QAction::triggered, this, [this]() {
        emit fullOptimizationRequested();
    });
    _showAsMeshAction = annotationMenu->addAction(tr("Show as mesh"));
    _showAsMeshAction->setEnabled(false);
    connect(_showAsMeshAction, &QAction::triggered, this, [this]() {
        emit showAsMeshRequested();
    });
    annotationMenu->addSeparator();
    _lasagnaDatasetMenu = annotationMenu->addMenu(tr("Lasagna dataset"));
    _lasagnaDatasetMenu->setToolTipsVisible(true);
    _lasagnaDatasetMenu->menuAction()->setToolTip(
        tr("Select the Lasagna dataset used for line annotation."));
    _fiberInferenceDatasetMenu = annotationMenu->addMenu(tr("Fiber dataset"));
    _fiberInferenceDatasetMenu->setToolTipsVisible(true);
    _fiberInferenceDatasetMenu->menuAction()->setToolTip(
        tr("Select the fiber inference dataset used for line annotation."));
    annotationMenu->addSeparator();
    // Length / Extrapolation live in the menu as embedded label+spinbox rows
    // (QWidgetAction); editing them does not close the menu. Spinbox edits are
    // uncommitted until the row's Apply button: the getters return the applied
    // values, and reopening the menu reverts any abandoned edit.
    const auto addSpinBoxMenuRow = [annotationMenu](const QString& label,
                                                    QSpinBox* spin) -> QPushButton* {
        auto* row = new QWidget(annotationMenu);
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(28, 3, 12, 3);
        rowLayout->setSpacing(8);
        rowLayout->addWidget(new QLabel(label, row));
        rowLayout->addStretch(1);
        spin->setParent(row);
        rowLayout->addWidget(spin);
        auto* apply = new QPushButton(QObject::tr("Apply"), row);
        apply->setEnabled(false);
        rowLayout->addWidget(apply);
        auto* action = new QWidgetAction(annotationMenu);
        action->setDefaultWidget(row);
        annotationMenu->addAction(action);
        return apply;
    };
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _appliedInitialCenterlineLengthVx =
            settings.value(vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX,
                           vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX_DEFAULT)
                .toInt();
        _appliedExtrapolationDistanceVx =
            settings.value(
                vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX,
                vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX_DEFAULT)
                .toInt();
    }
    _initialCenterlineLengthSpin = new QSpinBox;
    _initialCenterlineLengthSpin->setObjectName(
        QStringLiteral("lineAnnotationInitialCenterlineLengthSpinBox"));
    _initialCenterlineLengthSpin->setRange(100, 1000000);
    _initialCenterlineLengthSpin->setSingleStep(100);
    _initialCenterlineLengthSpin->setSuffix(tr(" vx"));
    _initialCenterlineLengthSpin->setToolTip(
        tr("Total length of a newly generated centerline, split equally around the seed."));
    _initialCenterlineLengthSpin->setValue(_appliedInitialCenterlineLengthVx);
    // Read back rather than trusting the setting: a stale or hand-edited value
    // outside the spinbox range would otherwise reach the optimizer unclamped
    // while the row displayed the clamped one.
    _appliedInitialCenterlineLengthVx = _initialCenterlineLengthSpin->value();
    auto* lengthApply = addSpinBoxMenuRow(tr("Length"), _initialCenterlineLengthSpin);
    connect(_initialCenterlineLengthSpin,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            [this, lengthApply](int value) {
                lengthApply->setEnabled(value != _appliedInitialCenterlineLengthVx);
            });
    connect(lengthApply, &QPushButton::clicked, this, [this, lengthApply]() {
        _appliedInitialCenterlineLengthVx = _initialCenterlineLengthSpin->value();
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::line_annotation::INITIAL_CENTERLINE_LENGTH_VX,
                          _appliedInitialCenterlineLengthVx);
        lengthApply->setEnabled(false);
    });
    _extrapolationDistanceSpin = new QSpinBox;
    _extrapolationDistanceSpin->setObjectName(
        QStringLiteral("lineAnnotationExtrapolationDistanceSpinBox"));
    _extrapolationDistanceSpin->setRange(0, 1000000);
    _extrapolationDistanceSpin->setSingleStep(100);
    _extrapolationDistanceSpin->setSuffix(tr(" vx"));
    _extrapolationDistanceSpin->setToolTip(
        tr("Distance generated beyond each outer control point."));
    _extrapolationDistanceSpin->setValue(_appliedExtrapolationDistanceVx);
    _appliedExtrapolationDistanceVx = _extrapolationDistanceSpin->value();
    auto* extrapolationApply =
        addSpinBoxMenuRow(tr("Extrapolation"), _extrapolationDistanceSpin);
    connect(_extrapolationDistanceSpin,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            [this, extrapolationApply](int value) {
                extrapolationApply->setEnabled(value != _appliedExtrapolationDistanceVx);
            });
    connect(extrapolationApply, &QPushButton::clicked, this, [this, extrapolationApply]() {
        _appliedExtrapolationDistanceVx = _extrapolationDistanceSpin->value();
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::line_annotation::EXTRAPOLATION_DISTANCE_VX,
                          _appliedExtrapolationDistanceVx);
        extrapolationApply->setEnabled(false);
        emit extrapolationDistanceChanged(_appliedExtrapolationDistanceVx);
    });
    // Reopening the menu discards uncommitted edits (setValue back to the
    // applied values also re-disables the Apply buttons via valueChanged).
    connect(annotationMenu, &QMenu::aboutToShow, this, [this, lengthApply, extrapolationApply]() {
        _initialCenterlineLengthSpin->setValue(_appliedInitialCenterlineLengthVx);
        _extrapolationDistanceSpin->setValue(_appliedExtrapolationDistanceVx);
        lengthApply->setEnabled(false);
        extrapolationApply->setEnabled(false);
    });
    annotationMenu->addSeparator();
    _mirrorCursorAction = annotationMenu->addAction(tr("Mirror cursor across panes"));
    _mirrorCursorAction->setCheckable(true);
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _mirrorCursorAction->setChecked(
            settings
                .value(vc3d::settings::line_annotation::MIRROR_CURSOR_ACROSS_PANES,
                       vc3d::settings::line_annotation::MIRROR_CURSOR_ACROSS_PANES_DEFAULT)
                .toBool());
    }
    _mirrorCursorAction->setToolTip(
        tr("Checked: hovering one generated pane draws the cursor cross in the other three.\n"
           "Unchecked: the cross stays in the hovered pane."));
    connect(_mirrorCursorAction, &QAction::toggled, this, [this](bool checked) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::line_annotation::MIRROR_CURSOR_ACROSS_PANES, checked);
        applyLinkedCursorMirroringToPanes();
    });
    _resetViewsAction = annotationMenu->addAction(tr("Reset views"));
    _resetViewsAction->setEnabled(false);
    connect(_resetViewsAction, &QAction::triggered, this, [this]() {
        resetGeneratedViews();
    });
    annotationMenuButton->setMenu(annotationMenu);
    buttonLayout->addWidget(annotationMenuButton);

    if (volumeSelectorFactory) {
        if (auto* volumeSelector = volumeSelectorFactory(buttonRow)) {
            volumeSelector->installEventFilter(this);
            buttonLayout->addWidget(volumeSelector);
        }
    }

    _fiberOptimizationCombo = new QComboBox(buttonRow);
    _fiberOptimizationCombo->setObjectName(
        QStringLiteral("lineAnnotationFiberOptimizationModeCombo"));
    _fiberOptimizationCombo->addItem(
        tr("Lasagna"),
        static_cast<int>(vc3d::line_annotation::FiberOptimizationMode::Lasagna));
    _fiberOptimizationCombo->addItem(
        tr("Fiber model"),
        static_cast<int>(vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d));
    installComboEventFilter(_fiberOptimizationCombo, this);
    buttonLayout->addWidget(_fiberOptimizationCombo);
    connect(_fiberOptimizationCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this](int) {
                emit fiberOptimizationModeChanged(fiberOptimizationMode());
            });

    rebuildDatasetMenus();

    auto* maxDistanceLabel = new QLabel(tr("Max extrap CP dist"), buttonRow);
    const QString maxDistanceTooltip = tr(
        "Maximum optimized-line arclength in base voxels for placing a control "
        "point beyond either outermost control point; 0 means unlimited.");
    maxDistanceLabel->setToolTip(maxDistanceTooltip);
    maxDistanceLabel->installEventFilter(this);
    buttonLayout->addWidget(maxDistanceLabel);
    _maxControlPointExtrapolationDistanceSpin = new QSpinBox(buttonRow);
    _maxControlPointExtrapolationDistanceSpin->setObjectName(
        QStringLiteral("lineAnnotationMaxControlDistanceSpinBox"));
    _maxControlPointExtrapolationDistanceSpin->setRange(0, 1000000);
    _maxControlPointExtrapolationDistanceSpin->setValue(0);
    _maxControlPointExtrapolationDistanceSpin->setSuffix(tr(" base vx"));
    _maxControlPointExtrapolationDistanceSpin->setSpecialValueText(tr("unlimited"));
    _maxControlPointExtrapolationDistanceSpin->setToolTip(maxDistanceTooltip);
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        _maxControlPointExtrapolationDistanceSpin->setValue(
            settings.value(vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX,
                           vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX_DEFAULT)
                .toInt());
    }
    _maxControlPointExtrapolationDistanceSpin->installEventFilter(this);
    buttonLayout->addWidget(_maxControlPointExtrapolationDistanceSpin);
    connect(_maxControlPointExtrapolationDistanceSpin,
            qOverload<int>(&QSpinBox::valueChanged),
            this,
            [this](int value) {
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(vc3d::settings::line_annotation::MAX_CONTROL_POINT_DISTANCE_VX,
                                  value);
                // The allowance also positions the keyboard pan's synthetic
                // boundary targets; rebase a running pan onto the new ones.
                rebaseArrowPanTargets();
                updateGeneratedDynamicOverlaysFast(false, false);
            });
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const double savedArrowPanSpeed =
            settings.value(vc3d::settings::line_annotation::ARROW_PAN_SPEED,
                           vc3d::settings::line_annotation::ARROW_PAN_SPEED_DEFAULT)
                .toDouble();
        _arrowPanCruiseSpeed =
            (std::isfinite(savedArrowPanSpeed) && savedArrowPanSpeed > 0.0)
                ? std::clamp(savedArrowPanSpeed,
                             vc3d::line_annotation::kGeneratedArrowPanMinimumSpeed,
                             vc3d::line_annotation::kGeneratedArrowPanMaximumSpeed)
                : vc3d::line_annotation::kGeneratedArrowPanDefaultSpeed;
    }
    // Belt and braces for focus loss: ActivationChange delivery varies by
    // window manager, so the application-state signal (which fires whenever the
    // whole app loses focus, e.g. Alt-Tab to another program) backs it up.
    connect(qGuiApp,
            &QGuiApplication::applicationStateChanged,
            this,
            [this](Qt::ApplicationState state) {
                if (state != Qt::ApplicationActive) {
                    stopArrowPanForFocusLoss();
                }
            });
    auto* tagsLabel = new QLabel(tr("Tags:"), buttonRow);
    tagsLabel->installEventFilter(this);
    buttonLayout->addWidget(tagsLabel);
    _tagRowWidget = new QWidget(buttonRow);
    _tagRowWidget->installEventFilter(this);
    _tagRowWidget->setStyleSheet(QStringLiteral(
        "QToolButton#lineAnnotationTagButton {"
        " border: 1px solid rgba(128, 128, 128, 140); border-radius: 9px;"
        " padding: 2px 10px; background: transparent; }"
        "QToolButton#lineAnnotationTagButton:checked {"
        " background-color: rgb(0, 190, 210); border-color: rgb(0, 190, 210);"
        " color: black; font-weight: 600; }"
        "QToolButton#lineAnnotationTagButton:disabled {"
        " border-color: rgba(128, 128, 128, 70); }"));
    _tagRowLayout = new QHBoxLayout(_tagRowWidget);
    _tagRowLayout->setContentsMargins(0, 0, 0, 0);
    _tagRowLayout->setSpacing(4);
    buttonLayout->addWidget(_tagRowWidget);
    setFiberTags({}, {}, false);
    // The side-strip intersection query still runs (it feeds the fiber-
    // intersection X markers); we just no longer show a progress indicator.
    // _sideStripIntersectionProgress stays null and its setters no-op.
    _fiberNameLabel = new QLabel(buttonRow);
    _fiberNameLabel->setObjectName(QStringLiteral("lineAnnotationFiberNameLabel"));
    _fiberNameLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    _fiberNameLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _fiberNameLabel->setMinimumWidth(0);
    _fiberNameLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    _fiberNameLabel->installEventFilter(this);
    buttonLayout->addWidget(_fiberNameLabel, 1);
    _layout->addWidget(buttonRow, 0);
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

LineAnnotationDialog::ReoptimizationMode LineAnnotationDialog::reoptimizationMode() const
{
    if (_autoReoptimizeAction && !_autoReoptimizeAction->isChecked()) {
        return ReoptimizationMode::NoOptimization;
    }
    return ReoptimizationMode::AutoReoptimize;
}

int LineAnnotationDialog::initialCenterlineLengthVx() const
{
    return _appliedInitialCenterlineLengthVx;
}

int LineAnnotationDialog::extrapolationDistanceVx() const
{
    return _appliedExtrapolationDistanceVx;
}

vc3d::line_annotation::FiberOptimizationMode
LineAnnotationDialog::fiberOptimizationMode() const
{
    if (!_fiberOptimizationCombo) {
        return vc3d::line_annotation::FiberOptimizationMode::Lasagna;
    }
    return static_cast<vc3d::line_annotation::FiberOptimizationMode>(
        _fiberOptimizationCombo->currentData().toInt());
}

void LineAnnotationDialog::setFiberOptimizationMode(
    vc3d::line_annotation::FiberOptimizationMode mode)
{
    if (!_fiberOptimizationCombo) {
        return;
    }
    const QSignalBlocker blocker(_fiberOptimizationCombo);
    const int index = _fiberOptimizationCombo->findData(static_cast<int>(mode));
    if (index >= 0) {
        _fiberOptimizationCombo->setCurrentIndex(index);
    }
}

void LineAnnotationDialog::setLasagnaDatasetOptions(
    std::vector<std::pair<std::string, std::string>> options,
    const std::string& selectedLocation)
{
    _lasagnaDatasetOptions = std::move(options);
    _selectedLasagnaDatasetLocation = selectedLocation;
    rebuildDatasetMenus();
}

void LineAnnotationDialog::setFiberInferenceDatasetOptions(
    std::vector<std::pair<std::string, std::string>> options,
    const std::string& selectedLocation)
{
    _fiberInferenceDatasetOptions = std::move(options);
    _selectedFiberInferenceDatasetLocation = selectedLocation;
    rebuildDatasetMenus();
}

void LineAnnotationDialog::rebuildDatasetMenus()
{
    auto populateMenu =
        [this](QMenu* menu,
               const std::vector<std::pair<std::string, std::string>>& options,
               const std::string& selected,
               bool fiberMenu) {
            if (!menu) {
                return;
            }
            menu->clear();
            if (options.empty()) {
                auto* action = menu->addAction(fiberMenu
                    ? tr("No fiber datasets attached")
                    : tr("No Lasagna datasets attached"));
                action->setEnabled(false);
                return;
            }
            for (const auto& [location, label] : options) {
                auto* action = menu->addAction(QString::fromStdString(label));
                action->setCheckable(true);
                action->setChecked(location == selected);
                action->setToolTip(QString::fromStdString(location));
                connect(action, &QAction::triggered, this, [this, location, fiberMenu]() {
                    if (fiberMenu) {
                        emit fiberInferenceDatasetSelectionChanged(location);
                    } else {
                        emit lasagnaDatasetSelectionChanged(location);
                    }
                });
            }
        };
    populateMenu(_lasagnaDatasetMenu,
                 _lasagnaDatasetOptions,
                 _selectedLasagnaDatasetLocation,
                 false);
    populateMenu(_fiberInferenceDatasetMenu,
                 _fiberInferenceDatasetOptions,
                 _selectedFiberInferenceDatasetLocation,
                 true);
}

int LineAnnotationDialog::maxControlPointExtrapolationDistanceVx() const
{
    return _maxControlPointExtrapolationDistanceSpin
        ? _maxControlPointExtrapolationDistanceSpin->value()
        : 0;
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
    // Rebase (not cancel) a running keyboard pan onto the new control-point
    // set: a deleted target is re-selected from what remains, while benign
    // refreshes leave the pan running.
    rebaseArrowPanTargets();
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
    bool requestSideStripIntersections,
    std::vector<GeneratedSpanAlignmentMetric> spanAlignmentMetrics)
{
    if (_closing || !_hasGeneratedViews) {
        return;
    }
    vc3d::line_annotation::replaceGeneratedBranchOverlayData(
        _generatedViews,
        std::move(controlPoints),
        std::move(branchLinePoints),
        std::move(branchLinks),
        std::move(spanAlignmentMetrics));
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    // Same rebase as setGeneratedControlPoints: this setter also delivers
    // asynchronous refreshes (span metrics), so cancelling here would kill
    // healthy pans; re-selecting the targets handles deleted ones.
    rebaseArrowPanTargets();
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

void LineAnnotationDialog::setOptimizationBusy(bool busy, bool blockInput)
{
    _optimizationBusy = busy;
    if (_fiberOptimizationCombo) {
        _fiberOptimizationCombo->setEnabled(!busy);
    }
    if (_extrapolationDistanceSpin) {
        _extrapolationDistanceSpin->setEnabled(!busy);
    }
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
    if (!_optimizationBadge) {
        // Passive counterpart of the overlay: a top-center pill that never
        // takes input, for solves the user may keep editing through.
        auto* badge = new QLabel(tr("Optimizing…"), content);
        badge->setObjectName(QStringLiteral("lineAnnotationOptimizationBadge"));
        badge->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        badge->setAttribute(Qt::WA_StyledBackground, true);
        badge->setAlignment(Qt::AlignCenter);
        badge->setStyleSheet(QStringLiteral(
            "#lineAnnotationOptimizationBadge {"
            " background-color: rgba(32, 32, 32, 190); color: white;"
            " font-weight: 600; border-radius: 4px; padding: 3px 12px; }"));
        badge->hide();
        _optimizationBadge = badge;
    }
    updateOptimizationOverlayGeometry();
    const bool showOverlay = busy && blockInput;
    const bool showBadge = busy && !blockInput;
    _optimizationOverlay->setVisible(showOverlay);
    _optimizationBadge->setVisible(showBadge);
    if (showOverlay) {
        _optimizationOverlay->raise();
    }
    if (showBadge) {
        _optimizationBadge->raise();
    }
}

void LineAnnotationDialog::setOptimizationStatus(bool optimized)
{
    _optimizationStatusOptimized = optimized;
    updateOptimizationStatusIndicator();
}

void LineAnnotationDialog::setFiberDisplayName(const QString& name)
{
    _fiberDisplayName = name;
    updateFiberNameLabel();
}

void LineAnnotationDialog::setFiberHvTag(const QString& tag)
{
    _fiberHvTag = tag.trimmed();
    updateFiberNameLabel();
}

void LineAnnotationDialog::setFiberTags(const std::vector<std::string>& knownTags,
                                        const std::vector<std::string>& activeTags,
                                        bool enabled)
{
    if (!_tagRowWidget || !_tagRowLayout) {
        return;
    }
    while (auto* item = _tagRowLayout->takeAt(0)) {
        // deleteLater, not delete: a rebuild can be triggered from a slot chain
        // that started in one of these buttons' own signals.
        if (auto* widget = item->widget()) {
            widget->hide();
            widget->deleteLater();
        }
        delete item;
    }

    const QString disabledToolTip =
        tr("Tags become editable once the fiber has been saved.");
    for (const auto& tag : knownTags) {
        const QString tagText = QString::fromStdString(tag);
        const bool active = std::find(activeTags.begin(), activeTags.end(), tag) !=
                            activeTags.end();
        auto* button = new QToolButton(_tagRowWidget);
        button->setObjectName(QStringLiteral("lineAnnotationTagButton"));
        button->setText(active ? QStringLiteral("✓ ") + tagText : tagText);
        button->setCheckable(true);
        button->setChecked(active);
        button->setEnabled(enabled);
        button->setToolTip(enabled
            ? tr("Toggle tag \"%1\" on the current fiber").arg(tagText)
            : disabledToolTip);
        button->installEventFilter(this);
        connect(button, &QToolButton::toggled, this, [this, tagText](bool checked) {
            // Deferred: the handler rebuilds this row, which must not delete the
            // button while its own toggled signal is still on the stack.
            QTimer::singleShot(0, this, [this, tagText, checked]() {
                emit fiberTagChangeRequested(tagText, checked);
            });
        });
        _tagRowLayout->addWidget(button);
    }

    auto* addButton = new QToolButton(_tagRowWidget);
    addButton->setObjectName(QStringLiteral("lineAnnotationAddTagButton"));
    addButton->setText(QStringLiteral("+"));
    addButton->setEnabled(enabled);
    addButton->setToolTip(enabled ? tr("Add a new tag to the current fiber")
                                  : disabledToolTip);
    addButton->installEventFilter(this);
    connect(addButton, &QToolButton::clicked, this, [this]() {
        const QString tag =
            QInputDialog::getText(this, tr("New tag"), tr("Tag name:")).trimmed();
        if (!tag.isEmpty()) {
            QTimer::singleShot(0, this, [this, tag]() {
                emit fiberTagChangeRequested(tag, true);
            });
        }
    });
    _tagRowLayout->addWidget(addButton);
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
        cancelArrowPan();
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

    if (_showAsMeshAction) {
        _showAsMeshAction->setEnabled(false);
    }
    if (_fullOptimizationAction) {
        _fullOptimizationAction->setEnabled(false);
    }
    if (_resetViewsAction) {
        _resetViewsAction->setEnabled(false);
    }

    clearGeneratedOverlayRefreshConnections();
    cancelArrowPan();
    // Drop all viewer references BEFORE deleting the widgets: destruction
    // delivers events through our eventFilter, which must not dereference a
    // half-destroyed viewer (QPointers only clear once ~QObject runs).
    _panes.clear();
    _stripViewers.clear();
    _overviewBar = nullptr;
    _currentCutOverlaySwapPending = false;
    _sideCutOverlaySwapPending = false;
    _stripOverlaySwapPending.clear();
    _pendingPlacementFocus.reset();
    clearFastGeneratedOverlayItemRefs();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
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
    if (_showAsMeshAction) {
        _showAsMeshAction->setEnabled(ok);
    }
    if (_fullOptimizationAction) {
        _fullOptimizationAction->setEnabled(ok);
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
            if (_closing) {
                return;
            }
            // Every update bumps the generation, including ones coalesced into
            // an already-queued callback, so a landing pass can record exactly
            // which updates its full rebuild covered.
            ++_generatedOverlayRefreshGeneration;
            if (_generatedOverlayRefreshQueued) {
                return;
            }
            _generatedOverlayRefreshQueued = true;
            QTimer::singleShot(16, this, [this]() {
                if (_closing) {
                    return;
                }
                _generatedOverlayRefreshQueued = false;
                if (_generatedOverlayRefreshGeneration ==
                    _generatedOverlayRefreshCoveredGeneration) {
                    // A landing's rebuildGeneratedOverlays(true) already
                    // covered every update this callback was queued for
                    // (including the intersection request whose preparation is
                    // the expensive part). Updates arriving after the landing
                    // rebuild bump the generation, so they are never skipped.
                    return;
                }
                if (_arrowPanDirection != 0) {
                    // During a keyboard pan every tick already rebuilds the
                    // dynamic overlays (via setCurrentLinePosition), so only
                    // the static strip overlays need to track the scrolling
                    // camera here. The side-strip intersection request (which
                    // clones geometry and snapshots+hashes every fiber per
                    // call) waits for the landing's full refresh.
                    rebuildGeneratedStaticStripOverlays();
                    return;
                }
                rebuildGeneratedOverlays();
            });
        }));
}

void LineAnnotationDialog::connectLinkedCursorMirroring(
    std::vector<QPointer<CChunkedVolumeViewer>> panes)
{
    _linkedCursorPanes = std::move(panes);
    for (const auto& panePtr : _linkedCursorPanes) {
        auto* pane = panePtr.data();
        if (!pane) {
            continue;
        }
        pane->setLinkedCursorAlwaysEnabled(true);
        _generatedOverlayRefreshConnections.push_back(connect(
            pane,
            &CChunkedVolumeViewer::sendMouseMoveVolume,
            this,
            [this, pane](cv::Vec3f volumePoint, Qt::MouseButtons, Qt::KeyboardModifiers, QPointF) {
                // Off-surface hovers emit non-finite positions; mirror those
                // as "no point" instead of a NaN readout.
                const bool finite = std::isfinite(volumePoint[0]) &&
                                    std::isfinite(volumePoint[1]) &&
                                    std::isfinite(volumePoint[2]);
                requestLinkedCursorMirror(
                    pane, finite ? std::optional<cv::Vec3f>(volumePoint) : std::nullopt);
            }));
        _generatedOverlayRefreshConnections.push_back(connect(
            pane->graphicsView(),
            &CVolumeViewerView::sendMouseLeftView,
            this,
            [this, pane]() { requestLinkedCursorMirror(pane, std::nullopt); }));
    }
    applyLinkedCursorMirroringToPanes();
}

void LineAnnotationDialog::applyLinkedCursorMirroringToPanes()
{
    const bool enabled = !_mirrorCursorAction || _mirrorCursorAction->isChecked();
    for (const auto& panePtr : _linkedCursorPanes) {
        auto* pane = panePtr.data();
        if (!pane) {
            continue;
        }
        pane->setLinkedCursorMirroringSuppressed(!enabled);
        if (!enabled) {
            // Same clear-on-disable as the global toggle: drops a mirrored cross
            // that is on screen right now, and with it the shared position
            // readout, which would otherwise sit frozen at the last hover.
            pane->setLinkedCursorVolumePoint(std::nullopt);
        }
    }
}

void LineAnnotationDialog::requestLinkedCursorMirror(CChunkedVolumeViewer* source,
                                                     const std::optional<cv::Vec3f>& point)
{
    _linkedCursorSource = source;
    _pendingLinkedCursorPoint = point;
    if (!_linkedCursorMirrorTimer) {
        _linkedCursorMirrorTimer = new QTimer(this);
        _linkedCursorMirrorTimer->setSingleShot(true);
        _linkedCursorMirrorTimer->setInterval(16);  // ~one global render tick
        connect(_linkedCursorMirrorTimer, &QTimer::timeout, this, [this]() {
            if (_closing) {
                return;
            }
            for (const auto& panePtr : _linkedCursorPanes) {
                auto* pane = panePtr.data();
                if (pane && pane != _linkedCursorSource.data()) {
                    pane->setLinkedCursorVolumePoint(_pendingLinkedCursorPoint);
                }
            }
        });
    }
    if (!_linkedCursorMirrorTimer->isActive()) {
        _linkedCursorMirrorTimer->start();
    }
}

void LineAnnotationDialog::clearGeneratedOverlayRefreshConnections()
{
    for (const auto& connection : _generatedOverlayRefreshConnections) {
        QObject::disconnect(connection);
    }
    _generatedOverlayRefreshConnections.clear();
    // Drop any pending cursor mirror along with the connections that feed it,
    // so it can't fire across a pane rebuild and stamp a pre-rebuild point
    // onto the new panes.
    if (_linkedCursorMirrorTimer) {
        _linkedCursorMirrorTimer->stop();
    }
    _linkedCursorPanes.clear();
    _linkedCursorSource.clear();
    _pendingLinkedCursorPoint.reset();
    _generatedOverlayRefreshQueued = false;
    _generatedOverlayRefreshGeneration = 0;
    _generatedOverlayRefreshCoveredGeneration = 0;
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
    GeneratedViews views,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_layout || views.linePoints.empty() ||
        views.lineUpVectors.size() != views.linePoints.size() ||
        !views.lineSurface || !views.lineSideSlice ||
        !views.currentCutSurface || !views.sideCutSurface) {
        return false;
    }

    // In-place update: every control-point placement re-materializes the views,
    // and the controller re-registers the new surfaces in CState under the SAME
    // names before calling us, so the live viewers have already adopted them via
    // surfaceChanged (view centers preserved). Rebuilding the panes -- the
    // previous behavior -- destroyed and recreated all four rendered viewers,
    // producing a
    // visible blank/jump on every placement.
    if (_hasGeneratedViews && _currentCutViewer && _sideCutViewer &&
        _stripViewers.size() == 2 && _stripViewers[0] && _stripViewers[1] &&
        _generatedViews.currentCutName == views.currentCutName &&
        _generatedViews.sideCutName == views.sideCutName &&
        _generatedViews.lineSurfaceName == views.lineSurfaceName &&
        _generatedViews.lineSideSliceName == views.lineSideSliceName) {
        // Snapshot the on-screen view data: each pane keeps drawing its overlays
        // from this until it adopts a rendered frame of the re-optimized
        // surfaces (hooks below), so overlays and image update together. Taken
        // before the adoption below, these copies of the previous state also
        // serve as the rollback if the cut frames cannot be built: the caller
        // restores the previous surfaces on failure, and this dialog must not
        // keep the new geometry metadata beside them.
        _heldGeneratedViews = _generatedViews;
        _heldControlIndex = _generatedControlIndex;
        _heldLinePosition = _currentLinePosition;

        const double previousLinePosition = _currentLinePosition;
        const float previousDisplayTangentSign = _displayTangentSign;
        const std::optional<cv::Vec3f> previousPendingPlacementFocus =
            _pendingPlacementFocus;
        _generatedViews = std::move(views);
        _displayTangentSign = vc3d::line_annotation::generatedDisplayTangentSign(
            _generatedViews.linePoints,
            _generatedViews.lineNormals);
        _generatedControlIndex =
            vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
                _generatedViews.controlPoints);
        const double maxLinePosition =
            static_cast<double>(_generatedViews.linePoints.size() - 1);
        // After a control-point placement, land the current position on the
        // control point that resulted from the click (positions renumber when
        // the line is re-optimized, so the old numeric position is ambiguous).
        double targetLinePosition = previousLinePosition;
        if (_pendingPlacementFocus) {
            if (const auto nearest =
                    vc3d::line_annotation::nearestGeneratedControlPointIndex(
                        _generatedViews.controlPoints, *_pendingPlacementFocus)) {
                const double controlPosition =
                    _generatedViews.controlPoints[*nearest].linePosition;
                if (std::isfinite(controlPosition)) {
                    targetLinePosition = controlPosition;
                }
            }
            _pendingPlacementFocus.reset();
        }
        _currentLinePosition =
            std::clamp(targetLinePosition, 0.0, maxLinePosition);
        // The one step of the in-place update that can fail, validated before
        // the overlay-swap flags, offsets and viewer properties are touched:
        // set earlier, those would describe an update that never happened, for
        // swap callbacks that are never installed. The two update calls write
        // only into the *new* cut surfaces, which the caller discards on
        // failure, so restoring the fields above is the whole rollback.
        if (!updatePlaneSurface(_generatedViews.currentCutSurface.get(),
                                _currentLinePosition) ||
            !updateSidePlaneSurface(_generatedViews.sideCutSurface.get(),
                                    _currentLinePosition)) {
            _generatedViews = _heldGeneratedViews;
            _generatedControlIndex = _heldControlIndex;
            _currentLinePosition = previousLinePosition;
            _displayTangentSign = previousDisplayTangentSign;
            _pendingPlacementFocus = previousPendingPlacementFocus;
            return false;
        }
        // The re-optimized views renumber line positions, so an in-flight
        // keyboard pan's velocity and targets refer to the old line: stop it.
        // After validation, deliberately — a rejected update leaves the old
        // line on screen, where that pan is still valid. No timer can fire
        // between here and the mutations above; this function is synchronous.
        cancelArrowPan();
        _currentCutOverlaySwapPending = true;
        _sideCutOverlaySwapPending = true;
        _stripOverlaySwapPending.assign(_stripViewers.size(), true);
        _currentCutNormalOffsetVx = 0.0;
        _sideCutNormalOffsetVx = 0.0;
        _currentCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
        _sideCutViewer->setProperty("vc_custom_normal_offset_vx", 0.0);
        _currentCutViewer->markSurfaceGeometryChanged();
        _currentCutViewer->renderVisible(true, "line annotation views updated");
        _sideCutViewer->markSurfaceGeometryChanged();
        _sideCutViewer->renderVisible(true, "line annotation views updated");
        for (const auto& stripViewer : _stripViewers) {
            if (stripViewer) {
                // Bump the strip's geometry epoch too, so the swap hooks below
                // can tell post-update frames from stale in-flight ones.
                stripViewer->markSurfaceGeometryChanged();
                stripViewer->renderVisible(true, "line annotation views updated");
            }
        }
        // Adopting the re-registered surfaces cleared each viewer's scene
        // (onSurfaceChanged does _scene->clear()), deleting the pooled overlay
        // items; drop our cached pointers so the rebuild recreates them.
        clearFastGeneratedOverlayItemRefs();
        // Per pane: swap to the new overlays only once the pane DISPLAYS a
        // frame of the post-update geometry. renderFrameCompleted fires at
        // every frame adoption, and a render submitted before the placement
        // can adopt first -- so gate on the displayed frame's geometry epoch
        // matching the viewer's current one instead of on the first adoption.
        const auto hookOverlaySwap = [this](CChunkedVolumeViewer* viewer,
                                            bool LineAnnotationDialog::*pendingFlag) {
            if (!viewer) {
                this->*pendingFlag = false;
                return;
            }
            auto connection = std::make_shared<QMetaObject::Connection>();
            *connection = connect(
                viewer,
                &CChunkedVolumeViewer::renderFrameCompleted,
                this,
                [this, viewer, connection, pendingFlag](std::uint64_t, double) {
                    if (_closing) {
                        QObject::disconnect(*connection);
                        return;
                    }
                    if (viewer->displayedSurfaceGeometryEpoch() !=
                        viewer->surfaceGeometryEpoch()) {
                        return;  // stale in-flight frame; keep waiting
                    }
                    QObject::disconnect(*connection);
                    this->*pendingFlag = false;
                    rebuildGeneratedStaticStripOverlays();
                    rebuildGeneratedDynamicOverlays();
                });
        };
        hookOverlaySwap(_currentCutViewer.data(),
                        &LineAnnotationDialog::_currentCutOverlaySwapPending);
        hookOverlaySwap(_sideCutViewer.data(),
                        &LineAnnotationDialog::_sideCutOverlaySwapPending);
        const auto hookStripOverlaySwap = [this](CChunkedVolumeViewer* viewer,
                                                 size_t stripIndex) {
            if (!viewer) {
                if (stripIndex < _stripOverlaySwapPending.size()) {
                    _stripOverlaySwapPending[stripIndex] = false;
                }
                return;
            }
            auto connection = std::make_shared<QMetaObject::Connection>();
            *connection = connect(
                viewer,
                &CChunkedVolumeViewer::renderFrameCompleted,
                this,
                [this, viewer, connection, stripIndex](std::uint64_t, double) {
                    if (_closing) {
                        QObject::disconnect(*connection);
                        return;
                    }
                    if (viewer->displayedSurfaceGeometryEpoch() !=
                        viewer->surfaceGeometryEpoch()) {
                        return;
                    }
                    QObject::disconnect(*connection);
                    if (stripIndex < _stripOverlaySwapPending.size()) {
                        _stripOverlaySwapPending[stripIndex] = false;
                    }
                    rebuildGeneratedStaticStripOverlays();
                    rebuildGeneratedDynamicOverlays();
                });
        };
        for (size_t i = 0; i < _stripViewers.size(); ++i) {
            hookStripOverlaySwap(_stripViewers[i].data(), i);
        }
        updatePauseIndicator();
        updateOptimizationStatusIndicator();
        updateUmbilicusNotice();
        rebuildGeneratedOverlays();
        if (_showAsMeshAction) {
            _showAsMeshAction->setEnabled(true);
        }
        if (_fullOptimizationAction) {
            _fullOptimizationAction->setEnabled(true);
        }
        if (_resetViewsAction) {
            _resetViewsAction->setEnabled(true);
        }
        return true;
    }

    if (_showAsMeshAction) {
        _showAsMeshAction->setEnabled(false);
    }
    if (_fullOptimizationAction) {
        _fullOptimizationAction->setEnabled(false);
    }
    if (_resetViewsAction) {
        _resetViewsAction->setEnabled(false);
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
    cancelArrowPan();
    // Drop all viewer references BEFORE deleting the widgets: destroying a
    // viewer synchronously delivers events (ChildRemoved/Hide/...) through our
    // eventFilter, which must not dereference a half-destroyed viewer via
    // _stripViewers/cut viewers (QPointers only clear once ~QObject runs).
    _panes.clear();
    _stripViewers.clear();
    _overviewBar = nullptr;
    _currentCutOverlaySwapPending = false;
    _sideCutOverlaySwapPending = false;
    _stripOverlaySwapPending.clear();
    _pendingPlacementFocus.reset();
    clearFastGeneratedOverlayItemRefs();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;
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

    _generatedViews = views;
    _displayTangentSign = vc3d::line_annotation::generatedDisplayTangentSign(
        _generatedViews.linePoints,
        _generatedViews.lineNormals);
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
    // This pane's top-left corner holds the one cursor-position readout shared
    // by all generated panes (fed by connectLinkedCursorMirroring).
    currentViewer->setProperty("vc_status_stats_top_right", true);
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
        [this](int steps, QPointF, Qt::KeyboardModifiers) {
            // Always step along the line; the cut plane never leaves it.
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
                    // Unlike a plain click this leaves follow untouched, so it
                    // must stop a keyboard pan itself.
                    cancelArrowPan();
                    emit generatedPredSnapPointRequested(_generatedViews.currentCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    if (!controlPointPlacementAllowedAt(_currentLinePosition)) {
                        return;
                    }
                    setCurrentCutFollowsStripMouse(true);
                    _pendingPlacementFocus = volumePoint;
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
    sideViewer->setProperty("vc_status_stats_top_right", true);
    sideViewer->setProperty("vc_hide_status_position", true);
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
                    // Unlike a plain click this leaves follow untouched, so it
                    // must stop a keyboard pan itself.
                    cancelArrowPan();
                    emit generatedPredSnapPointRequested(_generatedViews.sideCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    if (!controlPointPlacementAllowedAt(_currentLinePosition)) {
                        return;
                    }
                    setCurrentCutFollowsStripMouse(true);
                    _pendingPlacementFocus = volumePoint;
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

    // Schematic overview bar between the button row and the cut views: a
    // straight line with the control points, no volume rendering at all. Fixed
    // height; the annotation compresses horizontally as it grows. Registered in
    // _generatedContainers so the rebuild teardown deletes it like the splitter
    // containers.
    auto* overviewBar = new LineAnnotationOverviewBar(centralWidget());
    overviewBar->setObjectName(QStringLiteral("lineAnnotationOverviewBar"));
    overviewBar->setFixedHeight(kOverviewBarHeightPx);
    overviewBar->controlContextRequested = [this](double linePosition,
                                                  QPoint globalPos) {
        forwardOverviewControlContextMenu(linePosition, globalPos);
    };
    _layout->insertWidget(1, overviewBar, 0);
    _generatedContainers.push_back(overviewBar);
    _overviewBar = overviewBar;

    const std::pair<std::string, QString> stripSpecs[] = {
        {views.lineSurfaceName, views.lineSurfaceTitle},
        {views.lineSideSliceName, views.lineSideSliceTitle},
    };
    for (size_t stripIndex = 0; stripIndex < std::size(stripSpecs); ++stripIndex) {
        const auto& [surfaceName, title] = stripSpecs[stripIndex];
        auto* base = _viewerManager->createViewerInWidget(
            surfaceName,
            stripSplitter,
            ViewerManager::ViewerRole::Annotation);
        auto* viewer = base ? qobject_cast<CChunkedVolumeViewer*>(base->asQObject()) : nullptr;
        if (!viewer) {
            return false;
        }
        viewer->setObjectName(title);
        // The strips are camera-linked, so the zoom/scale stats are shown once
        // (top-right of the top strip); each keeps its own normal offset in
        // the top-left, and the shared cursor readout lives in the current cut.
        viewer->setProperty("vc_hide_status_position", true);
        viewer->setProperty("vc_hide_status_poi", true);
        if (stripIndex == 0) {
            viewer->setProperty("vc_status_stats_top_right", true);
        } else {
            viewer->setProperty("vc_hide_status_stats", true);
        }
        const bool haveStripCamera = stripIndex < stripCameras.size();
        auto stripCamera = haveStripCamera
            ? stripCameras[stripIndex]
            : generatedPaneCamera(viewer, camera);
        if (!haveStripCamera) {
            if (const auto center =
                    generatedStripSurfaceCenter(viewer, _currentLinePosition,
                                                &views.stripPositionMap)) {
                stripCamera.surfacePtrX = (*center)[0];
                stripCamera.surfacePtrY = (*center)[1];
            }
            if (const auto focusedScale = generatedStripScaleForLinePositionRange(
                    viewer,
                    views.initialStripLinePositionRange,
                    &views.stripPositionMap)) {
                stripCamera.scale = *focusedScale;
            }
            if (stripIndex < _savedStripZooms.size()) {
                stripCamera.scale = _savedStripZooms[stripIndex];
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
                            _pendingPlacementFocus = volumePoint;
                            emit generatedControlPointRequested(surfaceName, volumePoint, position);
                        }
                    }
                });
        stripSplitter->addWidget(viewer);
        _stripViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
        connectGeneratedOverlayRefresh(viewer);
        // Strips share their along-line position and zoom; overlaysUpdated is
        // the only notification pan/zoom emit, and syncLinkedStripCamera
        // no-ops when the cameras already agree.
        _generatedOverlayRefreshConnections.push_back(
            viewer->connectOverlaysUpdated(this, [this, viewer]() {
                syncLinkedStripCamera(viewer);
            }));
    }
    if (!_stripViewers.empty()) {
        syncLinkedStripCamera(_stripViewers.front().data());
    }
    connectLinkedCursorMirroring(
        {_currentCutViewer, _sideCutViewer, _stripViewers[0], _stripViewers[1]});

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

    updatePauseIndicator();
    updateOptimizationStatusIndicator();
    updateUmbilicusNotice();
    rebuildGeneratedOverlays();
    if (_showAsMeshAction) {
        _showAsMeshAction->setEnabled(true);
    }
    if (_fullOptimizationAction) {
        _fullOptimizationAction->setEnabled(true);
    }
    captureInitialGeneratedViewState();
    if (_resetViewsAction) {
        _resetViewsAction->setEnabled(true);
    }
    return true;
}

LineAnnotationDialog::GeneratedControlPointContextResult
LineAnnotationDialog::showGeneratedControlPointContextMenu(
    const std::string& surfaceName,
    CChunkedVolumeViewer* viewer,
    const QPointF& scenePoint,
    const QPoint& globalPos,
    const vc3d::line_annotation::GeneratedLinkCandidateMenuState& linkCandidateState,
    const vc3d::line_annotation::GeneratedLinkCandidateMenuState& splitCandidateState,
    const vc3d::line_annotation::GeneratedLinkCandidateMenuState& splitAndLinkCandidateState,
    const vc3d::line_annotation::GeneratedLinkCandidateMenuState& mergeCandidateState)
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
    options.stripPositionMap = _generatedViews.stripPositionMap;
    options.linkWithCandidateEnabled = linkCandidateState.enabled;
    options.linkWithCandidateLabel = linkCandidateState.label;
    options.mergeWithCandidateEnabled = mergeCandidateState.enabled;
    options.mergeWithCandidateLabel = mergeCandidateState.label;
    options.splitFromCandidateEnabled = splitCandidateState.enabled;
    options.splitFromCandidateLabel = splitCandidateState.label;
    options.splitFromCandidateAndLinkLabel = splitAndLinkCandidateState.label;
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
    options.mergeWithCandidate = [this, surfaceName](size_t controlPointIndex,
                                                     cv::Vec3f volumePoint) {
        emit generatedControlPointMergeWithCandidateRequested(surfaceName,
                                                              controlPointIndex,
                                                              volumePoint);
    };
    options.designateSplitCandidate = [this, surfaceName](size_t controlPointIndex,
                                                          cv::Vec3f volumePoint) {
        emit generatedControlPointSplitCandidateRequested(surfaceName,
                                                          controlPointIndex,
                                                          volumePoint);
    };
    options.splitFromCandidate = [this, surfaceName](size_t controlPointIndex,
                                                     cv::Vec3f volumePoint) {
        emit generatedControlPointSplitFromCandidateRequested(surfaceName,
                                                              controlPointIndex,
                                                              volumePoint);
    };
    options.splitFromCandidateAndLink = [this, surfaceName](size_t controlPointIndex,
                                                            cv::Vec3f volumePoint) {
        emit generatedControlPointSplitAndLinkFromCandidateRequested(surfaceName,
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
    options.setSegmentInterpolationGoal =
        [this, surfaceName](size_t firstControlPointIndex,
                            size_t secondControlPointIndex,
                            std::string goal) {
        emit generatedSegmentInterpolationGoalRequested(surfaceName,
                                                        firstControlPointIndex,
                                                        secondControlPointIndex,
                                                        goal);
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
    return vc3d::line_annotation::generatedLinePositionFromStripScene(
        viewer, scenePoint, &_generatedViews.stripPositionMap);
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
                                                  bool updateCurrentCutOverlay,
                                                  bool forceApply)
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
    const bool currentChanged =
        forceApply || std::abs(position - _currentLinePosition) >= 1.0e-3;
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
    // The scroll supersedes a running keyboard pan: stepping past its stop
    // target would otherwise make the next tick snap the position backward.
    cancelArrowPan();
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

void LineAnnotationDialog::startArrowPan(int direction)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty() || direction == 0) {
        return;
    }
    // Any new press invalidates the previous gesture's landing, even when this
    // press finds no target below: otherwise a stale landing flag paired with
    // a physical flag whose release a popup swallowed could hand the pan to a
    // key that is no longer held.
    _arrowPanEndedByLanding = false;
    if (_arrowPanDirection == direction) {
        // Same direction while already panning: nothing to recompute, just make
        // sure a coasting tap turns back into a hold.
        if (!_arrowPanKeyHeld) {
            _arrowPanKeyHeld = true;
            updateArrowPanStopTarget();
        }
        return;
    }

    const auto positions = arrowPanTargetPositions();
    // The gesture's floor: the first control point in the new direction from
    // here. A reversal recomputes it from the position at reversal.
    const auto firstAhead = (direction > 0)
        ? vc3d::line_annotation::nextGeneratedControlPointLinePosition(_currentLinePosition,
                                                                      positions)
        : vc3d::line_annotation::previousGeneratedControlPointLinePosition(_currentLinePosition,
                                                                          positions);
    if (!firstAhead) {
        // Nothing that way: a fresh press does nothing, a reversal just stops.
        if (_arrowPanDirection != 0) {
            cancelArrowPan();
        }
        return;
    }

    cancelControlPointPreviewAnimation();
    setCurrentCutFollowsStripMouse(false);
    const bool wasIdle = (_arrowPanDirection == 0);
    _arrowPanDirection = direction;
    _arrowPanKeyHeld = true;
    _arrowPanMinimumTarget = *firstAhead;
    if (wasIdle) {
        _arrowPanVelocity = 0.0;
        // Lock the strips onto the current-position line for the whole pan.
        centerStripsOnLinePosition(_currentLinePosition, true);
    }
    updateArrowPanStopTarget();
    if (!_arrowPanStopTarget) {
        return;
    }

    if (!_arrowPanTimer) {
        _arrowPanTimer = new QTimer(this);
        _arrowPanTimer->setInterval(kArrowPanTickMs);
        connect(_arrowPanTimer, &QTimer::timeout, this, &LineAnnotationDialog::tickArrowPan);
    }
    _arrowPanClock.start();
    if (!_arrowPanTimer->isActive()) {
        _arrowPanTimer->start();
    }
}

std::vector<double> LineAnnotationDialog::arrowPanTargetPositions() const
{
    auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    if (positions.empty() || _generatedViews.linePoints.empty()) {
        return positions;
    }
    // Add one target beyond each outer control at the configured base-voxel
    // arclength limit, clamped to the optimized line's extrapolated endpoint.
    const double maxDistanceBaseVoxels = static_cast<double>(
        maxControlPointExtrapolationDistanceVx());
    const double lastPosition =
        static_cast<double>(_generatedViews.linePoints.size() - 1);
    if (const auto left =
            vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
                _generatedViews.stripPositionMap.originalArclengths,
                positions,
                -1,
                0.0,
                maxDistanceBaseVoxels)) {
        positions.insert(positions.begin(), *left);
    }
    if (const auto right =
            vc3d::fiber_slice::controlExtrapolationBoundaryLinePosition(
                _generatedViews.stripPositionMap.originalArclengths,
                positions,
                1,
                lastPosition,
                maxDistanceBaseVoxels)) {
        positions.push_back(*right);
    }
    return positions;
}

void LineAnnotationDialog::releaseArrowPanKey(int direction)
{
    // Ignore the release of the key that a reversal already superseded.
    if (_arrowPanDirection == 0 || _arrowPanDirection != direction || !_arrowPanKeyHeld) {
        return;
    }
    _arrowPanKeyHeld = false;
    updateArrowPanStopTarget();
}

void LineAnnotationDialog::updateArrowPanStopTarget()
{
    if (_arrowPanDirection == 0) {
        _arrowPanStopTarget.reset();
        return;
    }
    const auto positions = arrowPanTargetPositions();
    double minimumTarget = _arrowPanMinimumTarget;
    if (_arrowPanKeyHeld && !positions.empty()) {
        // Held: cruise straight through the intermediate control points by
        // aiming at the far end, so the only braking is into the last target.
        minimumTarget = (_arrowPanDirection > 0) ? positions.back() : positions.front();
    }
    _arrowPanStopTarget = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions,
        _currentLinePosition,
        _arrowPanDirection,
        minimumTarget);
    if (!_arrowPanStopTarget) {
        cancelArrowPan();
        return;
    }
    if (static_cast<double>(_arrowPanDirection) *
            (*_arrowPanStopTarget - _currentLinePosition) < 0.0) {
        // Only a stale fallback (a mid-pan edit shrank the target set behind
        // us) can place the target against the travel; landing there would
        // teleport backward, so stop in place instead.
        cancelArrowPan();
    }
}

void LineAnnotationDialog::rebaseArrowPanTargets()
{
    if (_arrowPanDirection == 0) {
        return;
    }
    const auto positions = arrowPanTargetPositions();
    // The minimum target is the press-time promise; when an edit removed it
    // (or moved the boundary), re-promise from the current position so the
    // selector's fallback can never resurrect a deleted position.
    const bool minimumStillExists = std::any_of(
        positions.begin(), positions.end(), [this](double position) {
            return std::abs(position - _arrowPanMinimumTarget) <= 1.0e-9;
        });
    if (!minimumStillExists) {
        const auto firstAhead = (_arrowPanDirection > 0)
            ? vc3d::line_annotation::nextGeneratedControlPointLinePosition(
                  _currentLinePosition, positions)
            : vc3d::line_annotation::previousGeneratedControlPointLinePosition(
                  _currentLinePosition, positions);
        if (!firstAhead) {
            cancelArrowPan();
            return;
        }
        _arrowPanMinimumTarget = *firstAhead;
    }
    updateArrowPanStopTarget();
}

void LineAnnotationDialog::tickArrowPan()
{
    if (_closing || _arrowPanDirection == 0 || !_hasGeneratedViews ||
        _generatedViews.linePoints.empty()) {
        cancelArrowPan();
        return;
    }
    if (_arrowPanKeyHeld && QApplication::activePopupWidget()) {
        // A popup (hamburger or context menu, combo dropdown) grabs the
        // keyboard, so the key-up will never reach us; treat the grab as the
        // release so the hold cannot strand.
        _arrowKeyLeftDown = false;
        _arrowKeyRightDown = false;
        releaseArrowPanKey(_arrowPanDirection);
        if (_arrowPanDirection == 0) {
            return;
        }
    }
    const double dtSeconds = std::min(kArrowPanMaximumStepSeconds,
                                      static_cast<double>(_arrowPanClock.nsecsElapsed()) * 1.0e-9);
    _arrowPanClock.restart();
    if (dtSeconds <= 0.0) {
        return;
    }
    const double acceleration =
        _arrowPanCruiseSpeed / vc3d::line_annotation::kGeneratedArrowPanRampSeconds;
    const auto step = vc3d::line_annotation::generatedArrowPanStep(_currentLinePosition,
                                                                   _arrowPanVelocity,
                                                                   _arrowPanDirection,
                                                                   _arrowPanCruiseSpeed,
                                                                   acceleration,
                                                                   dtSeconds,
                                                                   _arrowPanStopTarget);
    _arrowPanVelocity = step.velocity;
    const double maxPosition = static_cast<double>(_generatedViews.linePoints.size() - 1);
    const double position = std::clamp(step.position, 0.0, maxPosition);
    const bool hitLineEnd = (position != step.position);
    if (step.landed) {
        finishArrowPan(position);
        return;
    }
    if (hitLineEnd) {
        // Only stop at the line end while the travel still points out of it. A
        // reversal pressed near the end is still shedding outward velocity, so
        // pin the position to the edge and keep integrating until the velocity
        // crosses zero and carries it back inside.
        const bool reversingBackInside =
            (step.position > maxPosition && _arrowPanDirection < 0) ||
            (step.position < 0.0 && _arrowPanDirection > 0);
        if (!reversingBackInside) {
            finishArrowPan(position);
            return;
        }
    }
    // Camera first, overlays second: the strip overlays bake the camera into
    // their scene coordinates (surfaceCoordsToScene subtracts surfacePtrX), so
    // recentering after the rebuild would draw the current-position line one
    // tick off-center and let the coalesced overlaysUpdated refresh redraw it
    // centered right after - a ~60 Hz flicker that reads as two green lines.
    centerStripsOnLinePosition(position, false);
    setCurrentLinePosition(position, false);
}

void LineAnnotationDialog::finishArrowPan(double position)
{
    cancelArrowPan();
    centerStripsOnLinePosition(position, false);
    // Force the apply: the final residual is often below the setter's 1e-3
    // no-op threshold, and the cut planes must land on the exact target too.
    setCurrentLinePosition(position, false, /*forceApply=*/true);
    // One full-quality pass at the landing: statics, dynamics (with the
    // current-cut overlay and span labels), and the side-strip intersection
    // refresh the pan ticks deferred.
    rebuildGeneratedOverlays(true);
    // The camera apply above queued the coalesced refresh callback; this pass
    // just did that work, so record the covered generation and let the
    // callback skip its (otherwise redundant) full rebuild. Any update after
    // this line bumps the generation and the callback runs normally.
    _generatedOverlayRefreshCoveredGeneration = _generatedOverlayRefreshGeneration;
    _arrowPanEndedByLanding = true;
}

void LineAnnotationDialog::cancelArrowPan()
{
    if (_arrowPanTimer) {
        _arrowPanTimer->stop();
    }
    _arrowPanDirection = 0;
    _arrowPanKeyHeld = false;
    _arrowPanVelocity = 0.0;
    _arrowPanStopTarget.reset();
    _arrowPanMinimumTarget = std::numeric_limits<double>::quiet_NaN();
    _arrowPanEndedByLanding = false;
}

void LineAnnotationDialog::adjustArrowPanCruiseSpeed(double factor)
{
    if (!std::isfinite(factor) || factor <= 0.0) {
        return;
    }
    const double updated = std::clamp(_arrowPanCruiseSpeed * factor,
                                      vc3d::line_annotation::kGeneratedArrowPanMinimumSpeed,
                                      vc3d::line_annotation::kGeneratedArrowPanMaximumSpeed);
    if (updated != _arrowPanCruiseSpeed) {
        _arrowPanCruiseSpeed = updated;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::line_annotation::ARROW_PAN_SPEED, updated);
    }
    // Flash the badge even when the value clamped, so the key press is answered.
    updateArrowPanSpeedIndicator();
    if (_arrowPanSpeedLabel) {
        _arrowPanSpeedLabel->show();
        _arrowPanSpeedLabel->raise();
    }
    if (!_arrowPanSpeedLabelTimer) {
        _arrowPanSpeedLabelTimer = new QTimer(this);
        _arrowPanSpeedLabelTimer->setSingleShot(true);
        connect(_arrowPanSpeedLabelTimer, &QTimer::timeout, this, [this]() {
            if (_arrowPanSpeedLabel) {
                _arrowPanSpeedLabel->hide();
            }
        });
    }
    _arrowPanSpeedLabelTimer->start(kArrowPanSpeedIndicatorHideMs);
}

void LineAnnotationDialog::updateArrowPanSpeedIndicator()
{
    CChunkedVolumeViewer* topStrip =
        _stripViewers.empty() ? nullptr : _stripViewers.front().data();
    if (!topStrip) {
        if (_arrowPanSpeedLabel) {
            _arrowPanSpeedLabel->hide();
        }
        return;
    }
    if (!_arrowPanSpeedLabel || _arrowPanSpeedLabel->parentWidget() != topStrip) {
        delete _arrowPanSpeedLabel;
        auto* label = new QLabel(topStrip);
        label->setObjectName(QStringLiteral("lineAnnotationArrowPanSpeedLabel"));
        label->setAttribute(Qt::WA_TransparentForMouseEvents);
        label->setStyleSheet(QStringLiteral(
            "QLabel { color: rgb(0, 245, 255); background-color: rgba(20, 20, 20, 160); "
            "border-radius: 4px; padding: 2px 8px; font-weight: bold; }"));
        label->hide();
        _arrowPanSpeedLabel = label;
    }
    _arrowPanSpeedLabel->setText(
        tr("pan speed %1 /s").arg(QString::number(_arrowPanCruiseSpeed, 'g', 3)));
    _arrowPanSpeedLabel->adjustSize();
    // Top-center of the top strip; the pause badge sits on the bottom strip.
    _arrowPanSpeedLabel->move(
        std::max(0, (topStrip->width() - _arrowPanSpeedLabel->width()) / 2),
        10);
}

void LineAnnotationDialog::centerStripsOnLinePosition(double linePosition, bool includeVertical)
{
    for (const auto& stripViewer : _stripViewers) {
        if (!stripViewer) {
            continue;
        }
        if (const auto center = generatedStripSurfaceCenter(
                stripViewer, linePosition, &_generatedViews.stripPositionMap)) {
            CChunkedVolumeViewer::CameraState camera = stripViewer->cameraState();
            camera.surfacePtrX = (*center)[0];
            if (includeVertical) {
                camera.surfacePtrY = (*center)[1];
            }
            stripViewer->applyCameraState(camera, false);
        }
        if (!includeVertical) {
            // Per-tick X recentering only touches the first strip: the linked
            // camera echo (syncLinkedStripCamera) already copies surfacePtrX to
            // the other strip synchronously, so applying it here too did the
            // same update twice per tick. The initial snap (includeVertical)
            // still visits every strip because Y is per-strip, not linked.
            break;
        }
    }
}

void LineAnnotationDialog::jumpToPreviousControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    cancelArrowPan();
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
    cancelArrowPan();
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
    cancelArrowPan();
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

void LineAnnotationDialog::resetGeneratedNormalOffsets()
{
    if (!_hasGeneratedViews) {
        return;
    }
    resetGeneratedCutNormalOffsets(true);
    for (const auto& strip : _stripViewers) {
        if (!strip) {
            continue;
        }
        const float offset = strip->normalOffset();
        if (offset != 0.0f) {
            strip->adjustSurfaceOffset(-offset);
        }
    }
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
    if (!follows) {
        // A hover update queued in the last coalescing tick must not fire into
        // a keyboard pan that pausing is about to start.
        _lineUpdatePending = false;
        if (_lineUpdateTimer) {
            _lineUpdateTimer->stop();
        }
    }
    if (follows && !wasFollowing) {
        // Un-pausing hands the current position back to the mouse, so it also
        // ends a keyboard pan. Space and the strip/cut click handlers all reach
        // the cancel through this one transition.
        cancelArrowPan();
        resetGeneratedCutNormalOffsets(true);
    }
    updatePauseIndicator();
}

void LineAnnotationDialog::updatePauseIndicator()
{
    CChunkedVolumeViewer* bottomStrip =
        _stripViewers.empty() ? nullptr : _stripViewers.back().data();
    if (!bottomStrip) {
        if (_pauseIndicator) {
            _pauseIndicator->hide();
        }
        return;
    }
    if (!_pauseIndicator || _pauseIndicator->parentWidget() != bottomStrip) {
        delete _pauseIndicator;
        auto* indicator = new QLabel(bottomStrip);
        indicator->setObjectName(QStringLiteral("lineAnnotationPauseIndicator"));
        indicator->setText(QStringLiteral("❚❚"));
        indicator->setStyleSheet(QStringLiteral(
            "QLabel { color: rgb(0, 245, 255); background-color: rgba(20, 20, 20, 160); "
            "border-radius: 6px; padding: 6px 14px; font-size: 32px; font-weight: bold; }"));
        indicator->setAttribute(Qt::WA_TransparentForMouseEvents);
        indicator->adjustSize();
        _pauseIndicator = indicator;
    }
    _pauseIndicator->move(
        std::max(0, (bottomStrip->width() - _pauseIndicator->width()) / 2), 10);
    _pauseIndicator->setVisible(!_currentCutFollowsStripMouse);
    _pauseIndicator->raise();
}

void LineAnnotationDialog::setUmbilicusNotice(const QString& notice)
{
    if (_umbilicusNotice == notice) {
        return;
    }
    _umbilicusNotice = notice;
    updateUmbilicusNotice();
}

void LineAnnotationDialog::updateUmbilicusNotice()
{
    CChunkedVolumeViewer* bottomStrip =
        _stripViewers.empty() ? nullptr : _stripViewers.back().data();
    if (!bottomStrip || _umbilicusNotice.isEmpty()) {
        if (_umbilicusNoticeLabel) {
            _umbilicusNoticeLabel->hide();
        }
        return;
    }
    if (!_umbilicusNoticeLabel ||
        _umbilicusNoticeLabel->parentWidget() != bottomStrip) {
        delete _umbilicusNoticeLabel;
        auto* label = new QLabel(bottomStrip);
        label->setObjectName(QStringLiteral("lineAnnotationUmbilicusNotice"));
        label->setAttribute(Qt::WA_TransparentForMouseEvents);
        label->setStyleSheet(QStringLiteral(
            "QLabel { color: rgb(255, 90, 90); background-color: rgba(20, 20, 20, 160); "
            "border-radius: 4px; padding: 2px 6px; font-size: 10px; }"));
        _umbilicusNoticeLabel = label;
    }
    _umbilicusNoticeLabel->setText(_umbilicusNotice);
    _umbilicusNoticeLabel->adjustSize();
    // Top-left: the pause badge owns the top centre and the optimization badge
    // the bottom right.
    _umbilicusNoticeLabel->move(10, 10);
    _umbilicusNoticeLabel->show();
    _umbilicusNoticeLabel->raise();
}

void LineAnnotationDialog::updateOptimizationStatusIndicator()
{
    CChunkedVolumeViewer* bottomStrip =
        _stripViewers.empty() ? nullptr : _stripViewers.back().data();
    if (!bottomStrip) {
        if (_optimizationStatusLabel) {
            _optimizationStatusLabel->hide();
        }
        return;
    }
    if (!_optimizationStatusLabel ||
        _optimizationStatusLabel->parentWidget() != bottomStrip) {
        delete _optimizationStatusLabel;
        auto* label = new QLabel(bottomStrip);
        label->setObjectName(QStringLiteral("lineAnnotationOptimizationStatusLabel"));
        label->setAttribute(Qt::WA_TransparentForMouseEvents);
        _optimizationStatusLabel = label;
    }
    _optimizationStatusLabel->setText(
        _optimizationStatusOptimized ? tr("optimized") : tr("not optimized"));
    _optimizationStatusLabel->setStyleSheet(QStringLiteral(
        "QLabel { color: %1; background-color: rgba(20, 20, 20, 160); "
        "border-radius: 4px; padding: 2px 8px; font-weight: bold; }")
        .arg(_optimizationStatusOptimized ? QStringLiteral("rgb(40, 220, 120)")
                                          : QStringLiteral("rgb(255, 80, 80)")));
    _optimizationStatusLabel->adjustSize();
    _optimizationStatusLabel->move(
        std::max(0, bottomStrip->width() - _optimizationStatusLabel->width() - 10),
        std::max(0, bottomStrip->height() - _optimizationStatusLabel->height() - 10));
    _optimizationStatusLabel->show();
    _optimizationStatusLabel->raise();
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
    std::vector<double> controlLinePositions;
    controlLinePositions.reserve(_generatedViews.controlPoints.size());
    for (const auto& control : _generatedViews.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
    }
    return vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
        _generatedViews.stripPositionMap.originalArclengths,
        linePosition,
        controlLinePositions,
        static_cast<double>(maxControlPointExtrapolationDistanceVx()));
}

vc3d::line_annotation::GeneratedCurrentLineMarkerState
LineAnnotationDialog::currentLineMarkerState() const
{
    if (maxControlPointExtrapolationDistanceVx() <= 0) {
        return vc3d::line_annotation::GeneratedCurrentLineMarkerState::Neutral;
    }
    std::vector<double> controlLinePositions;
    controlLinePositions.reserve(_generatedViews.controlPoints.size());
    for (const auto& control : _generatedViews.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
    }
    return vc3d::fiber_slice::linePositionWithinControlExtrapolationDistance(
               _generatedViews.stripPositionMap.originalArclengths,
               _currentLinePosition,
               controlLinePositions,
               static_cast<double>(maxControlPointExtrapolationDistanceVx()))
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
    bindNavigationShortcut(Qt::Key_R, &LineAnnotationDialog::snapPanesToOverviewCursor);
    bindNavigationShortcut(Qt::Key_T, &LineAnnotationDialog::jumpToNextControlPoint);
    bindNavigationShortcut(Qt::Key_B, &LineAnnotationDialog::resetGeneratedNormalOffsets);

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

    // Reset assigns position and follow state directly below, bypassing the
    // setter that would otherwise stop a running keyboard pan.
    cancelArrowPan();
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
    // _currentCutFollowsStripMouse was reset by direct assignment above; keep the
    // pause badge in sync.
    updatePauseIndicator();
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

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::zSliceOverlay(
    const GeneratedViews& views,
    const vc3d::line_annotation::GeneratedControlPointLinePositionIndex& controlIndex,
    double linePosition,
    bool emphasized,
    CChunkedVolumeViewer* viewer,
    PlaneSurface* plane) const
{
    (void)emphasized;
    return vc3d::line_annotation::makeGeneratedCrossSliceControlOverlayForPlane(
        views,
        linePosition,
        viewer,
        plane,
        &controlIndex);
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
        const bool sideStrip = i == 1;
        const std::string& key = sideStrip
            ? _generatedViews.lineSideSliceName
            : _generatedViews.lineSurfaceName;
        const bool swapPending = i < _stripOverlaySwapPending.size() &&
                                 _stripOverlaySwapPending[i];
        // Each rendered strip keeps its previous overlay until its own
        // re-optimized frame is displayed.
        const GeneratedViews& stripViews =
            swapPending ? _heldGeneratedViews : _generatedViews;
        GeneratedOverlay strip =
            vc3d::line_annotation::makeGeneratedStaticStripOverlay(stripViews);
        if (sideStrip) {
            strip.fiberIntersections = stripViews.fiberIntersections;
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

    // The schematic overview bar tracks the same inputs (control points +
    // current position) as the dynamic overlays; refresh it on the same cadence.
    updateOverviewBar();

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
               const std::string& surfaceName,
               size_t spanLabelCount) -> FastStripOverlayItems* {
        if (!viewer) {
            return nullptr;
        }
        if (index >= _fastStripOverlayItems.size()) {
            _fastStripOverlayItems.resize(index + 1);
        }
        auto& entry = _fastStripOverlayItems[index];
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
        const bool sideStrip = i == 1;
        const std::string& key = sideStrip
            ? _generatedViews.lineSideSliceName
            : _generatedViews.lineSurfaceName;
        const bool swapPending = i < _stripOverlaySwapPending.size() &&
                                 _stripOverlaySwapPending[i];
        const GeneratedViews& stripViews =
            swapPending ? _heldGeneratedViews : _generatedViews;
        auto* entry = ensureStripItems(
            i, viewer, key, stripViews.spanAlignmentMetrics.size());
        // While this pane's overlay swap is pending, map the current-position
        // marker (and span labels) through the held pre-update strip surface so
        // they stay consistent with the stale image instead of jumping a beat
        // ahead of it.
        QuadSurface* quad = nullptr;
        if (swapPending) {
            quad = sideStrip ? stripViews.lineSideSlice.get()
                             : stripViews.lineSurface.get();
        } else {
            quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        }
        const double stripPosition = swapPending
            ? _heldLinePosition
            : _currentLinePosition;
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
                                                                      stripPosition,
                                                                      &stripViews.stripPositionMap);
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
            struct VisibleSpanLabel {
                std::remove_reference_t<decltype(entry->spanLabels[0])>* items = nullptr;
                const vc3d::line_annotation::GeneratedSpanAlignmentMetric* metric = nullptr;
                double preferredLeft = 0.0;
                double left = 0.0;
                double width = 0.0;
                double height = 0.0;
                int row = 0;
            };
            std::vector<VisibleSpanLabel> visibleLabels;
            const QRect viewportRect = haveViewport ? viewport->rect() : QRect{};
            for (size_t labelIndex = 0; labelIndex < entry->spanLabels.size(); ++labelIndex) {
                auto& labelItems = entry->spanLabels[labelIndex];
                auto* background = labelItems.background;
                auto* text = labelItems.text;
                if (!background || !text) {
                    continue;
                }
                background->setVisible(false);
                text->setVisible(false);
                if (labelIndex >= stripViews.spanAlignmentMetrics.size() || !haveViewport) {
                    continue;
                }

                const auto& metric = stripViews.spanAlignmentMetrics[labelIndex];
                if (!shouldShowSpanAlignmentMetric(metric)) {
                    continue;
                }
                if (!std::isfinite(metric.firstControlLinePosition) ||
                    !std::isfinite(metric.secondControlLinePosition)) {
                    continue;
                }
                const QPointF firstScenePoint =
                    vc3d::line_annotation::generatedStripLinePositionToScene(
                        viewer,
                        quad,
                        metric.firstControlLinePosition,
                        &stripViews.stripPositionMap);
                const QPointF secondScenePoint =
                    vc3d::line_annotation::generatedStripLinePositionToScene(
                        viewer,
                        quad,
                        metric.secondControlLinePosition,
                        &stripViews.stripPositionMap);
                if (!std::isfinite(firstScenePoint.x()) ||
                    !std::isfinite(secondScenePoint.x())) {
                    continue;
                }
                const double firstX = view->mapFromScene(firstScenePoint).x();
                const double secondX = view->mapFromScene(secondScenePoint).x();
                const double visibleFirst = std::max<double>(
                    viewportRect.left(), std::min(firstX, secondX));
                const double visibleLast = std::min<double>(
                    viewportRect.right(), std::max(firstX, secondX));
                if (visibleLast < visibleFirst)
                    continue;
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
                background->setBrush(QBrush(backgroundColor));
                background->setRect(textRect.adjusted(-4.0, -2.0, 4.0, 2.0));
                background->setToolTip(text->toolTip());
                const double width = textRect.width() + 8.0;
                const double preferredCenter = std::clamp(
                    (firstX + secondX) * 0.5, visibleFirst, visibleLast);
                visibleLabels.push_back({
                    &labelItems,
                    &metric,
                    std::clamp(preferredCenter - width * 0.5,
                               static_cast<double>(viewportRect.left()),
                               std::max(static_cast<double>(viewportRect.left()),
                                        static_cast<double>(viewportRect.right()) - width)),
                    0.0,
                    width,
                    textRect.height() + 4.0,
                    0});
            }

            constexpr double gap = 4.0;
            double totalWidth = 0.0;
            for (const auto& label : visibleLabels)
                totalWidth += label.width;
            if (!visibleLabels.empty())
                totalWidth += gap * static_cast<double>(visibleLabels.size() - 1);
            const bool twoRows = totalWidth > viewportRect.width();
            if (twoRows) {
                for (size_t i = 0; i < visibleLabels.size(); ++i)
                    visibleLabels[i].row = static_cast<int>(i % 2);
            }
            for (int row = 0; row < (twoRows ? 2 : 1); ++row) {
                std::vector<size_t> indices;
                for (size_t i = 0; i < visibleLabels.size(); ++i) {
                    if (visibleLabels[i].row == row)
                        indices.push_back(i);
                }
                double next = viewportRect.left();
                for (size_t index : indices) {
                    auto& label = visibleLabels[index];
                    label.left = std::max(label.preferredLeft, next);
                    next = label.left + label.width + gap;
                }
                double right = viewportRect.right();
                for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
                    auto& label = visibleLabels[*it];
                    label.left = std::min(label.left, right - label.width);
                    label.left = std::max(label.left,
                                          static_cast<double>(viewportRect.left()));
                    right = label.left - gap;
                }
            }
            for (auto& label : visibleLabels) {
                const double viewportY = viewportRect.bottom() - 3.0 -
                    label.height - label.row * (label.height + gap);
                const QPointF scenePosition = view->mapToScene(QPoint(
                    static_cast<int>(std::llround(label.left + 4.0)),
                    static_cast<int>(std::llround(viewportY + 2.0))));
                label.items->text->setPos(scenePosition);
                const QPointF backgroundPosition = view->mapToScene(QPoint(
                    static_cast<int>(std::llround(label.left)),
                    static_cast<int>(std::llround(viewportY))));
                label.items->background->setPos(backgroundPosition);
                label.items->background->setVisible(true);
                label.items->text->setVisible(true);
            }
        }
    }

    if (_sideCutViewer && _generatedViews.sideCutSurface) {
        // Draw the full line on the side view by projecting each line point onto the plane and
        // connecting consecutive points (linear interpolation), in addition to the current-position
        // marker and nearby control points from the cross-slice overlay. During an in-place
        // update, draw from the held pre-update views until this pane's new frame lands.
        const GeneratedViews& sideViews =
            _sideCutOverlaySwapPending ? _heldGeneratedViews : _generatedViews;
        const auto& sideIndex =
            _sideCutOverlaySwapPending ? _heldControlIndex : _generatedControlIndex;
        const double sidePosition =
            _sideCutOverlaySwapPending ? _heldLinePosition : _currentLinePosition;
        GeneratedOverlay sideOverlay = zSliceOverlay(sideViews,
                                                     sideIndex,
                                                     sidePosition,
                                                     true,
                                                     _sideCutViewer,
                                                     sideViews.sideCutSurface.get());
        sideOverlay.linePoints = sideViews.linePoints;
        // Highlight the live cursor position on the line. The cross-slice overlay's emphasized
        // marker otherwise sits at the static focus/seed point; override it to the current
        // position so the highlight tracks the cursor as it moves along the line. Use the
        // (possibly held) sideViews line points so the marker matches the displayed image.
        const cv::Vec3f currentPoint = vc3d::line_annotation::interpolatedGeneratedLinePoint(
            sideViews.linePoints, sidePosition);
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
        !_fastCurrentCutOverlayItems.splitCandidatePoints ||
        !_fastCurrentCutOverlayItems.branchControlPoints ||
        !_fastCurrentCutOverlayItems.pendingBranchControlPoints ||
        !_fastCurrentCutOverlayItems.sameHvBranchControlPoints ||
        !_fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints ||
        !_fastCurrentCutOverlayItems.fiberIntersections ||
        !_fastCurrentCutOverlayItems.linkCandidateFiberIntersections ||
        !_fastCurrentCutOverlayItems.branchLinkFiberIntersections ||
        !_fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections ||
        !_fastCurrentCutOverlayItems.fiberIntersectionConnectors ||
        !_fastCurrentCutOverlayItems.ghostControlPointPrev ||
        !_fastCurrentCutOverlayItems.ghostControlPointNext) {
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

        QPen splitCandidatePen(QColor(235, 60, 60, 245));
        splitCandidatePen.setWidthF(2.0);
        QBrush splitCandidateBrush(QColor(235, 60, 60, 175));
        _fastCurrentCutOverlayItems.splitCandidatePoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.splitCandidatePoints->setPen(splitCandidatePen);
        _fastCurrentCutOverlayItems.splitCandidatePoints->setBrush(splitCandidateBrush);
        _fastCurrentCutOverlayItems.splitCandidatePoints->setZValue(163.5);

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

        QPen sameHvBranchControlPen(QColor(255, 140, 0, 245));
        sameHvBranchControlPen.setWidthF(2.0);
        QBrush sameHvBranchControlBrush(QColor(255, 140, 0, 175));
        _fastCurrentCutOverlayItems.sameHvBranchControlPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.sameHvBranchControlPoints->setPen(sameHvBranchControlPen);
        _fastCurrentCutOverlayItems.sameHvBranchControlPoints->setBrush(
            sameHvBranchControlBrush);
        _fastCurrentCutOverlayItems.sameHvBranchControlPoints->setZValue(162.0);

        QPen sameHvPendingBranchControlPen(QColor(255, 190, 120, 245));
        sameHvPendingBranchControlPen.setWidthF(2.0);
        QBrush sameHvPendingBranchControlBrush(QColor(255, 190, 120, 175));
        _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints->setPen(
            sameHvPendingBranchControlPen);
        _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints->setBrush(
            sameHvPendingBranchControlBrush);
        _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints->setZValue(162.5);

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

        // Hollow dashed rings, deliberately unlike the solid control markers:
        // they sit at a fictional, parallax-shifted spot until they land.
        QPen ghostControlPen(QColor(255, 230, 0));
        ghostControlPen.setWidthF(1.5);
        ghostControlPen.setStyle(Qt::DashLine);
        _fastCurrentCutOverlayItems.ghostControlPointPrev = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.ghostControlPointPrev->setPen(ghostControlPen);
        _fastCurrentCutOverlayItems.ghostControlPointPrev->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.ghostControlPointPrev->setZValue(155.0);

        _fastCurrentCutOverlayItems.ghostControlPointNext = new QGraphicsPathItem();
        _fastCurrentCutOverlayItems.ghostControlPointNext->setPen(ghostControlPen);
        _fastCurrentCutOverlayItems.ghostControlPointNext->setBrush(Qt::NoBrush);
        _fastCurrentCutOverlayItems.ghostControlPointNext->setZValue(155.0);

        viewer->setOverlayGroup(kGeneratedDynamicCurrentCutOverlayKey,
                                {_fastCurrentCutOverlayItems.centerPoint,
                                 _fastCurrentCutOverlayItems.controlPoints,
                                 _fastCurrentCutOverlayItems.seedPoints,
                                 _fastCurrentCutOverlayItems.linkCandidatePoints,
                                 _fastCurrentCutOverlayItems.splitCandidatePoints,
                                 _fastCurrentCutOverlayItems.branchControlPoints,
                                 _fastCurrentCutOverlayItems.pendingBranchControlPoints,
                                 _fastCurrentCutOverlayItems.sameHvBranchControlPoints,
                                 _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints,
                                 _fastCurrentCutOverlayItems.fiberIntersections,
                                 _fastCurrentCutOverlayItems.linkCandidateFiberIntersections,
                                 _fastCurrentCutOverlayItems.branchLinkFiberIntersections,
                                 _fastCurrentCutOverlayItems.pendingBranchLinkFiberIntersections,
                                 _fastCurrentCutOverlayItems.fiberIntersectionConnectors,
                                 _fastCurrentCutOverlayItems.ghostControlPointPrev,
                                 _fastCurrentCutOverlayItems.ghostControlPointNext});
    }

    // During an in-place update, draw from the held pre-update views until this
    // pane's first re-optimized frame lands.
    const GeneratedViews& cutViews =
        _currentCutOverlaySwapPending ? _heldGeneratedViews : _generatedViews;
    const auto& cutIndex =
        _currentCutOverlaySwapPending ? _heldControlIndex : _generatedControlIndex;
    const double cutPosition =
        _currentCutOverlaySwapPending ? _heldLinePosition : _currentLinePosition;

    QPointF centerScenePoint;
    if (normalOffsetActive(_currentCutNormalOffsetVx)) {
        centerScenePoint = viewer->volumeToScene(
            vc3d::line_annotation::interpolatedGeneratedLinePoint(
                cutViews.linePoints, cutPosition));
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
    QPainterPath splitCandidatePath;
    QPainterPath branchControlPath;
    QPainterPath pendingBranchControlPath;
    QPainterPath sameHvBranchControlPath;
    QPainterPath sameHvPendingBranchControlPath;
    const double lineRadius =
        std::max(0.5, (_viewerManager ? _viewerManager->zScrollSensitivity() : 1.0) * 0.5);
    const double lower = cutPosition - lineRadius;
    const double upper = cutPosition + lineRadius;
    const auto& indices = cutIndex.sortedControlIndices;
    const auto positionForIndex = [&cutViews](size_t controlIndex) {
        return cutViews.controlPoints[controlIndex].linePosition;
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
        if (controlIndex >= cutViews.controlPoints.size()) {
            continue;
        }
        const auto& control = cutViews.controlPoints[controlIndex];
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
        if (control.isSplitCandidate) {
            splitCandidatePath.addEllipse(scenePoint, control.isSeed ? 11.0 : 10.0,
                                          control.isSeed ? 11.0 : 10.0);
        } else if (control.isLinkCandidate) {
            linkCandidatePath.addEllipse(scenePoint, control.isSeed ? 11.0 : 10.0,
                                         control.isSeed ? 11.0 : 10.0);
        } else if (control.hasPendingLinks) {
            (control.hasSameHvPendingLinks ? sameHvPendingBranchControlPath
                                           : pendingBranchControlPath)
                .addEllipse(scenePoint, 12.0, 12.0);
        } else if (control.hasBranches) {
            (control.hasSameHvBranches ? sameHvBranchControlPath : branchControlPath)
                .addEllipse(scenePoint, 12.0, 12.0);
        } else if (control.isSeed) {
            seedPath.addEllipse(scenePoint, 11.0, 11.0);
        } else {
            controlPath.addEllipse(scenePoint, 10.0, 10.0);
        }
    }
    _fastCurrentCutOverlayItems.controlPoints->setPath(controlPath);
    _fastCurrentCutOverlayItems.seedPoints->setPath(seedPath);
    _fastCurrentCutOverlayItems.linkCandidatePoints->setPath(linkCandidatePath);
    _fastCurrentCutOverlayItems.splitCandidatePoints->setPath(splitCandidatePath);
    _fastCurrentCutOverlayItems.branchControlPoints->setPath(branchControlPath);
    _fastCurrentCutOverlayItems.pendingBranchControlPoints->setPath(pendingBranchControlPath);
    _fastCurrentCutOverlayItems.sameHvBranchControlPoints->setPath(sameHvBranchControlPath);
    _fastCurrentCutOverlayItems.sameHvPendingBranchControlPoints->setPath(
        sameHvPendingBranchControlPath);

    // Parallax ghosts: the nearest control point behind and ahead of the cursor
    // within the visibility distance slide in horizontally from the side they
    // will arrive from and land on the solid marker as their delta hits 0.
    double ghostMaxSceneOffset = 0.0;
    if (auto* ghostView = viewer->graphicsView()) {
        if (auto* ghostViewport = ghostView->viewport();
            ghostViewport && ghostViewport->width() > 0 && ghostViewport->height() > 0) {
            const QRect ghostViewportRect = ghostViewport->rect();
            const QPointF leftScene = ghostView->mapToScene(ghostViewportRect.topLeft());
            const QPointF rightScene = ghostView->mapToScene(ghostViewportRect.topRight());
            if (std::isfinite(leftScene.x()) && std::isfinite(rightScene.x())) {
                ghostMaxSceneOffset = std::abs(rightScene.x() - leftScene.x()) *
                                      kGeneratedGhostMaxOffsetViewportFraction;
            }
        }
    }

    const auto ghostPathForDirection = [&](int direction) {
        QPainterPath ghostPath;
        const auto ghost = vc3d::line_annotation::generatedParallaxGhost(
            cutViews.controlPoints,
            cutIndex,
            cutPosition,
            direction,
            kGeneratedGhostSlideRangeLinePositions,
            kGeneratedGhostVisibilityRadiusMultiplier * lineRadius);
        if (!ghost || ghost->controlIndex >= cutViews.controlPoints.size()) {
            return std::make_pair(ghostPath, 0.0);
        }
        const auto& control = cutViews.controlPoints[ghost->controlIndex];
        if (!finitePoint(control.point)) {
            return std::make_pair(ghostPath, 0.0);
        }
        const QPointF landingScenePoint = viewer->volumeToScene(control.point);
        if (!std::isfinite(landingScenePoint.x()) || !std::isfinite(landingScenePoint.y())) {
            return std::make_pair(ghostPath, 0.0);
        }
        // Higher line position is to the right in the strips; keep that sense here.
        const QPointF ghostScenePoint(
            landingScenePoint.x() + ghost->offsetFraction * ghostMaxSceneOffset,
            landingScenePoint.y());
        ghostPath.addEllipse(ghostScenePoint, kGeneratedGhostRadius, kGeneratedGhostRadius);
        return std::make_pair(ghostPath, ghost->opacity);
    };

    const auto [nextGhostPath, nextGhostOpacity] = ghostPathForDirection(1);
    _fastCurrentCutOverlayItems.ghostControlPointNext->setPath(nextGhostPath);
    _fastCurrentCutOverlayItems.ghostControlPointNext->setOpacity(nextGhostOpacity);

    const auto [prevGhostPath, prevGhostOpacity] = ghostPathForDirection(-1);
    _fastCurrentCutOverlayItems.ghostControlPointPrev->setPath(prevGhostPath);
    _fastCurrentCutOverlayItems.ghostControlPointPrev->setOpacity(prevGhostOpacity);

    QPainterPath fiberIntersectionPath;
    QPainterPath linkCandidateFiberIntersectionPath;
    QPainterPath branchLinkFiberIntersectionPath;
    QPainterPath pendingBranchLinkFiberIntersectionPath;
    QPainterPath fiberIntersectionConnectorPath;
    auto* currentCutPlane = cutViews.currentCutSurface.get();
    const std::optional<float> intersectionThreshold =
        (currentCutPlane && !cutViews.fiberIntersections.empty())
            ? vc3d::line_annotation::generatedCrossSliceControlPointDistanceThreshold(viewer)
            : std::nullopt;
    if (intersectionThreshold) {
        constexpr qreal kIntersectionArm = 7.5;
        for (const auto& intersection : cutViews.fiberIntersections) {
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
    // One sign for the whole fiber (see generatedDisplayTangentSign): the cut
    // planes are posed from this tangent, so the displayed left/right must not
    // follow the stored point order.
    return normalizedOrNan(tangent) * _displayTangentSign;
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

cv::Vec3f LineAnnotationDialog::interpolatedOrientedNormal(double linePosition) const
{
    const cv::Vec3f nan{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    const auto& normals = _generatedViews.lineNormals;
    if (normals.empty() || !std::isfinite(linePosition)) {
        return nan;
    }

    const double position = std::clamp(
        linePosition, 0.0, static_cast<double>(normals.size() - 1));
    const int lower = static_cast<int>(std::floor(position));
    const int upper = std::min<int>(lower + 1, static_cast<int>(normals.size()) - 1);
    const cv::Vec3f lowerNormal = normals[static_cast<size_t>(lower)];
    cv::Vec3f upperNormal = normals[static_cast<size_t>(upper)];
    if (!finitePoint(lowerNormal) || !finitePoint(upperNormal) ||
        cv::norm(lowerNormal) <= 1.0e-6f ||
        cv::norm(upperNormal) <= 1.0e-6f) {
        return nan;
    }
    if (lowerNormal.dot(upperNormal) < 0.0f) {
        upperNormal = -upperNormal;
    }

    const float t = static_cast<float>(position - static_cast<double>(lower));
    return normalizedOrNan(lowerNormal * (1.0f - t) + upperNormal * t);
}

cv::Vec3f LineAnnotationDialog::interpolatedLineNormal(double linePosition,
                                                       const cv::Vec3f& tangent) const
{
    const cv::Vec3f nan{std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
    cv::Vec3f up = interpolatedOrientedNormal(linePosition);
    if (!finitePoint(up)) {
        return nan;
    }
    up -= tangent * up.dot(tangent);
    // A tiny projection means the sheet normal runs nearly along the tangent
    // (extreme bend): the direction is unstable, so let the caller fall back.
    if (cv::norm(up) <= 0.1f) {
        return nan;
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
    // Display up for the cut: the sampled sheet normal, sign-oriented away
    // from the scroll center (with the scene's y-down rendering this shows
    // the sheet horizontal with the fiber's center-facing surface at the top
    // of the view) -- and the orientation is a pure function of the line
    // position. The transported line up remains the fallback where samples
    // are missing or, at extreme bends, nearly parallel to the tangent.
    cv::Vec3f upHint = interpolatedLineNormal(linePosition, tangent);
    if (!finitePoint(upHint)) {
        upHint = interpolatedLineUp(linePosition, tangent);
    }
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

bool LineAnnotationDialog::updateSidePlaneSurface(PlaneSurface* plane, double linePosition) const
{
    if (!plane) {
        return false;
    }
    const cv::Vec3f origin = interpolatedLinePoint(linePosition);
    const cv::Vec3f tangent = interpolatedLineTangent(linePosition);
    if (!finitePoint(origin) || !finitePoint(tangent) ||
        cv::norm(tangent) <= 1.0e-6f) {
        return false;
    }
    // The side cut is the current cut's frame rotated a quarter turn: the
    // plane spanned by the line tangent and the (away-from-center oriented)
    // sheet normal, viewed with the fiber vertical. A windowed PCA fit of
    // the line was used here before, but its plane tumbles where the fiber's
    // cross-fiber wander is nearly isotropic (no well-defined local fiber
    // plane); the sheet frame is always defined and keeps both cut views
    // consistent. normal = sheetNormal x tangent puts the stored sheet
    // normal on the view's x axis, so the direction shown at the top of the
    // current cut (toward the scroll center) points screen-left.
    cv::Vec3f sheetNormal = interpolatedLineNormal(linePosition, tangent);
    if (!finitePoint(sheetNormal)) {
        sheetNormal = interpolatedLineUp(linePosition, tangent);
    }
    if (!finitePoint(sheetNormal) || cv::norm(sheetNormal) <= 1.0e-6f) {
        return false;
    }
    const cv::Vec3f normal = normalizedOrNan(sheetNormal.cross(tangent));
    if (!finitePoint(normal)) {
        return false;
    }
    plane->setFromNormalAndUp(origin, normal, tangent);
    return true;
}

QPointF LineAnnotationDialog::stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                                       QuadSurface* surface,
                                                       double linePosition) const
{
    return vc3d::line_annotation::generatedStripLinePositionToScene(
        viewer, surface, linePosition, &_generatedViews.stripPositionMap);
}

void LineAnnotationDialog::keyPressEvent(QKeyEvent* event)
{
    if (handleKeyPress(event)) {
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void LineAnnotationDialog::keyReleaseEvent(QKeyEvent* event)
{
    if (handleKeyRelease(event)) {
        return;
    }
    QMainWindow::keyReleaseEvent(event);
}

void LineAnnotationDialog::changeEvent(QEvent* event)
{
    QMainWindow::changeEvent(event);
    if (event && event->type() == QEvent::ActivationChange && !isActiveWindow()) {
        stopArrowPanForFocusLoss();
    }
}

bool LineAnnotationDialog::event(QEvent* event)
{
    if (event && event->type() == QEvent::WindowDeactivate) {
        stopArrowPanForFocusLoss();
    }
    return QMainWindow::event(event);
}

void LineAnnotationDialog::stopArrowPanForFocusLoss()
{
    // The key-up goes to whichever window took focus (Alt-Tab mid-hold), and
    // an inactive window must not keep rendering a pan at all: braking into
    // the next target can mean minutes of four-pane rendering at low speeds
    // with sparse control points. Cancel outright, like hideEvent.
    _arrowKeyLeftDown = false;
    _arrowKeyRightDown = false;
    cancelArrowPan();
}

void LineAnnotationDialog::hideEvent(QHideEvent* event)
{
    QMainWindow::hideEvent(event);
    // Hiding the workspace in place (an embedding tab switch) fires no
    // activation change, and the key-up then goes elsewhere; a pan running in
    // a hidden view is pure waste, so stop it outright.
    _arrowKeyLeftDown = false;
    _arrowKeyRightDown = false;
    cancelArrowPan();
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

bool LineAnnotationDialog::placeControlPointAtCurrentLinePosition()
{
    // A click cannot reach a busy dialog because the optimization overlay
    // covers the panes, but the keys still arrive; the controller would reject
    // the request and leave _pendingPlacementFocus behind for the running
    // optimization to land on.
    if (!_hasGeneratedViews || !_currentCutViewer || _optimizationBusy ||
        !controlPointPlacementAllowedAt(_currentLinePosition)) {
        return false;
    }
    // The current cut's plane origin, i.e. exactly the blue dot the key aims at.
    const cv::Vec3f volumePoint = interpolatedLinePoint(_currentLinePosition);
    if (!finitePoint(volumePoint)) {
        return false;
    }
    // Unlike a click in the cut pane this leaves hover-follow as the user set it
    // -- the key is meant to be tapped mid-pan, where follow is deliberately
    // paused -- so, like the shift-click snap, it has to stop the pan itself:
    // the placement renumbers the line positions the pan is steering by.
    cancelArrowPan();
    _pendingPlacementFocus = volumePoint;
    emit generatedControlPointRequested(_generatedViews.currentCutName,
                                        volumePoint,
                                        _currentLinePosition);
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
    if (event->key() == Qt::Key_B && event->modifiers() == Qt::NoModifier &&
        !event->isAutoRepeat()) {
        resetGeneratedNormalOffsets();
        event->accept();
        return true;
    }
    // Handled here rather than as a QShortcut so keyboardFocusIsTextEntry() can
    // keep "0" typeable in the menu's spinboxes and the toolbar combos. The
    // keypad variants carry KeypadModifier (numpad / always, numpad 0 with Num
    // Lock on), so mask it out rather than require NoModifier outright.
    if ((event->key() == Qt::Key_Slash || event->key() == Qt::Key_0) &&
        (event->modifiers() & ~Qt::KeypadModifier) == Qt::NoModifier &&
        !event->isAutoRepeat() && !keyboardFocusIsTextEntry()) {
        if (placeControlPointAtCurrentLinePosition()) {
            event->accept();
            return true;
        }
    }
    // Arrows drive the control-point pan here; accepting all four also keeps the
    // viewers' own 64 px arrow panning out of this dialog.
    const bool arrowKey = event->key() == Qt::Key_Left || event->key() == Qt::Key_Right ||
                          event->key() == Qt::Key_Up || event->key() == Qt::Key_Down;
    if (arrowKey && _hasGeneratedViews && !keyboardFocusIsTextEntry()) {
        if (event->modifiers() == Qt::NoModifier) {
            switch (event->key()) {
            case Qt::Key_Left:
            case Qt::Key_Right:
                // Auto-repeat is the keyboard's own hold; the integrator already has one.
                if (!event->isAutoRepeat()) {
                    (event->key() == Qt::Key_Left ? _arrowKeyLeftDown
                                                  : _arrowKeyRightDown) = true;
                    startArrowPan(event->key() == Qt::Key_Left ? -1 : 1);
                }
                event->accept();
                return true;
            case Qt::Key_Up:
            case Qt::Key_Down:
                adjustArrowPanCruiseSpeed(
                    event->key() == Qt::Key_Up
                        ? vc3d::line_annotation::kGeneratedArrowPanSpeedStep
                        : 1.0 / vc3d::line_annotation::kGeneratedArrowPanSpeedStep);
                event->accept();
                return true;
            default:
                break;
            }
        } else if (_arrowPanDirection != 0) {
            // A modifier picked up mid-gesture must not leak arrow repeats to
            // the viewers' own panning while the keyboard pan owns the keys.
            event->accept();
            return true;
        }
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

bool LineAnnotationDialog::handleKeyRelease(QKeyEvent* event)
{
    if (!event || !_hasGeneratedViews) {
        return false;
    }
    if (event->key() != Qt::Key_Left && event->key() != Qt::Key_Right) {
        return false;
    }
    const int direction = (event->key() == Qt::Key_Left) ? -1 : 1;
    if (!event->isAutoRepeat()) {
        (direction < 0 ? _arrowKeyLeftDown : _arrowKeyRightDown) = false;
    }
    // The release that ends an active hold must be honored even when focus
    // moved into a text widget mid-gesture; otherwise the hold is stranded and
    // the pan cruises on to the boundary.
    const bool endsActiveHold = (_arrowPanDirection == direction) && _arrowPanKeyHeld;
    if (!endsActiveHold && keyboardFocusIsTextEntry()) {
        return false;
    }
    // Auto-repeat delivers a release before every repeated press; only the real
    // key-up ends the hold. Modifiers are not checked here: a modifier picked up
    // mid-gesture must not strand the pan in its held state.
    if (!event->isAutoRepeat()) {
        releaseArrowPanKey(direction);
        // Hand the pan back to the other horizontal key if it is still held
        // (hold Right, tap Left to peek back, keep holding Right). Only when
        // this release ended a live hold or the pan finished by landing - a
        // bare physical flag must not revive a pan that space or an edit
        // already cancelled - and never while a text widget owns the keyboard.
        const bool otherStillDown = (direction < 0) ? _arrowKeyRightDown : _arrowKeyLeftDown;
        const bool panEndedNaturally = (_arrowPanDirection == 0) && _arrowPanEndedByLanding;
        if ((endsActiveHold || panEndedNaturally) && otherStillDown &&
            !keyboardFocusIsTextEntry()) {
            startArrowPan(-direction);
        }
    }
    event->accept();
    return true;
}

bool LineAnnotationDialog::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == _fiberNameLabel && event->type() == QEvent::Resize) {
        updateFiberNameLabel();
    }
    if (_pauseIndicator && watched == _pauseIndicator->parentWidget() &&
        event->type() == QEvent::Resize) {
        updatePauseIndicator();
    }
    if (_optimizationStatusLabel &&
        watched == _optimizationStatusLabel->parentWidget() &&
        event->type() == QEvent::Resize) {
        updateOptimizationStatusIndicator();
    }
    if (_arrowPanSpeedLabel && watched == _arrowPanSpeedLabel->parentWidget() &&
        event->type() == QEvent::Resize) {
        updateArrowPanSpeedIndicator();
    }
    if (event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (handleKeyPress(keyEvent)) {
            return true;
        }
    }
    // The dialog is the event filter on every pane widget (viewer, graphics
    // view, viewport, splitters), so the key-up that ends an arrow hold arrives
    // here whichever of them holds focus.
    if (event->type() == QEvent::KeyRelease) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (handleKeyRelease(keyEvent)) {
            return true;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void LineAnnotationDialog::updateOverviewBar()
{
    auto* bar = static_cast<LineAnnotationOverviewBar*>(_overviewBar.data());
    if (!bar || !_hasGeneratedViews) {
        return;
    }

    std::vector<LineAnnotationOverviewBar::ControlDot> dots;
    dots.reserve(_generatedViews.controlPoints.size());
    for (const auto& control : _generatedViews.controlPoints) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        LineAnnotationOverviewBar::ControlDot dot;
        dot.linePosition = control.linePosition;
        // Same state palette as the cut-view overlays.
        if (control.isSplitCandidate) {
            dot.color = QColor(235, 60, 60);
        } else if (control.isLinkCandidate) {
            dot.color = QColor(60, 235, 120);
        } else if (control.hasPendingLinks) {
            dot.color = control.hasSameHvPendingLinks ? QColor(255, 190, 120)
                                                      : QColor(80, 150, 255);
        } else if (control.hasBranches) {
            dot.color = control.hasSameHvBranches ? QColor(255, 140, 0)
                                                  : QColor(210, 95, 255);
        } else {
            dot.color = QColor(255, 230, 0);
        }
        dot.radius = control.isSeed ? 6.0 : 4.5;
        dots.push_back(dot);
    }
    bar->setLineData(_generatedViews.linePoints.size(), std::move(dots));

    QColor markerColor(0, 245, 255);
    switch (currentLineMarkerState()) {
    case vc3d::line_annotation::GeneratedCurrentLineMarkerState::Allowed:
        markerColor = QColor(40, 220, 120);
        break;
    case vc3d::line_annotation::GeneratedCurrentLineMarkerState::Blocked:
        markerColor = QColor(255, 70, 70);
        break;
    default:
        break;
    }
    bar->setCurrentPosition(_currentLinePosition, markerColor);
}

void LineAnnotationDialog::forwardOverviewControlContextMenu(double linePosition,
                                                             QPoint globalPos)
{
    if (!_hasGeneratedViews || _stripViewers.empty()) {
        return;
    }
    CChunkedVolumeViewer* strip = _stripViewers.back().data();
    auto* quad = strip ? dynamic_cast<QuadSurface*>(strip->currentSurface()) : nullptr;
    auto* view = strip ? strip->graphicsView() : nullptr;
    if (!quad || !view) {
        return;
    }
    const QPointF scenePoint = vc3d::line_annotation::generatedStripLinePositionToScene(
        strip, quad, linePosition, &_generatedViews.stripPositionMap);
    if (!std::isfinite(scenePoint.x()) || !std::isfinite(scenePoint.y())) {
        return;
    }
    // Emit the strip view's own context-menu signal: CWindow routes it through
    // the controller, which supplies link-candidate state and shows the same
    // menu an in-viewer Ctrl+right-click would. The synthesized scene point
    // round-trips to the clicked control point's line position, and the menu
    // pops at the overview-bar cursor via globalPos.
    QMetaObject::invokeMethod(
        view,
        "sendAnnotationContextMenuRequested",
        Qt::DirectConnection,
        Q_ARG(QPointF, scenePoint),
        Q_ARG(QPoint, globalPos),
        Q_ARG(Qt::KeyboardModifiers, Qt::KeyboardModifiers(Qt::ControlModifier)));
}

void LineAnnotationDialog::snapPanesToOverviewCursor()
{
    auto* bar = static_cast<LineAnnotationOverviewBar*>(_overviewBar.data());
    if (!_hasGeneratedViews || !bar) {
        return;
    }
    const QPoint localPos = bar->mapFromGlobal(QCursor::pos());
    if (!bar->rect().contains(localPos)) {
        return;
    }
    const auto position = bar->linePositionAtLocalX(localPos.x());
    if (!position) {
        return;
    }
    // The snap supersedes a running keyboard pan; without the cancel the next
    // tick would drag the position straight back toward the pan's old target.
    cancelArrowPan();
    setCurrentLinePosition(*position);
    // Recenter the bottom strip on the snapped position (keeping its zoom), so the
    // current-position marker can't end up outside a zoomed-in viewport.
    for (const auto& stripViewer : _stripViewers) {
        if (!stripViewer) {
            continue;
        }
        if (const auto center = generatedStripSurfaceCenter(
                stripViewer, *position, &_generatedViews.stripPositionMap)) {
            CChunkedVolumeViewer::CameraState camera = stripViewer->cameraState();
            camera.surfacePtrX = (*center)[0];
            camera.surfacePtrY = (*center)[1];
            stripViewer->applyCameraState(camera, false);
        }
    }
}

void LineAnnotationDialog::syncLinkedStripCamera(CChunkedVolumeViewer* source)
{
    if (_syncingStripCameras || _closing || !source) {
        return;
    }
    const auto sourceCamera = source->cameraState();
    for (const auto& stripViewer : _stripViewers) {
        auto* other = stripViewer.data();
        if (!other || other == source) {
            continue;
        }
        auto camera = other->cameraState();
        // Exact comparison on purpose: the fields are copied verbatim, so the
        // converged state is bit-identical and this is the loop terminator for
        // the overlaysUpdated echo from applyCameraState.
        if (camera.surfacePtrX == sourceCamera.surfacePtrX &&
            camera.scale == sourceCamera.scale) {
            continue;
        }
        camera.surfacePtrX = sourceCamera.surfacePtrX;
        camera.scale = sourceCamera.scale;
        _syncingStripCameras = true;
        other->applyCameraState(camera, false);
        _syncingStripCameras = false;
    }
}

void LineAnnotationDialog::updateOptimizationOverlayGeometry()
{
    if (!centralWidget()) {
        return;
    }
    if (_optimizationOverlay) {
        _optimizationOverlay->setGeometry(centralWidget()->rect());
        if (_optimizationOverlay->isVisible()) {
            _optimizationOverlay->raise();
        }
    }
    if (_optimizationBadge) {
        const QSize badgeSize = _optimizationBadge->sizeHint();
        const QRect content = centralWidget()->rect();
        _optimizationBadge->setGeometry(
            content.center().x() - badgeSize.width() / 2,
            content.top() + 8,
            badgeSize.width(),
            badgeSize.height());
        if (_optimizationBadge->isVisible()) {
            _optimizationBadge->raise();
        }
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

    const QString hvSuffix = _fiberHvTag.isEmpty()
        ? QString()
        : QStringLiteral("  [%1]").arg(_fiberHvTag);
    const QFontMetrics metrics = _fiberNameLabel->fontMetrics();
    const int nameWidth = _fiberNameLabel->contentsRect().width() -
        (hvSuffix.isEmpty() ? 0 : metrics.horizontalAdvance(hvSuffix));
    _fiberNameLabel->setText(
        vc3d::adaptFiberNameToWidth(fullName, metrics, nameWidth) + hvSuffix);
    _fiberNameLabel->setToolTip(fullName + hvSuffix);
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
