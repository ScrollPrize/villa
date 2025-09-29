#include "DrawingWidget.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QSlider>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QColorDialog>
#include <QMessageBox>
#include <QFileDialog>
#include <QPainter>
#include <QPainterPath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Logging.hpp"

#include <filesystem>
#include <fstream>

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;
using PathRenderMode = ViewerOverlayControllerBase::PathRenderMode;






DrawingWidget::DrawingWidget(QWidget* parent)
    : QWidget(parent)
    , fVpkg(nullptr)
    , currentVolume(nullptr)
    , chunkCache(nullptr)
    , currentPathId(0)
    , brushSize(3.0f)
    , opacity(1.0f)
    , eraserMode(false)
    , brushShape(PathBrushShape::Circle)
    , isDrawing(false)
    , currentZSlice(0)
    , drawingModeActive(false)
    , temporaryEraserMode(false)
    , originalEraserMode(false)
{
    qRegisterMetaType<PathPrimitive>("PathPrimitive");
    qRegisterMetaType<QList<PathPrimitive>>("QList<PathPrimitive>");
    setupUI();
    
    // Initialize with some default colors for path IDs
    pathIdColors[0] = QColor(255, 100, 100);  // Red
    pathIdColors[1] = QColor(100, 255, 100);  // Green
    pathIdColors[2] = QColor(100, 100, 255);  // Blue
    pathIdColors[3] = QColor(255, 255, 100);  // Yellow
    pathIdColors[4] = QColor(255, 100, 255);  // Magenta
    
    updateColorPreview();
}

DrawingWidget::~DrawingWidget() = default;

void DrawingWidget::setupUI()
{
    auto mainLayout = new QVBoxLayout(this);
    
    // Drawing mode toggle button
    toggleModeButton = new QPushButton("Enable Drawing Mode (D)", this);
    toggleModeButton->setCheckable(true);
    toggleModeButton->setToolTip("Toggle drawing mode on/off (D key)");
    mainLayout->addWidget(toggleModeButton);
    
    // Info label
    infoLabel = new QLabel("Drawing mode disabled", this);
    mainLayout->addWidget(infoLabel);
    
    mainLayout->addSpacing(10);
    
    // Path ID control with color preview
    auto pathIdLayout = new QHBoxLayout();
    pathIdLayout->addWidget(new QLabel("Path ID:", this));
    
    pathIdSpinBox = new QSpinBox(this);
    pathIdSpinBox->setRange(0, 999);
    pathIdSpinBox->setValue(0);
    pathIdSpinBox->setToolTip("Current path ID - all drawn paths will use this ID");
    pathIdLayout->addWidget(pathIdSpinBox);
    
    colorButton = new QPushButton(this);
    colorButton->setFixedSize(30, 30);
    colorButton->setToolTip("Click to change color for this path ID");
    pathIdLayout->addWidget(colorButton);
    
    pathIdLayout->addStretch();
    mainLayout->addLayout(pathIdLayout);
    
    // Brush size control
    auto brushSizeLayout = new QHBoxLayout();
    brushSizeLayout->addWidget(new QLabel("Brush Size:", this));
    
    brushSizeSlider = new QSlider(Qt::Horizontal, this);
    brushSizeSlider->setRange(1, 50);
    brushSizeSlider->setValue(3);
    brushSizeSlider->setToolTip("Size of the drawing brush in pixels");
    brushSizeLayout->addWidget(brushSizeSlider);
    
    brushSizeLabel = new QLabel("3", this);
    brushSizeLabel->setMinimumWidth(30);
    brushSizeLayout->addWidget(brushSizeLabel);
    
    mainLayout->addLayout(brushSizeLayout);
    
    // Opacity control
    auto opacityLayout = new QHBoxLayout();
    opacityLayout->addWidget(new QLabel("Opacity:", this));
    
    opacitySlider = new QSlider(Qt::Horizontal, this);
    opacitySlider->setRange(0, 100);
    opacitySlider->setValue(100);
    opacitySlider->setToolTip("Transparency of drawn paths (0-100%)");
    opacityLayout->addWidget(opacitySlider);
    
    opacityLabel = new QLabel("100%", this);
    opacityLabel->setMinimumWidth(40);
    opacityLayout->addWidget(opacityLabel);
    
    mainLayout->addLayout(opacityLayout);
    
    // Eraser mode
    eraserCheckBox = new QCheckBox("Eraser Mode", this);
    eraserCheckBox->setToolTip("Toggle eraser mode to remove drawn areas\n"
                                "Tip: Hold Shift while drawing to temporarily erase");
    mainLayout->addWidget(eraserCheckBox);
    
    // Brush shape
    auto brushShapeLayout = new QHBoxLayout();
    brushShapeLayout->addWidget(new QLabel("Brush Shape:", this));
    
    brushShapeGroup = new QButtonGroup(this);
    
    circleRadio = new QRadioButton("Circle", this);
    circleRadio->setChecked(true);
    brushShapeGroup->addButton(circleRadio, 0);
    brushShapeLayout->addWidget(circleRadio);
    
    squareRadio = new QRadioButton("Square", this);
    brushShapeGroup->addButton(squareRadio, 1);
    brushShapeLayout->addWidget(squareRadio);
    
    brushShapeLayout->addStretch();
    mainLayout->addLayout(brushShapeLayout);
    
    // Action buttons
    mainLayout->addSpacing(10);
    
    clearAllButton = new QPushButton("Clear All Paths", this);
    clearAllButton->setToolTip("Remove all drawn paths");
    mainLayout->addWidget(clearAllButton);
    
    saveAsMaskButton = new QPushButton("Save as Mask", this);
    saveAsMaskButton->setToolTip("Export drawn paths as segmentation mask");
    mainLayout->addWidget(saveAsMaskButton);
    
    mainLayout->addStretch();
    
    // Connect signals
    connect(pathIdSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &DrawingWidget::onPathIdChanged);
    connect(brushSizeSlider, &QSlider::valueChanged,
            this, &DrawingWidget::onBrushSizeChanged);
    connect(opacitySlider, &QSlider::valueChanged,
            this, &DrawingWidget::onOpacityChanged);
    connect(eraserCheckBox, &QCheckBox::toggled,
            this, &DrawingWidget::onEraserToggled);
    connect(brushShapeGroup, QOverload<int>::of(&QButtonGroup::idClicked),
            this, &DrawingWidget::onBrushShapeChanged);
    connect(colorButton, &QPushButton::clicked,
            this, &DrawingWidget::onColorButtonClicked);
    connect(clearAllButton, &QPushButton::clicked,
            this, &DrawingWidget::onClearAllClicked);
    connect(saveAsMaskButton, &QPushButton::clicked,
            this, &DrawingWidget::onSaveAsMaskClicked);
    connect(toggleModeButton, &QPushButton::toggled,
            [this](bool checked) {
                drawingModeActive = checked;
                toggleModeButton->setText(checked ? 
                    "Disable Drawing Mode (D)" : "Enable Drawing Mode (D)");
                updateUI();
                emit sendDrawingModeActive(checked);
            });
    
    // Set size policy
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    
}

void DrawingWidget::setVolumePkg(std::shared_ptr<VolumePkg> vpkg)
{
    fVpkg = vpkg;
    updateUI();
}

void DrawingWidget::setCurrentVolume(std::shared_ptr<Volume> volume)
{
    currentVolume = volume;
    updateUI();
}

void DrawingWidget::setCache(ChunkCache* cache)
{
    chunkCache = cache;
}

void DrawingWidget::onVolumeChanged(std::shared_ptr<Volume> vol)
{
    setCurrentVolume(vol);
}

void DrawingWidget::onVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId)
{
    currentVolume = vol;
    currentVolumeId = volumeId;
    updateUI();
}

void DrawingWidget::toggleDrawingMode()
{
    toggleModeButton->setChecked(!toggleModeButton->isChecked());
}

void DrawingWidget::onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (!drawingModeActive || button != Qt::LeftButton) {
        return;
    }
    
    if (!isValidVolumePoint(vol_point)) {
        return;
    }
    
    // Check for shift key to enable temporary eraser mode
    if (modifiers & Qt::ShiftModifier) {
        temporaryEraserMode = true;
        originalEraserMode = eraserMode;
        eraserMode = true;
    }
    
    startDrawing(vol_point);
}

void DrawingWidget::onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    // Check if mouse button is still pressed and we're in drawing mode
    if (!(buttons & Qt::LeftButton) || !drawingModeActive) {
        return;
    }
    
    // Handle invalid points
    if (!isValidVolumePoint(vol_point)) {
        // If we were drawing and hit an invalid point, save the current segment
        if (isDrawing && currentPath.points.size() >= 2) {
            // Save the current path segment with its path ID and color
            drawnPaths.append(currentPath);
            
            // Process paths to apply eraser operations
            QList<PathPrimitive> processedPaths = processPathsWithErasers(drawnPaths);
            emit sendPathsChanged(processedPaths);
            
            // Mark that we're no longer drawing but remember the path settings
            isDrawing = false;
        }
        return;
    }
    
    // If we're back in a valid region but not currently drawing
    if (!isDrawing) {
        // Start a new segment with the same path ID and color
        isDrawing = true;
        // Keep the same path settings but clear the points
        currentPath.points.clear();
        currentPath.points.push_back(vol_point);
        lastPoint = vol_point;
        
        // Show the new segment
        QList<PathPrimitive> allPaths = drawnPaths;
        allPaths.append(currentPath);
        
        // Process paths to apply eraser operations
        QList<PathPrimitive> processedPaths = processPathsWithErasers(allPaths);
        emit sendPathsChanged(processedPaths);
        return;
    }
    
    addPointToPath(vol_point);
}

void DrawingWidget::onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (button != Qt::LeftButton || !isDrawing) {
        return;
    }
    
    finalizePath();
    
    // Restore original eraser mode if we were using temporary eraser mode
    if (temporaryEraserMode) {
        eraserMode = originalEraserMode;
        temporaryEraserMode = false;
        
        // Update UI to reflect the restored state
        eraserCheckBox->setChecked(originalEraserMode);
    }
}

void DrawingWidget::updateCurrentZSlice(int z)
{
    currentZSlice = z;
}

void DrawingWidget::onSurfacesLoaded()
{
    updateUI();
}

void DrawingWidget::onPathIdChanged(int value)
{
    currentPathId = value;
    updateColorPreview();
    infoLabel->setText(QString("Drawing with Path ID %1").arg(currentPathId));
}

void DrawingWidget::onBrushSizeChanged(int value)
{
    brushSize = static_cast<float>(value);
    brushSizeLabel->setText(QString::number(value));
    
    // Update cursor if drawing mode is active
    if (drawingModeActive) {
        emit sendDrawingModeActive(true);
    }
}

void DrawingWidget::onOpacityChanged(int value)
{
    opacity = value / 100.0f;
    opacityLabel->setText(QString("%1%").arg(value));
}

void DrawingWidget::onEraserToggled(bool checked)
{
    eraserMode = checked;
    infoLabel->setText(eraserMode ? "Eraser Mode Active" : "Drawing Mode Active");
}

void DrawingWidget::onBrushShapeChanged()
{
    brushShape = circleRadio->isChecked() ? 
        PathBrushShape::Circle : PathBrushShape::Square;
    
    // Update cursor if drawing mode is active
    if (drawingModeActive) {
        emit sendDrawingModeActive(true);
    }
}

void DrawingWidget::onClearAllClicked()
{
    if (!drawnPaths.isEmpty()) {
        auto reply = QMessageBox::question(this, "Clear All Paths", 
            "Are you sure you want to clear all drawn paths?",
            QMessageBox::Yes | QMessageBox::No);
        
        if (reply == QMessageBox::Yes) {
            clearAllPaths();
        }
    }
}

void DrawingWidget::onSaveAsMaskClicked()
{
    if (!currentVolume || drawnPaths.isEmpty()) {
        QMessageBox::warning(this, "No Data", 
            "No paths to save or volume not loaded.");
        return;
    }
    
    QString filename = QFileDialog::getSaveFileName(this, 
        "Save Mask", QString(), "TIFF Files (*.tif *.tiff)");
    
    if (!filename.isEmpty()) {
        cv::Mat mask = generateMask();
        saveMask(mask, filename.toStdString());
        
        emit sendStatusMessageAvailable(
            QString("Mask saved to %1").arg(filename), 5000);
    }
}

void DrawingWidget::onColorButtonClicked()
{
    QColor current = getColorForPathId(currentPathId);
    QColor newColor = QColorDialog::getColor(current, this, 
        QString("Choose Color for Path ID %1").arg(currentPathId));
    
    if (newColor.isValid()) {
        pathIdColors[currentPathId] = newColor;
        updateColorPreview();
    }
}

void DrawingWidget::clearAllPaths()
{
    drawnPaths.clear();
    emit sendPathsChanged(drawnPaths);
    infoLabel->setText("All paths cleared");
}

void DrawingWidget::updateUI()
{
    bool hasVolume = currentVolume != nullptr;
    saveAsMaskButton->setEnabled(hasVolume && !drawnPaths.isEmpty());
    
    // Update info label based on drawing mode
    if (drawingModeActive) {
        if (eraserMode) {
            infoLabel->setText("Drawing mode: Eraser Active");
        } else {
            infoLabel->setText(QString("Drawing mode: Path ID %1").arg(currentPathId));
        }
    } else {
        infoLabel->setText("Drawing mode disabled");
    }
}

void DrawingWidget::startDrawing(cv::Vec3f startPoint)
{
    isDrawing = true;
    currentPath = PathPrimitive();
    currentPath.points.clear();
    currentPath.points.push_back(startPoint);
    currentPath.color = getColorForPathId(currentPathId);
    currentPath.lineWidth = brushSize;
    currentPath.opacity = opacity;
    currentPath.isEraser = eraserMode;
    currentPath.brushShape = brushShape;
    currentPath.pathId = currentPathId;
    currentPath.renderMode = PathRenderMode::LineStrip;
    currentPath.pointRadius = std::max(1.0f, brushSize * 0.5f);
    currentPath.z = eraserMode ? 26.0 : 25.0;
    
    lastPoint = startPoint;
    
    // Show temporary path
    QList<PathPrimitive> allPaths = drawnPaths;
    allPaths.append(currentPath);
    
    QList<PathPrimitive> processedPaths = processPathsWithErasers(allPaths);
    emit sendPathsChanged(processedPaths);
}

void DrawingWidget::addPointToPath(cv::Vec3f point)
{
    if (!isDrawing) {
        return;
    }
    
    // Only add if there's some distance from the last point
    float distance = cv::norm(point - lastPoint);
    
    if (distance > 0.5f) {  // Minimum distance threshold
        currentPath.points.push_back(point);
        lastPoint = point;
        
        // Update display periodically
        if (currentPath.points.size() % 5 == 0) {
            QList<PathPrimitive> allPaths = drawnPaths;
            allPaths.append(currentPath);
            
            QList<PathPrimitive> processedPaths = processPathsWithErasers(allPaths);
            emit sendPathsChanged(processedPaths);
        }
    }
}

void DrawingWidget::finalizePath()
{
    if (!isDrawing || currentPath.points.size() < 2) {
        isDrawing = false;
        return;
    }
    
    // Add the path to the collection
    drawnPaths.append(currentPath);
    
    isDrawing = false;
    currentPath.points.clear();
    
    // Update UI
    updateUI();
    
    // Process paths to apply eraser operations
    QList<PathPrimitive> processedPaths = processPathsWithErasers(drawnPaths);
    emit sendPathsChanged(processedPaths);
    
    // Update info
    infoLabel->setText(QString("%1 paths drawn").arg(drawnPaths.size()));
}

QColor DrawingWidget::getColorForPathId(int pathId)
{
    if (!pathIdColors.contains(pathId)) {
        // Generate a new color for this ID
        int hue = (pathId * 137) % 360;  
        pathIdColors[pathId] = QColor::fromHsv(hue, 200, 255);
    }
    return pathIdColors[pathId];
}

void DrawingWidget::updateColorPreview()
{
    QColor color = getColorForPathId(currentPathId);
    colorButton->setStyleSheet(QString("background-color: %1; border: 1px solid black;")
        .arg(color.name()));
}

cv::Mat DrawingWidget::generateMask()
{
    if (!currentVolume || drawnPaths.isEmpty()) {
        return cv::Mat();
    }
    
    // Generate a 2D mask at the current Z slice 
    const int width = currentVolume->sliceWidth();
    const int height = currentVolume->sliceHeight();
    
    QImage maskImage(width, height, QImage::Format_Grayscale8);
    maskImage.fill(0); // Start with black (background)
    
    QPainter painter(&maskImage);
    painter.setRenderHint(QPainter::Antialiasing, false); // No antialiasing for masks
    
    // Process paths to apply eraser operations
    QList<PathPrimitive> processedPaths = processPathsWithErasers(drawnPaths);
    
    // Process paths in order
    for (const auto& path : processedPaths) {
        if (path.points.size() < 2) {
            continue;
        }
        
        // Build QPainterPath from 3D points (using only X,Y coordinates)
        QPainterPath painterPath;
        bool firstPoint = true;
        bool hasPointsOnSlice = false;
        
        for (const auto& pt3d : path.points) {
            // Only include points that are on or near the current Z slice
            if (std::abs(pt3d[2] - currentZSlice) < 1.0f) {
                if (firstPoint) {
                    painterPath.moveTo(pt3d[0], pt3d[1]);
                    firstPoint = false;
                } else {
                    painterPath.lineTo(pt3d[0], pt3d[1]);
                }
                hasPointsOnSlice = true;
            }
        }
        
        if (!hasPointsOnSlice) {
            continue;
        }
        
        QPen pen;
        pen.setWidthF(path.lineWidth);
        pen.setCapStyle(path.brushShape == PathBrushShape::Square ? 
                        Qt::SquareCap : Qt::RoundCap);
        pen.setJoinStyle(path.brushShape == PathBrushShape::Square ? 
                         Qt::MiterJoin : Qt::RoundJoin);
        
        // For eraser, use black (0). For drawing, use the path ID + 1
        // (since 0 is reserved for background)
        int grayValue = path.isEraser ? 0 : std::min(path.pathId + 1, 255);
        pen.setColor(QColor(grayValue, grayValue, grayValue));
        
        painter.setPen(pen);
        painter.drawPath(painterPath);
    }
    
    painter.end();
    
    cv::Mat mask(height, width, CV_8UC1);
    
    for (int y = 0; y < height; ++y) {
        const uchar* qImageLine = maskImage.scanLine(y);
        uchar* cvMatLine = mask.ptr<uchar>(y);
        std::memcpy(cvMatLine, qImageLine, width);
    }
    
    return mask;
}

void DrawingWidget::saveMask(const cv::Mat& mask, const std::string& filename)
{
    if (mask.empty()) {
        return;
    }
    
    // Save as TIFF
    cv::imwrite(filename, mask);
}

bool DrawingWidget::isValidVolumePoint(const cv::Vec3f& point) const
{
    // Check if we have a valid volume
    if (!currentVolume) {
        return false;
    }
    
    // Check for invalid marker value (-1)
    if (point[0] < 0 || point[1] < 0 || point[2] < 0) {
        return false;
    }
    
    // Check if the point is within volume bounds
    const float width = static_cast<float>(currentVolume->sliceWidth());
    const float height = static_cast<float>(currentVolume->sliceHeight());
    const float depth = static_cast<float>(currentVolume->numSlices());
    
    if (point[0] >= width || point[1] >= height || point[2] >= depth) {
        return false;
    }
    
    return true;
}

float DrawingWidget::pointToSegmentDistance(const cv::Vec3f& point, const cv::Vec3f& segStart, const cv::Vec3f& segEnd) const
{
    cv::Vec3f segVec = segEnd - segStart;
    float segLengthSq = segVec.dot(segVec);
    
    if (segLengthSq < 0.0001f) {
        // Segment is essentially a point
        return cv::norm(point - segStart);
    }
    
    // Calculate parameter t for the closest point on the line segment
    float t = std::max(0.0f, std::min(1.0f, (point - segStart).dot(segVec) / segLengthSq));
    
    // Find the closest point on the segment
    cv::Vec3f projection = segStart + t * segVec;
    
    // Return distance from point to the projection
    return cv::norm(point - projection);
}

bool DrawingWidget::isPointInEraserBrush(const cv::Vec3f& point, const cv::Vec3f& eraserPoint,
                                          float eraserRadius, PathBrushShape brushShape) const
{
    // For now, we only consider 2D distance (X,Y) as we're drawing on slices
    float dx = point[0] - eraserPoint[0];
    float dy = point[1] - eraserPoint[1];
    
    if (brushShape == PathBrushShape::Circle) {
        float distSq = dx * dx + dy * dy;
        return distSq <= (eraserRadius * eraserRadius);
    } else { // SQUARE
        return std::abs(dx) <= eraserRadius && std::abs(dy) <= eraserRadius;
    }
}

QList<PathPrimitive> DrawingWidget::processPathsWithErasers(const QList<PathPrimitive>& rawPaths) const
{
    QList<PathPrimitive> processedPaths;
    
    // Process paths in chronological order
    for (const auto& path : rawPaths) {
        if (!path.isEraser) {
            // For non-eraser paths, add them to processed list
            processedPaths.append(path);
        } else {
            // For eraser paths, process all previous paths
            QList<PathPrimitive> updatedPaths;
            
            for (const auto& targetPath : processedPaths) {
                // Skip if this is also an eraser path (shouldn't happen, but be safe)
                if (targetPath.isEraser) {
                    updatedPaths.append(targetPath);
                    continue;
                }
                
                // Process this target path against the eraser
                PathPrimitive currentSegment = targetPath;
                currentSegment.points.clear();
                QList<PathPrimitive> segments;
                
                bool inErasedSection = false;
                
                for (size_t i = 0; i < targetPath.points.size(); ++i) {
                    const cv::Vec3f& point = targetPath.points[i];
                    bool isErased = false;
                    
                    // Check if this point is within eraser influence
                    for (size_t j = 0; j < path.points.size(); ++j) {
                        if (isPointInEraserBrush(point, path.points[j], path.lineWidth / 2.0f, path.brushShape)) {
                            isErased = true;
                            break;
                        }
                        
                        // Also check line segments between eraser points
                        if (j > 0) {
                            float dist = pointToSegmentDistance(point, path.points[j-1], path.points[j]);
                            if (dist <= path.lineWidth / 2.0f) {
                                isErased = true;
                                break;
                            }
                        }
                    }
                    
                    if (isErased && !inErasedSection) {
                        // Entering erased section
                        if (currentSegment.points.size() >= 2) {
                            segments.append(currentSegment);
                            currentSegment = targetPath;
                            currentSegment.points.clear();
                        }
                        inErasedSection = true;
                    } else if (!isErased && inErasedSection) {
                        // Exiting erased section
                        inErasedSection = false;
                    }
                    
                    if (!isErased) {
                        currentSegment.points.push_back(point);
                    }
                }
                
                // Add final segment if it has enough points
                if (currentSegment.points.size() >= 2) {
                    segments.append(currentSegment);
                }
                
                // Add all segments from this path
                updatedPaths.append(segments);
            }
            
            processedPaths = updatedPaths;
        }
    }
    
    return processedPaths;
}
