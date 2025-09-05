#pragma once

#include <QWidget>
#include <QList>
#include <QMap>
#include <QColor>
#include <memory>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>
#include <opencv2/core.hpp>

#include "PathData.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Slicing.hpp"


/**
 * @brief Widget for freehand drawing on volume surfaces
 * 
 * This widget provides tools for drawing masks and annotations on volume surfaces
 * with configurable brush settings, eraser mode, and mask export functionality.
 */
class DrawingWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit DrawingWidget(QWidget* parent = nullptr);
    ~DrawingWidget();
    
    /** Set the volume package */
    void setVolumePkg(std::shared_ptr<VolumePkg> vpkg);
    
    /** Set the current volume */
    void setCurrentVolume(std::shared_ptr<Volume> volume);
    
    /** Set the cache for volume data access */
    void setCache(ChunkCache* cache);
    
    /** Clear all drawn paths */
    void clearAllPaths();
    
    /** Get current path ID */
    int getCurrentPathId() const { return currentPathId; }
    
    /** Get current brush size */
    float getBrushSize() const { return brushSize; }
    
    /** Check if in eraser mode */
    bool isEraserMode() const { return eraserMode; }
    
    /** Check if drawing mode is active */
    bool isDrawingModeActive() const { return drawingModeActive; }
    
    /** Get current brush shape */
    PathData::BrushShape getBrushShape() const { return brushShape; }
    
    /** Toggle drawing mode */
    void toggleDrawingMode();

public slots:
    /** Handle volume change */
    void onVolumeChanged(std::shared_ptr<Volume> vol);
    void onVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId);
    
    /** Handle mouse events from volume viewers */
    void onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    
    /** Handle Z-slice changes */
    void updateCurrentZSlice(int z);
    
    /** Handle surface reload */
    void onSurfacesLoaded();

signals:
    /** Emitted when paths change */
    void sendPathsChanged(const QList<PathData>& paths);
    
    /** Emitted to show status messages */
    void sendStatusMessageAvailable(const QString& message, int timeout);
    
    /** Emitted when drawing mode is active/inactive */
    void sendDrawingModeActive(bool active);

private slots:
    /** UI control handlers */
    void onPathIdChanged(int value);
    void onBrushSizeChanged(int value);
    void onOpacityChanged(int value);
    void onEraserToggled(bool checked);
    void onBrushShapeChanged();
    void onClearAllClicked();
    void onSaveAsMaskClicked();
    void onColorButtonClicked();

private:
    /** Initialize UI components */
    void setupUI();
    
    /** Update UI based on current state */
    void updateUI();
    
    /** Start drawing a new path */
    void startDrawing(cv::Vec3f startPoint);
    
    /** Add point to current path */
    void addPointToPath(cv::Vec3f point);
    
    /** Finalize current path */
    void finalizePath();
    
    /** Get or create color for path ID */
    QColor getColorForPathId(int pathId);
    
    /** Update the color preview button */
    void updateColorPreview();
    
    /** Generate mask from drawn paths */
    cv::Mat generateMask();
    
    /** Save mask to file */
    void saveMask(const cv::Mat& mask, const std::string& filename);
    
    /** Check if a volume point is valid (within bounds and not -1) */
    bool isValidVolumePoint(const cv::Vec3f& point) const;
    
    /** Process paths to apply eraser operations */
    QList<PathData> processPathsWithErasers(const QList<PathData>& rawPaths) const;
    
    /** Calculate distance from point to line segment */
    float pointToSegmentDistance(const cv::Vec3f& point, const cv::Vec3f& segStart, const cv::Vec3f& segEnd) const;
    
    /** Check if a point is within eraser brush */
    bool isPointInEraserBrush(const cv::Vec3f& point, const cv::Vec3f& eraserPoint, 
                              float eraserRadius, PathData::BrushShape brushShape) const;

private:
    // Volume data
    std::shared_ptr<VolumePkg> fVpkg;
    std::shared_ptr<Volume> currentVolume;
    std::string currentVolumeId;
    ChunkCache* chunkCache;
    
    // Drawing state
    int currentPathId;
    float brushSize;
    float opacity;
    bool eraserMode;
    PathData::BrushShape brushShape;
    
    // Path management
    QList<PathData> drawnPaths;
    PathData currentPath;
    bool isDrawing;
    cv::Vec3f lastPoint;
    int currentZSlice;
    
    // Color management
    QMap<int, QColor> pathIdColors;
    
    // Drawing mode state
    bool drawingModeActive;
    
    // Temporary eraser state (for shift-to-erase)
    bool temporaryEraserMode;
    bool originalEraserMode;
    
    // UI elements
    QLabel* infoLabel;
    QPushButton* toggleModeButton;
    QSpinBox* pathIdSpinBox;
    QSlider* brushSizeSlider;
    QLabel* brushSizeLabel;
    QSlider* opacitySlider;
    QLabel* opacityLabel;
    QCheckBox* eraserCheckBox;
    QButtonGroup* brushShapeGroup;
    QRadioButton* circleRadio;
    QRadioButton* squareRadio;
    QPushButton* colorButton;
    QPushButton* clearAllButton;
    QPushButton* saveAsMaskButton;
};


