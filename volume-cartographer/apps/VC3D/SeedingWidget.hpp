#pragma once

#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QProcess>
#include <QPointer>
#include <QToolButton>
#include <QFutureWatcher>
#include <opencv2/core/mat.hpp>
#include <memory>

#include "elements/ProgressUtil.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"

class CChunkedVolumeViewer;
class CState;
class ViewerManager;
class SurfacePatchIndex;
class QuadSurface;

class SeedingWidget : public QWidget {
    Q_OBJECT
    
public:
    explicit SeedingWidget(VCCollection* point_collection, CState* state, QWidget* parent = nullptr);
    ~SeedingWidget();
    
    enum class RelWindingIntersectionSource {
        CurrentVolume = 0,
        Patches = 1
    };

    void setState(CState* state);
    void setViewerManager(ViewerManager* viewerManager);

    // --- Agent bridge headless entry points (SPEC §15.2) ---
    // The seeding action entry points are private slots; these one-line public
    // wrappers let the agent bridge invoke the non-blocking ones without a Qt
    // signal connection. Only the genuinely non-blocking actions are wrapped:
    // preview-rays (synchronous point math), cast-rays (QtConcurrent background),
    // and reset-points (pure UI reset). onRunSegmentationClicked /
    // onExpandSeedsClicked / analyzePaths spin a nested QApplication::
    // processEvents loop until child processes finish and are deliberately NOT
    // wrapped (they would violate the bridge's no-nested-event-loop rule, SPEC
    // §1.3). setDialogsSuppressed() gates the precondition QMessageBox::warning
    // calls the wrapped slots would otherwise raise (a static QMessageBox spins a
    // nested event loop), for the bridge's lifetime only.
    void runPreviewRays() { onPreviewRaysClicked(); }
    void runCastRays() { onCastRaysClicked(); }
    void runResetPoints() { onResetPointsClicked(); }
    void setDialogsSuppressed(bool suppressed) { _dialogsSuppressed = suppressed; }
    [[nodiscard]] bool dialogsSuppressed() const { return _dialogsSuppressed; }

signals:
    void sendPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void sendStatusMessageAvailable(QString text, int timeout);
    void relWindingAnnotationModeChanged(bool active);
    
public slots:
    void onSurfacesLoaded();  // Called when surfaces have been loaded/reloaded
    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    
public slots:
    void onVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId);
    void updateCurrentZSlice(int z);
    void onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void setRelWindingAnnotationMode(bool active);
    void setRelWindingIntersectionSource(int source);
    void setRelWindingPatchTolerance(double tolerance);
    
private slots:
    void onPreviewRaysClicked();
    void onClearPreviewClicked();
    void onCastRaysClicked();
    void onCastRaysFinished();
    void onClearPeaksClicked();
    void onRunSegmentationClicked();
    void onExpandSeedsClicked();
    void onResetPointsClicked();
    void onCancelClicked();
    void onNeuralTraceClicked();
    void onNeuralCheckpointBrowseClicked();
    
private:
    // Mode enum
    enum class Mode {
        PointMode,
        DrawMode
    };
    
    void setupUI();
    void computeDistanceTransform();
    void castRays();
    void findPeaksAlongRay(const cv::Vec2f& rayDir, const cv::Vec3f& startPoint);
    void runSegmentation();
    QString findExecutablePath();
    QString findNeuralTracePyPath();
    QString findPythonExecutable();
    void updateParameterPreview();
    void updateModeUI();
    void analyzePaths();
    void findPeaksAlongPath(const ViewerOverlayControllerBase::PathPrimitive& path);
    void startDrawing(cv::Vec3f startPoint);
    void addPointToPath(cv::Vec3f point);
    void finalizePath();
    // Relative winding annotation helpers
    void finalizePathLabelWraps(bool shiftHeld);
    void findPeaksAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    int findPatchIntersectionsAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    std::vector<std::shared_ptr<QuadSurface>> relWindingPatchSurfaces() const;
    QColor generatePathColor();
    void displayPaths();
    void updatePointsDisplay();
    void updateInfoLabel();
    void updateButtonStates();
    
private:
    
    // UI elements
    QLabel* infoLabel;
    QComboBox* collectionComboBox;
    QDoubleSpinBox* angleStepSpinBox;
    QSpinBox* processesSpinBox;
    QSpinBox* ompThreadsSpinBox;
    QSpinBox* thresholdSpinBox;  // Intensity threshold for peak detection
    QSpinBox* windowSizeSpinBox; // Window size for peak detection
    QSpinBox* maxRadiusSpinBox;  // Max radius for ray casting
    QSpinBox* expansionIterationsSpinBox; // Number of expansion iterations
    
    // Layout and label references for hiding/showing
    QHBoxLayout* maxRadiusLayout;
    QLabel* maxRadiusLabel;
    QHBoxLayout* angleStepLayout;
    QLabel* angleStepLabel;
    
    QString executablePath;
    
    QPushButton* previewRaysButton;
    QPushButton* clearPreviewButton;
    QPushButton* castRaysButton;
    QPushButton* clearPeaksButton;
    QPushButton* runSegmentationButton;
    QPushButton* expandSeedsButton;
    QPushButton* resetPointsButton;
    QPushButton* cancelButton;
    QProgressBar* progressBar;
    ProgressUtil* progressUtil;
    
    // Data
    bool _dialogsSuppressed{false};  // agent bridge suppresses precondition dialogs
    CState* _state{nullptr};
    ViewerManager* _viewerManager{nullptr};
    QPointer<CChunkedVolumeViewer> _relWindingDrawViewer;
    int currentZSlice;
    VCCollection* _point_collection;
    cv::Mat distanceTransform;
    
    // Drawing mode data
    Mode currentMode;
    QList<ViewerOverlayControllerBase::PathPrimitive> paths;  
    bool isDrawing;
    ViewerOverlayControllerBase::PathPrimitive currentPath;
    int colorIndex;
    bool labelWrapsMode = false; // special mode built on DrawMode
    RelWindingIntersectionSource _relWindingIntersectionSource{RelWindingIntersectionSource::CurrentVolume};
    double _relWindingPatchTolerance{1.0};
    
    // Process management
    QList<QPointer<QProcess>> runningProcesses;
    bool jobsRunning;

    // Async cast rays
    QFutureWatcher<void>* _castRaysWatcher{nullptr};
    std::vector<cv::Vec3f> _castRaysPeaks;  // Collected by background thread
    bool _castRaysWasPointMode{true};  // Track which mode triggered the cast

    // Neural trace UI and state
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralMaxSize{nullptr};
    QSpinBox* _spinNeuralStepsPerCrop{nullptr};
    QPushButton* _btnNeuralTrace{nullptr};
    QString _neuralCheckpointPath;
    QString _neuralPythonPath;
    int _neuralVolumeScale{0};
    int _neuralMaxSize{60};
    int _neuralStepsPerCrop{1};
};
