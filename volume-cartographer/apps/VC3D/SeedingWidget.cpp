#include "SeedingWidget.hpp"
#include "OpenedVolumes.hpp"
#include "ToolExecutable.hpp"
#include "VolumeCommandPath.hpp"
#include "ObjectStoreRegions.hpp"
#include "SurfaceCatalog.hpp"
#include <iostream>
#include "CState.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QApplication>
#include <QCoreApplication>
#include <QFileInfo>
#include <QFileDialog>
#include <QGroupBox>
#include <QSettings>
#include <QDir>
#include <QFile>

#include <opencv2/imgproc.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/OpenCvCompat.hpp"

#include "vc/ui/VCCollection.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_set>
#include "vc_reader/array.hpp"

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;
using PathRenderMode = ViewerOverlayControllerBase::PathRenderMode;


namespace {

struct SegmentTriangleHit {
    float pathT = 0.0f;
    float distanceSq = std::numeric_limits<float>::max();
    cv::Vec3f point{0, 0, 0};
};

float clampFloat(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

SegmentTriangleHit segmentSegmentClosestOnFirst(const cv::Vec3f& p1,
                                                const cv::Vec3f& q1,
                                                const cv::Vec3f& p2,
                                                const cv::Vec3f& q2)
{
    const cv::Vec3f d1 = q1 - p1;
    const cv::Vec3f d2 = q2 - p2;
    const cv::Vec3f r = p1 - p2;
    const float a = d1.dot(d1);
    const float e = d2.dot(d2);
    const float f = d2.dot(r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= 1e-12f && e <= 1e-12f) {
        return {0.0f, (p1 - p2).dot(p1 - p2), p1};
    }
    if (a <= 1e-12f) {
        t = clampFloat(f / e, 0.0f, 1.0f);
    } else {
        const float c = d1.dot(r);
        if (e <= 1e-12f) {
            s = clampFloat(-c / a, 0.0f, 1.0f);
        } else {
            const float b = d1.dot(d2);
            const float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = clampFloat((b * f - c * e) / denom, 0.0f, 1.0f);
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clampFloat(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clampFloat((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    const cv::Vec3f c1 = p1 + d1 * s;
    const cv::Vec3f c2 = p2 + d2 * t;
    return {s, (c1 - c2).dot(c1 - c2), c1};
}

} // namespace






SeedingWidget::SeedingWidget(VCCollection* point_collection, CState* state, QWidget* parent)
    : QWidget(parent)
    , currentZSlice(0)
    , isDrawing(false)
    , colorIndex(0)
    , jobsRunning(false)
    , _point_collection(point_collection)
    , _state(state)
    , progressUtil(new ProgressUtil(nullptr, this))
{
    qRegisterMetaType<PathPrimitive>("PathPrimitive");
    qRegisterMetaType<QList<PathPrimitive>>("QList<PathPrimitive>");
    setupUI();

    if (_point_collection) {
        connect(_point_collection, &VCCollection::collectionsAdded, this, &SeedingWidget::onCollectionsAdded);
        connect(_point_collection, &VCCollection::collectionChanged, this, &SeedingWidget::onCollectionChanged);
        connect(_point_collection, &VCCollection::collectionRemoved, this, &SeedingWidget::onCollectionRemoved);
        onCollectionChanged(0); // Initial population
    }
    
    // Automatically find the executable path
    executablePath = vc3d::findToolExecutable(QStringLiteral("vc_grow_seg_from_seed"));
}

SeedingWidget::~SeedingWidget()
{
    if (jobsRunning) {
        onCancelClicked();
    }
}

void SeedingWidget::setupUI()
{
    // Main layout
    auto mainLayout = new QVBoxLayout(this);
    
    // Info label
    infoLabel = new QLabel("Turn on 'Draw paths' to draw paths in a viewer", this);
    mainLayout->addWidget(infoLabel);
    
    // Drawing toggle
    drawPathsButton = new QPushButton("Draw paths", this);
    drawPathsButton->setCheckable(true);
    drawPathsButton->setToolTip("While on, left-drag in a viewer to draw a path");
    mainLayout->addWidget(drawPathsButton);
    
    // Collection selector
    auto collectionLayout = new QHBoxLayout();
    collectionLayout->addWidget(new QLabel("Source Collection:", this));
    collectionComboBox = new QComboBox(this);
    collectionLayout->addWidget(collectionComboBox);
    mainLayout->addLayout(collectionLayout);
    
    // Processes control
    auto processesLayout = new QHBoxLayout();
    processesLayout->addWidget(new QLabel("Parallel Processes:", this));
    processesSpinBox = new QSpinBox(this);
    processesSpinBox->setRange(1, 256);
    processesSpinBox->setValue(16);
    processesLayout->addWidget(processesSpinBox);
    mainLayout->addLayout(processesLayout);

    // OMP threads control
    auto ompLayout = new QHBoxLayout();
    ompLayout->addWidget(new QLabel("OMP Threads:", this));
    ompThreadsSpinBox = new QSpinBox(this);
    ompThreadsSpinBox->setRange(0, 256);
    ompThreadsSpinBox->setValue(0);
    ompThreadsSpinBox->setToolTip("If greater than 0, prefixes commands with OMP_NUM_THREADS before execution");
    ompLayout->addWidget(ompThreadsSpinBox);
    mainLayout->addLayout(ompLayout);

    // Intensity threshold control
    auto thresholdLayout = new QHBoxLayout();
    thresholdLayout->addWidget(new QLabel("Intensity Threshold:", this));
    thresholdSpinBox = new QSpinBox(this);
    thresholdSpinBox->setRange(1, 255);
    thresholdSpinBox->setValue(20); 
    thresholdSpinBox->setToolTip("Minimum intensity value for peak detection");
    thresholdLayout->addWidget(thresholdSpinBox);
    mainLayout->addLayout(thresholdLayout);
    
    // Expansion iterations control
    auto expansionLayout = new QHBoxLayout();
    expansionLayout->addWidget(new QLabel("Expansion Iterations:", this));
    expansionIterationsSpinBox = new QSpinBox(this);
    expansionIterationsSpinBox->setRange(1, 5000000);
    expansionIterationsSpinBox->setValue(1000000);
    expansionIterationsSpinBox->setToolTip("Number of expansion iterations to run");
    expansionLayout->addWidget(expansionIterationsSpinBox);
    mainLayout->addLayout(expansionLayout);
    
    // Buttons
    auto analyzeLayout = new QHBoxLayout();
    analyzePathsButton = new QPushButton("Analyze Paths", this);
    analyzePathsButton->setEnabled(false);
    analyzePathsButton->setToolTip("Find intensity peaks along the drawn paths (into 'seeding_peaks')");
    analyzeLayout->addWidget(analyzePathsButton);
    clearPeaksButton = new QPushButton("Clear Peaks", this);
    analyzeLayout->addWidget(clearPeaksButton);
    mainLayout->addLayout(analyzeLayout);
    
    runSegmentationButton = new QPushButton("Run Seeding", this);
    runSegmentationButton->setEnabled(false);
    mainLayout->addWidget(runSegmentationButton);
    
    expandSeedsButton = new QPushButton("Expand Seeds", this);
    expandSeedsButton->setEnabled(false);
    expandSeedsButton->setToolTip("Run seed expansion using expand.json");
    mainLayout->addWidget(expandSeedsButton);
    
    resetPointsButton = new QPushButton("Reset Points", this);
    resetPointsButton->setEnabled(false);
    mainLayout->addWidget(resetPointsButton);

    // Cancel button (only visible when jobs are running)
    cancelButton = new QPushButton("Cancel", this);
    cancelButton->setVisible(false);
    cancelButton->setToolTip("Cancel running jobs");
    mainLayout->addWidget(cancelButton);
    
    // Progress bar
    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setVisible(false);
    mainLayout->addWidget(progressBar);
    progressUtil->setProgressBar(progressBar);
    ProgressUtil::ProgressBarOptions progressDefaults;
    progressDefaults.textMode = ProgressUtil::ProgressTextMode::Percent;
    progressDefaults.percentPrecision = 0;
    progressUtil->configureProgressBar(progressDefaults);
    
    // Connect signals
    connect(drawPathsButton, &QPushButton::toggled, this, [this](bool checked) {
        setDrawEnabled(checked);
        // Turning drawing off also ends relative winding annotation.
        if (!checked && labelWrapsMode) {
            setRelWindingAnnotationMode(false);
        }
    });
    connect(analyzePathsButton, &QPushButton::clicked, this, &SeedingWidget::onAnalyzePathsClicked);
    connect(clearPeaksButton, &QPushButton::clicked, this, &SeedingWidget::onClearPeaksClicked);
    connect(runSegmentationButton, &QPushButton::clicked, this, &SeedingWidget::onRunSegmentationClicked);
    connect(expandSeedsButton, &QPushButton::clicked, this, &SeedingWidget::onExpandSeedsClicked);
    connect(resetPointsButton, &QPushButton::clicked, this, &SeedingWidget::onResetPointsClicked);
    connect(cancelButton, &QPushButton::clicked, this, &SeedingWidget::onCancelClicked);

    // Set size policy
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

void SeedingWidget::setState(CState* state)
{
    _state = state;
    updateButtonStates();
}

void SeedingWidget::setViewerManager(ViewerManager* viewerManager)
{
    _viewerManager = viewerManager;
}

void SeedingWidget::setRelWindingIntersectionSource(int source)
{
    _relWindingIntersectionSource = source == static_cast<int>(RelWindingIntersectionSource::Patches)
        ? RelWindingIntersectionSource::Patches
        : RelWindingIntersectionSource::CurrentVolume;
}

void SeedingWidget::setRelWindingPatchTolerance(double tolerance)
{
    _relWindingPatchTolerance = std::max(0.0, tolerance);
}

void SeedingWidget::onCollectionsAdded(const std::vector<uint64_t>& collectionIds)
{
    onCollectionChanged(0);
}

void SeedingWidget::onCollectionChanged(uint64_t collectionId)
{
    if (!_point_collection) return;

    collectionComboBox->clear();
    const auto& collections = _point_collection->getAllCollections();
    for (const auto& pair : collections) {
        collectionComboBox->addItem(QString::fromStdString(pair.second.name), QVariant::fromValue(pair.first));
    }
}

void SeedingWidget::onCollectionRemoved(uint64_t collectionId)
{
    onCollectionChanged(0);
}

void SeedingWidget::onVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId)
{
    updateButtonStates();
}

void SeedingWidget::updateCurrentZSlice(int z)
{
    currentZSlice = z;
}

void SeedingWidget::onClearPeaksClicked()
{
    if (const auto id = _point_collection->getCollectionId("seeding_peaks"))
        _point_collection->clearCollection(*id);
    infoLabel->setText("Peak points cleared.");
}

void SeedingWidget::onAnalyzePathsClicked()
{
    QString errorMessage;
    if (!runAnalyzePathsHeadless(&errorMessage) && !errorMessage.isEmpty()) {
        QMessageBox::warning(this, tr("Warning"), errorMessage);
    }
}

void SeedingWidget::setDrawEnabled(bool enabled)
{
    _drawEnabled = enabled;
    if (drawPathsButton->isChecked() != enabled) {
        const QSignalBlocker blocker(drawPathsButton);
        drawPathsButton->setChecked(enabled);
    }
    if (!labelWrapsMode) {
        infoLabel->setText(enabled ? QString("Drawing: %1 path(s); left-drag in a viewer to add one").arg(paths.size())
                                   : QString("Turn on 'Draw paths' to draw paths in a viewer"));
    }
    updateModeUI();
    displayPaths();
}

// Trilinear values of the current volume's working level at XYZ points (vc_reader's
// interpolated floats, clamped to the 8-bit range).
std::vector<float> SeedingWidget::sampleIntensities(const Volume& volume, const std::vector<cv::Vec3f>& points) const
{
    const auto level = _state->openedVolumes()->level(volume, 0);
    std::vector<double> zyx;
    zyx.reserve(3 * points.size());
    for (const auto& p : points)
        zyx.insert(zyx.end(), {double(p[2]), double(p[1]), double(p[0])});
    std::vector<float> values(points.size());
    level.array->array->sample_linear(zyx.data(), points.size(), level.array->meta.type, false, values.data());
    for (auto& v : values)
        v = std::clamp(v, 0.0f, 255.0f);
    return values;
}

void SeedingWidget::computeDistanceTransform()
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!currentVolume) {
        return;
    }

    // Read the current slice of the volume's working level
    const auto level = _state->openedVolumes()->level(*currentVolume, 0);
    const auto& meta = level.array->meta;
    const int width = int(meta.shape[2]);
    const int height = int(meta.shape[1]);
    cv::Mat_<float> values(height, width);
    level.array->array->read_many_float({{currentZSlice, 0, 0}}, {{1, height, width}},
                                        {reinterpret_cast<std::uint8_t*>(values.ptr<float>())}, meta.type, false);
    cv::Mat_<uint8_t> sliceData;
    values.convertTo(sliceData, CV_8U);
    
    // Threshold the slice to create a binary image for distance transform
    cv::Mat binaryImage;
    cv::threshold(sliceData, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Compute the distance transform
    cv::distanceTransform(binaryImage, distanceTransform,
                          vc::opencv::distanceL2, cv::DIST_MASK_PRECISE);
}

void SeedingWidget::onRunSegmentationClicked()
{
    // The shared batch launcher reports precondition failures synchronously and
    // drains child processes through the event loop.
    QString err;
    if (!runSegmentationHeadless(&err) && !err.isEmpty()) {
        QMessageBox::warning(this, tr("Seeding"), err);
    }
}

bool SeedingWidget::runSegmentationHeadless(QString* errorMessage)
{
    auto setError = [&](const QString& msg) { if (errorMessage) *errorMessage = msg; };

    if (jobsRunning) {
        setError(tr("A seeding batch is already running."));
        return false;
    }
    if (executablePath.isEmpty()) {
        setError(tr("vc_grow_seg_from_seed executable not found."));
        return false;
    }

    auto fVpkg = _state ? _state->vpkg() : nullptr;
    auto currentVolume = _state ? _state->currentVolume() : nullptr;

    // Get the selected collection name from the combo box
    std::string sourceCollection = collectionComboBox->currentText().toStdString();
    if (sourceCollection.empty()) {
        setError(tr("Please select a source collection."));
        return false;
    }

    // Combine analysis points and user-placed points for segmentation
    std::vector<ColPoint> allPoints = _point_collection->getPoints(sourceCollection);

    if (allPoints.empty() || !fVpkg) {
        setError(tr("No points available for segmentation or volume package not loaded."));
        return false;
    }

    const int numProcesses = processesSpinBox->value();
    const int totalPoints = static_cast<int>(allPoints.size());
    const int ompThreads = ompThreadsSpinBox ? ompThreadsSpinBox->value() : 0;

    // New surfaces go to the object store; seed.json lives in the project directory.
    QString storeError;
    const auto store = batchStore(&storeError);
    if (!store) {
        setError(storeError);
        return false;
    }
    const std::filesystem::path projectDir(fVpkg->directory().string());
    const std::filesystem::path seedJsonPath = projectDir / "seed.json";

    if (!std::filesystem::exists(seedJsonPath)) {
        setError(tr("seed.json not found in volume package."));
        return false;
    }

    if (!currentVolume) {
        setError(tr("No current volume selected."));
        return false;
    }

    // Validate before begin(); otherwise an early return leaves a phantom batch.
    _batchVolumePath = vc3d::commandPathForVolume(currentVolume);
    if (_batchVolumePath.isEmpty()) {
        setError(tr("Could not resolve a volume path for segmentation "
                    "(no local mirror and no remote locator)."));
        return false;
    }

    // Keep per-batch configuration in member state for asynchronous callbacks.
    _batch.begin(QStringLiteral("run"), totalPoints);
    _batchPoints = std::move(allPoints);
    _batchNextIndex = 0;
    _batchOmpThreads = ompThreads;
    _batchStore = *store;
    _batchConfigJson = seedJsonPath;
    _batchWorkingDir = QString::fromStdString(projectDir.string());

    // Update UI
    infoLabel->setText(tr("Running segmentation jobs..."));
    runSegmentationButton->setEnabled(false);
    expandSeedsButton->setEnabled(false);

    // Show cancel button and set jobs running
    jobsRunning = true;
    cancelButton->setVisible(true);
    runningProcesses.clear();

    ProgressUtil::ProgressBarOptions barOptions;
    barOptions.textMode = ProgressUtil::ProgressTextMode::DoneRemaining;
    barOptions.fontPointSize = 11;
    progressUtil->startProgress(totalPoints, &barOptions);

    // Start initial batch of processes; the rest chain in via the finished
    // callback with bounded concurrency.
    for (int i = 0; i < std::min(numProcesses, totalPoints); i++) {
        startSegmentationProcessForPoint(_batchNextIndex++);
    }
    return true;
}

void SeedingWidget::setSurfaceCatalog(SurfaceCatalog* catalog)
{
    _surfaceCatalog = catalog;
    // Expansion needs stored surfaces to grow from.
    if (_surfaceCatalog) connect(_surfaceCatalog, &SurfaceCatalog::rowsChanged, this, &SeedingWidget::updateButtonStates);
}

std::optional<SurfaceCatalog::ToolStore> SeedingWidget::batchStore(QString* error) const
{
    if (!_surfaceCatalog) {
        *error = tr("Attach an object store and open a volume first.");
        return std::nullopt;
    }
    try {
        return _surfaceCatalog->toolStore();
    } catch (const std::exception& e) {
        *error = QString::fromUtf8(e.what());
        return std::nullopt;
    }
}

void SeedingWidget::startSegmentationProcessForPoint(int pointIndex)
{
    if (pointIndex >= _batch.total() || !jobsRunning) {
        return;
    }

    const auto& point = _batchPoints[pointIndex];
    const QStringList toolArgs = QStringList()
                  << QStringLiteral("-v") << _batchVolumePath
                  << QStringLiteral("-p") << QString::fromStdString(_batchConfigJson.string())
                  << QStringLiteral("--tag") << QStringLiteral("paths")
                  << QStringLiteral("--seed")
                  << QString::number(point.p[0])
                  << QString::number(point.p[1])
                  << QString::number(point.p[2]);

    std::cout << "Starting seeding job " << pointIndex << std::endl;
    launchBatchProcess(pointIndex, toolArgs);
}

void SeedingWidget::handleBatchProcessFinished(vc3d::ToolProcess* run, int index,
                                               const vc3d::ToolProcess::Result& result)
{
    if (!jobsRunning) {
        return;
    }

    const bool isExpand = (_batch.kind() == QLatin1String("expand"));

    // Aggregate the outcome, deduping on the stable child index — not the run
    // itself, since run->deleteLater() below frees each finished child mid-batch and
    // a later one can be allocated at the same address, misreading as already-terminal.
    const QString label = isExpand ? tr("Expansion iteration %1").arg(index)
                                   : tr("Segmentation for point %1").arg(index);
    if (!_batch.recordTerminal(index, result.failedToStart, result.crashed, result.exitCode, result.tail, label)) {
        return;
    }

    const bool failed = !result.ok;
    if (failed) {
        std::cerr << _batch.failureMessages().constLast().toStdString() << std::endl;
    } else {
        std::cout << (isExpand ? "Completed expansion iteration "
                               : "Completed segmentation for point ")
                  << index << std::endl;
    }

    progressUtil->updateProgress(_batch.completed());
    emit seedingBatchProgressChanged(_batch.kind(), _batch.completed(), _batch.total());

    runningProcesses.removeOne(run);
    run->deleteLater();

    // Continue the remaining batch after an individual failure unless canceled.
    if (_batchNextIndex < _batch.total() && jobsRunning && !_batch.cancelRequested()) {
        if (isExpand) {
            startExpansionProcessForIteration(_batchNextIndex++);
        } else {
            startSegmentationProcessForPoint(_batchNextIndex++);
        }
    }

    if (_batch.isComplete()) {
        finalizeSeedingBatch();
    }
}

void SeedingWidget::finalizeSeedingBatch()
{
    // Common terminal teardown for a run/expand batch or its cancel. The
    // batch-finished signal is only emitted for a started batch (non-empty kind).
    // Idempotent: cancel + the last child completion can both reach here.
    if (_batch.finalized()) {
        return;
    }

    // Capture kind before finalize() clears it.
    const QString kind = _batch.kind();
    const SeedingBatchTracker::Result r = _batch.finalize();
    const int completed = r.completed;
    const int total = r.total;
    const bool canceled = r.canceled;
    const bool success = r.success;
    const QString message = r.message;

    progressUtil->stopProgress();
    cancelButton->setVisible(false);
    jobsRunning = false;
    runningProcesses.clear();

    if (!kind.isEmpty() && success) {
        if (kind == QLatin1String("expand")) {
            infoLabel->setText(tr("Expansion complete after %1 iterations.").arg(total));
            emit sendStatusMessageAvailable(tr("Completed %1 expansion iterations").arg(total), 5000);
        } else {
            infoLabel->setText(tr("Segmentation complete for %1 points.").arg(total));
            emit sendStatusMessageAvailable(tr("Completed segmentation for %1 points").arg(total), 5000);
        }
    } else if (canceled) {
        infoLabel->setText(tr("Jobs cancelled by user"));
        emit sendStatusMessageAvailable(tr("Jobs cancelled by user"), 3000);
    } else if (!kind.isEmpty()) {
        infoLabel->setText(message);
        emit sendStatusMessageAvailable(message, 5000);
    }

    runSegmentationButton->setEnabled(true);
    expandSeedsButton->setEnabled(true);
    updateButtonStates();

    // Drop the latched batch config; nothing must dangle past this point.
    // (_batch retains its finalized outcome for idempotency; begin() resets it.)
    _batchPoints.clear();
    _batchPoints.shrink_to_fit();

    if (!kind.isEmpty()) {
        emit seedingBatchFinished(kind, success, canceled, completed, total, message);
    }
}


void SeedingWidget::onResetPointsClicked()
{
    if (_point_collection) {
        if (const auto id = _point_collection->getCollectionId("seeding_peaks"))
            _point_collection->clearCollection(*id);
    }
    
    // Clear paths
    paths.clear();
    
    // Reset UI
    analyzePathsButton->setEnabled(false);
    runSegmentationButton->setEnabled(false);
    expandSeedsButton->setEnabled(false);
    resetPointsButton->setEnabled(false);
    infoLabel->setText("Draw paths, then click 'Analyze Paths'");
    
    // Redraw to clear paths
    displayPaths();
}

void SeedingWidget::updateModeUI()
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    const bool hasPaths = !paths.empty();
    resetPointsButton->setText(_drawEnabled ? "Clear All Paths" : "Reset Points");
    analyzePathsButton->setEnabled(hasPaths && currentVolume != nullptr);
    resetPointsButton->setEnabled(hasPaths || !_point_collection->getPoints("seeding_peaks").empty());
}

void SeedingWidget::analyzePaths()
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!currentVolume || paths.empty()) {
        return;
    }
    
    // Reset previous peaks
    if (const auto id = _point_collection->getCollectionId("seeding_peaks"))
        _point_collection->clearCollection(*id);
    
    // Compute distance transform once
    computeDistanceTransform();
    
    int totalPaths = paths.size();
    int pathIndex = 0;
    
    ProgressUtil::ProgressBarOptions options;
    options.textMode = ProgressUtil::ProgressTextMode::Fraction;
    progressUtil->startProgress(totalPaths, &options);

    // For each path
    for (const auto& path : paths) {
        // Analyze along this path
        findPeaksAlongPath(path);

        // Update progress
        pathIndex++;
        progressUtil->updateProgress(pathIndex);
        // This synchronous computation must not pump a re-entrant event loop.
    }

    progressUtil->stopProgress();
    
    // Enable segmentation button if we found peaks
    runSegmentationButton->setEnabled(!_point_collection->getPoints("seeding_peaks").empty());
    
    // Update visualization
    displayPaths();
    
    // Update UI
    infoLabel->setText(QString("Found %1 peaks along %2 paths").arg(_point_collection->getPoints("seeding_peaks").size()).arg(paths.size()));
    emit sendStatusMessageAvailable(
        QString("Analyzed %1 paths and found %2 intensity peaks").arg(paths.size()).arg(_point_collection->getPoints("seeding_peaks").size()),
        5000);
}

bool SeedingWidget::runAnalyzePathsHeadless(QString* errorMessage, int* pathsAnalyzed,
                                            int* peaksFound)
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!currentVolume) {
        if (errorMessage) *errorMessage = tr("No current volume selected.");
        return false;
    }
    if (paths.empty()) {
        if (errorMessage) *errorMessage = tr("No paths to analyze. Draw paths first.");
        return false;
    }

    // analyzePaths is synchronous and completes before this returns.
    analyzePaths();

    if (pathsAnalyzed) *pathsAnalyzed = static_cast<int>(paths.size());
    if (peaksFound) {
        *peaksFound = static_cast<int>(_point_collection->getPoints("seeding_peaks").size());
    }
    return true;
}

void SeedingWidget::findPeaksAlongPath(const PathPrimitive& path)
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!currentVolume || path.points.empty()) {
        return;
    }
    
    // densify the path so we don't skip over small surfaces when drawing
    PathPrimitive densifiedPath = path.densify(0.5f); // Sample every 0.5 pixels

    // Get data bounds for bounds checking
    auto [width, height, depth] = currentVolume->shapeXyz();
    float bx0 = 0, by0 = 0, bz0 = 0;
    float bx1 = static_cast<float>(width), by1 = static_cast<float>(height), bz1 = static_cast<float>(depth);

    std::vector<float> intensities;
    std::vector<cv::Vec3f> positions;

    // Read intensity values at each point along the (denser) path
    for (const auto& pt : densifiedPath.points) {
        // Check bounds against data region
        if (pt[0] >= bx0 && pt[0] < bx1 &&
            pt[1] >= by0 && pt[1] < by1 &&
            pt[2] >= bz0 && pt[2] < bz1) {
            
            positions.push_back(pt);
        }
    }
    intensities = sampleIntensities(*currentVolume, positions);
    
    if (intensities.empty()) {
        return;
    }
    
    // Place a single point at the center of each above-threshold segment along the path
    const int thr = thresholdSpinBox->value();
    bool inside = false;
    size_t seg_start = 0;

    auto sampleDistAt = [&](const cv::Vec3f& p) -> float {
        if (distanceTransform.empty()) return 0.0f;
        int x = std::max(0, std::min(distanceTransform.cols - 1, int(std::round(p[0]))));
        int y = std::max(0, std::min(distanceTransform.rows - 1, int(std::round(p[1]))));
        return distanceTransform.at<float>(y, x);
    };

    auto addCenterOfSegment = [&](size_t s, size_t e) {
        if (e < s || s >= positions.size() || e >= positions.size()) return;

        const float distEps = 1e-4f;
        const float intensityEps = 1e-3f;

        size_t center_idx = s + (e - s) / 2;
        size_t best_idx = center_idx;
        float best_dt = sampleDistAt(positions[best_idx]);
        float best_intensity = intensities[best_idx];

        for (size_t k = s; k <= e; ++k) {
            float dt = sampleDistAt(positions[k]);
            float intensity = intensities[k];

            bool betterDistance = dt > best_dt + distEps;
            bool similarDistance = std::fabs(dt - best_dt) <= distEps;
            bool betterIntensity = intensity > best_intensity + intensityEps;
            bool similarIntensity = std::fabs(intensity - best_intensity) <= intensityEps;

            if (betterDistance || (similarDistance && betterIntensity)) {
                best_idx = k;
                best_dt = dt;
                best_intensity = intensity;
                continue;
            }

            if (similarDistance && similarIntensity) {
                size_t currentOffset = (best_idx > center_idx) ? (best_idx - center_idx) : (center_idx - best_idx);
                size_t newOffset = (k > center_idx) ? (k - center_idx) : (center_idx - k);
                if (newOffset < currentOffset) {
                    best_idx = k;
                    best_dt = dt;
                    best_intensity = intensity;
                }
            }
        }

        _point_collection->addPoint("seeding_peaks", positions[best_idx]);
    };

    for (size_t i = 0; i < intensities.size(); ++i) {
        bool curr_inside = intensities[i] >= thr;
        if (curr_inside && !inside) {
            inside = true;
            seg_start = i;
        }
        // If leaving a segment or at the end, close it and add center point
        bool at_end = (i + 1 == intensities.size());
        bool next_inside = at_end ? false : (intensities[i + 1] >= thr);
        if (inside && (!next_inside || at_end)) {
            size_t seg_end = i;
            addCenterOfSegment(seg_start, seg_end);
            inside = false;
        }
    }
}

void SeedingWidget::startDrawing(cv::Vec3f startPoint)
{
    isDrawing = true;
    currentPath.points.clear();
    currentPath.points.push_back(startPoint);
    currentPath.color = generatePathColor();
    currentPath.lineWidth = 2.0f;
    currentPath.opacity = 1.0f;
    currentPath.isEraser = false;
    currentPath.brushShape = PathBrushShape::Circle;
    currentPath.renderMode = PathRenderMode::LineStrip;
    currentPath.pointRadius = 2.5f;
    currentPath.z = 25.0;

    // Show temporary path
    displayPaths();
}

void SeedingWidget::addPointToPath(cv::Vec3f point)
{
    if (!isDrawing) {
        return;
    }
    
    if (currentPath.points.empty()) {
        currentPath.points.push_back(point);
    } else {
        // Only add if there's some distance from the last point
        cv::Vec3f lastPoint = currentPath.points.back();
        float distance = cv::norm(point - lastPoint);
        
        // Add point if it's far enough (reduces density but keeps path smooth)
        if (distance > 1.0f) {
            currentPath.points.push_back(point);
            
            // Only update display every few points
            if (currentPath.points.size() % 5 == 0) {
                displayPaths();
            }
        }
    }
}

void SeedingWidget::finalizePath()
{
    if (!isDrawing || currentPath.points.size() < 2) {
        isDrawing = false;
        return;
    }
    
    // Add the path to the collection
    paths.push_back(currentPath);
    
    isDrawing = false;
    currentPath.points.clear();
    
    // Update UI
    updateModeUI();
    
    // Always display paths when finalizing to ensure the complete path is shown
    displayPaths();
    
    // Update info
    infoLabel->setText(QString("Drawing: %1 path(s)").arg(paths.size()));
}

void SeedingWidget::finalizePathLabelWraps(bool shiftHeld)
{
    const bool usePatchIntersections =
        _relWindingIntersectionSource == RelWindingIntersectionSource::Patches;
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!isDrawing || currentPath.points.size() < 2 || (!usePatchIntersections && !currentVolume)) {
        isDrawing = false;
        currentPath.points.clear();
        displayPaths();
        return;
    }

    if (!usePatchIntersections) {
        // Ensure distance transform available for center-of-band selection.
        computeDistanceTransform();
    }

    // Create a new target collection.
    std::string newColName = _point_collection->generateNewCollectionName("wraps");
    uint64_t newColId = _point_collection->addCollection(newColName);

    int addedPoints = 0;
    if (usePatchIntersections) {
        addedPoints = findPatchIntersectionsAlongPathToCollection(currentPath, newColName);
    } else {
        findPeaksAlongPathToCollection(currentPath, newColName);
        const auto& collections = _point_collection->getAllCollections();
        const auto it = collections.find(newColId);
        if (it != collections.end()) {
            addedPoints = static_cast<int>(it->second.points.size());
        }
    }

    if (addedPoints > 0) {
        // Auto fill winding annotations in chosen direction.
        _point_collection->autoFillWindingNumbers(
            newColId,
            shiftHeld ? VCCollection::WindingFillMode::Decremental : VCCollection::WindingFillMode::Incremental
        );
    }

    // Reset drawing path and refresh view.
    isDrawing = false;
    currentPath.points.clear();
    displayPaths();

    infoLabel->setText(QString("Rel winding annotation: added %1 point(s) to '%2' (%3)")
                           .arg(addedPoints)
                           .arg(QString::fromStdString(newColName))
                           .arg(shiftHeld ? "decreasing" : "increasing"));
    updateButtonStates();
}

void SeedingWidget::findPeaksAlongPathToCollection(const PathPrimitive& path, const std::string& collectionName)
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;
    if (!currentVolume || path.points.empty()) {
        return;
    }

    PathPrimitive densifiedPath = path.densify(0.5f);

    auto [width, height, depth] = currentVolume->shapeXyz();
    float bx0 = 0, by0 = 0, bz0 = 0;
    float bx1 = static_cast<float>(width), by1 = static_cast<float>(height), bz1 = static_cast<float>(depth);

    std::vector<float> intensities;
    std::vector<cv::Vec3f> positions;

    for (const auto& pt : densifiedPath.points) {
        if (pt[0] >= bx0 && pt[0] < bx1 &&
            pt[1] >= by0 && pt[1] < by1 &&
            pt[2] >= bz0 && pt[2] < bz1) {
            positions.push_back(pt);
        }
    }
    intensities = sampleIntensities(*currentVolume, positions);

    if (intensities.empty()) return;

    const int thr = thresholdSpinBox->value();
    bool inside = false;
    size_t seg_start = 0;

    auto sampleDistAt = [&](const cv::Vec3f& p) -> float {
        if (distanceTransform.empty()) return 0.0f;
        int x = std::max(0, std::min(distanceTransform.cols - 1, int(std::round(p[0]))));
        int y = std::max(0, std::min(distanceTransform.rows - 1, int(std::round(p[1]))));
        return distanceTransform.at<float>(y, x);
    };

    auto addCenterOfSegmentToCollection = [&](size_t s, size_t e) {
        if (e < s || s >= positions.size() || e >= positions.size()) return;
        size_t best_idx = s;
        float best_dt = sampleDistAt(positions[s]);
        for (size_t k = s; k <= e; ++k) {
            float dt = sampleDistAt(positions[k]);
            if (dt > best_dt) {
                best_dt = dt;
                best_idx = k;
            }
        }
        _point_collection->addPoint(collectionName, positions[best_idx]);
    };

    for (size_t i = 0; i < intensities.size(); ++i) {
        bool curr_inside = intensities[i] >= thr;
        if (curr_inside && !inside) {
            inside = true;
            seg_start = i;
        }
        bool at_end = (i + 1 == intensities.size());
        bool next_inside = at_end ? false : (intensities[i + 1] >= thr);
        if (inside && (!next_inside || at_end)) {
            size_t seg_end = i;
            addCenterOfSegmentToCollection(seg_start, seg_end);
            inside = false;
        }
    }
}

int SeedingWidget::findPatchIntersectionsAlongPathToCollection(const PathPrimitive& path,
                                                               const std::string& collectionName)
{
    if (!_viewerManager || !_point_collection || path.points.size() < 2 || !_relWindingDrawViewer) {
        return 0;
    }

    auto* plane = dynamic_cast<PlaneSurface*>(_relWindingDrawViewer->currentSurface());
    if (!plane) {
        return 0;
    }

    struct PathHit {
        std::string surface;
        float pathDistance = 0.0f;
        float distanceSq = std::numeric_limits<float>::max();
        cv::Vec3f point{0, 0, 0};
    };

    const float tolerance = static_cast<float>(std::max(0.0, _relWindingPatchTolerance));
    const float toleranceSq = tolerance * tolerance;

    std::vector<cv::Vec3f> pathPlane;
    pathPlane.reserve(path.points.size());
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    for (const cv::Vec3f& point : path.points) {
        const cv::Vec3f projected = plane->project(point, 1.0f, 1.0f);
        if (!std::isfinite(projected[0]) || !std::isfinite(projected[1])) {
            continue;
        }
        pathPlane.push_back(cv::Vec3f(projected[0], projected[1], 0.0f));
        minX = std::min(minX, projected[0]);
        minY = std::min(minY, projected[1]);
        maxX = std::max(maxX, projected[0]);
        maxY = std::max(maxY, projected[1]);
    }
    if (pathPlane.size() < 2) {
        return 0;
    }

    const float roiPad = std::max(2.0f, tolerance + 2.0f);
    cv::Rect planeRoi{static_cast<int>(std::floor(minX - roiPad)),
                      static_cast<int>(std::floor(minY - roiPad)),
                      std::max(1, static_cast<int>(std::ceil(maxX - minX + roiPad * 2.0f))),
                      std::max(1, static_cast<int>(std::ceil(maxY - minY + roiPad * 2.0f)))};

    // Where surfaces cross the drawing plane around the path, from the object
    // store (the surface being edited through its draft).
    std::map<std::string, std::vector<std::array<cv::Vec3f, 2>>> intersections;
    if (_surfaceCatalog && _surfaceCatalog->volume()) {
        auto corner = [&](float x, float y) {
            const cv::Vec3f p = plane->coord({0, 0, 0}, {x, y, 0.0f});
            return vc::fibers::Point{p[0], p[1], p[2]};
        };
        const auto region = vc3d::store::viewRegion(corner(planeRoi.x, planeRoi.y),
                                                    corner(planeRoi.x + planeRoi.width, planeRoi.y),
                                                    corner(planeRoi.x, planeRoi.y + planeRoi.height), 1.0);
        if (region) {
            try {
                const auto crossings = SurfaceCatalog::region(*_surfaceCatalog->session(), _surfaceCatalog->sources(),
                                                              _surfaceCatalog->voxels(),
                                                              vc3d::store::regionSlice(*region), {});
                for (const auto& run : crossings.runs) {
                    auto& out = intersections[run.objectUuid];
                    for (size_t i = 1; i < run.points.size(); ++i)
                        out.push_back({cv::Vec3f(run.points[i - 1][0], run.points[i - 1][1], run.points[i - 1][2]),
                                       cv::Vec3f(run.points[i][0], run.points[i][1], run.points[i][2])});
                }
            } catch (const std::exception& error) {
                Logger()->warn("Relative winding surface query failed: {}", error.what());
            }
        }
    }
    if (intersections.empty()) {
        return 0;
    }

    std::vector<PathHit> hits;
    float cumulativeDistance = 0.0f;
    for (std::size_t i = 0; i + 1 < pathPlane.size(); ++i) {
        const cv::Vec3f p0 = pathPlane[i];
        const cv::Vec3f p1 = pathPlane[i + 1];
        const float segmentLength = cv::norm(p1 - p0);
        if (!std::isfinite(segmentLength) || segmentLength <= 1e-6f) {
            continue;
        }

        for (const auto& [surface, segments] : intersections) {
            for (const auto& segment : segments) {
                const cv::Vec3f aProj = plane->project(segment[0], 1.0f, 1.0f);
                const cv::Vec3f bProj = plane->project(segment[1], 1.0f, 1.0f);
                if (!std::isfinite(aProj[0]) || !std::isfinite(aProj[1]) ||
                    !std::isfinite(bProj[0]) || !std::isfinite(bProj[1])) {
                    continue;
                }
                const cv::Vec3f a(aProj[0], aProj[1], 0.0f);
                const cv::Vec3f b(bProj[0], bProj[1], 0.0f);
                const auto hit = segmentSegmentClosestOnFirst(p0, p1, a, b);
                if (hit.distanceSq <= toleranceSq) {
                    hits.push_back({surface,
                                    cumulativeDistance + hit.pathT * segmentLength,
                                    hit.distanceSq,
                                    plane->coord({0, 0, 0}, {hit.point[0], hit.point[1], 0.0f})});
                }
            }
        }

        cumulativeDistance += segmentLength;
    }

    if (hits.empty()) {
        return 0;
    }

    std::sort(hits.begin(), hits.end(), [](const PathHit& a, const PathHit& b) {
        if (a.pathDistance != b.pathDistance) {
            return a.pathDistance < b.pathDistance;
        }
        return a.surface < b.surface;
    });

    const float mergeDistance = std::max(1e-3f, tolerance);
    std::vector<PathHit> selected;
    selected.reserve(hits.size());
    for (const PathHit& hit : hits) {
        bool merged = false;
        for (PathHit& existing : selected) {
            if (existing.surface == hit.surface &&
                std::abs(existing.pathDistance - hit.pathDistance) <= mergeDistance) {
                if (hit.distanceSq < existing.distanceSq) {
                    existing = hit;
                }
                merged = true;
                break;
            }
        }
        if (!merged) {
            selected.push_back(hit);
        }
    }

    std::sort(selected.begin(), selected.end(), [](const PathHit& a, const PathHit& b) {
        return a.pathDistance < b.pathDistance;
    });

    for (const PathHit& hit : selected) {
        _point_collection->addPoint(collectionName, hit.point);
    }
    return static_cast<int>(selected.size());
}

void SeedingWidget::setRelWindingAnnotationMode(bool active)
{
    if (labelWrapsMode == active) {
        return;
    }

    labelWrapsMode = active;
    if (labelWrapsMode) {
        infoLabel->setText("Rel winding annotation: draw a path; release to create a labeled collection. Hold Shift for decreasing order.");
        setDrawEnabled(true);
    } else {
        // Revert to the default text; drawing stays as it is.
        setDrawEnabled(_drawEnabled);
    }
    emit relWindingAnnotationModeChanged(labelWrapsMode);
}

QColor SeedingWidget::generatePathColor()
{
    // Generate distinct colors for paths
    static const QColor colors[] = {
        QColor(255, 100, 100),  // Red
        QColor(100, 255, 100),  // Green
        QColor(100, 100, 255),  // Blue
        QColor(255, 255, 100),  // Yellow
        QColor(255, 100, 255),  // Magenta
        QColor(100, 255, 255),  // Cyan
        QColor(255, 165, 0),    // Orange
        QColor(128, 0, 128),    // Purple
        QColor(0, 128, 128),    // Teal
        QColor(255, 192, 203)   // Pink
    };
    
    colorIndex = (colorIndex + 1) % 10;
    return colors[colorIndex];
}

void SeedingWidget::displayPaths()
{
    // Prepare the list of paths to send
    QList<PathPrimitive> allPaths = paths;
    
    // Add the current drawing path if we're actively drawing
    if (isDrawing && !currentPath.points.empty()) {
        allPaths.append(currentPath);
    }
    
    // Send the paths for line rendering
    emit sendPathsChanged(allPaths);
    
    // Send peaks as red points (they should still be displayed as individual points)
}

void SeedingWidget::updateButtonStates()
{
    auto currentVolume = _state ? _state->currentVolume() : nullptr;

    // Enable segmentation once the analysis found peaks
    bool hasAnyPoints = !_point_collection->getPoints("seeding_peaks").empty();
    runSegmentationButton->setEnabled(hasAnyPoints && currentVolume != nullptr);

    // Enable expansion if we have a volume AND at least one stored surface
    const bool hasSurfaces = _surfaceCatalog && !_surfaceCatalog->rows().empty();
    expandSeedsButton->setEnabled(currentVolume != nullptr && hasSurfaces);

    // Enable reset if we have any points or paths
    bool hasAnyData = hasAnyPoints || !paths.empty();
    resetPointsButton->setEnabled(hasAnyData);

    analyzePathsButton->setEnabled(!paths.empty() && currentVolume != nullptr);
}

void SeedingWidget::onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (!_drawEnabled || button != Qt::LeftButton) {
        return;
    }

    _relWindingDrawViewer = qobject_cast<CChunkedVolumeViewer*>(sender());
    startDrawing(vol_point);
}

void SeedingWidget::onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    if (!_drawEnabled || !isDrawing || !(buttons & Qt::LeftButton)) {
        return;
    }
    
    addPointToPath(vol_point);
}

void SeedingWidget::onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (!_drawEnabled || button != Qt::LeftButton || !isDrawing) {
        return;
    }

    if (labelWrapsMode) {
        bool shiftHeld = modifiers.testFlag(Qt::ShiftModifier);
        finalizePathLabelWraps(shiftHeld);
    } else {
        finalizePath();
    }
}

void SeedingWidget::onExpandSeedsClicked()
{
    // Interactive slot: delegate to the non-blocking headless core (see
    // onRunSegmentationClicked).
    QString err;
    if (!runExpandSeedsHeadless(&err) && !err.isEmpty()) {
        QMessageBox::warning(this, tr("Seeding"), err);
    }
}

bool SeedingWidget::runExpandSeedsHeadless(QString* errorMessage)
{
    auto setError = [&](const QString& msg) { if (errorMessage) *errorMessage = msg; };

    if (jobsRunning) {
        setError(tr("A seeding batch is already running."));
        return false;
    }
    if (executablePath.isEmpty()) {
        setError(tr("vc_grow_seg_from_seed executable not found."));
        return false;
    }

    auto fVpkg = _state ? _state->vpkg() : nullptr;
    auto currentVolume = _state ? _state->currentVolume() : nullptr;

    if (!fVpkg || !currentVolume) {
        setError(tr("Volume package or volume not loaded."));
        return false;
    }

    const int numProcesses = processesSpinBox->value();
    const int expansionIterations = expansionIterationsSpinBox->value();
    const int ompThreads = ompThreadsSpinBox ? ompThreadsSpinBox->value() : 0;

    // Expansion grows from the volume's stored surfaces into new ones;
    // expand.json lives in the project directory.
    QString storeError;
    const auto store = batchStore(&storeError);
    if (!store) {
        setError(storeError);
        return false;
    }
    const std::filesystem::path projectDir(fVpkg->directory().string());
    const std::filesystem::path expandJsonPath = projectDir / "expand.json";

    if (!std::filesystem::exists(expandJsonPath)) {
        setError(tr("expand.json not found in volume package."));
        return false;
    }

    // Validate before begin() so an early return cannot leave a phantom batch.
    _batchVolumePath = vc3d::commandPathForVolume(currentVolume);
    if (_batchVolumePath.isEmpty()) {
        setError(tr("Could not resolve a volume path for expansion "
                    "(no local mirror and no remote locator)."));
        return false;
    }

    // Keep per-batch configuration in member state for asynchronous callbacks.
    _batch.begin(QStringLiteral("expand"), expansionIterations);
    _batchPoints.clear();
    _batchNextIndex = 0;
    _batchOmpThreads = ompThreads;
    _batchStore = *store;
    _batchConfigJson = expandJsonPath;
    _batchWorkingDir = QString::fromStdString(projectDir.string());

    // Update UI
    infoLabel->setText(tr("Running expansion jobs..."));
    expandSeedsButton->setEnabled(false);
    runSegmentationButton->setEnabled(false);

    // Show cancel button and set jobs running
    jobsRunning = true;
    cancelButton->setVisible(true);
    runningProcesses.clear();

    ProgressUtil::ProgressBarOptions barOptions;
    barOptions.textMode = ProgressUtil::ProgressTextMode::DoneRemaining;
    barOptions.fontPointSize = 11;
    barOptions.prefix = tr("Expansion ");
    progressUtil->startProgress(expansionIterations, &barOptions);

    // Start initial batch of processes; the rest chain in via the finished
    // callback with bounded concurrency.
    for (int i = 0; i < std::min(numProcesses, expansionIterations); i++) {
        startExpansionProcessForIteration(_batchNextIndex++);
    }
    return true;
}

void SeedingWidget::startExpansionProcessForIteration(int iterationIndex)
{
    if (iterationIndex >= _batch.total() || !jobsRunning) {
        return;
    }

    const QStringList toolArgs = QStringList()
                  << QStringLiteral("-v") << _batchVolumePath
                  << QStringLiteral("-p") << QString::fromStdString(_batchConfigJson.string())
                  << QStringLiteral("--tag") << QStringLiteral("paths");

    std::cout << "Starting expansion job " << iterationIndex << std::endl;
    launchBatchProcess(iterationIndex, toolArgs);
}


void SeedingWidget::launchBatchProcess(int index, const QStringList& toolArgs)
{
    auto* run = new vc3d::ToolProcess(this);
    connect(run, &vc3d::ToolProcess::finished, this,
            [this, run, index](const vc3d::ToolProcess::Result& result) { handleBatchProcessFinished(run, index, result); });
    // Tracked before start() so a failed start is handled as part of this batch.
    runningProcesses.append(QPointer<vc3d::ToolProcess>(run));
    vc3d::ToolProcess::Launch launch;
    launch.tool = executablePath;
    launch.args = toolArgs;
    launch.ompThreads = _batchOmpThreads;
    launch.workingDirectory = _batchWorkingDir;
    launch.store = QString::fromStdString(_batchStore.store.string());
    launch.volume = QString::fromStdString(_batchStore.volume.volume);
    launch.voxelScale = _batchStore.volume.scale;
    launch.lowPriority = true;
    if (!run->start(launch)) {
        vc3d::ToolProcess::Result failed;
        failed.failedToStart = true;
        failed.error = run->error();
        handleBatchProcessFinished(run, index, failed);
        return;
    }
    std::cout << "  " << run->command().toStdString() << std::endl;
}

void SeedingWidget::cancelSeedingBatchHeadless()
{
    if (!jobsRunning) {
        return;
    }

    // Stop launching children and distinguish cancellation from failure.
    jobsRunning = false;
    _batch.requestCancel();

    // No callbacks after cancellation; deleting a run kills its process (bounded wait).
    for (const QPointer<vc3d::ToolProcess>& run : runningProcesses) {
        if (!run) continue;
        run->disconnect(this);
        delete run.data();
    }
    runningProcesses.clear();

    // Common terminal teardown + seedingBatchFinished(kind,false,canceled=true,...)
    // when a run/expand batch was active.
    finalizeSeedingBatch();
}

void SeedingWidget::onCancelClicked()
{
    // The shared cancel path performs the bounded child-process teardown.
    cancelSeedingBatchHeadless();
}

void SeedingWidget::onSurfacesLoaded()
{
    // Update button states when surfaces are loaded/reloaded
    updateButtonStates();
}
