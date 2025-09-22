#include "SeedingWidget.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QProcess>
#include <QProcessEnvironment>
#include <QApplication>
#include <QCoreApplication>
#include <QFileInfo>

#include <opencv2/imgproc.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Logging.hpp"

#include "vc/ui/VCCollection.hpp"
#include "CSurfaceCollection.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <functional>






SeedingWidget::SeedingWidget(VCCollection* point_collection, CSurfaceCollection* surface_collection, QWidget* parent)
    : QWidget(parent)
    , fVpkg(nullptr)
    , currentVolume(nullptr)
    , chunkCache(nullptr)
    , currentZSlice(0)
    , currentMode(Mode::PointMode)
    , isDrawing(false)
    , colorIndex(0)
    , jobsRunning(false)
    , _point_collection(point_collection)
    , _surface_collection(surface_collection)
{
    setupUI();

    if (_point_collection) {
        connect(_point_collection, &VCCollection::collectionAdded, this, &SeedingWidget::onCollectionAdded);
        connect(_point_collection, &VCCollection::collectionChanged, this, &SeedingWidget::onCollectionChanged);
        connect(_point_collection, &VCCollection::collectionRemoved, this, &SeedingWidget::onCollectionRemoved);
        onCollectionChanged(0); // Initial population
    }
    
    // Automatically find the executable path
    executablePath = findExecutablePath();
    if (executablePath.isEmpty()) {
        QMessageBox::warning(this, "Warning",
            "Could not find vc_grow_seg_from_seed executable. "
            "Please ensure it is built and in your PATH or in the build directory.");
    }
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
    infoLabel = new QLabel("Set a focus point to begin", this);
    mainLayout->addWidget(infoLabel);
    
    // Mode toggle button
    auto modeButton = new QPushButton("Switch to Draw Mode", this);
    modeButton->setToolTip("Toggle between point mode and draw mode");
    mainLayout->addWidget(modeButton);
    
    // Collection selector
    auto collectionLayout = new QHBoxLayout();
    collectionLayout->addWidget(new QLabel("Source Collection:", this));
    collectionComboBox = new QComboBox(this);
    collectionLayout->addWidget(collectionComboBox);
    mainLayout->addLayout(collectionLayout);
    
    // Max radius control
    maxRadiusLayout = new QHBoxLayout();
    maxRadiusLabel = new QLabel("Max Radius (pixels):", this);
    maxRadiusLayout->addWidget(maxRadiusLabel);
    maxRadiusSpinBox = new QSpinBox(this);
    maxRadiusSpinBox->setRange(50, 20000);
    maxRadiusSpinBox->setValue(1500);
    maxRadiusSpinBox->setSingleStep(250);
    maxRadiusSpinBox->setToolTip("Maximum distance from center point for ray casting");
    maxRadiusLayout->addWidget(maxRadiusSpinBox);
    mainLayout->addLayout(maxRadiusLayout);
    
    // Angle step control
    angleStepLayout = new QHBoxLayout();
    angleStepLabel = new QLabel("Angle Step (degrees):", this);
    angleStepLayout->addWidget(angleStepLabel);
    angleStepSpinBox = new QDoubleSpinBox(this);
    angleStepSpinBox->setRange(1.0, 180.0);
    angleStepSpinBox->setValue(15.0);
    angleStepSpinBox->setSingleStep(1.0);
    angleStepLayout->addWidget(angleStepSpinBox);
    mainLayout->addLayout(angleStepLayout);
    
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
    
    // Window size control for peak detection
    auto windowSizeLayout = new QHBoxLayout();
    windowSizeLayout->addWidget(new QLabel("Peak Detection Window:", this));
    windowSizeSpinBox = new QSpinBox(this);
    windowSizeSpinBox->setRange(1, 10);
    windowSizeSpinBox->setValue(7);  // Default window size
    windowSizeSpinBox->setToolTip("Size of window for local maxima detection (larger values detect broader peaks)");
    windowSizeLayout->addWidget(windowSizeSpinBox);
    mainLayout->addLayout(windowSizeLayout);
    
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
    auto previewLayout = new QHBoxLayout();
    previewRaysButton = new QPushButton("Show Preview Points", this);
    previewRaysButton->setEnabled(true);
    previewLayout->addWidget(previewRaysButton);
    clearPreviewButton = new QPushButton("Clear", this);
    previewLayout->addWidget(clearPreviewButton);
    mainLayout->addLayout(previewLayout);

    auto castLayout = new QHBoxLayout();
    castRaysButton = new QPushButton("Cast Rays", this);
    castRaysButton->setEnabled(false);
    castLayout->addWidget(castRaysButton);
    clearPeaksButton = new QPushButton("Clear", this);
    castLayout->addWidget(clearPeaksButton);
    mainLayout->addLayout(castLayout);
    
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

    // Label Wraps mode toggle
    labelWrapsButton = new QPushButton("Start Label Wraps", this);
    labelWrapsButton->setToolTip("Enable 'Label Wraps' mode: draw a line to auto-create a new collection with winding labels. Hold Shift for decreasing order.");
    mainLayout->addWidget(labelWrapsButton);
    
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
    
    // Connect signals
    connect(modeButton, &QPushButton::clicked, [this, modeButton]() {
        if (currentMode == Mode::PointMode) {
            currentMode = Mode::DrawMode;
            modeButton->setText("Switch to Point Mode");
            infoLabel->setText("Draw Mode: Click and drag to draw paths");
        } else {
            currentMode = Mode::PointMode;
            modeButton->setText("Switch to Draw Mode");
            infoLabel->setText("Point Mode: Set a focus point to begin");
            // Turning off Draw Mode should also disable Label Wraps
            if (labelWrapsMode) {
                labelWrapsMode = false;
                if (labelWrapsButton) labelWrapsButton->setText("Start Label Wraps");
            }
        }
        updateModeUI();
        displayPaths();
    });
    connect(previewRaysButton, &QPushButton::clicked, this, &SeedingWidget::onPreviewRaysClicked);
    connect(clearPreviewButton, &QPushButton::clicked, this, &SeedingWidget::onClearPreviewClicked);
    connect(castRaysButton, &QPushButton::clicked, this, &SeedingWidget::onCastRaysClicked);
    connect(clearPeaksButton, &QPushButton::clicked, this, &SeedingWidget::onClearPeaksClicked);
    connect(runSegmentationButton, &QPushButton::clicked, this, &SeedingWidget::onRunSegmentationClicked);
    connect(expandSeedsButton, &QPushButton::clicked, this, &SeedingWidget::onExpandSeedsClicked);
    connect(resetPointsButton, &QPushButton::clicked, this, &SeedingWidget::onResetPointsClicked);
    connect(cancelButton, &QPushButton::clicked, this, &SeedingWidget::onCancelClicked);
    connect(labelWrapsButton, &QPushButton::clicked, [this]() {
        setLabelWrapsMode(!labelWrapsMode);
    });
    
    // Connect parameter changes to preview update
    connect(maxRadiusSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &SeedingWidget::updateParameterPreview);
    connect(angleStepSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &SeedingWidget::updateParameterPreview);
    
    // Set size policy
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
}

void SeedingWidget::setVolumePkg(std::shared_ptr<VolumePkg> vpkg)
{
    std::cout << "SeedingWidget::setVolumePkg called - vpkg: " << (vpkg ? "valid" : "null") << std::endl;
    fVpkg = vpkg;
    updateButtonStates();
}

void SeedingWidget::setCurrentVolume(std::shared_ptr<Volume> volume)
{
    currentVolume = volume;
    updateButtonStates();
}

void SeedingWidget::setCache(ChunkCache* cache)
{
    chunkCache = cache;
}


void SeedingWidget::onCollectionAdded(uint64_t collectionId)
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
    std::cout << "SeedingWidget::onVolumeChanged called - volume: " << (vol ? "valid" : "null")
              << ", volumeId: " << volumeId << std::endl;
    currentVolume = vol;
    currentVolumeId = volumeId;
    updateButtonStates();
}

void SeedingWidget::updateCurrentZSlice(int z)
{
    currentZSlice = z;
}

void SeedingWidget::onClearPreviewClicked()
{
    _point_collection->clearCollection(_point_collection->getCollectionId("ray_preview"));
    infoLabel->setText("Preview points cleared.");
}

void SeedingWidget::onClearPeaksClicked()
{
    _point_collection->clearCollection(_point_collection->getCollectionId("seeding_peaks"));
    infoLabel->setText("Peak points cleared.");
}

void SeedingWidget::onPreviewRaysClicked()
{
    if (currentMode != Mode::PointMode || !currentVolume) {
        return;
    }

    POI* focus_poi = _surface_collection->poi("focus");
    if (!focus_poi) {
        QMessageBox::warning(this, "Warning", "No focus point set. Please set a focus point before previewing rays.");
        return;
    }

    _point_collection->clearCollection(_point_collection->getCollectionId("ray_preview"));

    const double angleStep = angleStepSpinBox->value();
    const int numSteps = static_cast<int>(360.0 / angleStep);
    const int maxRadius = maxRadiusSpinBox->value();
    const cv::Vec3f& startPoint = focus_poi->p;

    std::vector<cv::Vec3f> preview_points;

    const int pointsPerRay = 10;
    for (int i = 0; i < numSteps; i++) {
        const double angle = i * angleStep * M_PI / 180.0;
        const cv::Vec2f rayDir(cos(angle), sin(angle));

        for (int j = 1; j <= pointsPerRay; ++j) {
            const float dist = (static_cast<float>(j) / pointsPerRay) * maxRadius;
            cv::Vec3f pointOnRay;
            pointOnRay[0] = startPoint[0] + dist * rayDir[0];
            pointOnRay[1] = startPoint[1] + dist * rayDir[1];
            pointOnRay[2] = startPoint[2];
            preview_points.push_back(pointOnRay);
        }
    }

    if (!preview_points.empty()) {
        _point_collection->addPoints("ray_preview", preview_points);
    }

    infoLabel->setText(QString("Previewing %1 rays.").arg(numSteps));
}

void SeedingWidget::onCastRaysClicked()
{
    if (currentMode == Mode::PointMode) {
        if (!currentVolume) {
            return;
        }
        
        // Reset previous peaks
        _point_collection->clearCollection(_point_collection->getCollectionId("seeding_peaks"));
        
        // Compute distance transform for the current slice
        computeDistanceTransform();
        
        // Cast rays and find peaks
        castRays();
        
        // Enable segmentation button if we found peaks
        runSegmentationButton->setEnabled(!_point_collection->getAllCollections().at(_point_collection->getCollectionId("seeding_peaks")).points.empty());
        
        // Update UI with clearer instructions about the displayed points
        infoLabel->setText(QString("Found %1 peaks (shown in red). Review points then click 'Run Segmentation'.").arg(_point_collection->getAllCollections().at(_point_collection->getCollectionId("seeding_peaks")).points.size()));
        emit sendStatusMessageAvailable(
            QString("Cast %1 rays and found %2 intensity peaks. Points are displayed for review.").arg(360.0 / angleStepSpinBox->value()).arg(_point_collection->getAllCollections().at(_point_collection->getCollectionId("seeding_peaks")).points.size()),
            5000);
    } else {
        // Draw mode - analyze paths
        analyzePaths();
    }
}

void SeedingWidget::computeDistanceTransform()
{
    if (!currentVolume) {
        return;
    }
    
    // Get the current slice data
    const int width = currentVolume->sliceWidth();
    const int height = currentVolume->sliceHeight();
    
    cv::Mat_<uint8_t> sliceData(height, width);
    
    // Extract the slice data from the volume
    cv::Mat_<cv::Vec3f> coords(height, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            coords(y, x) = cv::Vec3f(x, y, currentZSlice);
        }
    }
    
    // Read the slice data using the volume's dataset
    readInterpolated3D(sliceData, currentVolume->zarrDataset(0), coords, chunkCache);
    
    // Threshold the slice to create a binary image for distance transform
    cv::Mat binaryImage;
    cv::threshold(sliceData, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Compute the distance transform
    cv::distanceTransform(binaryImage, distanceTransform, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    
    // Normalize the distance transform for visualization if needed
    cv::Mat distNormalized;
    cv::normalize(distanceTransform, distNormalized, 0, 255, cv::NORM_MINMAX);
    distNormalized.convertTo(distNormalized, CV_8UC1);
}

void SeedingWidget::castRays()
{
    if (!currentVolume) {
        return;
    }
    
    // Setup progress tracking
    progressBar->setVisible(true);
    progressBar->setValue(0);
    
    // Cast rays at regular angle steps
    const double angleStep = angleStepSpinBox->value();
    const int numSteps = static_cast<int>(360.0 / angleStep);
    
    for (int i = 0; i < numSteps; i++) {
        // Calculate ray direction in 2D (will be used for XY plane)
        const double angle = i * angleStep * M_PI / 180.0;
        const cv::Vec2f rayDir(cos(angle), sin(angle));
        
        // Find peaks along this ray in 3D
        POI* focus_poi = _surface_collection->poi("focus");
        if (!focus_poi) {
            QMessageBox::warning(this, "Warning", "No focus point set. Please set a focus point before casting rays.");
            return;
        }
        findPeaksAlongRay(rayDir, focus_poi->p);
        
        // Update progress
        progressBar->setValue((i + 1) * 100 / numSteps);
        QApplication::processEvents();
    }
    
    progressBar->setVisible(false);
}

void SeedingWidget::findPeaksAlongRay(
    const cv::Vec2f& rayDir, 
    const cv::Vec3f& startPoint)
{
    if (!currentVolume) {
        return;
    }
    
    const int maxRadius = maxRadiusSpinBox->value();
    const int width = currentVolume->sliceWidth();
    const int height = currentVolume->sliceHeight();
    const int depth = currentVolume->numSlices();
    
    std::vector<float> intensities;
    std::vector<cv::Vec3f> positions;
    
    // Get the window size from the spinbox
    const int window = windowSizeSpinBox->value();
    
    // Trace ray up to max radius (assuming ray is in XY plane for now)
    for (int dist = 1; dist < maxRadius; dist++) {
        cv::Vec3f point;
        point[0] = startPoint[0] + dist * rayDir[0];
        point[1] = startPoint[1] + dist * rayDir[1];
        point[2] = startPoint[2]; // Keep Z constant for now (ray in XY plane)
        
        // Check bounds
        if (point[0] < 0 || point[0] >= width || 
            point[1] < 0 || point[1] >= height ||
            point[2] < 0 || point[2] >= depth) {
            break;
        }
        
        // Read intensity at this 3D point
        cv::Mat_<cv::Vec3f> coord(1, 1);
        coord(0, 0) = point;
        
        cv::Mat_<uint8_t> intensity(1, 1);
        readInterpolated3D(intensity, currentVolume->zarrDataset(0), coord, chunkCache);
        
        // Store intensity and position
        intensities.push_back(intensity(0, 0));
        positions.push_back(point);
    }
    
    if (intensities.empty()) {
        return;
    }
    
    // Place a single point at the center of each above-threshold segment
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
        // Prefer the position with the largest distance-transform value within the segment (center of band)
        size_t best_idx = s;
        float best_dt = sampleDistAt(positions[s]);
        for (size_t k = s; k <= e; ++k) {
            float dt = sampleDistAt(positions[k]);
            if (dt > best_dt) {
                best_dt = dt;
                best_idx = k;
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

void SeedingWidget::onRunSegmentationClicked()
{
    std::cout << "SeedingWidget::onRunSegmentationClicked - START" << std::endl;
    std::cout << "  currentVolume: " << (currentVolume ? "valid" : "null") << std::endl;
    std::cout << "  currentVolumeId: " << currentVolumeId << std::endl;
    std::cout << "  fVpkg: " << (fVpkg ? "valid" : "null") << std::endl;
    
    // Get the selected collection name from the combo box
    std::string sourceCollection = collectionComboBox->currentText().toStdString();
    if (sourceCollection.empty()) {
        QMessageBox::warning(this, "Warning", "Please select a source collection.");
        return;
    }

    // Combine analysis points and user-placed points for segmentation
    std::vector<ColPoint> allPoints = _point_collection->getPoints(sourceCollection);
    
    if (allPoints.empty() || !fVpkg) {
        QMessageBox::warning(this, "Error", "No points available for segmentation or volume package not loaded.");
        return;
    }
    
    // Update UI
    progressBar->setVisible(true);
    progressBar->setValue(0);
    infoLabel->setText("Running segmentation jobs...");
    runSegmentationButton->setEnabled(false);
    expandSeedsButton->setEnabled(false);
    
    // Show cancel button and set jobs running
    jobsRunning = true;
    cancelButton->setVisible(true);
    runningProcesses.clear();
    
    const int numProcesses = processesSpinBox->value();
    const int totalPoints = static_cast<int>(allPoints.size());
    const int ompThreads = ompThreadsSpinBox ? ompThreadsSpinBox->value() : 0;

    // Get paths
    std::filesystem::path pathsDir;
    std::filesystem::path seedJsonPath;
    
    if (fVpkg->hasSegmentations() && !fVpkg->segmentationIDs().empty()) {
        auto segID = fVpkg->segmentationIDs()[0];
        auto seg = fVpkg->segmentation(segID);
        pathsDir = seg->path().parent_path();
        seedJsonPath = pathsDir.parent_path() / "seed.json";
    } else {
        if (!fVpkg->hasVolumes()) {
            QMessageBox::warning(this, "Error", "No volumes in volume package.");
            progressBar->setVisible(false);
            runSegmentationButton->setEnabled(true);
            return;
        }
        
        auto vol = fVpkg->volume();
        std::filesystem::path vpkgPath = vol->path().parent_path().parent_path();
        pathsDir = vpkgPath / "paths";
        seedJsonPath = vpkgPath / "seed.json";
        
        if (!std::filesystem::exists(pathsDir)) {
            QMessageBox::warning(this, "Error", "Segmentation paths directory not found in volume package.");
            progressBar->setVisible(false);
            runSegmentationButton->setEnabled(true);
            return;
        }
    }
    
    if (!std::filesystem::exists(seedJsonPath)) {
        QMessageBox::warning(this, "Error", "seed.json not found in volume package.");
        progressBar->setVisible(false);
        runSegmentationButton->setEnabled(true);
        return;
    }
    
    if (!currentVolume) {
        QMessageBox::warning(this, "Error", "No current volume selected.");
        progressBar->setVisible(false);
        runSegmentationButton->setEnabled(true);
        return;
    }
    
    std::filesystem::path volumePath = currentVolume->path();
    QString workingDir = QString::fromStdString(pathsDir.parent_path().string());
    
    // Track completion
    int completedJobs = 0;
    int nextPointIndex = 0;
    
    // Lambda to start a process for a point
    std::function<void(int)> startProcessForPoint = [&](int pointIndex) {
        if (pointIndex >= totalPoints || !jobsRunning) {
            return;
        }
        
        const auto& point = allPoints[pointIndex];
        QProcess* process = new QProcess(this);
        process->setProcessChannelMode(QProcess::MergedChannels);
        process->setWorkingDirectory(workingDir);

        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        if (ompThreads > 0) {
            env.insert("OMP_NUM_THREADS", QString::number(ompThreads));
        }
        process->setProcessEnvironment(env);

        // Connect finished signal
        connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [this, process, pointIndex, &completedJobs, &nextPointIndex, &startProcessForPoint, totalPoints]
            (int exitCode, QProcess::ExitStatus exitStatus) {
                if (!jobsRunning) {
                    return;
                }
                
                // Log result
                if (exitCode != 0) {
                    std::cerr << "Process for point " << pointIndex << " failed with exit code: " << exitCode << std::endl;
                } else {
                    std::cout << "Completed segmentation for point " << pointIndex << std::endl;
                }
                
                // Update progress
                completedJobs++;
                progressBar->setValue(completedJobs * 100 / totalPoints);
                
                // Remove from running list (QPointer will handle null checking)
                runningProcesses.removeOne(process);
                process->deleteLater();
                
                // Start next process if available
                if (nextPointIndex < totalPoints && jobsRunning) {
                    startProcessForPoint(nextPointIndex++);
                }
                
                // Check if all done
                if (completedJobs >= totalPoints) {
                    progressBar->setVisible(false);
                    cancelButton->setVisible(false);
                    jobsRunning = false;
                    runningProcesses.clear();
                    infoLabel->setText(QString("Segmentation complete for %1 points.").arg(totalPoints));
                    runSegmentationButton->setEnabled(true);
                    expandSeedsButton->setEnabled(true);
                    updateButtonStates();
                    emit sendStatusMessageAvailable(QString("Completed segmentation for %1 points").arg(totalPoints), 5000);
                }
            });
        
        // Start the process
        QStringList previewParts;
        if (ompThreads > 0) {
            previewParts << QString("OMP_NUM_THREADS=%1").arg(ompThreads);
        }
        previewParts << executablePath
                     << QString("\"%1\"").arg(QString::fromStdString(volumePath.string()))
                     << QString("\"%1\"").arg(QString::fromStdString(pathsDir.string()))
                     << QString("\"%1\"").arg(QString::fromStdString(seedJsonPath.string()))
                     << QString::number(point.p[0])
                     << QString::number(point.p[1])
                     << QString::number(point.p[2]);

        std::cout << "Starting job " << pointIndex << ": " << previewParts.join(' ').toStdString() << std::endl;
        
        process->start("nice", QStringList() << "-n" << "19" << "ionice" << "-c" << "3" << executablePath <<
                      QString::fromStdString(volumePath.string()) <<
                      QString::fromStdString(pathsDir.string()) <<
                      QString::fromStdString(seedJsonPath.string()) <<
                      QString::number(point.p[0]) <<
                      QString::number(point.p[1]) <<
                      QString::number(point.p[2]));
        
        runningProcesses.append(QPointer<QProcess>(process));
    };
    
    // Start initial batch of processes
    for (int i = 0; i < std::min(numProcesses, totalPoints); i++) {
        startProcessForPoint(nextPointIndex++);
    }
    
    // Process events until all jobs complete or cancelled
    while (jobsRunning && completedJobs < totalPoints) {
        QApplication::processEvents(QEventLoop::AllEvents, 100);
    }
}


void SeedingWidget::onResetPointsClicked()
{
    if (_point_collection) {
        _point_collection->clearCollection(_point_collection->getCollectionId("seeding_peaks"));
        _point_collection->clearCollection(_point_collection->getCollectionId("seeding_seeds"));
    }
    
    // Clear paths
    paths.clear();
    
    // Reset UI
    castRaysButton->setEnabled(false);
    runSegmentationButton->setEnabled(false);
    expandSeedsButton->setEnabled(false);
    resetPointsButton->setEnabled(false);
    infoLabel->setText("Click 'Cast Rays' to begin");
    
    // Redraw to clear points and paths
    updatePointsDisplay();
    displayPaths();
}

QString SeedingWidget::findExecutablePath()
{
    // vc_grow_seg_from_seed should be in the same directory as the VC3D application
    QString execPath = QCoreApplication::applicationDirPath() + "/vc_grow_seg_from_seed";
    
    QFileInfo fileInfo(execPath);
    if (fileInfo.exists() && fileInfo.isExecutable()) {
        return fileInfo.absoluteFilePath();
    }
    
    // If not found, return empty string
    return QString();
}

void SeedingWidget::updateParameterPreview()
{
    auto focus_points = _point_collection->getPoints("focus");
    if (focus_points.empty() || !currentVolume) {
        return;
    }
    auto& center_point = focus_points[0].p;

    // Clear previous preview points
    _point_collection->clearCollection(_point_collection->getCollectionId("seeding_preview"));
    
    // Get parameters
    const double angleStep = angleStepSpinBox->value();
    const int numRays = static_cast<int>(360.0 / angleStep);
    const int maxRadius = maxRadiusSpinBox->value();
    
    // Add points along the radius circle
    for (int i = 0; i < numRays; ++i) {
        const double angle = i * angleStep * M_PI / 180.0;
        const cv::Vec2f rayDir(cos(angle), sin(angle));
        
        // Add points along the radius circle
        float x = center_point[0] + maxRadius * rayDir[0];
        float y = center_point[1] + maxRadius * rayDir[1];
        
        _point_collection->addPoint("seeding_preview", {x, y, center_point[2]});
        
        // Add intermediate points for better visualization
        for (int r = maxRadius / 4; r < maxRadius; r += maxRadius / 4) {
            x = center_point[0] + r * rayDir[0];
            y = center_point[1] + r * rayDir[1];
            
            _point_collection->addPoint("seeding_preview", {x, y, center_point[2]});
        }
    }
    
    // Update info label
    infoLabel->setText(QString("Seed at (%1, %2, %3) | Preview: %4 rays, radius %5px")
                           .arg(center_point[0])
                           .arg(center_point[1])
                           .arg(center_point[2])
                           .arg(numRays)
                           .arg(maxRadius));
}

void SeedingWidget::updateModeUI()
{
    if (currentMode == Mode::PointMode) {
        castRaysButton->setText("Cast Rays");
        resetPointsButton->setText("Reset Points");
        
        // Show radius and angle controls in point mode
        maxRadiusLabel->setVisible(true);
        maxRadiusSpinBox->setVisible(true);
        angleStepLabel->setVisible(true);
        angleStepSpinBox->setVisible(true);
        
        // Enable/disable based on state
        castRaysButton->setEnabled(currentVolume != nullptr);
        resetPointsButton->setEnabled(true);
    } else { // DrawMode
        castRaysButton->setText("Analyze Paths");
        resetPointsButton->setText("Clear All Paths");
        
        // Hide radius and angle controls in draw mode
        maxRadiusLabel->setVisible(false);
        maxRadiusSpinBox->setVisible(false);
        angleStepLabel->setVisible(false);
        angleStepSpinBox->setVisible(false);
        
        // Enable/disable based on paths
        bool hasPaths = !paths.empty();
        castRaysButton->setEnabled(hasPaths && currentVolume != nullptr);
        resetPointsButton->setEnabled(hasPaths);
    }
}

void SeedingWidget::analyzePaths()
{
    if (!currentVolume || paths.empty()) {
        return;
    }
    
    // Reset previous peaks
    _point_collection->clearCollection(_point_collection->getCollectionId("seeding_peaks"));
    
    // Compute distance transform once
    computeDistanceTransform();
    
    // Setup progress tracking
    progressBar->setVisible(true);
    progressBar->setValue(0);
    
    int totalPaths = paths.size();
    int pathIndex = 0;
    
    // For each path
    for (const auto& path : paths) {
        // Analyze along this path
        findPeaksAlongPath(path);
        
        // Update progress
        pathIndex++;
        progressBar->setValue(pathIndex * 100 / totalPaths);
        QApplication::processEvents();
    }
    
    progressBar->setVisible(false);
    
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

void SeedingWidget::findPeaksAlongPath(const PathData& path)
{
    if (!currentVolume || path.points.empty()) {
        return;
    }
    
    // densify the path so we don't skip over small surfaces when drawing 
    PathData densifiedPath = path.densify(0.5f); // Sample every 0.5 pixels
    
    // Get volume dimensions for bounds checking
    const int width = currentVolume->sliceWidth();
    const int height = currentVolume->sliceHeight();
    const int depth = currentVolume->numSlices();
    
    std::vector<float> intensities;
    std::vector<cv::Vec3f> positions;
    
    // Read intensity values at each point along the (denser) path
    for (const auto& pt : densifiedPath.points) {
        // Check bounds
        if (pt[0] >= 0 && pt[0] < width && 
            pt[1] >= 0 && pt[1] < height && 
            pt[2] >= 0 && pt[2] < depth) {
            
            // Create a single-point coordinate matrix for reading
            cv::Mat_<cv::Vec3f> coord(1, 1);
            coord(0, 0) = pt;
            
            // Read the intensity value at this 3D point
            cv::Mat_<uint8_t> intensity(1, 1);
            readInterpolated3D(intensity, currentVolume->zarrDataset(0), coord, chunkCache);
            
            intensities.push_back(intensity(0, 0));
            positions.push_back(pt);
        }
    }
    
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
        // Prefer the position with the largest distance-transform value within the segment (center of band)
        size_t best_idx = s;
        float best_dt = sampleDistAt(positions[s]);
        for (size_t k = s; k <= e; ++k) {
            float dt = sampleDistAt(positions[k]);
            if (dt > best_dt) {
                best_dt = dt;
                best_idx = k;
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
    infoLabel->setText(QString("Draw Mode: %1 path(s)").arg(paths.size()));
}

void SeedingWidget::finalizePathLabelWraps(bool shiftHeld)
{
    if (!isDrawing || currentPath.points.size() < 2 || !currentVolume) {
        isDrawing = false;
        currentPath.points.clear();
        displayPaths();
        return;
    }

    // Ensure distance transform available for center-of-band selection
    computeDistanceTransform();

    // Create a new target collection
    std::string newColName = _point_collection->generateNewCollectionName("wraps");
    uint64_t newColId = _point_collection->addCollection(newColName);

    // Analyze the just-drawn path and add peaks to the new collection
    findPeaksAlongPathToCollection(currentPath, newColName);

    // Auto fill winding annotations in chosen direction
    _point_collection->autoFillWindingNumbers(
        newColId,
        shiftHeld ? VCCollection::WindingFillMode::Decremental : VCCollection::WindingFillMode::Incremental
    );

    // Reset drawing path and refresh view
    isDrawing = false;
    currentPath.points.clear();
    displayPaths();

    infoLabel->setText(QString("Label Wraps: added points to '%1' (%2)")
                           .arg(QString::fromStdString(newColName))
                           .arg(shiftHeld ? "decreasing" : "increasing"));
    updateButtonStates();
}

void SeedingWidget::findPeaksAlongPathToCollection(const PathData& path, const std::string& collectionName)
{
    if (!currentVolume || path.points.empty()) {
        return;
    }

    PathData densifiedPath = path.densify(0.5f);

    const int width = currentVolume->sliceWidth();
    const int height = currentVolume->sliceHeight();
    const int depth = currentVolume->numSlices();

    std::vector<float> intensities;
    std::vector<cv::Vec3f> positions;

    for (const auto& pt : densifiedPath.points) {
        if (pt[0] >= 0 && pt[0] < width &&
            pt[1] >= 0 && pt[1] < height &&
            pt[2] >= 0 && pt[2] < depth) {
            cv::Mat_<cv::Vec3f> coord(1, 1);
            coord(0, 0) = pt;
            cv::Mat_<uint8_t> intensity(1, 1);
            readInterpolated3D(intensity, currentVolume->zarrDataset(0), coord, chunkCache);
            intensities.push_back(intensity(0, 0));
            positions.push_back(pt);
        }
    }

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

void SeedingWidget::setLabelWrapsMode(bool active)
{
    labelWrapsMode = active;
    if (labelWrapsMode) {
        currentMode = Mode::DrawMode;
        infoLabel->setText("Label Wraps: draw a path; release to create a labeled collection. Hold Shift for decreasing order.");
        labelWrapsButton->setText("Stop Label Wraps");
    } else {
        // Revert to default UI language; keep currentMode as-is
        infoLabel->setText("Point Mode: Set a focus point to begin");
        labelWrapsButton->setText("Start Label Wraps");
    }
    updateModeUI();
    displayPaths();
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
    QList<PathData> allPaths = paths;
    
    // Add the current drawing path if we're actively drawing
    if (isDrawing && !currentPath.points.empty()) {
        allPaths.append(currentPath);
    }
    
    // Send the paths for line rendering
    emit sendPathsChanged(allPaths);
    
    // Send peaks as red points (they should still be displayed as individual points)
}

void SeedingWidget::updatePointsDisplay()
{
    // Send both analysis results (red) and user points (blue) for display
}

void SeedingWidget::updateInfoLabel()
{
    QString infoText;
    
    if (currentMode == Mode::PointMode) {
        auto focus_points = _point_collection->getPoints("focus");
        if (!focus_points.empty()) {
            auto& p = focus_points[0].p;
            infoText = QString("Seed: (%1, %2, %3) | Analysis: %4 pts | User: %5 pts")
                           .arg(p[0])
                           .arg(p[1])
                           .arg(p[2])
                           .arg(_point_collection->getPoints("seeding_peaks").size())
                           .arg(_point_collection->getPoints("seeding_seeds").size());
        } else {
            infoText = "Point Mode: Set a focus point to begin";
        }
    } else {
        infoText = QString("Draw Mode: %1 paths drawn").arg(paths.size());
    }
    infoLabel->setText(infoText);
}

void SeedingWidget::updateButtonStates()
{
    // Enable segmentation if we have any points (analysis results OR user points)
    bool hasAnyPoints = !_point_collection->getPoints("seeding_peaks").empty() || !_point_collection->getPoints("seeding_seeds").empty();
    runSegmentationButton->setEnabled(hasAnyPoints && currentVolume != nullptr);
    
    // Enable expansion if we have a volume AND at least one segmentation
    bool hasSegmentations = fVpkg && fVpkg->hasSegmentations();
    expandSeedsButton->setEnabled(currentVolume != nullptr && hasSegmentations);
    
    // Enable reset if we have any points or paths
    bool hasAnyData = hasAnyPoints || !paths.empty();
    resetPointsButton->setEnabled(hasAnyData);
    
    if (currentMode == Mode::PointMode) {
        castRaysButton->setEnabled(currentVolume != nullptr);
    } else {
        castRaysButton->setEnabled(!paths.empty());
    }
}

void SeedingWidget::onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (currentMode != Mode::DrawMode || button != Qt::LeftButton) {
        return;
    }
    
    startDrawing(vol_point);
}

void SeedingWidget::onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    if (currentMode != Mode::DrawMode || !isDrawing || !(buttons & Qt::LeftButton)) {
        return;
    }
    
    addPointToPath(vol_point);
}

void SeedingWidget::onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (currentMode != Mode::DrawMode || button != Qt::LeftButton || !isDrawing) {
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
    if (!fVpkg || !currentVolume) {
        QMessageBox::warning(this, "Error", "Volume package or volume not loaded.");
        return;
    }
    
    // Update UI
    progressBar->setVisible(true);
    progressBar->setValue(0);
    infoLabel->setText("Running expansion jobs...");
    expandSeedsButton->setEnabled(false);
    runSegmentationButton->setEnabled(false);
    
    // Show cancel button and set jobs running
    jobsRunning = true;
    cancelButton->setVisible(true);
    runningProcesses.clear();
    
    const int numProcesses = processesSpinBox->value();
    const int expansionIterations = expansionIterationsSpinBox->value();
    const int ompThreads = ompThreadsSpinBox ? ompThreadsSpinBox->value() : 0;

    // Get paths
    std::filesystem::path pathsDir;
    std::filesystem::path expandJsonPath;
    
    if (fVpkg->hasSegmentations() && !fVpkg->segmentationIDs().empty()) {
        auto segID = fVpkg->segmentationIDs()[0];
        auto seg = fVpkg->segmentation(segID);
        pathsDir = seg->path().parent_path();
        expandJsonPath = pathsDir.parent_path() / "expand.json";
    } else {
        if (!fVpkg->hasVolumes()) {
            QMessageBox::warning(this, "Error", "No volumes in volume package.");
            progressBar->setVisible(false);
            expandSeedsButton->setEnabled(true);
            return;
        }
        
        auto vol = fVpkg->volume();
        std::filesystem::path vpkgPath = vol->path().parent_path().parent_path();
        pathsDir = vpkgPath / "paths";
        expandJsonPath = vpkgPath / "expand.json";
        
        if (!std::filesystem::exists(pathsDir)) {
            QMessageBox::warning(this, "Error", "Segmentation paths directory not found in volume package.");
            progressBar->setVisible(false);
            expandSeedsButton->setEnabled(true);
            return;
        }
    }
    
    if (!std::filesystem::exists(expandJsonPath)) {
        QMessageBox::warning(this, "Error", "expand.json not found in volume package.");
        progressBar->setVisible(false);
        expandSeedsButton->setEnabled(true);
        return;
    }
    
    std::filesystem::path volumePath = currentVolume->path();
    QString workingDir = QString::fromStdString(pathsDir.parent_path().string());
    
    // Track completion
    int completedJobs = 0;
    int nextIterationIndex = 0;
    
    // Lambda to start an expansion process
    std::function<void(int)> startExpansionProcess = [&](int iterationIndex) {
        if (iterationIndex >= expansionIterations || !jobsRunning) {
            return;
        }
        
        QProcess* process = new QProcess(this);
        process->setProcessChannelMode(QProcess::MergedChannels);
        process->setWorkingDirectory(workingDir);

        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        if (ompThreads > 0) {
            env.insert("OMP_NUM_THREADS", QString::number(ompThreads));
        }
        process->setProcessEnvironment(env);

        // Connect finished signal
        connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [this, process, iterationIndex, &completedJobs, &nextIterationIndex, &startExpansionProcess, expansionIterations]
            (int exitCode, QProcess::ExitStatus exitStatus) {
                if (!jobsRunning) {
                    return;
                }
                
                // Log result
                if (exitCode != 0) {
                    std::cerr << "Expansion iteration " << iterationIndex << " failed with exit code: " << exitCode << std::endl;
                } else {
                    std::cout << "Completed expansion iteration " << iterationIndex << std::endl;
                }
                
                // Update progress
                completedJobs++;
                progressBar->setValue(completedJobs * 100 / expansionIterations);
                
                // Remove from running list (QPointer will handle null checking)
                runningProcesses.removeOne(process);
                process->deleteLater();
                
                // Start next process if available
                if (nextIterationIndex < expansionIterations && jobsRunning) {
                    startExpansionProcess(nextIterationIndex++);
                }
                
                // Check if all done
                if (completedJobs >= expansionIterations) {
                    progressBar->setVisible(false);
                    cancelButton->setVisible(false);
                    jobsRunning = false;
                    runningProcesses.clear();
                    infoLabel->setText(QString("Expansion complete after %1 iterations.").arg(expansionIterations));
                    expandSeedsButton->setEnabled(true);
                    runSegmentationButton->setEnabled(true);
                    updateButtonStates();
                    emit sendStatusMessageAvailable(QString("Completed %1 expansion iterations").arg(expansionIterations), 5000);
                }
            });
        
        // Start the process
        QStringList previewParts;
        if (ompThreads > 0) {
            previewParts << QString("OMP_NUM_THREADS=%1").arg(ompThreads);
        }
        previewParts << executablePath
                     << QString("\"%1\"").arg(QString::fromStdString(volumePath.string()))
                     << QString("\"%1\"").arg(QString::fromStdString(pathsDir.string()))
                     << QString("\"%1\"").arg(QString::fromStdString(expandJsonPath.string()));

        std::cout << "Starting expansion job " << iterationIndex << ": " << previewParts.join(' ').toStdString() << std::endl;
        
        process->start("nice", QStringList() << "-n" << "19" << "ionice" << "-c" << "3" << executablePath <<
                      QString::fromStdString(volumePath.string()) <<
                      QString::fromStdString(pathsDir.string()) <<
                      QString::fromStdString(expandJsonPath.string()));
        
        runningProcesses.append(QPointer<QProcess>(process));
    };
    
    // Start initial batch of processes
    for (int i = 0; i < std::min(numProcesses, expansionIterations); i++) {
        startExpansionProcess(nextIterationIndex++);
    }
    
    // Process events until all jobs complete or cancelled
    while (jobsRunning && completedJobs < expansionIterations) {
        QApplication::processEvents(QEventLoop::AllEvents, 100);
    }
}

void SeedingWidget::onCancelClicked()
{
    if (!jobsRunning || runningProcesses.isEmpty()) {
        return;
    }
    
    // Set flag to stop any new processes from starting
    jobsRunning = false;
    
    // using qpointer here to avoid dangling pointers
    for (const QPointer<QProcess>& processPtr : runningProcesses) {
        if (processPtr) {
            QProcess* process = processPtr.data();
            
            // Disconnect all signals to prevent callbacks after cancellation
            process->disconnect();
            
            // Terminate the process if it's still running
            if (process->state() != QProcess::NotRunning) {
                process->terminate();
                // Give it a chance to terminate gracefully
                if (!process->waitForFinished(1000)) {
                    // Force kill if it didn't terminate
                    process->kill();
                    process->waitForFinished(1000);
                }
            }
            
            // Schedule deletion
            process->deleteLater();
        }
    }
    
    // Clear the list
    runningProcesses.clear();
    
    // Update UI
    cancelButton->setVisible(false);
    progressBar->setVisible(false);
    progressBar->setValue(0);
    infoLabel->setText("Jobs cancelled by user");
    
    // Re-enable buttons
    runSegmentationButton->setEnabled(true);
    expandSeedsButton->setEnabled(true);
    updateButtonStates();
    
    emit sendStatusMessageAvailable("Jobs cancelled by user", 3000);
}

void SeedingWidget::onSurfacesLoaded()
{
    // Update button states when surfaces are loaded/reloaded
    updateButtonStates();
}




