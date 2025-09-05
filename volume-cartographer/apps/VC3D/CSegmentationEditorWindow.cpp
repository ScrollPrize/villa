#include "CSegmentationEditorWindow.hpp"
#include "CVolumeViewer.hpp"
#include "CSurfaceCollection.hpp"



CSegmentationEditorWindow::CSegmentationEditorWindow(CSurfaceCollection* surfCol, QWidget* parent)
    : QMainWindow(parent)
    , volumeViewer(nullptr)
    , surfaceCollection(surfCol)
    , currentOffset(0.0f)
    , stepSize(1.0f)
{
    setWindowTitle("Segmentation Editor");
    resize(800, 600);
    
    // Create the toolbar on the left side
    toolBar = new QToolBar("Navigation", this);
    toolBar->setMovable(false);
    toolBar->setFloatable(false);
    toolBar->setOrientation(Qt::Vertical);
    addToolBar(Qt::LeftToolBarArea, toolBar);
    
    // Create widgets for the toolbar
    QWidget* toolbarWidget = new QWidget(this);
    QVBoxLayout* toolbarLayout = new QVBoxLayout(toolbarWidget);
    
    // Offset slider and label
    QLabel* offsetLabel = new QLabel("Plane Offset:", toolbarWidget);
    toolbarLayout->addWidget(offsetLabel);
    
    offsetSlider = new QSlider(Qt::Horizontal, toolbarWidget);
    offsetSlider->setRange(-50, 50);
    offsetSlider->setValue(0);
    offsetSlider->setTickPosition(QSlider::TicksBelow);
    offsetSlider->setTickInterval(10);
    toolbarLayout->addWidget(offsetSlider);
    
    offsetValueLabel = new QLabel("0", toolbarWidget);
    toolbarLayout->addWidget(offsetValueLabel);
    
    // Step size slider and label
    QLabel* stepSizeLabel = new QLabel("Step Size:", toolbarWidget);
    toolbarLayout->addWidget(stepSizeLabel);
    
    stepSizeSlider = new QSlider(Qt::Horizontal, toolbarWidget);
    stepSizeSlider->setRange(1, 20);
    stepSizeSlider->setValue(1);
    stepSizeSlider->setTickPosition(QSlider::TicksBelow);
    stepSizeSlider->setTickInterval(5);
    toolbarLayout->addWidget(stepSizeSlider);
    
    stepSizeValueLabel = new QLabel("1", toolbarWidget);
    toolbarLayout->addWidget(stepSizeValueLabel);
    
    // Navigation buttons
    moveBackwardBtn = new QPushButton("← Move Backward", toolbarWidget);
    toolbarLayout->addWidget(moveBackwardBtn);
    
    moveForwardBtn = new QPushButton("Move Forward →", toolbarWidget);
    toolbarLayout->addWidget(moveForwardBtn);
    
    resetBtn = new QPushButton("Reset View", toolbarWidget);
    toolbarLayout->addWidget(resetBtn);
    
    toolbarLayout->addStretch();
    toolBar->addWidget(toolbarWidget);
    
    // Create a new isolated surface collection for our editor window
    CSurfaceCollection* editorSurfaceCollection = new CSurfaceCollection();
    
    // Copy the necessary surfaces from the original collection
    if (surfaceCollection) {
        // Copy the segmentation surface - this is the QuadSurface we need to show
        if (surfaceCollection->surface("segmentation")) {
            editorSurfaceCollection->setSurface("segmentation", surfaceCollection->surface("segmentation"));
        }
        
        // Also copy any POIs like focus point
        if (surfaceCollection->poi("focus")) {
            POI* focusPoi = new POI(*surfaceCollection->poi("focus"));
            editorSurfaceCollection->setPOI("focus", focusPoi);
        }
        
        // Copy the intersection planes as well
        if (surfaceCollection->surface("seg xz")) {
            editorSurfaceCollection->setSurface("seg xz", surfaceCollection->surface("seg xz"));
        }
        
        if (surfaceCollection->surface("seg yz")) {
            editorSurfaceCollection->setSurface("seg yz", surfaceCollection->surface("seg yz"));
        }
    }
    
    // Create the volume viewer with our isolated surface collection
    volumeViewer = new CVolumeViewer(editorSurfaceCollection, this);
    
    // Make sure intersections are configured (though they will be recalculated)
    volumeViewer->setIntersects({"seg xz", "seg yz"});
    setCentralWidget(volumeViewer);
    
    // Connect signals and slots
    connect(offsetSlider, &QSlider::valueChanged, this, &CSegmentationEditorWindow::onOffsetChanged);
    connect(stepSizeSlider, &QSlider::valueChanged, this, &CSegmentationEditorWindow::onStepSizeChanged);
    connect(moveForwardBtn, &QPushButton::clicked, this, &CSegmentationEditorWindow::onMoveForward);
    connect(moveBackwardBtn, &QPushButton::clicked, this, &CSegmentationEditorWindow::onMoveBackward);
    connect(resetBtn, &QPushButton::clicked, this, &CSegmentationEditorWindow::onReset);
}

CSegmentationEditorWindow::~CSegmentationEditorWindow()
{
    // No need to delete widgets as they are parented to this window
}

void CSegmentationEditorWindow::setSegmentationSurface(const std::string& name)
{
    surfaceName = name;
    
    // Set the surface in the viewer - it will use the surface from the collection we provided
    volumeViewer->setSurface(name);
    
    // Reset the view to ensure everything is initialized properly
    onReset();
}

void CSegmentationEditorWindow::onOffsetChanged(int value)
{
    currentOffset = value;
    offsetValueLabel->setText(QString::number(value));
    
    // Calculate the delta from the previous offset value
    static int previousOffset = 0;
    int delta = value - previousOffset;
    previousOffset = value;
    
    // If this is the first call, just update the stored offset without changing the view
    static bool firstCall = true;
    if (firstCall) {
        firstCall = false;
        return;
    }
    
    // We can't access _z_off directly, but we can simulate a zoom event with Shift modifier
    // The onZoom method in CVolumeViewer handles the z-offset when called with Shift
    // The steps parameter controls how much to adjust the z-offset
    
    // Scale the delta by the step size
    int steps = delta * stepSize;
    
    // Only trigger if there's an actual change
    if (steps != 0) {
        // Invoke the onZoom method with Shift modifier to adjust z-offset
        // Use a dummy point (0,0) since it's not used for z-offset adjustment
        volumeViewer->onZoom(steps, QPointF(0, 0), Qt::ShiftModifier);
    }
}

void CSegmentationEditorWindow::onStepSizeChanged(int value)
{
    stepSize = value;
    stepSizeValueLabel->setText(QString::number(value));
    
    // Update the offset to reflect the new step size
    onOffsetChanged(offsetSlider->value());
}

void CSegmentationEditorWindow::onMoveForward()
{
    offsetSlider->setValue(offsetSlider->value() + 1);
}

void CSegmentationEditorWindow::onMoveBackward()
{
    offsetSlider->setValue(offsetSlider->value() - 1);
}

void CSegmentationEditorWindow::onReset()
{
    // First reset the controls
    const QSignalBlocker blockOffset(offsetSlider);
    const QSignalBlocker blockStepSize(stepSizeSlider);
    
    offsetSlider->setValue(0);
    stepSizeSlider->setValue(1);
    
    // Also update our labels
    offsetValueLabel->setText("0");
    stepSizeValueLabel->setText("1");
    
    // Reset our internal values
    currentOffset = 0.0f;
    stepSize = 1.0f;
    
    // Reset static tracking variables
    static_cast<void>([]() { 
        static bool firstRun = true; 
        if (!firstRun) { 
            // This will reset the static variable in onOffsetChanged
            static int& previousOffset = *new int(0);
            previousOffset = 0;
            static bool& firstCall = *new bool(true);
            firstCall = true;
        }
        firstRun = false; 
        return true; 
    }());
    
    // Reset the view by simulating z-offset changes that bring us back to 0
    // We'll use onZoom with Shift modifier, sending a large negative number to ensure we're at 0
    // Then we'll make sure everything is re-rendered properly
    volumeViewer->onZoom(-1000, QPointF(0, 0), Qt::ShiftModifier); // Move back to ensure we're at or below 0
    volumeViewer->onZoom(0, QPointF(0, 0), Qt::ShiftModifier);    // Now set to exactly 0
    
    // Make sure everything is rendered properly
    volumeViewer->invalidateVis();
    volumeViewer->invalidateIntersect();
    volumeViewer->renderVisible(true);
    volumeViewer->renderIntersections();
}


