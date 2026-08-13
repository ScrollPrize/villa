#include "viewer_controls/panels/ViewerInkDetectionPanel.hpp"

#include "ViewerManager.hpp"
#include "VolumeViewerCmaps.hpp"
#include "elements/LabeledControlRow.hpp"
#include "overlays/InkDetectionOverlayController.hpp"

#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QVBoxLayout>

#include <filesystem>

ViewerInkDetectionPanel::ViewerInkDetectionPanel(ViewerManager* viewerManager, QWidget* parent)
    : QWidget(parent)
    , _viewerManager(viewerManager)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto* detectionRow = new LabeledControlRow(tr("Detection"), this);
    _detectionCombo = new QComboBox(detectionRow);
    detectionRow->addControl(_detectionCombo, 1);
    layout->addWidget(detectionRow);

    auto* colormapRow = new LabeledControlRow(tr("Colormap"), this);
    _colormapCombo = new QComboBox(colormapRow);
    colormapRow->addControl(_colormapCombo, 1);
    layout->addWidget(colormapRow);

    auto* opacityRow = new LabeledControlRow(tr("Opacity"), this);
    _opacitySlider = new QSlider(Qt::Horizontal, opacityRow);
    _opacitySlider->setRange(0, 100);
    _opacitySlider->setSingleStep(5);
    _opacityValue = new QLabel(opacityRow);
    _opacityValue->setMinimumWidth(36);
    opacityRow->addControl(_opacitySlider, 1);
    opacityRow->addControl(_opacityValue);
    layout->addWidget(opacityRow);

    auto* flipRow = new LabeledControlRow(tr("Flip"), this);
    _horizontalFlipButton = new QPushButton(tr("H Flip"), flipRow);
    _horizontalFlipButton->setCheckable(true);
    _horizontalFlipButton->setToolTip(tr("Mirror the ink detection overlay horizontally."));
    _verticalFlipButton = new QPushButton(tr("V Flip"), flipRow);
    _verticalFlipButton->setCheckable(true);
    _verticalFlipButton->setToolTip(tr("Mirror the ink detection overlay vertically."));
    flipRow->addControl(_horizontalFlipButton);
    flipRow->addControl(_verticalFlipButton);
    flipRow->addStretch(1);
    layout->addWidget(flipRow);

    populateColormaps();
    populateDetections();

    if (auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr) {
        _opacitySlider->setValue(overlay->opacity());
        _opacityValue->setText(QStringLiteral("%1%").arg(overlay->opacity()));
        _horizontalFlipButton->setChecked(overlay->horizontalFlip());
        _verticalFlipButton->setChecked(overlay->verticalFlip());
        connect(overlay, &InkDetectionOverlayController::availableDetectionsChanged,
                this, &ViewerInkDetectionPanel::populateDetections);
        connect(overlay, &InkDetectionOverlayController::selectionChanged,
                this, &ViewerInkDetectionPanel::updateControlState);
        connect(overlay, &InkDetectionOverlayController::opacityChanged, this, [this](int opacity) {
            if (_opacitySlider) {
                const QSignalBlocker blocker(_opacitySlider);
                _opacitySlider->setValue(opacity);
            }
            if (_opacityValue) {
                _opacityValue->setText(QStringLiteral("%1%").arg(opacity));
            }
            updateControlState();
        });
        connect(overlay, &InkDetectionOverlayController::flipChanged,
                this, [this](bool horizontal, bool vertical) {
            if (_horizontalFlipButton) {
                const QSignalBlocker blocker(_horizontalFlipButton);
                _horizontalFlipButton->setChecked(horizontal);
            }
            if (_verticalFlipButton) {
                const QSignalBlocker blocker(_verticalFlipButton);
                _verticalFlipButton->setChecked(vertical);
            }
        });
    } else {
        _opacitySlider->setValue(70);
        _opacityValue->setText(QStringLiteral("70%"));
    }

    connect(_detectionCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr;
        if (!overlay || index < 0 || !_detectionCombo) {
            return;
        }
        const QString path = _detectionCombo->itemData(index).toString();
        if (path.isEmpty()) {
            overlay->clearSelection();
        } else {
            overlay->setSelectedPath(std::filesystem::path(path.toStdString()));
        }
        updateControlState();
    });

    connect(_colormapCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr;
        if (!overlay || index < 0 || !_colormapCombo) {
            return;
        }
        overlay->setColormapId(_colormapCombo->itemData(index).toString().toStdString());
    });

    connect(_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        if (_opacityValue) {
            _opacityValue->setText(QStringLiteral("%1%").arg(value));
        }
        if (auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr) {
            overlay->setOpacity(value);
        }
    });

    connect(_horizontalFlipButton, &QPushButton::toggled, this, [this](bool checked) {
        if (auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr) {
            overlay->setHorizontalFlip(checked);
        }
    });

    connect(_verticalFlipButton, &QPushButton::toggled, this, [this](bool checked) {
        if (auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr) {
            overlay->setVerticalFlip(checked);
        }
    });

    updateControlState();
}

void ViewerInkDetectionPanel::populateDetections()
{
    if (!_detectionCombo) {
        return;
    }

    const QSignalBlocker blocker(_detectionCombo);
    const QString previous = _detectionCombo->currentData().toString();
    _detectionCombo->clear();
    _detectionCombo->addItem(tr("None"), QString());

    auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr;
    int selectedIndex = 0;
    if (overlay) {
        const QString selectedPath = QString::fromStdString(overlay->selectedPath().string());
        const QString target = selectedPath.isEmpty() ? previous : selectedPath;
        int row = 1;
        for (const auto& option : overlay->options()) {
            const QString path = QString::fromStdString(option.localPath.string());
            _detectionCombo->addItem(option.label, path);
            if (!target.isEmpty() && path == target) {
                selectedIndex = row;
            }
            ++row;
        }
    }
    _detectionCombo->setCurrentIndex(selectedIndex);
    updateControlState();
}

void ViewerInkDetectionPanel::populateColormaps()
{
    if (!_colormapCombo) {
        return;
    }
    const QSignalBlocker blocker(_colormapCombo);
    _colormapCombo->clear();
    _colormapCombo->addItem(tr("Grayscale"), QString());
    const auto& entries = volume_viewer_cmaps::entries(volume_viewer_cmaps::EntryScope::SharedOnly);
    for (const auto& entry : entries) {
        _colormapCombo->addItem(entry.label, QString::fromStdString(entry.id));
    }
    _colormapCombo->setCurrentIndex(0);
}

void ViewerInkDetectionPanel::updateControlState()
{
    auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr;
    const bool hasOverlay = overlay != nullptr;
    const bool hasDetections = hasOverlay && !overlay->options().empty();
    const bool hasSelection = hasOverlay && !overlay->selectedPath().empty();
    const bool singleChannel = hasSelection && overlay->selectedIsSingleChannel();

    if (_detectionCombo) {
        _detectionCombo->setEnabled(hasOverlay && hasDetections);
    }
    if (_colormapCombo) {
        _colormapCombo->setEnabled(singleChannel);
    }
    if (_opacitySlider) {
        _opacitySlider->setEnabled(hasSelection);
    }
    if (_horizontalFlipButton) {
        _horizontalFlipButton->setEnabled(hasSelection);
    }
    if (_verticalFlipButton) {
        _verticalFlipButton->setEnabled(hasSelection);
    }
}
