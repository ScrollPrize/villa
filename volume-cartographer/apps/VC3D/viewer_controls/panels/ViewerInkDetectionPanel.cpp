#include "viewer_controls/panels/ViewerInkDetectionPanel.hpp"

#include "ViewerManager.hpp"
#include "elements/LabeledControlRow.hpp"
#include "overlays/InkDetectionOverlayController.hpp"

#include "Colormaps.hpp"
#include "VCSettings.hpp"
#include "overlays/InkPreviewOverlayController.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
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
    if (auto* preview = createLivePreviewGroup()) {
        layout->addWidget(preview);
    }

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
    connect(_colormapCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        auto* preview = _viewerManager ? _viewerManager->inkPreviewOverlay() : nullptr;
        if (preview && index >= 0 && _colormapCombo) {
            preview->setColormapId(_colormapCombo->itemData(index).toString().toStdString());
        }
    });

    connect(_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        if (_opacityValue) {
            _opacityValue->setText(QStringLiteral("%1%").arg(value));
        }
        if (auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr) {
            overlay->setOpacity(value);
        }
    });
    connect(_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        if (auto* preview = _viewerManager ? _viewerManager->inkPreviewOverlay() : nullptr) {
            preview->setOpacity(value);
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
    for (const auto& entry : vc::entries(vc::EntryScope::SharedOnly)) {
        _colormapCombo->addItem(QString::fromStdString(entry.label), QString::fromStdString(entry.id));
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
    auto* preview = _viewerManager ? _viewerManager->inkPreviewOverlay() : nullptr;
    const bool previewOn = preview && preview->enabled();
    if (_colormapCombo) {
        _colormapCombo->setEnabled(singleChannel || previewOn);
    }
    if (_opacitySlider) {
        _opacitySlider->setEnabled(hasSelection || previewOn);
    }
    if (_horizontalFlipButton) {
        _horizontalFlipButton->setEnabled(hasSelection);
    }
    if (_verticalFlipButton) {
        _verticalFlipButton->setEnabled(hasSelection);
    }
}

QWidget* ViewerInkDetectionPanel::createLivePreviewGroup()
{
    auto* preview = _viewerManager ? _viewerManager->inkPreviewOverlay() : nullptr;
    if (!preview) {
        return nullptr;
    }

    auto* group = new QGroupBox(tr("Live preview"), this);
    group->setToolTip(tr("Ink model output for the part of the segment on screen, computed by an ink preview "
                         "server you started (docs/ink_preview.md)."));
    auto* groupLayout = new QVBoxLayout(group);
    groupLayout->setSpacing(6);

    auto* enable = new QCheckBox(tr("Show the live preview"), group);
    enable->setChecked(preview->enabled());
    enable->setToolTip(tr("Ask the server for the visible part of the segmentation view once the view settles."));
    groupLayout->addWidget(enable);

    auto* serverRow = new LabeledControlRow(tr("Server"), group);
    auto* serverEdit = new QLineEdit(serverRow);
    serverEdit->setPlaceholderText(QString::fromLatin1(vc3d::settings::ink_preview::SERVER_URL_DEFAULT));
    serverEdit->setToolTip(tr("Address of the ink preview server, as http://HOST:PORT."));
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        serverEdit->setText(settings
                                .value(vc3d::settings::ink_preview::SERVER_URL,
                                       QString::fromLatin1(vc3d::settings::ink_preview::SERVER_URL_DEFAULT))
                                .toString());
    }
    // An empty address means the default one, which the placeholder shows.
    const auto applyServer = [preview, serverEdit]() {
        const QString text = serverEdit->text().trimmed();
        preview->setServerUrl(
            QUrl(text.isEmpty() ? QString::fromLatin1(vc3d::settings::ink_preview::SERVER_URL_DEFAULT) : text));
    };
    applyServer();
    serverRow->addControl(serverEdit, 1);
    groupLayout->addWidget(serverRow);

    auto* reversed = new QCheckBox(tr("Reverse layers"), group);
    reversed->setChecked(preview->reversed());
    reversed->setToolTip(tr("Feed the model the layers in reverse order. This is the control view: model output "
                            "that depends on the real layer order should fade. It is also the fix when a "
                            "surface's normals point the other way."));
    groupLayout->addWidget(reversed);

    auto* levelRow = new LabeledControlRow(tr("Level"), group);
    _previewLevelCombo = new QComboBox(levelRow);
    _previewLevelCombo->setToolTip(tr("Volume level the layers are sampled from. Auto picks the level whose voxel "
                                      "size is nearest the model's training voxel size."));
    levelRow->addControl(_previewLevelCombo, 1);
    groupLayout->addWidget(levelRow);

    _previewStatus = new QLabel(group);
    _previewStatus->setWordWrap(true);
    _previewStatus->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _previewStatus->setText(preview->statusText());
    groupLayout->addWidget(_previewStatus);

    populatePreviewLevels();

    connect(enable, &QCheckBox::toggled, this, [this, preview](bool checked) {
        if (checked && _opacitySlider) {
            preview->setOpacity(_opacitySlider->value());
        }
        preview->setEnabled(checked);
    });
    connect(preview, &InkPreviewOverlayController::enabledChanged, this, [this, enable](bool on) {
        const QSignalBlocker blocker(enable);
        enable->setChecked(on);
        updateControlState();
    });
    connect(serverEdit, &QLineEdit::textChanged, this, [](const QString& text) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        settings.setValue(vc3d::settings::ink_preview::SERVER_URL, text.trimmed());
    });
    connect(serverEdit, &QLineEdit::editingFinished, this, applyServer);
    connect(reversed, &QCheckBox::toggled, preview, &InkPreviewOverlayController::setReversed);
    connect(_previewLevelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, preview](int index) {
        if (index >= 0 && _previewLevelCombo) {
            preview->setLevelChoice(_previewLevelCombo->itemData(index).toInt());
        }
    });
    connect(preview, &InkPreviewOverlayController::levelsChanged,
            this, &ViewerInkDetectionPanel::populatePreviewLevels);
    connect(preview, &InkPreviewOverlayController::statusChanged, this, [this](const QString& text, bool error) {
        if (_previewStatus) {
            _previewStatus->setText(text);
            _previewStatus->setStyleSheet(error ? QStringLiteral("color: #c0392b;") : QString());
        }
    });
    // The Space key toggles the preview's opacity; the shared slider follows it.
    connect(preview, &InkPreviewOverlayController::opacityChanged, this, [this](int opacity) {
        if (_opacitySlider) {
            const QSignalBlocker blocker(_opacitySlider);
            _opacitySlider->setValue(opacity);
        }
        if (_opacityValue) {
            _opacityValue->setText(QStringLiteral("%1%").arg(opacity));
        }
        auto* overlay = _viewerManager ? _viewerManager->inkDetectionOverlay() : nullptr;
        if (overlay && !overlay->hasLoadedSelection()) {
            overlay->setOpacity(opacity);
        }
    });

    return group;
}

void ViewerInkDetectionPanel::populatePreviewLevels()
{
    auto* preview = _viewerManager ? _viewerManager->inkPreviewOverlay() : nullptr;
    if (!preview || !_previewLevelCombo) {
        return;
    }
    const QSignalBlocker blocker(_previewLevelCombo);
    _previewLevelCombo->clear();
    // With Auto chosen, the level it picked.
    const auto resolved = preview->levelChoice() == InkPreviewOverlayController::kAutoLevel
                              ? preview->resolvedLevel()
                              : std::nullopt;
    const QString autoText = resolved ? tr("Auto (level %1)").arg(*resolved) : tr("Auto");
    _previewLevelCombo->addItem(autoText, InkPreviewOverlayController::kAutoLevel);
    int current = 0;
    for (const auto& info : preview->levels()) {
        const QString text = info.voxelUm > 0.0
                                 ? tr("%1 (%2 µm)").arg(info.level).arg(info.voxelUm, 0, 'g', 4)
                                 : QString::number(info.level);
        _previewLevelCombo->addItem(text, info.level);
        if (info.level == preview->levelChoice()) {
            current = _previewLevelCombo->count() - 1;
        }
    }
    _previewLevelCombo->setCurrentIndex(current);
}
