#include "viewer_controls/panels/ViewerNormalVisualizationPanel.hpp"

#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QCheckBox>
#include <QGridLayout>
#include <QLabel>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>

#include <algorithm>

namespace
{

void moveGridLayoutItems(QGridLayout* from, QGridLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setHorizontalSpacing(from->horizontalSpacing());
    to->setVerticalSpacing(from->verticalSpacing());
    for (int column = 0; column < from->columnCount(); ++column) {
        to->setColumnStretch(column, from->columnStretch(column));
        to->setColumnMinimumWidth(column, from->columnMinimumWidth(column));
    }
    for (int row = 0; row < from->rowCount(); ++row) {
        to->setRowStretch(row, from->rowStretch(row));
        to->setRowMinimumHeight(row, from->rowMinimumHeight(row));
    }
    for (int index = from->count() - 1; index >= 0; --index) {
        int row = 0;
        int column = 0;
        int rowSpan = 1;
        int columnSpan = 1;
        from->getItemPosition(index, &row, &column, &rowSpan, &columnSpan);
        if (auto* item = from->takeAt(index)) {
            if (newParent) {
                if (auto* widget = item->widget()) {
                    widget->setParent(newParent);
                } else if (auto* layout = item->layout()) {
                    layout->setParent(newParent);
                }
            }
            to->addItem(item, row, column, rowSpan, columnSpan, item->alignment());
        }
    }
}

} // namespace

ViewerNormalVisualizationPanel::ViewerNormalVisualizationPanel(const UiRefs& uiRefs,
                                                               ViewerManager* viewerManager,
                                                               QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
    , _viewerManager(viewerManager)
{
    auto* layout = new QGridLayout(this);
    moveGridLayoutItems(qobject_cast<QGridLayout*>(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr),
                        layout,
                        this);
    setupControls();
}

void ViewerNormalVisualizationPanel::setupControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (auto* chkShowNormals = _uiRefs.showSurfaceNormals) {
        bool showNormals = settings.value(vc3d::settings::viewer::SHOW_SURFACE_NORMALS,
                                          vc3d::settings::viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        {
            QSignalBlocker blocker(chkShowNormals);
            chkShowNormals->setChecked(showNormals);
        }
        updateControlsEnabled(showNormals);

        connect(chkShowNormals, &QCheckBox::toggled, this, [this](bool checked) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::SHOW_SURFACE_NORMALS, checked ? "1" : "0");
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([checked](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setShowSurfaceNormals(checked);
                    }
                });
            }
            updateControlsEnabled(checked);
            emit statusMessageRequested(checked ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
        });
    }

    if (auto* sliderArrowLength = _uiRefs.normalArrowLengthSlider) {
        int savedScale = settings.value(vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
                                        vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE_DEFAULT).toInt();
        savedScale = std::clamp(savedScale, sliderArrowLength->minimum(), sliderArrowLength->maximum());
        {
            QSignalBlocker blocker(sliderArrowLength);
            sliderArrowLength->setValue(savedScale);
        }
        if (_uiRefs.normalArrowLengthValueLabel) {
            _uiRefs.normalArrowLengthValueLabel->setText(tr("%1%").arg(savedScale));
        }
        const float scaleFloat = static_cast<float>(savedScale) / 100.0f;
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([scaleFloat](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setNormalArrowLengthScale(scaleFloat);
                }
            });
        }

        connect(sliderArrowLength, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_ARROW_LENGTH_SCALE, value);
            if (_uiRefs.normalArrowLengthValueLabel) {
                _uiRefs.normalArrowLengthValueLabel->setText(tr("%1%").arg(value));
            }
            const float scaleFloat = static_cast<float>(value) / 100.0f;
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([scaleFloat](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setNormalArrowLengthScale(scaleFloat);
                    }
                });
            }
        });
    }

    if (auto* sliderMaxArrows = _uiRefs.normalMaxArrowsSlider) {
        int savedMaxArrows = settings.value(vc3d::settings::viewer::NORMAL_MAX_ARROWS,
                                            vc3d::settings::viewer::NORMAL_MAX_ARROWS_DEFAULT).toInt();
        savedMaxArrows = std::clamp(savedMaxArrows, sliderMaxArrows->minimum(), sliderMaxArrows->maximum());
        {
            QSignalBlocker blocker(sliderMaxArrows);
            sliderMaxArrows->setValue(savedMaxArrows);
        }
        if (_uiRefs.normalMaxArrowsValueLabel) {
            _uiRefs.normalMaxArrowsValueLabel->setText(QString::number(savedMaxArrows));
        }
        if (_viewerManager) {
            _viewerManager->forEachBaseViewer([savedMaxArrows](VolumeViewerBase* viewer) {
                if (viewer) {
                    viewer->setNormalMaxArrows(savedMaxArrows);
                }
            });
        }

        connect(sliderMaxArrows, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_MAX_ARROWS, value);
            if (_uiRefs.normalMaxArrowsValueLabel) {
                _uiRefs.normalMaxArrowsValueLabel->setText(QString::number(value));
            }
            if (_viewerManager) {
                _viewerManager->forEachBaseViewer([value](VolumeViewerBase* viewer) {
                    if (viewer) {
                        viewer->setNormalMaxArrows(value);
                    }
                });
            }
        });
    }
}

void ViewerNormalVisualizationPanel::updateControlsEnabled(bool enabled)
{
    QWidget* controls[] = {
        _uiRefs.normalArrowLengthLabel,
        _uiRefs.normalArrowLengthSlider,
        _uiRefs.normalArrowLengthValueLabel,
        _uiRefs.normalMaxArrowsLabel,
        _uiRefs.normalMaxArrowsSlider,
        _uiRefs.normalMaxArrowsValueLabel,
    };
    for (auto* widget : controls) {
        if (widget) {
            widget->setEnabled(enabled);
        }
    }
}
