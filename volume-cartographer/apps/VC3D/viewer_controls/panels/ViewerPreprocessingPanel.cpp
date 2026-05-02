#include "viewer_controls/panels/ViewerPreprocessingPanel.hpp"

#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QLabel>
#include <QLayout>
#include <QScrollArea>
#include <QSlider>
#include <QString>
#include <QVBoxLayout>

#include <algorithm>
#include <cstdint>

namespace
{

void moveLayoutItems(QLayout* from, QLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setSpacing(from->spacing());
    while (auto* item = from->takeAt(0)) {
        if (newParent) {
            if (auto* widget = item->widget()) {
                widget->setParent(newParent);
            } else if (auto* layout = item->layout()) {
                layout->setParent(newParent);
            }
        }
        to->addItem(item);
    }
}

} // namespace

ViewerPreprocessingPanel::ViewerPreprocessingPanel(const UiRefs& uiRefs,
                                                   ViewerManager* viewerManager,
                                                   QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
    , _viewerManager(viewerManager)
{
    if (_uiRefs.scrollArea && _uiRefs.scrollArea->widget() == _uiRefs.contents) {
        _uiRefs.scrollArea->takeWidget();
    }

    auto* layout = new QVBoxLayout(this);
    moveLayoutItems(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr, layout, this);

    setupControls();
}

void ViewerPreprocessingPanel::setupControls()
{
    if (!_uiRefs.isoCutoff) {
        return;
    }

    connect(_uiRefs.isoCutoff, &QSlider::valueChanged, this, [this](int value) {
        if (_uiRefs.isoCutoffValue) {
            _uiRefs.isoCutoffValue->setText(QString::number(value));
        }
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachBaseViewer([value](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            s.params.isoCutoff = static_cast<uint8_t>(std::clamp(value, 0, 255));
            viewer->setCompositeRenderSettings(s);
        });
    });
}
