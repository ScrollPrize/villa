#pragma once

#include <QComboBox>
#include <QEvent>

namespace vc3d {

class VolumePathTooltipFilter final : public QObject
{
public:
    explicit VolumePathTooltipFilter(QComboBox* combo) : QObject(combo) {}
protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (event->type() == QEvent::ToolTip) {
            auto* combo = static_cast<QComboBox*>(watched);
            combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
        }
        return false;
    }
};

inline void refreshVolumeComboTooltip(QComboBox* combo)
{
    if (!combo->property("vcVolumePathTooltip").toBool()) {
        combo->setProperty("vcVolumePathTooltip", true);
        combo->installEventFilter(new VolumePathTooltipFilter(combo));
        QObject::connect(combo, &QComboBox::currentIndexChanged, combo, [combo] {
            combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
        });
    }
    // Population often blocks signals, so initialize explicitly as well.
    combo->setToolTip(combo->currentData(Qt::ToolTipRole).toString());
}

} // namespace vc3d
