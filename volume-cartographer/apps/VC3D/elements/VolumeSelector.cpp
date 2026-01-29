#include "elements/VolumeSelector.hpp"

#include <QHBoxLayout>

VolumeSelector::VolumeSelector(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    _label = new QLabel(tr("Volume:"), this);
    _combo = new QComboBox(this);

    layout->addWidget(_label);
    layout->addWidget(_combo, 1);
}

void VolumeSelector::setLabelText(const QString& text)
{
    if (_label) {
        _label->setText(text);
    }
}

void VolumeSelector::setVolumes(const QVector<VolumeOption>& volumes, const QString& defaultVolumeId)
{
    if (!_combo) {
        return;
    }

    _combo->clear();

    int defaultIndex = -1;
    for (int i = 0; i < volumes.size(); ++i) {
        const auto& opt = volumes[i];
        const QString label = opt.name.isEmpty()
            ? opt.id
            : tr("%1 (%2)").arg(opt.name, opt.id);
        _combo->addItem(label, opt.path);
        _combo->setItemData(i, opt.id, Qt::UserRole + 1);
        if (defaultIndex == -1 && !defaultVolumeId.isEmpty() && opt.id == defaultVolumeId) {
            defaultIndex = i;
        }
    }

    if (_combo->count() > 0) {
        _combo->setCurrentIndex(defaultIndex >= 0 ? defaultIndex : 0);
        _combo->setEnabled(true);
    } else {
        _combo->setEnabled(false);
    }
}

QString VolumeSelector::selectedVolumeId() const
{
    if (!_combo) {
        return QString();
    }
    return _combo->currentData(Qt::UserRole + 1).toString();
}

QString VolumeSelector::selectedVolumePath() const
{
    if (!_combo) {
        return QString();
    }
    return _combo->currentData(Qt::UserRole).toString();
}

bool VolumeSelector::hasVolumes() const
{
    return _combo && _combo->count() > 0;
}
