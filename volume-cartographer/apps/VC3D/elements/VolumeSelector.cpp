#include "elements/VolumeSelector.hpp"
#include "VolumePathLabels.hpp"
#include "VolumeComboTooltip.hpp"

#include <QAbstractItemView>
#include <QHBoxLayout>

VolumeSelector::VolumeSelector(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    _label = new QLabel(tr("Volume:"), this);
    _combo = new QComboBox(this);
    vc3d::refreshVolumeComboTooltip(_combo);

    layout->addWidget(_label);
    layout->addWidget(_combo, 1);

    // Allow the selector to shrink arbitrarily; long items should elide instead of enforcing a wide minimum.
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMinimumWidth(0);
    _label->setMinimumWidth(0);

    _combo->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    _combo->setMinimumWidth(0);
    _combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    _combo->setMinimumContentsLength(0);
    if (auto* v = _combo->view()) {
        v->setTextElideMode(Qt::ElideRight);
    }
}

void VolumeSelector::setLabelText(const QString& text)
{
    if (_label) {
        _label->setText(text);
    }
}

void VolumeSelector::setLabelVisible(bool visible)
{
    if (_label) {
        _label->setVisible(visible);
    }
}

void VolumeSelector::setVolumes(const QVector<VolumeOption>& volumes, const QString& defaultVolumeId)
{
    if (!_combo) {
        return;
    }

    _combo->clear();

    int defaultIndex = -1;
    QStringList paths;
    for (const auto& opt : volumes) paths.push_back(opt.path);
    const auto suffixes = vc3d::distinguishingVolumePaths(paths);
    for (int i = 0; i < volumes.size(); ++i) {
        const auto& opt = volumes[i];
        QString label = opt.name.isEmpty()
            ? opt.id
            : tr("%1 (%2)").arg(opt.name, opt.id);
        if (suffixes[i].contains('/'))
            label += QStringLiteral(" [%1]").arg(suffixes[i]);
        const int row = _combo->count();
        _combo->addItem(label, opt.path);
        _combo->setItemData(row, opt.id, Qt::UserRole + 1);
        _combo->setItemData(row, vc3d::volumePathForDisplay(opt.path), Qt::ToolTipRole);
        if (defaultIndex == -1 && !defaultVolumeId.isEmpty() && opt.id == defaultVolumeId) {
            defaultIndex = row;
        }
    }

    if (_combo->count() > 0) {
        _combo->setCurrentIndex(defaultIndex >= 0 ? defaultIndex : 0);
        _combo->setEnabled(true);
    } else {
        _combo->setEnabled(false);
    }
    vc3d::refreshVolumeComboTooltip(_combo);
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
    if (!_combo) {
        return false;
    }

    for (int i = 0; i < _combo->count(); ++i) {
        const QString id = _combo->itemData(i, Qt::UserRole + 1).toString();
        const QString path = _combo->itemData(i, Qt::UserRole).toString();
        if (!id.isEmpty() || !path.isEmpty()) {
            return true;
        }
    }

    return false;
}
