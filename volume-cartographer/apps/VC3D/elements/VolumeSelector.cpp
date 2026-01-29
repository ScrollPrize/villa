#include "elements/VolumeSelector.hpp"

#include <QHBoxLayout>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>

VolumeSelector::VolumeSelector(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    _label = new QLabel(tr("Volume:"), this);
    _combo = new QComboBox(this);
    _browseButton = new QToolButton(this);
    _browseButton->setText(QStringLiteral("…"));
    _browseButton->setToolTip(tr("Browse for a volume path"));
    _browseButton->setVisible(false);

    layout->addWidget(_label);
    layout->addWidget(_combo, 1);
    layout->addWidget(_browseButton);

    connect(_browseButton, &QToolButton::clicked, this, [this]() {
        if (!_combo) {
            return;
        }

        QString title = _browseDialogTitle;
        if (title.isEmpty()) {
            title = tr("Select volume path");
        }

        QString startPath = selectedVolumePath();
        if (startPath.isEmpty()) {
            startPath = QDir::homePath();
        }

        QString pickedPath;
        if (_browseMode == BrowseMode::File) {
            pickedPath = QFileDialog::getOpenFileName(this, title, startPath);
        } else {
            pickedPath = QFileDialog::getExistingDirectory(this, title, startPath);
        }

        if (pickedPath.isEmpty()) {
            return;
        }

        const QString cleanedPath = QDir::cleanPath(pickedPath);
        int existingIndex = -1;
        for (int i = 0; i < _combo->count(); ++i) {
            if (_combo->itemData(i, Qt::UserRole).toString() == cleanedPath) {
                existingIndex = i;
                break;
            }
        }

        if (existingIndex >= 0) {
            _combo->setCurrentIndex(existingIndex);
            return;
        }

        const QFileInfo info(cleanedPath);
        const QString label = info.fileName().isEmpty() ? cleanedPath : info.fileName();
        const int row = _combo->count();
        _combo->addItem(label, cleanedPath);
        _combo->setItemData(row, QString(), Qt::UserRole + 1);
        _combo->setCurrentIndex(row);
    });
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

void VolumeSelector::setAllowNone(bool allow, const QString& label)
{
    _allowNone = allow;
    if (!label.isEmpty()) {
        _noneLabel = label;
    } else if (!_allowNone) {
        _noneLabel.clear();
    }
}

void VolumeSelector::setBrowseEnabled(bool enabled)
{
    _browseEnabled = enabled;
    if (_browseButton) {
        _browseButton->setVisible(enabled);
    }
}

void VolumeSelector::setBrowseDialogTitle(const QString& title)
{
    _browseDialogTitle = title;
}

void VolumeSelector::setBrowseMode(BrowseMode mode)
{
    _browseMode = mode;
}

void VolumeSelector::setVolumes(const QVector<VolumeOption>& volumes, const QString& defaultVolumeId)
{
    if (!_combo) {
        return;
    }

    _combo->clear();

    int defaultIndex = -1;
    if (_allowNone) {
        const QString label = _noneLabel.isEmpty() ? tr("None") : _noneLabel;
        _combo->addItem(label, QVariant());
        _combo->setItemData(0, QString(), Qt::UserRole + 1);
        if (defaultVolumeId.isEmpty()) {
            defaultIndex = 0;
        }
    }

    for (int i = 0; i < volumes.size(); ++i) {
        const auto& opt = volumes[i];
        const QString label = opt.name.isEmpty()
            ? opt.id
            : tr("%1 (%2)").arg(opt.name, opt.id);
        const int row = _combo->count();
        _combo->addItem(label, opt.path);
        _combo->setItemData(row, opt.id, Qt::UserRole + 1);
        if (defaultIndex == -1 && !defaultVolumeId.isEmpty() && opt.id == defaultVolumeId) {
            defaultIndex = row;
        }
    }

    if (_combo->count() > 0) {
        if (defaultIndex < 0 && _allowNone) {
            defaultIndex = 0;
        }
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
