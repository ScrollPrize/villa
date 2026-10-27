#include "viewer_controls/panels/VolumeRegistrationsPanel.hpp"

#include "CState.hpp"
#include "ObjectStoreEvents.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <set>

namespace {
using vcr::catalog::TransformStatus;

QString statusText(const vc::objects::VolumeTransforms::Related& related)
{
    if (related.status == TransformStatus::Ambiguous) return QObject::tr("Ambiguous: choose a route");
    if (related.chosen) return QObject::tr("Chosen route");
    if (related.routes.size() > 1) return QObject::tr("Resolved");
    return QObject::tr("One route");
}
}

VolumeRegistrationsPanel::VolumeRegistrationsPanel(CState* state, QWidget* parent)
    : QWidget(parent), _state(state)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    _heading = new QLabel(this);
    _heading->setWordWrap(true);
    layout->addWidget(_heading);

    _list = new QTreeWidget(this);
    _list->setColumnCount(3);
    _list->setHeaderLabels({tr("Volume"), tr("Status"), tr("Route")});
    _list->setRootIsDecorated(false);
    _list->header()->setStretchLastSection(true);
    _list->setToolTip(tr("Volumes the object store maps into the open volume. Their surfaces, fibers and "
                         "other objects are shown here through the store's mapping."));
    layout->addWidget(_list);

    auto* buttons = new QHBoxLayout();
    _choose = new QPushButton(tr("Choose Route…"), this);
    _choose->setToolTip(tr("Pick which of the store's routes maps the selected volume into the open one."));
    _register = new QPushButton(tr("Register Transform…"), this);
    _register->setToolTip(tr("Add an affine from another volume to the open volume to the object store."));
    buttons->addWidget(_choose);
    buttons->addWidget(_register);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    _status = new QLabel(this);
    _status->setWordWrap(true);
    layout->addWidget(_status);

    connect(_list, &QTreeWidget::itemSelectionChanged, this, &VolumeRegistrationsPanel::updateButtons);
    connect(_choose, &QPushButton::clicked, this, &VolumeRegistrationsPanel::chooseRoute);
    connect(_register, &QPushButton::clicked, this, &VolumeRegistrationsPanel::registerTransform);
    if (_state) {
        connect(_state, &CState::volumeChanged, this, [this]() { refresh(); });
        connect(_state, &CState::objectStoreChanged, this, [this](const vc::objects::StoreSessionEvent& event) {
            if (event.kind == vc::objects::StoreSessionEvent::Kind::Hub) return;
            if (vc3d::store::touches(event, {"volume_transform", "volume_transform_choice"})) refresh();
        });
    }
    refresh();
}

void VolumeRegistrationsPanel::refresh()
{
    _list->clear();
    _related.clear();
    _graph = {};
    const auto volume = _state ? _state->currentVolume() : nullptr;
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    if (!volume || !attachment) {
        _heading->setText(tr("Open a volume and attach an object store to see its registrations."));
        _status->clear();
        updateButtons();
        return;
    }
    _heading->setText(tr("Volumes mapped into %1").arg(QString::fromStdString(volume->id())));
    try {
        _state->objectStoreSession()->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto transforms = db.volumeTransforms(txn);
            _graph = transforms.graph;
            _related = transforms.related(volume->id());
        });
        _status->setText(_related.empty() ? tr("No other volume maps into this one.") : QString());
    } catch (const std::exception& error) {
        _status->setText(tr("Cannot read registrations: %1").arg(QString::fromUtf8(error.what())));
    }
    for (const auto& related : _related) {
        auto* item = new QTreeWidgetItem(_list);
        item->setText(0, QString::fromStdString(related.volume));
        item->setText(1, statusText(related));
        if (related.status == TransformStatus::Resolved) {
            const auto& path = related.chosen ? *related.chosen : related.routes.front().path;
            item->setText(2, QString::fromStdString(_graph.describe(path)));
        } else {
            item->setText(2, tr("%n route(s)", nullptr, int(related.routes.size())));
        }
        item->setToolTip(2, item->text(2));
    }
    _list->resizeColumnToContents(0);
    updateButtons();
}

void VolumeRegistrationsPanel::updateButtons()
{
    const auto attachment = _state ? _state->objectStoreAttachment() : std::nullopt;
    const bool writable = attachment && _state->currentVolume() && _state->objectStoreSession()->writable(*attachment);
    const int row = _list->currentIndex().isValid() ? _list->currentIndex().row() : -1;
    _choose->setEnabled(writable && row >= 0 && row < int(_related.size()) && _related[size_t(row)].routes.size() > 1);
    _register->setEnabled(writable);
}

void VolumeRegistrationsPanel::registerTransform()
{
    const auto volume = _state->currentVolume();
    const auto attachment = _state->objectStoreAttachment();
    if (!volume || !attachment) return;

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Register Transform"));
    auto* form = new QFormLayout(&dialog);
    auto* source = new QComboBox(&dialog);
    source->setEditable(true);
    std::set<std::string> volumes;
    if (const auto pkg = _state->vpkg())
        for (const auto& id : pkg->volumeIDs()) volumes.insert(id);
    for (const auto& registration : _graph.registrations()) {
        volumes.insert(registration.source);
        volumes.insert(registration.target);
    }
    volumes.erase(volume->id());
    for (const auto& id : volumes) source->addItem(QString::fromStdString(id));
    source->setCurrentText({});
    form->addRow(tr("Source volume"), source);
    auto* name = new QLineEdit(&dialog);
    name->setPlaceholderText(tr("e.g. the registration's provenance"));
    form->addRow(tr("Name"), name);
    auto* matrix = new QPlainTextEdit(&dialog);
    matrix->setPlaceholderText(QStringLiteral("1 0 0 0\n0 1 0 0\n0 0 1 0"));
    form->addRow(tr("Matrix"), matrix);
    form->addRow(new QLabel(tr("Three or four rows of four numbers mapping source-volume voxels (x, y, z) to "
                               "voxels of %1.").arg(QString::fromStdString(volume->id())), &dialog));
    auto* box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(box);
    connect(box, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(box, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;
    try {
        const auto affine = vc3d::registrations::parseAffine(matrix->toPlainText().toStdString());
        const auto from = source->currentText().trimmed().toStdString();
        const auto label = name->text().trimmed().toStdString();
        vc::objects::registerVolumeTransform(*_state->objectStoreSession(), *attachment, from, volume->id(), affine,
                                             label.empty() ? from + " → " + volume->id() : label);
        _state->pollObjectStoreChanges();
    } catch (const std::exception& error) {
        QMessageBox::warning(this, tr("Register Transform"), QString::fromUtf8(error.what()));
    }
}

void VolumeRegistrationsPanel::chooseRoute()
{
    const auto volume = _state->currentVolume();
    const auto attachment = _state->objectStoreAttachment();
    const int row = _list->currentIndex().row();
    if (!volume || !attachment || row < 0 || row >= int(_related.size())) return;
    const auto related = _related[size_t(row)];
    QStringList routes;
    int current = 0;
    for (size_t i = 0; i < related.routes.size(); ++i) {
        routes << QStringLiteral("%1. %2").arg(i + 1).arg(
            QString::fromStdString(_graph.describe(related.routes[i].path)));
        if (related.chosen && *related.chosen == related.routes[i].path) current = int(i);
    }
    bool accepted = false;
    const auto picked = QInputDialog::getItem(this, tr("Choose Route"),
        tr("Route from %1 to %2:").arg(QString::fromStdString(related.volume), QString::fromStdString(volume->id())),
        routes, current, false, &accepted);
    if (!accepted) return;
    try {
        vc::objects::chooseVolumeTransform(*_state->objectStoreSession(), *attachment, related.volume, volume->id(),
                                         related.routes[size_t(routes.indexOf(picked))].path);
        _state->pollObjectStoreChanges();
    } catch (const std::exception& error) {
        QMessageBox::warning(this, tr("Choose Route"), QString::fromUtf8(error.what()));
    }
}
