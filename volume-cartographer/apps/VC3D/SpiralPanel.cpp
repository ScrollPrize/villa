#include "SpiralPanel.hpp"

#include "SpiralServiceManager.hpp"
#include "VCSettings.hpp"
#include "elements/CollapsibleSettingsGroup.hpp"
#include "elements/VolumeSelector.hpp"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

SpiralPanel::SpiralPanel(SpiralServiceManager* service, QWidget* parent)
    : QWidget(parent), _service(service)
{
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(4, 4, 4, 4);
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    auto* contents = new QWidget(scroll);
    auto* layout = new QVBoxLayout(contents);

    auto makeSection = [this, contents, layout](const QString& title,
                                                const QString& objectName,
                                                const QString& settingsKey) {
        auto* group = new CollapsibleSettingsGroup(title, contents);
        group->setObjectName(objectName);
        layout->addWidget(group);
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        group->setExpanded(settings.value(settingsKey, true).toBool());
        connect(group, &CollapsibleSettingsGroup::toggled, this,
                [settingsKey](bool expanded) {
                    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                    settings.setValue(settingsKey, expanded);
                });
        return group;
    };

    auto* pathsGroup = makeSection(tr("Dataset and fit geometry"),
                                   QStringLiteral("spiralDatasetGeometryGroup"),
                                   QStringLiteral("spiral/groups/dataset_geometry_expanded"));
    auto* pathsContents = new QWidget(pathsGroup->contentWidget());
    auto* pathsForm = new QFormLayout(pathsContents);
    pathsGroup->contentLayout()->addWidget(pathsContents);
    addPathRow(pathsForm, "dataset_root", tr("Dataset root"), true);
    auto* refill = new QPushButton(tr("Refill from Dataset Root"), pathsContents);
    pathsForm->addRow(refill);
    addPathRow(pathsForm, "umbilicus", tr("Umbilicus"), false);

    auto* pclContainer = new QWidget(pathsContents);
    auto* pclLayout = new QVBoxLayout(pclContainer);
    pclLayout->setContentsMargins(0, 0, 0, 0);
    _pclList = new QListWidget(pclContainer);
    _pclList->setObjectName(QStringLiteral("spiralPclList"));
    _pclList->setSelectionMode(QAbstractItemView::SingleSelection);
    _pclList->setMinimumHeight(90);
    pclLayout->addWidget(_pclList);
    auto* pclInputRow = new QHBoxLayout;
    _pclRole = new QComboBox(pclContainer);
    _pclRole->setObjectName(QStringLiteral("spiralPclRole"));
    _pclRole->addItem(tr("Absolute"), QStringLiteral("absolute"));
    _pclRole->addItem(tr("Patch overlap"), QStringLiteral("patch_overlap"));
    _pclRole->addItem(tr("Relative"), QStringLiteral("relative"));
    _pclRole->addItem(tr("Same winding"), QStringLiteral("same_winding"));
    _pclPath = new QLineEdit(pclContainer);
    _pclPath->setObjectName(QStringLiteral("spiralPclPath"));
    _pclPath->setPlaceholderText(tr("PCL path"));
    auto* browsePcl = new QToolButton(pclContainer);
    browsePcl->setText(QStringLiteral("…"));
    browsePcl->setToolTip(tr("Select PCL file"));
    auto* addPcl = new QPushButton(QStringLiteral("+"), pclContainer);
    addPcl->setObjectName(QStringLiteral("spiralAddPcl"));
    addPcl->setToolTip(tr("Add PCL"));
    _removePcl = new QPushButton(QStringLiteral("-"), pclContainer);
    _removePcl->setObjectName(QStringLiteral("spiralRemovePcl"));
    _removePcl->setToolTip(tr("Remove selected PCL"));
    _removePcl->setEnabled(false);
    pclInputRow->addWidget(_pclRole);
    pclInputRow->addWidget(_pclPath, 1);
    pclInputRow->addWidget(browsePcl);
    pclInputRow->addWidget(addPcl);
    pclInputRow->addWidget(_removePcl);
    pclLayout->addLayout(pclInputRow);
    pathsForm->addRow(tr("PCLs"), pclContainer);

    addPathRow(pathsForm, "fibers", tr("Fibers"), true);
    addPathRow(pathsForm, "tracks_dbm", tr("Tracks DBM"), false);
    addPathRow(pathsForm, "verified_patches", tr("Verified patches"), true);
    addPathRow(pathsForm, "unverified_patches", tr("Unverified patches"), true);
    addPathRow(pathsForm, "outer_shell", tr("Outer shell"), true);

    auto* lasagnaSection = makeSection(tr("Lasagna inputs"),
                                       QStringLiteral("spiralLasagnaInputsGroup"),
                                       QStringLiteral("spiral/groups/lasagna_inputs_expanded"));
    auto* lasagnaContents = new QWidget(lasagnaSection->contentWidget());
    auto* lasagnaForm = new QFormLayout(lasagnaContents);
    lasagnaSection->contentLayout()->addWidget(lasagnaContents);
    addPathRow(lasagnaForm, "normal_x", tr("Normal X"), true);
    addPathRow(lasagnaForm, "normal_y", tr("Normal Y"), true);
    addPathRow(lasagnaForm, "gradient_magnitude", tr("Gradient magnitude"), true);
    _lasagnaGroup = new QLineEdit(QStringLiteral("4"), lasagnaContents);
    _lasagnaScale = new QSpinBox(lasagnaContents);
    _lasagnaScale->setRange(1, 1024);
    _lasagnaScale->setValue(4);
    _storageBackend = new QComboBox(lasagnaContents);
    _storageBackend->addItem(tr("Auto"), QStringLiteral("auto"));
    _storageBackend->addItem(tr("Memory mapped"), QStringLiteral("mmap"));
    _storageBackend->addItem(tr("Dense CUDA (legacy)"), QStringLiteral("dense_cuda"));
    addPathRow(lasagnaForm, "cache_directory", tr("Cache directory"), true);
    lasagnaForm->addRow(tr("Zarr group"), _lasagnaGroup);
    lasagnaForm->addRow(tr("Coordinate scale"), _lasagnaScale);
    lasagnaForm->addRow(tr("Storage backend"), _storageBackend);

    auto* outputGroup = makeSection(tr("Fit and output"),
                                    QStringLiteral("spiralFitOutputGroup"),
                                    QStringLiteral("spiral/groups/fit_output_expanded"));
    auto* outputContents = new QWidget(outputGroup->contentWidget());
    auto* outputForm = new QFormLayout(outputContents);
    outputGroup->contentLayout()->addWidget(outputContents);
    addPathRow(outputForm, "output_directory", tr("Output directory"), true);
    addPathRow(outputForm, "checkpoint", tr("Checkpoint"), false);
    addPathRow(outputForm, "scroll_zarr", tr("Scroll/render Zarr"), true);
    _zBegin = new QSpinBox(outputContents); _zBegin->setRange(0, 1000000); _zBegin->setValue(4000);
    _zEnd = new QSpinBox(outputContents); _zEnd->setRange(1, 1000000); _zEnd->setValue(17000);
    _scrollName = new QLineEdit(QStringLiteral("s1"), outputContents);
    _outwardSense = new QComboBox(outputContents); _outwardSense->addItems({QStringLiteral("CW"), QStringLiteral("ACW")});
    _voxelSize = new QDoubleSpinBox(outputContents); _voxelSize->setRange(0.001, 10000); _voxelSize->setDecimals(4); _voxelSize->setValue(9.6);
    _legacyCheckpointStep = new QSpinBox(outputContents); _legacyCheckpointStep->setRange(0, 1000000000);
    _renderVolumeScale = new QSpinBox(outputContents); _renderVolumeScale->setRange(1, 4096); _renderVolumeScale->setValue(16);
    _savePngVisualizations = new QCheckBox(tr("Save diagnostic PNG visualizations"), outputContents);
    _savePngVisualizations->setChecked(false);
    _runTag = new QLineEdit(outputContents);
    _advanced = new QPlainTextEdit(QStringLiteral("{}"), outputContents); _advanced->setMaximumHeight(90);
    outputForm->addRow(tr("z begin"), _zBegin);
    outputForm->addRow(tr("z end"), _zEnd);
    outputForm->addRow(tr("Scroll name"), _scrollName);
    outputForm->addRow(tr("Outward sense"), _outwardSense);
    outputForm->addRow(tr("Voxel size (µm)"), _voxelSize);
    outputForm->addRow(tr("Legacy checkpoint step"), _legacyCheckpointStep);
    outputForm->addRow(tr("Run tag"), _runTag);
    outputForm->addRow(tr("Render-volume scale"), _renderVolumeScale);
    outputForm->addRow(_savePngVisualizations);
    outputForm->addRow(tr("Advanced config JSON"), _advanced);

    auto* displayGroup = makeSection(tr("Display"),
                                     QStringLiteral("spiralDisplayGroup"),
                                     QStringLiteral("spiral/groups/display_expanded"));
    auto* displayContents = new QWidget(displayGroup->contentWidget());
    auto* displayLayout = new QVBoxLayout(displayContents);
    displayGroup->contentLayout()->addWidget(displayContents);
    _volumeSelector = new VolumeSelector(displayContents);
    displayLayout->addWidget(_volumeSelector);
    for (const auto& item : std::initializer_list<std::pair<const char*, const char*>>{
             {"output", "Output"}, {"fibers", "Fibers"}, {"tracks", "Tracks"},
             {"pcls", "Winding/PCL inputs"}, {"verified", "Verified patches"},
             {"unverified", "Unverified patches"}, {"shell", "Shell"}, {"lasagna", "Lasagna inputs"}}) {
        auto* check = new QCheckBox(tr(item.second), displayContents);
        const QString key = QString::fromLatin1(item.first);
        _visibilityChecks[key] = check;
        check->setChecked(key == QStringLiteral("output"));
        connect(check, &QCheckBox::toggled, this, [this, key = QString::fromLatin1(item.first)](bool shown) {
            emit visibilityChanged(key, shown);
        });
        displayLayout->addWidget(check);
    }

    auto* runGroup = makeSection(tr("Run and status"),
                                 QStringLiteral("spiralRunStatusGroup"),
                                 QStringLiteral("spiral/groups/run_status_expanded"));
    auto* runContents = new QWidget(runGroup->contentWidget());
    auto* runLayout = new QVBoxLayout(runContents);
    runGroup->contentLayout()->addWidget(runContents);
    auto* controls = new QHBoxLayout;
    _load = new QPushButton(tr("Load/Reload Inputs"), runContents);
    _iterations = new QSpinBox(runContents); _iterations->setRange(1, 1000000); _iterations->setValue(100);
    _run = new QPushButton(tr("Run"), runContents);
    _stop = new QPushButton(tr("Stop after iteration"), runContents); _stop->setEnabled(false);
    _save = new QPushButton(tr("Save Checkpoint"), runContents); _save->setEnabled(false);
    controls->addWidget(_load); controls->addWidget(_iterations); controls->addWidget(_run);
    controls->addWidget(_stop); controls->addWidget(_save);
    runLayout->addLayout(controls);
    _state = new QLabel(tr("Service stopped"), runContents);
    _metrics = new QLabel(runContents);
    _warnings = new QLabel(runContents); _warnings->setWordWrap(true);
    runLayout->addWidget(_state); runLayout->addWidget(_metrics); runLayout->addWidget(_warnings);
    layout->addStretch(1);
    scroll->setWidget(contents);
    rootLayout->addWidget(scroll);

    connect(_paths["dataset_root"], &QLineEdit::editingFinished, this, [this]() {
        _pendingDatasetRoot = _paths["dataset_root"]->text();
        _service->ensureStarted();
        if (_service->isReady()) _service->resolveDataset(_pendingDatasetRoot);
    });
    connect(refill, &QPushButton::clicked, this, [this]() {
        _pendingDatasetRoot = _paths["dataset_root"]->text();
        _service->ensureStarted();
        if (_service->isReady()) _service->resolveDataset(_pendingDatasetRoot);
    });
    auto appendPcl = [this]() {
        const QString path = _pclPath->text().trimmed();
        if (path.isEmpty()) return;
        addPclItem(path, _pclRole->currentData().toString());
        _pclPath->clear();
        _hasManualEdits = true;
        markReloadRequired();
    };
    connect(addPcl, &QPushButton::clicked, this, appendPcl);
    connect(_pclPath, &QLineEdit::returnPressed, this, appendPcl);
    connect(browsePcl, &QToolButton::clicked, this, [this]() {
        const QString chosen = QFileDialog::getOpenFileName(
            this, tr("Select PCL file"), _pclPath->text(), tr("JSON files (*.json);;All files (*)"));
        if (!chosen.isEmpty()) _pclPath->setText(chosen);
    });
    connect(_pclList, &QListWidget::itemSelectionChanged, this, [this]() {
        _removePcl->setEnabled(_pclList->currentItem() != nullptr);
    });
    connect(_removePcl, &QPushButton::clicked, this, [this]() {
        if (auto* item = _pclList->takeItem(_pclList->currentRow())) {
            delete item;
            _hasManualEdits = true;
            markReloadRequired();
        }
    });
    connect(_service, &SpiralServiceManager::serviceStateChanged, this, [this](const QString& state) {
        _state->setText(tr("Service: %1").arg(state));
        if (state == tr("Ready") && !_pendingDatasetRoot.isEmpty()) _service->resolveDataset(_pendingDatasetRoot);
    });
    connect(_service, &SpiralServiceManager::datasetResolved, this, [this](const QJsonObject& value) {
        applyResolution(value, !_hasManualEdits);
        _pendingDatasetRoot.clear();
    });
    connect(_service, &SpiralServiceManager::sessionStatusChanged, this, &SpiralPanel::updateStatus);
    connect(_service, &SpiralServiceManager::sessionAccepted, this,
            [this](const QJsonObject&, qint64) {
                _hasSession = true;
                _reloadRequired = false;
                for (auto it = _visibilityChecks.begin(); it != _visibilityChecks.end(); ++it)
                    it.value()->setChecked(it.key() == QStringLiteral("output"));
            });
    connect(_service, &SpiralServiceManager::errorOccurred, this, [this](const QString& error) {
        _warnings->setText(error);
    });
    connect(_load, &QPushButton::clicked, this, [this]() {
        QJsonParseError error;
        const QJsonDocument advanced = QJsonDocument::fromJson(_advanced->toPlainText().toUtf8(), &error);
        if (error.error != QJsonParseError::NoError || !advanced.isObject()) {
            QMessageBox::warning(this, tr("Invalid advanced configuration"),
                                 tr("Advanced config must be a JSON object: %1").arg(error.errorString()));
            return;
        }
        persist();
        emit pythonOutputRequested();
        _service->loadSession(sessionRequest());
    });
    connect(_run, &QPushButton::clicked, this, [this]() {
        emit pythonOutputRequested();
        _service->runIterations(_iterations->value());
    });
    connect(_stop, &QPushButton::clicked, _service, &SpiralServiceManager::stopAfterIteration);
    connect(_save, &QPushButton::clicked, this, [this]() {
        const QString initial = QDir(_paths["output_directory"]->text())
            .filePath(QStringLiteral("checkpoint_manual.ckpt"));
        const QString path = QFileDialog::getSaveFileName(this, tr("Save Spiral checkpoint"),
                                                          initial, tr("Checkpoint (*.ckpt)"));
        if (!path.isEmpty()) _service->saveCheckpoint(path);
    });
    connect(_volumeSelector->comboBox(), qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { emit volumeSelected(_volumeSelector->selectedVolumeId()); });
    for (QSpinBox* spin : {_zBegin, _zEnd, _lasagnaScale, _legacyCheckpointStep, _renderVolumeScale})
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { markReloadRequired(); });
    connect(_voxelSize, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double) { markReloadRequired(); });
    for (QLineEdit* edit : {_lasagnaGroup, _scrollName, _runTag})
        connect(edit, &QLineEdit::textEdited, this, [this](const QString&) { markReloadRequired(); });
    connect(_outwardSense, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { markReloadRequired(); });
    connect(_storageBackend, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { markReloadRequired(); });
    connect(_savePngVisualizations, &QCheckBox::toggled, this,
            [this](bool) { markReloadRequired(); });
    connect(_advanced, &QPlainTextEdit::textChanged, this, &SpiralPanel::markReloadRequired);
    restore();
    _service->ensureStarted();
}

QLineEdit* SpiralPanel::addPathRow(QFormLayout* form, const QString& key, const QString& label, bool directory)
{
    auto* container = new QWidget(form->parentWidget());
    auto* row = new QHBoxLayout(container); row->setContentsMargins(0, 0, 0, 0);
    auto* edit = new QLineEdit(container);
    auto* browse = new QToolButton(container); browse->setText(QStringLiteral("…"));
    row->addWidget(edit, 1); row->addWidget(browse);
    form->addRow(label, container);
    _paths[key] = edit; _pathDirectories[key] = directory;
    connect(edit, &QLineEdit::textEdited, this, [this, key](const QString&) {
        if (!_applyingResolution) {
            if (key != QStringLiteral("dataset_root")) _hasManualEdits = true;
            markReloadRequired();
        }
    });
    connect(browse, &QToolButton::clicked, this, [this, edit, directory, key]() {
        const QString chosen = directory
            ? QFileDialog::getExistingDirectory(this, tr("Select directory"), edit->text())
            : QFileDialog::getOpenFileName(this, tr("Select file"), edit->text());
        if (!chosen.isEmpty()) {
            edit->setText(chosen);
            if (key != QStringLiteral("dataset_root")) _hasManualEdits = true;
            markReloadRequired();
        }
    });
    return edit;
}

void SpiralPanel::addPclItem(const QString& path, const QString& role, bool required)
{
    if (path.trimmed().isEmpty()) return;
    const int roleIndex = _pclRole->findData(role);
    const QString roleLabel = roleIndex >= 0 ? _pclRole->itemText(roleIndex) : role;
    auto* item = new QListWidgetItem(tr("%1 — %2").arg(roleLabel, path), _pclList);
    item->setData(Qt::UserRole, path);
    item->setData(Qt::UserRole + 1, role);
    item->setData(Qt::UserRole + 2, required);
    item->setToolTip(path);
}

void SpiralPanel::setVolumes(const QVector<VolumeSelector::VolumeOption>& volumes, const QString& selectedId)
{
    _volumeSelector->setVolumes(volumes, selectedId);
}

void SpiralPanel::applyResolution(const QJsonObject& resolution, bool force)
{
    if (!force && _hasManualEdits) {
        if (QMessageBox::question(this, tr("Refill Spiral paths"),
                tr("Replace manually edited path rows with the Dataset Root proposals?")) != QMessageBox::Yes) return;
    }
    _applyingResolution = true;
    const QJsonObject resolved = resolution.value("resolved").toObject();
    for (auto it = resolved.begin(); it != resolved.end(); ++it)
        if (_paths.contains(it.key())) _paths[it.key()]->setText(it.value().toString());
    _pclList->clear();
    for (const QJsonValue& value : resolution.value("pcl_inputs").toArray()) {
        const QJsonObject item = value.toObject();
        addPclItem(item.value("path").toString(), item.value("role").toString(),
                   item.value("required").toBool());
    }
    _applyingResolution = false; _hasManualEdits = false;
    markReloadRequired();
    const QStringList missing = resolution.value("missing_required").toVariant().toStringList();
    _warnings->setText(missing.isEmpty() ? QString() : tr("Missing required: %1").arg(missing.join(", ")));
}

QJsonObject SpiralPanel::sessionRequest() const
{
    QJsonObject paths;
    for (auto it = _paths.begin(); it != _paths.end(); ++it)
        paths[it.key()] = it.value()->text();
    QJsonArray pcls;
    for (int row = 0; row < _pclList->count(); ++row) {
        const QListWidgetItem* item = _pclList->item(row);
        pcls.append(QJsonObject{{"path", item->data(Qt::UserRole).toString()},
                                {"role", item->data(Qt::UserRole + 1).toString()},
                                {"required", item->data(Qt::UserRole + 2).toBool()}});
    }
    paths["pcls"] = pcls;
    QJsonParseError parseError;
    const QJsonDocument advanced = QJsonDocument::fromJson(_advanced->toPlainText().toUtf8(), &parseError);
    QJsonObject config = advanced.isObject() ? advanced.object() : QJsonObject{};
    config[QStringLiteral("save_png_visualizations")] = _savePngVisualizations->isChecked();
    QJsonObject run{{"z_begin", _zBegin->value()}, {"z_end", _zEnd->value()},
                    {"scroll_name", _scrollName->text()}, {"outward_sense", _outwardSense->currentText()},
                    {"voxel_size_um", _voxelSize->value()}, {"lasagna_group", _lasagnaGroup->text()},
                    {"lasagna_scale", _lasagnaScale->value()},
                    {"storage_backend", _storageBackend->currentData().toString()},
                    {"legacy_checkpoint_step", _legacyCheckpointStep->value()},
                    {"run_tag", _runTag->text()},
                    {"render_volume_scale", _renderVolumeScale->value()},
                    {"config", config}};
    return {{"paths", paths}, {"run", run}, {"preview", QJsonObject{{"first_winding", 10}}}};
}

void SpiralPanel::updateStatus(const QJsonObject& status)
{
    const QString state = status.value("state").toString();
    _state->setText(tr("Session: %1 — %2 — iteration %3/%4")
        .arg(state, status.value("phase").toString())
        .arg(status.value("current_iteration").toInteger())
        .arg(status.value("target_iteration").toInteger()));
    const QJsonObject metrics = status.value("latest_metrics").toObject();
    _metrics->setText(metrics.contains("total_loss") ? tr("Loss: %1").arg(metrics.value("total_loss").toDouble()) : QString());
    QStringList diagnostics;
    const QString error = status.value(QStringLiteral("error")).toString();
    if (!error.isEmpty()) diagnostics.push_back(error);
    for (const QJsonValue& warning : status.value(QStringLiteral("warnings")).toArray()) {
        const QString text = warning.toString().trimmed();
        if (!text.isEmpty()) diagnostics.push_back(text);
    }
    _warnings->setText(diagnostics.join(QStringLiteral("\n\n")));
    const bool runnable = state == "Ready" || state == "Paused";
    _run->setEnabled(runnable && !_reloadRequired);
    _stop->setEnabled(state == "Running"); _save->setEnabled(runnable);
}

void SpiralPanel::markReloadRequired()
{
    if (!_hasSession || _applyingResolution) return;
    _reloadRequired = true;
    _run->setEnabled(false);
    _state->setText(tr("Reload required — fit inputs or training configuration changed"));
}

void SpiralPanel::persist() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    for (auto it = _paths.begin(); it != _paths.end(); ++it) settings.setValue("spiral/paths/" + it.key(), it.value()->text());
    QJsonArray pcls;
    for (int row = 0; row < _pclList->count(); ++row) {
        const QListWidgetItem* item = _pclList->item(row);
        pcls.append(QJsonObject{{"path", item->data(Qt::UserRole).toString()},
                                {"role", item->data(Qt::UserRole + 1).toString()},
                                {"required", item->data(Qt::UserRole + 2).toBool()}});
    }
    settings.setValue(QStringLiteral("spiral/pcls"),
                      QJsonDocument(pcls).toJson(QJsonDocument::Compact));
    settings.setValue("spiral/z_begin", _zBegin->value()); settings.setValue("spiral/z_end", _zEnd->value());
    settings.setValue("spiral/scroll_name", _scrollName->text());
    settings.setValue("spiral/outward_sense", _outwardSense->currentText());
    settings.setValue("spiral/voxel_size_um", _voxelSize->value());
    settings.setValue("spiral/lasagna_group", _lasagnaGroup->text());
    settings.setValue("spiral/lasagna_scale", _lasagnaScale->value());
    settings.setValue("spiral/storage_backend", _storageBackend->currentData().toString());
    settings.setValue("spiral/legacy_checkpoint_step", _legacyCheckpointStep->value());
    settings.setValue("spiral/run_tag", _runTag->text());
    settings.setValue("spiral/render_volume_scale", _renderVolumeScale->value());
    settings.setValue("spiral/save_png_visualizations", _savePngVisualizations->isChecked());
    settings.setValue("spiral/iterations", _iterations->value());
    settings.setValue("spiral/advanced_config", _advanced->toPlainText());
}

void SpiralPanel::restore()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _applyingResolution = true;
    for (auto it = _paths.begin(); it != _paths.end(); ++it) it.value()->setText(settings.value("spiral/paths/" + it.key()).toString());
    const QByteArray savedPcls = settings.value(QStringLiteral("spiral/pcls")).toByteArray();
    const QJsonDocument pclDocument = QJsonDocument::fromJson(savedPcls);
    if (pclDocument.isArray()) {
        for (const QJsonValue& value : pclDocument.array()) {
            const QJsonObject item = value.toObject();
            addPclItem(item.value(QStringLiteral("path")).toString(),
                       item.value(QStringLiteral("role")).toString(),
                       item.value(QStringLiteral("required")).toBool());
        }
    } else {
        // Import settings written by the original four-row PCL UI once.
        for (const auto& pair : std::initializer_list<std::pair<const char*, const char*>>{
                 {"pcl_absolute", "absolute"}, {"pcl_patch_overlap", "patch_overlap"},
                 {"pcl_relative", "relative"}, {"pcl_same_winding", "same_winding"}}) {
            const QString path = settings.value(
                QStringLiteral("spiral/paths/") + QString::fromLatin1(pair.first)).toString();
            addPclItem(path, QString::fromLatin1(pair.second));
        }
    }
    _zBegin->setValue(settings.value("spiral/z_begin", 4000).toInt());
    _zEnd->setValue(settings.value("spiral/z_end", 17000).toInt());
    _scrollName->setText(settings.value("spiral/scroll_name", "s1").toString());
    _outwardSense->setCurrentText(settings.value("spiral/outward_sense", "CW").toString());
    _voxelSize->setValue(settings.value("spiral/voxel_size_um", 9.6).toDouble());
    _lasagnaGroup->setText(settings.value("spiral/lasagna_group", "4").toString());
    _lasagnaScale->setValue(settings.value("spiral/lasagna_scale", 4).toInt());
    const int backend = _storageBackend->findData(settings.value("spiral/storage_backend", "auto").toString());
    if (backend >= 0) _storageBackend->setCurrentIndex(backend);
    _legacyCheckpointStep->setValue(settings.value("spiral/legacy_checkpoint_step", 0).toInt());
    _runTag->setText(settings.value("spiral/run_tag").toString());
    _renderVolumeScale->setValue(settings.value("spiral/render_volume_scale", 16).toInt());
    _savePngVisualizations->setChecked(
        settings.value("spiral/save_png_visualizations", false).toBool());
    _iterations->setValue(settings.value("spiral/iterations", 100).toInt());
    _advanced->setPlainText(settings.value("spiral/advanced_config", "{}").toString());
    _applyingResolution = false;
}
