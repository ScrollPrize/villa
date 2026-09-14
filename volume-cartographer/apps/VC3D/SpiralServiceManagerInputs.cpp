#include "SpiralServiceManager.hpp"
#include "SpiralArtifactCache.hpp"
#include "SpiralInputCopy.hpp"
#include <QFutureWatcher>
#include <QPromise>
#include <limits>
#include <QtConcurrent/QtConcurrent>

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>
#include <QSet>
#include <QUuid>

namespace {
QString uuid() { return QUuid::createUuid().toString(QUuid::WithoutBraces); }

using vc3d::spiral::copyInput;

QJsonObject readDocument(const QString& path, QString& error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        error = QObject::tr("Cannot read %1").arg(path);
        return {};
    }
    QJsonParseError parse;
    const auto document = QJsonDocument::fromJson(file.readAll(), &parse);
    if (parse.error != QJsonParseError::NoError || !document.isObject())
        error = QObject::tr("Invalid JSON in %1: %2").arg(path, parse.errorString());
    return document.object();
}

bool writeDocument(const QString& path, const QJsonObject& document)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QSaveFile file(path);
    const auto bytes = QJsonDocument(document).toJson();
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
}
}

void SpiralServiceManager::reportInputPreparation(const QString& message)
{
    emit inputPreparationProgress(message);
    if (!message.isEmpty()) emit logMessage(message);
}

void SpiralServiceManager::claimInputWorkspace()
{
    reportInputPreparation(tr("Reading the service's input catalog…"));
    const auto failed = [this](const QString& error) {
        reportInputPreparation({});
        emit errorOccurred(error);
    };
    get(QStringLiteral("/session/input-catalog"), Timeout::Load,
        [this, failed](const QJsonObject& catalog) {
            const auto workspace = catalog.value(QStringLiteral("workspace_id")).toString();
            if (!_inputWorkspaceId.isEmpty() && workspace != _inputWorkspaceId && (hasInputDrafts() || _inputCommand)) {
                _inputOwner = false;
                failed(tr("The service editing workspace changed. Local drafts are preserved; reconnect to their original service before applying them."));
                return;
            }
    reportInputPreparation(catalog.value(QStringLiteral("ready")).toBool()
        ? tr("Requesting editing access. The service is checking dataset inputs for changes…")
        : tr("Requesting editing access. The service is preparing snapshots of dataset fibers, patches and point collections…"));
    postWithRetry(QStringLiteral("/session/editing/claim"),
        {{QStringLiteral("command_id"), commandId()}}, Timeout::Load, 2,
        [this, failed](const QJsonObject& response) {
            const auto workspace = response.value(QStringLiteral("workspace_id")).toString();
            if (!_inputWorkspaceId.isEmpty() && workspace != _inputWorkspaceId && (hasInputDrafts() || _inputCommand)) {
                failed(tr("The service editing workspace changed. Local drafts are preserved; reconnect to their original service before applying them."));
                return;
            }
            if (!_inputWorkspaceId.isEmpty() && workspace != _inputWorkspaceId)
                clearInputWorkspace();
            _inputWorkspaceId = workspace;
            reportInputPreparation(tr("Editing access acquired. Loading the input list…"));
            get(QStringLiteral("/session/input-catalog"), Timeout::Command,
                [this](const QJsonObject& catalog) {
                    installInputCatalog(catalog.value(QStringLiteral("inputs")).toArray());
                    _inputOwner = true;
                    reportInputPreparation({});
                    emit logMessage(tr("Editing workspace ready"));
                    emit inputDraftsChanged();
                    if (_inputCommand) resumeInputCommand();
                }, failed);
        }, [this, failed](const QString& error) {
            _inputOwner = false;
            failed(error);
            refreshInputCatalog(); // Other clients can observe the catalog.
        });
        }, failed);
}

void SpiralServiceManager::refreshInputCatalog()
{
    get(QStringLiteral("/session/input-catalog"), Timeout::Command,
        [this](const QJsonObject& response) {
            const auto workspace = response.value(QStringLiteral("workspace_id")).toString();
            if (workspace != _inputWorkspaceId) {
                if (hasInputDrafts() || _inputCommand) return;
                clearInputWorkspace();
                _inputWorkspaceId = workspace;
            }
            // Reconcile a retained command before interpreting newer cursors.
            if (_inputCommand && _inputOwner) {
                resumeInputCommand();
                return;
            }
            installInputCatalog(response.value(QStringLiteral("inputs")).toArray());
        });
}

void SpiralServiceManager::installInputCatalog(const QJsonArray& inputs)
{
    for (const auto& value : inputs) {
        const auto input = value.toObject();
        const auto id = input.value(QStringLiteral("id")).toString();
        _inputCatalog[id] = input;
        const auto accepted = input.value(QStringLiteral("accepted_revision")).toInteger();
        auto draft = _inputDrafts.value(id);
        if (!draft) _inputOrder.push_back(id);
        if (!draft || (!draft->dirty() && draft->accepted() != quint64(accepted))) {
            QJsonObject manifest = input;
            manifest[QStringLiteral("path")] = input.value(QStringLiteral("content")).toObject().value(QStringLiteral("path"));
            // This content belongs to the service; restores reference it without uploading a host path.
            manifest[QStringLiteral("restore_revision")] = input.value(QStringLiteral("accepted_revision"));
            draft = std::make_shared<vc3d::spiral::InputDraft>(id,
                vc3d::spiral::InputDraftContent{manifest, input.value(QStringLiteral("deleted")).toBool()}, accepted);
            _inputDrafts[id] = draft;
        }
        // Local content may have advanced while this catalog request ran.
        // Only acknowledgements tied to a captured submission clear it.
        if (draft->accepted() == quint64(accepted)) {
            draft->reconcileServiceCursors(input.value(QStringLiteral("applied_revision")).toInteger(),
                                           input.value(QStringLiteral("persisted_revision")).toInteger());
        }
    }
    emit inputDraftsChanged();
}

QString SpiralServiceManager::logicalInputId(const QString& kind, const QString& alias,
                                             const QString& role, const QString& targetCollection)
{
    if (_inputDrafts.contains(alias)) return alias;
    const QString key = kind + QLatin1Char(':') + role + QLatin1Char(':') + alias;
    if (_inputAliases.contains(key)) return _inputAliases[key];
    for (auto it = _inputCatalog.cbegin(); it != _inputCatalog.cend(); ++it) {
        const auto& input = it.value();
        if (input.value(QStringLiteral("kind")).toString() != kind) continue;
        const QFileInfo source(input.value(QStringLiteral("source")).toString());
        const bool match = kind == QStringLiteral("pcl")
            ? !targetCollection.isEmpty() && input.value(QStringLiteral("role")).toString() == role
                && QString::number(input.value(QStringLiteral("collection_id")).toInteger()) == targetCollection
            : (kind == QStringLiteral("fiber") ? source.completeBaseName() : source.fileName()) == alias;
        if (match) return _inputAliases[key] = it.key();
    }
    return _inputAliases[key] = uuid();
}

void SpiralServiceManager::stageInput(const QString& kind, const QString& path, const QString& alias,
                                      const QString& role, const QString& targetCollection, bool deleted)
{
    QString error;
    QJsonObject document;
    if (kind == QStringLiteral("pcl") && !deleted) document = readDocument(path, error);
    auto collections = document.value(QStringLiteral("collections")).toObject();
    QStringList keys = collections.keys();
    if (kind != QStringLiteral("pcl") || deleted || keys.isEmpty()) keys = {QString()};
    for (const auto& key : keys) {
        const QString logicalAlias = keys.size() == 1 ? alias : alias + QLatin1Char('_') + key;
        const QString id = logicalInputId(kind, logicalAlias, role, targetCollection);
        QString localPath = path;
        if (kind == QStringLiteral("pcl") && !deleted && error.isEmpty()) {
            auto single = document;
            single[QStringLiteral("collections")] = QJsonObject{{key, collections.value(key)}};
            localPath = QDir(_inputCopies.path()).filePath(QStringLiteral("drafts/%1/%2.json").arg(id, uuid()));
            if (!writeDocument(localPath, single)) error = tr("Cannot capture PCL draft %1").arg(alias);
        }
        if (kind != QStringLiteral("pcl") && !deleted && error.isEmpty()) {
            const auto captured = QDir(_inputCopies.path()).filePath(
                QStringLiteral("drafts/%1/%2/%3").arg(id, uuid(), QFileInfo(path).fileName()));
            if (copyInput(path, captured, error)) localPath = captured;
        }
        if (!deleted && !QFileInfo::exists(localPath) && error.isEmpty()) error = tr("Input file is missing: %1").arg(localPath);
        QJsonObject manifest{{QStringLiteral("kind"), kind}, {QStringLiteral("path"), localPath},
            {QStringLiteral("role"), role}, {QStringLiteral("name"), alias}, {QStringLiteral("alias"), alias}};
        auto draft = _inputDrafts.value(id);
        if (!draft) {
            _inputOrder.push_back(id);
            draft = std::make_shared<vc3d::spiral::InputDraft>(id, vc3d::spiral::InputDraftContent{manifest, false});
            _inputDrafts[id] = draft;
        }
        if (deleted) draft->remove();
        else draft->edit({manifest, false}, error);
        if (!error.isEmpty()) _inputErrors[id] = error;
        else _inputErrors.remove(id);
    }
    emit inputDraftsChanged();
    emit inputDraftStaged(alias);
}

void SpiralServiceManager::stagePatch(const QString& directory, const QString& inputId)
{
    stageInput(QStringLiteral("patch"), directory, inputId, QStringLiteral("verified"));
}

void SpiralServiceManager::stageJsonInput(const QString& kind, const QString& path,
                                          const QString& inputId, const QString& role)
{
    stageInput(kind, path, inputId, role);
}

void SpiralServiceManager::stagePclReplacement(vc3d::spiral::PclRole role, const QString& path,
    const QString& inputId, const QString& operation, const QString& target)
{
    stageInput(QStringLiteral("pcl"), path, inputId, vc3d::spiral::pclRoleName(role), target,
               operation == QStringLiteral("delete_collection"));
}

void SpiralServiceManager::removeInputDraft(const QString& id)
{
    if (auto draft = _inputDrafts.value(id)) draft->remove();
    emit inputDraftsChanged();
}

void SpiralServiceManager::restoreInputDraft(const QString& id)
{
    if (auto draft = _inputDrafts.value(id)) draft->restore();
    emit inputDraftsChanged();
}

void SpiralServiceManager::discardInputDraft(const QString& id)
{
    if (auto draft = _inputDrafts.value(id)) {
        const auto alias = draft->snapshot().content.manifest.value(QStringLiteral("alias")).toString(id);
        if (!draft->accepted()) { _inputDrafts.remove(id); _inputOrder.removeAll(id); }
        else draft->discardLocalChanges();
        emit inputDraftDiscarded(alias);
    }
    _inputErrors.remove(id);
    emit inputDraftsChanged();
}

bool SpiralServiceManager::hasInputDrafts() const
{
    for (const auto& draft : _inputDrafts) if (draft->needsCommit()) return true;
    return false;
}

QJsonArray SpiralServiceManager::inputDraftStatus() const
{
    if (!_inputRowsDirty) return _inputRowsCache;
    QJsonArray rows;
    for (const auto& orderedId : _inputOrder) {
        auto it = _inputDrafts.constFind(orderedId);
        if (it == _inputDrafts.cend()) continue;
        const auto& draft = *it.value();
        const auto snapshot = draft.snapshot();
        QJsonObject row = _inputCatalog.value(it.key());
        for (auto field = snapshot.content.manifest.begin(); field != snapshot.content.manifest.end(); ++field)
            row[field.key()] = field.value();
        row[QStringLiteral("id")] = it.key();
        row[QStringLiteral("deleted")] = draft.deleted();
        row[QStringLiteral("can_restore")] = draft.canRestore();
        row[QStringLiteral("dirty")] = draft.dirty();
        row[QStringLiteral("committed")] = !draft.needsCommit();
        row[QStringLiteral("accepted_revision")] = qint64(draft.accepted());
        row[QStringLiteral("applied_revision")] = qint64(draft.applied());
        row[QStringLiteral("persisted_revision")] = qint64(draft.persisted());
        row[QStringLiteral("session_changed")] =
            row.value(QStringLiteral("session_changed")).toBool()
            || draft.accepted() != 1 || snapshot.localRevision > 1;
        QString state = draft.dirty() ? tr("local") : draft.applied() < draft.accepted() ? tr("accepted") : tr("applied");
        if (_inputCommand) for (const auto& selected : _inputCommand->batch.entries)
            if (selected.id == it.key()) state = _inputCommand->applied ? tr("committing") : tr("applying");
        QString error = !draft.valid() ? draft.validationError() : _inputErrors.value(it.key());
        if (error.isEmpty() && !draft.dirty()) {
            for (const auto& value : row.value(QStringLiteral("errors")).toArray()) {
                const auto stageError = value.toObject();
                if (stageError.value(QStringLiteral("revision")).toInteger() == qint64(draft.accepted()))
                    error = stageError.value(QStringLiteral("message")).toString();
            }
        }
        if (!error.isEmpty()) state = (_inputCommand && _inputCommand->batch.outcomeUnknown)
            ? tr("awaiting outcome") : (error.contains(QStringLiteral("changed"), Qt::CaseInsensitive)
                || error.contains(QStringLiteral("conflict"), Qt::CaseInsensitive)) ? tr("conflicted") : tr("failed");
        row[QStringLiteral("state")] = state;
        row[QStringLiteral("error")] = error;
        rows.append(row);
    }
    _inputRowsCache = rows;
    _inputRowsDirty = false;
    return rows;
}

QString SpiralServiceManager::workingCopy(const QString& source, QString* error)
{
    const auto canonical = QFileInfo(source).absoluteFilePath();
    if (_workingCopies.contains(canonical)) return _workingCopies[canonical];
    const auto destination = QDir(_inputCopies.path()).filePath(
        QStringLiteral("working/%1/%2").arg(uuid(), QFileInfo(source).fileName()));
    QString message;
    if (!copyInput(source, destination, message)) {
        if (error) *error = message;
        return {};
    }
    _workingCopies[canonical] = destination;
    return destination;
}

void SpiralServiceManager::invalidateWorkingCopy(const QString& source)
{
    const auto key = QFileInfo(source).absoluteFilePath();
    _workingCopies.remove(key);
    _workingCopyDirectories.remove(key);
}

void SpiralServiceManager::workingCopyAsync(const QString& source, FetchPreviewFileCallback done)
{
    // Path construction is cheap; all source filesystem traversal stays in the worker.
    const auto key = QFileInfo(source).absoluteFilePath();
    if (_workingCopies.contains(key)) {
        done(_workingCopies.value(key), {});
        return;
    }
    auto* watcher = _workingCopyJobs.value(key);
    const auto generation = _workingCopyGeneration;
    const bool startCopy = !watcher;
    if (!watcher) {
        watcher = new QFutureWatcher<vc3d::spiral::InputCopyResult>(this);
        _workingCopyJobs[key] = watcher;
        connect(watcher, &QFutureWatcherBase::progressTextChanged, this,
                [this, generation](const QString& text) {
                    if (generation == _workingCopyGeneration)
                        emit inputCopyProgress(_workingCopyJobs.size(), text);
                });
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher, key, generation]() {
            watcher->deleteLater();
            if (generation != _workingCopyGeneration) return;
            _workingCopyJobs.remove(key);
            if (!watcher->isCanceled()) {
                const auto result = watcher->result();
                if (result.error.isEmpty()) {
                    _workingCopies[key] = result.path;
                    _workingCopyDirectories[key] = result.directory;
                }
            }
            emit inputCopyProgress(_workingCopyJobs.size(), tr("Preparing input working copies…"));
        });
    }
    connect(watcher, &QFutureWatcherBase::finished, this,
            [this, watcher, generation, done = std::move(done)]() {
                if (generation != _workingCopyGeneration || watcher->isCanceled()) return;
                const auto result = watcher->result();
                done(result.error.isEmpty() ? result.path : QString(), result.error);
            });
    if (!startCopy) return;
    emit inputCopyProgress(_workingCopyJobs.size(), tr("Copying input files…"));
    watcher->setFuture(QtConcurrent::run([key](QPromise<vc3d::spiral::InputCopyResult>& promise) {
        vc3d::spiral::InputCopyResult result;
        // The worker owns its temporary directory even if the UI is destroyed.
        result.directory = std::shared_ptr<QTemporaryDir>(new QTemporaryDir,
            [](QTemporaryDir* directory) {
                // Removing a large working tree must not block the UI either.
                (void)QtConcurrent::run([directory]() { delete directory; });
            });
        if (!result.directory->isValid()) {
            result.error = QObject::tr("Cannot create an input working directory");
        } else {
            result.path = QDir(result.directory->path()).filePath(QFileInfo(key).fileName());
            int files = 0;
            qint64 bytes = 0;
            QElapsedTimer elapsed;
            elapsed.start();
            promise.setProgressRange(0, std::numeric_limits<int>::max());
            copyInput(key, result.path, result.error, [&](qint64 size) {
                if (promise.isCanceled()) return false;
                if (size >= 0) { ++files; bytes += size; }
                if (elapsed.elapsed() >= 100) {
                    promise.setProgressValueAndText(files,
                        QObject::tr("Copying inputs: %1 files, %2 MiB")
                            .arg(files).arg(bytes / (1024 * 1024)));
                    elapsed.restart();
                }
                return true;
            });
        }
        promise.addResult(result);
    }));
}

void SpiralServiceManager::commitInputs() { applyInputDrafts(true); }

void SpiralServiceManager::applyInputDrafts(bool commit, const QStringList& selection)
{
    if (!_inputOwner) {
        emit inputBatchFinished(tr("This connection does not own dataset editing"));
        return;
    }
    if (_inputCommand) {
        _inputCommand->commit |= commit;
        resumeInputCommand();
        return;
    }
    const auto selected = selection.isEmpty() ? _inputSelection : selection;
    const QSet<QString> selectedSet(selected.begin(), selected.end());
    QVector<vc3d::spiral::InputDraft*> drafts;
    QStringList localDeletions;
    QJsonArray revisions;
    for (const auto& orderedId : _inputOrder) {
        auto it = _inputDrafts.find(orderedId);
        if (it == _inputDrafts.end()) continue;
        if ((!selected.isEmpty() || _inputSelectionExplicit) && !selectedSet.contains(it.key())) continue;
        if (!it.value()->accepted() && it.value()->deleted()) {
            if (commit) localDeletions.push_back(it.key());
            continue;
        }
        drafts.push_back(it.value().get());
        if (it.value()->accepted() > it.value()->persisted())
            revisions.append(QJsonObject{{QStringLiteral("id"), it.key()},
                {QStringLiteral("revision"), qint64(it.value()->accepted())}});
    }
    const auto captured = _inputSubmission.begin(commandId(), drafts);
    if (!captured.errors.isEmpty()) {
        for (auto it = captured.errors.cbegin(); it != captured.errors.cend(); ++it) _inputErrors[it.key()] = it.value();
        emit inputDraftsChanged();
        emit inputBatchFinished(tr("Repair or exclude the invalid selected drafts"));
        return;
    }
    auto command = std::make_shared<DraftCommand>();
    command->batch = captured.batch.value_or(vc3d::spiral::InputDraftBatch{commandId(), {}, false});
    command->commit = commit;
    command->commitId = commandId();
    command->revisions = revisions;
    command->localDeletions = localDeletions;
    _inputCommand = command;
    QJsonArray changes;
    for (const auto& snapshot : command->batch.entries) {
        const auto& manifest = snapshot.content.manifest;
        const auto kind = manifest.value(QStringLiteral("kind")).toString();
        QJsonObject change{{QStringLiteral("id"), snapshot.id},
            {QStringLiteral("kind"), kind}, {QStringLiteral("role"), manifest.value(QStringLiteral("role"))},
            {QStringLiteral("name"), manifest.value(QStringLiteral("name"))},
            {QStringLiteral("expected_revision"), qint64(snapshot.expectedAccepted)}};
        if (snapshot.content.deleted) {
            change[QStringLiteral("deleted")] = true;
        } else if (manifest.contains(QStringLiteral("restore_revision"))) {
            change[QStringLiteral("restore_revision")] = manifest.value(QStringLiteral("restore_revision"));
        } else {
            DraftTransfer transfer;
            transfer.id = snapshot.id;
            transfer.uploadId = QUuid::createUuid().toString(QUuid::Id128);
            transfer.directory = QDir(_inputCopies.path()).filePath(QStringLiteral("submissions/%1").arg(transfer.uploadId));
            const auto path = manifest.value(QStringLiteral("path")).toString();
            QString error;
            const bool directory = QFileInfo(path).isDir();
            const auto target = directory ? transfer.directory : QDir(transfer.directory).filePath(QFileInfo(path).fileName());
            if (!copyInput(path, target, error)) { failInputCommand(error, {{QStringLiteral("error"), error}}); return; }
            QJsonArray files;
            QDirIterator iterator(transfer.directory, QDir::Files | QDir::Hidden, QDirIterator::Subdirectories);
            while (iterator.hasNext()) {
                QFile file(iterator.next());
                if (!file.open(QIODevice::ReadOnly)) { failInputCommand(tr("Cannot read captured input"), {{QStringLiteral("error"), "read"}}); return; }
                QCryptographicHash hash(QCryptographicHash::Sha256);
                hash.addData(&file);
                files.append(QJsonObject{{QStringLiteral("name"), QDir(transfer.directory).relativeFilePath(file.fileName())},
                    {QStringLiteral("size"), file.size()}, {QStringLiteral("sha256"), QString::fromLatin1(hash.result().toHex())}});
            }
            transfer.manifest = {{QStringLiteral("upload_id"), transfer.uploadId}, {QStringLiteral("id"), snapshot.id},
                {QStringLiteral("kind"), kind}, {QStringLiteral("files"), files}};
            if (kind == QStringLiteral("pcl")) transfer.manifest[QStringLiteral("role")] = manifest.value(QStringLiteral("role"));
            change[QStringLiteral("upload_id")] = transfer.uploadId;
            command->transfers.push_back(transfer);
        }
        changes.append(change);
    }
    QJsonArray unchanged;
    for (const auto& value : command->revisions) {
        const auto id = value.toObject().value(QStringLiteral("id")).toString();
        bool changed = false;
        for (const auto& snapshot : command->batch.entries) changed |= snapshot.id == id;
        if (!changed) unchanged.append(value);
    }
    command->request = {{QStringLiteral("command_id"), command->batch.commandId},
        {QStringLiteral("changes"), changes}, {QStringLiteral("revisions"), unchanged}};
    emit inputDraftsChanged();
    resumeInputCommand();
}

void SpiralServiceManager::resumeInputCommand()
{
    if (!_inputCommand || _inputCommandBusy || !_inputOwner || !isReady()) return;
    _inputCommandBusy = true;
    if (_inputCommand->applied) { persistInputCommand(); return; }
    transferInput(0);
}

void SpiralServiceManager::transferInput(int index)
{
    const auto command = _inputCommand;
    if (!command) return;
    if (index == command->transfers.size()) { sendInputChanges(); return; }
    const auto transfer = command->transfers[index];
    postWithRetry(QStringLiteral("/session/inputs"), transfer.manifest, Timeout::Command, 2,
        [this, command, index, transfer](const QJsonObject&) {
            if (_inputCommand != command) return;
            get(QStringLiteral("/session/inputs/%1").arg(transfer.uploadId), Timeout::Command,
                [this, command, index, transfer](const QJsonObject& status) {
                    if (_inputCommand != command) return;
                    if (status.value(QStringLiteral("state")).toString() == QStringLiteral("finalized")) {
                        transferInput(index + 1); return;
                    }
                    for (const auto& value : status.value(QStringLiteral("files")).toArray()) {
                        const auto fileStatus = value.toObject();
                        if (fileStatus.value(QStringLiteral("received")).toBool()) continue;
                        const auto name = fileStatus.value(QStringLiteral("name")).toString();
                        const auto offset = fileStatus.value(QStringLiteral("offset")).toInteger();
                        auto* file = new QFile(QDir(transfer.directory).filePath(name));
                        if (!file->open(QIODevice::ReadOnly) || !file->seek(offset)) {
                            delete file; failInputCommand(tr("Cannot read captured input bytes"), {{QStringLiteral("error"), "read"}}); return;
                        }
                        auto request = makeRequest(QStringLiteral("/session/inputs/%1/files/%2?offset=%3")
                            .arg(transfer.uploadId, name).arg(offset), int(Timeout::LongCommand));
                        request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/octet-stream"));
                        request.setHeader(QNetworkRequest::ContentLengthHeader, file->size() - offset);
                        auto* reply = _network->put(request, file);
                        file->setParent(reply);
                        const auto generation = _connectionGeneration;
                        connect(reply, &QNetworkReply::finished, this, [this, reply, generation, command, index]() {
                            handleReply(reply, generation,
                                [this, command, index](const QJsonObject&) { if (_inputCommand == command) transferInput(index); },
                                {}, [this, command, index](const QString& error, const QJsonObject& body) {
                                    // QNetworkAccessManager may replay a PUT after a lost
                                    // response. Reconcile the server offset before resending.
                                    if (_inputCommand == command && body.value(QStringLiteral("http_status")).toInt() == 409
                                        && error.contains(QStringLiteral("offset"), Qt::CaseInsensitive)) {
                                        transferInput(index);
                                        return;
                                    }
                                    failInputCommand(error, body);
                                });
                        });
                        return;
                    }
                    postWithRetry(QStringLiteral("/session/inputs/%1/finalize").arg(transfer.uploadId), {}, Timeout::Command, 2,
                        [this, command, index](const QJsonObject&) { if (_inputCommand == command) transferInput(index + 1); }, {},
                        [this](const QString& error, const QJsonObject& body) { failInputCommand(error, body); });
                }, [this](const QString& error) { failInputCommand(error); });
        }, {}, [this](const QString& error, const QJsonObject& body) { failInputCommand(error, body); });
}

void SpiralServiceManager::sendInputChanges()
{
    if (!_inputCommand) return;
    if (_inputCommand->batch.entries.isEmpty()) {
        if (_inputCommand->revisions.isEmpty()) { finishInputCommand(); return; }
        postWithRetry(QStringLiteral("/session/apply-inputs"),
            {{QStringLiteral("command_id"), _inputCommand->batch.commandId},
             {QStringLiteral("revisions"), _inputCommand->revisions}}, Timeout::LongCommand, 2,
            [this](const QJsonObject& response) { finishInputChanges(response); }, {},
            [this](const QString& error, const QJsonObject& body) { failInputCommand(error, body); });
        return;
    }
    postWithRetry(QStringLiteral("/session/input-changes"), _inputCommand->request, Timeout::LongCommand, 2,
        [this](const QJsonObject& response) { finishInputChanges(response); }, {},
        [this](const QString& error, const QJsonObject& body) { failInputCommand(error, body); });
}

void SpiralServiceManager::finishInputChanges(const QJsonObject& response)
{
    if (!_inputCommand) return;
    const auto revisions = response.value(QStringLiteral("revisions")).toArray();
    for (const auto& value : revisions) {
        const auto pair = value.toObject();
        const auto id = pair.value(QStringLiteral("id")).toString();
        const auto revision = pair.value(QStringLiteral("revision")).toInteger();
        if (auto draft = _inputDrafts.value(id)) {
            for (const auto& snapshot : _inputCommand->batch.entries)
                if (snapshot.id == id) draft->acknowledgeAccepted(snapshot, revision);
        }
        for (int index = _inputCommand->revisions.size() - 1; index >= 0; --index)
            if (_inputCommand->revisions[index].toObject().value(QStringLiteral("id")).toString() == id)
                _inputCommand->revisions.removeAt(index);
        _inputCommand->revisions.append(pair);
    }
    if (!response.value(QStringLiteral("applied")).toBool()) {
        installInputCatalog(response.value(QStringLiteral("catalog")).toArray());
        failInputCommand(tr("The selected batch could not be applied: %1").arg(
            QString::fromUtf8(QJsonDocument(response.value(QStringLiteral("errors")).toObject()).toJson(QJsonDocument::Compact))),
            {{QStringLiteral("error"), "application"}});
        return;
    }
    for (const auto& value : _inputCommand->revisions) {
        const auto pair = value.toObject();
        if (auto draft = _inputDrafts.value(pair.value(QStringLiteral("id")).toString()))
            draft->acknowledgeApplied(pair.value(QStringLiteral("revision")).toInteger());
    }
    installInputCatalog(response.value(QStringLiteral("catalog")).toArray());
    _inputCommand->applied = true;
    if (_inputCommand->commit) persistInputCommand();
    else finishInputCommand();
}

void SpiralServiceManager::persistInputCommand()
{
    if (!_inputCommand) return;
    if (!_inputCommand->commit || _inputCommand->revisions.isEmpty()) { finishInputCommand(); return; }
    emit inputDraftsChanged();
    postWithRetry(QStringLiteral("/session/commit-inputs"),
        {{QStringLiteral("command_id"), _inputCommand->commitId},
         {QStringLiteral("revisions"), _inputCommand->revisions}}, Timeout::LongCommand, 2,
        [this](const QJsonObject& response) {
            if (!_inputCommand) return;
            for (const auto& value : _inputCommand->revisions) {
                const auto pair = value.toObject();
                if (auto draft = _inputDrafts.value(pair.value(QStringLiteral("id")).toString()))
                    draft->acknowledgePersisted(pair.value(QStringLiteral("revision")).toInteger());
            }
            installInputCatalog(response.value(QStringLiteral("catalog")).toArray());
            fetchAdvertisedDataset();
            finishInputCommand();
        }, {}, [this](const QString& error, const QJsonObject& body) { failInputCommand(error, body); });
}

void SpiralServiceManager::failInputCommand(const QString& error, const QJsonObject& body)
{
    if (!_inputCommand) return;
    for (const auto& snapshot : _inputCommand->batch.entries) _inputErrors[snapshot.id] = error;
    _inputCommandBusy = false;
    // A timeout or a retained transaction is an unknown outcome. Keep the
    // captured bytes and command IDs so Retry/reconnect attaches to it.
    const bool unknown = body.isEmpty() || body.contains(QStringLiteral("command_id"))
        || body.value(QStringLiteral("http_status")).toInt() >= 500;
    if (unknown) { _inputSubmission.transportInterrupted(); _inputCommand->batch.outcomeUnknown = true; }
    else {
        _inputSubmission.reconciled(_inputCommand->batch.commandId);
        _inputCommand.reset();
    }
    emit inputDraftsChanged();
    emit inputBatchFinished(error);
    emit errorOccurred(error);
    for (const auto& conflict : body.value(QStringLiteral("conflicts")).toArray())
        emit inputConflict(conflict.toObject());
}

void SpiralServiceManager::finishInputCommand()
{
    if (!_inputCommand) return;
    const auto command = std::move(_inputCommand);
    for (const auto& id : command->localDeletions) discardInputDraft(id);
    _inputCommandBusy = false;
    _inputSubmission.reconciled(command->batch.commandId);
    QStringList aliases, committed;
    for (const auto& snapshot : command->batch.entries) {
        _inputErrors.remove(snapshot.id);
        const auto alias = snapshot.content.manifest.value(QStringLiteral("alias")).toString(snapshot.id);
        const auto current = _inputDrafts.value(snapshot.id);
        if (current && current->snapshot().localRevision == snapshot.localRevision
            && !aliases.contains(alias)) aliases.push_back(alias);
    }
    for (const auto& value : command->revisions) {
        const auto id = value.toObject().value(QStringLiteral("id")).toString();
        const auto draft = _inputDrafts.value(id);
        const auto alias = draft ? draft->snapshot().content.manifest.value(QStringLiteral("alias")).toString(id) : id;
        if (draft && !draft->dirty() && draft->persisted() == draft->accepted()
            && !committed.contains(alias)) committed.push_back(alias);
    }
    for (const auto& alias : aliases) emit inputUploadFinished(alias, {});
    if (command->commit) emit commitInputsFinished(committed, {});
    emit inputBatchFinished({});
    emit inputDraftsChanged();
    if (_afterInputCommand) { auto next = std::move(_afterInputCommand); next(); }
}

void SpiralServiceManager::resolveInputConflict(const QJsonObject& conflict, const QString& action)
{
    const auto id = conflict.value(QStringLiteral("id")).toString();
    const auto draft = _inputDrafts.value(id);
    if (!draft) return;
    const auto captured = draft->snapshot();
    postWithRetry(QStringLiteral("/session/resolve-input"),
        {{QStringLiteral("command_id"), commandId()}, {QStringLiteral("id"), id},
         {QStringLiteral("expected_revision"), conflict.value(QStringLiteral("expected_revision")).toInteger(draft->accepted())},
         {QStringLiteral("review_token"), conflict.value(QStringLiteral("review_token"))},
         {QStringLiteral("action"), action == QStringLiteral("save_as_new") ? QStringLiteral("use_current") : action}},
        Timeout::LongCommand, 2,
        [this, id, captured, action](const QJsonObject& response) {
            if (!response.value(QStringLiteral("resolved")).toBool()) { emit errorOccurred(tr("Current input could not be applied")); return; }
            {
                for (const auto& value : response.value(QStringLiteral("catalog")).toArray()) {
                    const auto input = value.toObject();
                    if (input.value(QStringLiteral("id")).toString() != id) continue;
                    auto manifest = input;
                    manifest[QStringLiteral("path")] = input.value(QStringLiteral("content")).toObject().value(QStringLiteral("path"));
                    manifest[QStringLiteral("restore_revision")] = input.value(QStringLiteral("accepted_revision"));
                    if (auto current = _inputDrafts.value(id))
                        current->reconcileReviewedContent({manifest, input.value(QStringLiteral("deleted")).toBool()},
                            input.value(QStringLiteral("accepted_revision")).toInteger(),
                            input.value(QStringLiteral("applied_revision")).toInteger(),
                            input.value(QStringLiteral("persisted_revision")).toInteger(),
                            action != QStringLiteral("apply_local_after_review")
                                && current->snapshot().localRevision == captured.localRevision);
                }
            }
            if (action == QStringLiteral("save_as_new")) {
                const auto newId = uuid();
                auto content = captured.content;
                content.manifest.remove(QStringLiteral("restore_revision"));
                content.manifest[QStringLiteral("name")] = newId;
                content.manifest[QStringLiteral("alias")] = newId;
                _inputOrder.push_back(newId);
                _inputDrafts[newId] = std::make_shared<vc3d::spiral::InputDraft>(newId, content);
            }
            _inputErrors.remove(id);
            installInputCatalog(response.value(QStringLiteral("catalog")).toArray());
        }, [this](const QString& error) { emit errorOccurred(error); });
}

void SpiralServiceManager::editInputDraft(const QString& id)
{
    const auto draft = _inputDrafts.value(id);
    if (!draft || draft->deleted()) return;
    const auto snapshot = draft->snapshot();
    const auto input = _inputCatalog.value(id, snapshot.content.manifest);
    if (input.value(QStringLiteral("kind")).toString() == QStringLiteral("pcl")) {
        auto editorInput = input;
        editorInput[QStringLiteral("alias")] = snapshot.content.manifest.value(QStringLiteral("alias"));
        emit inputEditorRequested(editorInput, {});
        return;
    }
    auto open = [this, input](const QString& source) {
        const bool fiber = input.value(QStringLiteral("kind")).toString() == QStringLiteral("fiber");
        workingCopyAsync(fiber ? QFileInfo(source).absolutePath() : source,
            [this, input, source, fiber](const QString& working, const QString& error) {
                if (working.isEmpty()) emit errorOccurred(error);
                else emit inputEditorRequested(input, fiber ? QDir(working).filePath(QFileInfo(source).fileName()) : working);
            });
    };
    const auto local = snapshot.content.manifest.value(QStringLiteral("path")).toString();
    if (draft->dirty() && QFileInfo::exists(local)) { open(local); return; }
    get(QStringLiteral("/session/input-content/%1/%2").arg(id).arg(draft->accepted()), Timeout::Command,
        [this, input, open](const QJsonObject& response) {
            const auto workspace = response.value(QStringLiteral("workspace_id")).toString();
            const auto artifact = response.value(QStringLiteral("artifact")).toObject().value(QStringLiteral("id")).toString();
            _artifactCache->fetchArtifact(workspace, artifact,
                [this, input, open](const QString& path, const QString& error, bool) {
                    if (!error.isEmpty()) { emit errorOccurred(error); return; }
                    open(input.value(QStringLiteral("kind")).toString() == QStringLiteral("patch")
                         ? QFileInfo(path).absolutePath() : path);
                });
        });
}

void SpiralServiceManager::discardInputWorkspace(std::function<void()> done)
{
    if (_inputCommand) {
        _afterInputCommand = [this, done]() { discardInputWorkspace(done); };
        resumeInputCommand();
        return;
    }
    QJsonArray revisions;
    for (auto it = _inputDrafts.cbegin(); it != _inputDrafts.cend(); ++it)
        if (it.value()->accepted() > it.value()->persisted())
            revisions.append(QJsonObject{{QStringLiteral("id"), it.key()},
                {QStringLiteral("revision"), qint64(it.value()->accepted())}});
    postWithRetry(QStringLiteral("/session/discard-inputs"),
        {{QStringLiteral("command_id"), commandId()}, {QStringLiteral("revisions"), revisions}}, Timeout::LongCommand, 2,
        [this, done](const QJsonObject& response) {
            if (!response.value(QStringLiteral("discarded")).toBool()) { emit errorOccurred(tr("The inputs could not be discarded")); return; }
            _inputDrafts.clear();
            _inputOrder.clear();
            _inputCatalog.clear();
            _inputAliases.clear();
            _inputErrors.clear();
            _inputSelection.clear();
            _inputSelectionExplicit = false;
            emit inputDraftsChanged();
            if (done) done();
        }, [this](const QString& error) { emit errorOccurred(error); });
}

void SpiralServiceManager::releaseInputWorkspace(std::function<void()> done)
{
    if (!_inputOwner) { if (done) done(); return; }
    // Stop local continuations before releasing their server-side workspace.
    if (_inputCommand) _inputSubmission.reconciled(_inputCommand->batch.commandId);
    _inputCommand.reset();
    _inputCommandBusy = false;
    _afterInputCommand = {};
    cancelWorkingCopies();
    postWithRetry(QStringLiteral("/session/editing/release"),
        {{QStringLiteral("command_id"), commandId()}}, Timeout::LongCommand, 2,
        [this, done](const QJsonObject&) {
            clearInputWorkspace();
            if (done) done();
        }, [this](const QString& error) { emit errorOccurred(error); });
}

void SpiralServiceManager::cancelWorkingCopies()
{
    ++_workingCopyGeneration;
    for (auto* job : _workingCopyJobs) job->cancel();
    _workingCopyJobs.clear();
    emit inputCopyProgress(0, {});
}

void SpiralServiceManager::clearInputWorkspace()
{
    cancelWorkingCopies();
    _workingCopies.clear();
    emit inputCopyProgress(0, {});
    _inputOwner = false;
    _inputWorkspaceId.clear();
    _inputDrafts.clear();
    _inputOrder.clear();
    _inputCatalog.clear();
    _inputAliases.clear();
    _inputErrors.clear();
    _inputSelection.clear();
    _inputSelectionExplicit = false;
    emit inputWorkspaceReleased();
    emit inputDraftsChanged();
    // Editors must detach before background cleanup removes their files.
    _workingCopyDirectories.clear();
}
