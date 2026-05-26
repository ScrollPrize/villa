#include "LasagnaBatchWindow.hpp"

#include "LasagnaServiceManager.hpp"

#include <QDateTime>
#include <QFileInfo>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QJsonObject>
#include <QAbstractItemView>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStringList>
#include <QTableView>
#include <QVBoxLayout>
#include <QHBoxLayout>

namespace
{
QString jobStateText(const QJsonObject& job)
{
    const QString state = job[QStringLiteral("state")].toString();
    if (state == QStringLiteral("upload")) {
        return QObject::tr("Upload");
    }
    if (state == QStringLiteral("waiting")) {
        const int pos = job[QStringLiteral("queue_position")].toInt();
        return pos > 0 ? QObject::tr("Waiting #%1").arg(pos) : QObject::tr("Waiting");
    }
    if (state == QStringLiteral("running")) {
        const double progress = job[QStringLiteral("overall_progress")].toDouble();
        return QObject::tr("Running %1%").arg(progress * 100.0, 0, 'f', 1);
    }
    if (state == QStringLiteral("finished")) {
        return QObject::tr("Finished");
    }
    if (state == QStringLiteral("cancelled")) {
        return QObject::tr("Cancelled");
    }
    if (state == QStringLiteral("error")) {
        const QString error = job[QStringLiteral("error")].toString();
        return error.isEmpty() ? QObject::tr("Error") : QObject::tr("Error: %1").arg(error);
    }
    return state;
}

QString submittedTimeText(const QJsonObject& job)
{
    const qint64 seconds = static_cast<qint64>(job[QStringLiteral("submitted_at")].toDouble());
    if (seconds <= 0) {
        return {};
    }
    return QDateTime::fromSecsSinceEpoch(seconds).toString(Qt::ISODate);
}

QString outputNameText(const QJsonObject& job)
{
    const QString outputName = job[QStringLiteral("output_name")].toString().trimmed();
    if (!outputName.isEmpty()) {
        return outputName;
    }
    const QString outputDir = job[QStringLiteral("output_dir")].toString().trimmed();
    return outputDir.isEmpty() ? QString() : QFileInfo(outputDir).fileName();
}
}

LasagnaBatchWindow::LasagnaBatchWindow(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    _model = new QStandardItemModel(this);
    _model->setHorizontalHeaderLabels({
        tr("Order"),
        tr("Source"),
        tr("Config"),
        tr("Output"),
        tr("Progress"),
        tr("Submitted"),
    });

    _table = new QTableView(this);
    _table->setModel(_model);
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setSelectionMode(QAbstractItemView::SingleSelection);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    layout->addWidget(_table, 1);
    connect(_table->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, [this]() { updateActionState(); });
    connect(_table, &QTableView::doubleClicked, this, [this](const QModelIndex& index) {
        if (!index.isValid()) {
            return;
        }
        const QJsonObject job = jobAtRow(index.row());
        if (job[QStringLiteral("state")].toString() != QStringLiteral("finished")) {
            return;
        }
        const QString outputName = outputNameText(job);
        if (!outputName.isEmpty()) {
            emit finishedOutputActivated(outputName);
        }
    });

    auto* buttonRow = new QHBoxLayout();
    _upButton = new QPushButton(tr("Up"), this);
    _downButton = new QPushButton(tr("Down"), this);
    _cancelButton = new QPushButton(tr("Cancel"), this);
    buttonRow->addWidget(_upButton);
    buttonRow->addWidget(_downButton);
    buttonRow->addWidget(_cancelButton);
    buttonRow->addStretch(1);
    layout->addLayout(buttonRow);

    connect(&LasagnaServiceManager::instance(), &LasagnaServiceManager::jobsUpdated,
            this, &LasagnaBatchWindow::setJobs);

    connect(_upButton, &QPushButton::clicked, this, [this]() {
        const int row = selectedRow();
        if (row <= 0 || !isWaitingRow(row)) return;
        int prev = row - 1;
        while (prev >= 0 && !isWaitingRow(prev)) {
            --prev;
        }
        if (prev >= 0) {
            const QString jobId = selectedJobId();
            const QString beforeJobId = jobAtRow(prev)[QStringLiteral("job_id")].toString();
            optimisticallyMoveBefore(jobId, beforeJobId);
            LasagnaServiceManager::instance().moveJobBefore(
                jobId,
                beforeJobId);
        }
    });

    connect(_downButton, &QPushButton::clicked, this, [this]() {
        const int row = selectedRow();
        if (row < 0 || !isWaitingRow(row)) return;
        int next = row + 1;
        while (next < _jobs.size() && !isWaitingRow(next)) {
            ++next;
        }
        if (next < _jobs.size()) {
            const QString jobId = selectedJobId();
            int afterNext = next + 1;
            while (afterNext < _jobs.size() && !isWaitingRow(afterNext)) {
                ++afterNext;
            }
            const QString beforeJobId = afterNext < _jobs.size()
                ? jobAtRow(afterNext)[QStringLiteral("job_id")].toString()
                : QString();
            optimisticallyMoveBefore(jobId, beforeJobId);
            LasagnaServiceManager::instance().moveJobBefore(
                jobId,
                beforeJobId);
        }
    });

    connect(_cancelButton, &QPushButton::clicked, this, [this]() {
        const QString jobId = selectedJobId();
        if (!jobId.isEmpty()) {
            optimisticallyCancel(jobId);
            LasagnaServiceManager::instance().cancelJob(jobId);
        }
    });

    updateActionState();
}

void LasagnaBatchWindow::setJobs(const QJsonArray& jobs)
{
    applyJobs(jobs);
}

void LasagnaBatchWindow::applyJobs(const QJsonArray& jobs, const QString& preferredSelection)
{
    const QString selectedId = preferredSelection.isEmpty() ? selectedJobId() : preferredSelection;
    const int scrollValue = _table->verticalScrollBar()->value();
    QItemSelectionModel* selectionModel = _table->selectionModel();
    const QSignalBlocker selectionBlocker(selectionModel);

    _jobs = jobs;

    for (int targetRow = 0; targetRow < jobs.size(); ++targetRow) {
        const QJsonObject job = jobs[targetRow].toObject();
        const QString jobId = job[QStringLiteral("job_id")].toString();
        int existingRow = rowForJobId(jobId);
        if (existingRow < 0) {
            QList<QStandardItem*> rowItems{
                new QStandardItem(),
                new QStandardItem(),
                new QStandardItem(),
                new QStandardItem(),
                new QStandardItem(),
                new QStandardItem(),
            };
            _model->insertRow(targetRow, rowItems);
            existingRow = targetRow;
        } else if (existingRow != targetRow) {
            QList<QStandardItem*> rowItems = _model->takeRow(existingRow);
            _model->insertRow(targetRow, rowItems);
            existingRow = targetRow;
        }
        updateRow(existingRow, job);
    }

    for (int row = _model->rowCount() - 1; row >= 0; --row) {
        QStandardItem* idItem = _model->item(row, 0);
        const QString jobId = idItem ? idItem->data(Qt::UserRole).toString() : QString();
        bool stillPresent = false;
        for (const QJsonValue& value : jobs) {
            if (value.toObject()[QStringLiteral("job_id")].toString() == jobId) {
                stillPresent = true;
                break;
            }
        }
        if (!stillPresent) {
            _model->removeRow(row);
        }
    }

    const int selectedRow = rowForJobId(selectedId);
    if (selectedRow >= 0) {
        const QModelIndex index = _model->index(selectedRow, 0);
        _table->setCurrentIndex(index);
        if (selectionModel) {
            selectionModel->select(index, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows);
        }
    } else if (!selectedId.isEmpty()) {
        _table->clearSelection();
        _table->setCurrentIndex(QModelIndex());
    }
    _table->verticalScrollBar()->setValue(scrollValue);
    updateActionState();
}

void LasagnaBatchWindow::updateRow(int row, const QJsonObject& job)
{
    const QString state = job[QStringLiteral("state")].toString();
    const int queuePos = job[QStringLiteral("queue_position")].toInt();
    const QString order = queuePos > 0 ? QString::number(queuePos) : state;
    const QString jobId = job[QStringLiteral("job_id")].toString();
    const QStringList values{
        order,
        job[QStringLiteral("source")].toString(),
        job[QStringLiteral("config_name")].toString(),
        outputNameText(job),
        jobStateText(job),
        submittedTimeText(job),
    };
    for (int col = 0; col < values.size(); ++col) {
        QStandardItem* item = _model->item(row, col);
        if (!item) {
            item = new QStandardItem();
            _model->setItem(row, col, item);
        }
        if (item->text() != values[col]) {
            item->setText(values[col]);
        }
        item->setData(jobId, Qt::UserRole);
    }
}

QString LasagnaBatchWindow::selectedJobId() const
{
    const int row = selectedRow();
    if (row < 0 || row >= _model->rowCount()) {
        return {};
    }
    QStandardItem* item = _model->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

int LasagnaBatchWindow::selectedRow() const
{
    const QModelIndex index = _table->currentIndex();
    return index.isValid() ? index.row() : -1;
}

int LasagnaBatchWindow::rowForJobId(const QString& jobId) const
{
    if (jobId.isEmpty()) {
        return -1;
    }
    for (int row = 0; row < _model->rowCount(); ++row) {
        QStandardItem* item = _model->item(row, 0);
        if (item && item->data(Qt::UserRole).toString() == jobId) {
            return row;
        }
    }
    return -1;
}

QJsonObject LasagnaBatchWindow::jobAtRow(int row) const
{
    if (row < 0 || row >= _jobs.size()) {
        return {};
    }
    return _jobs[row].toObject();
}

bool LasagnaBatchWindow::isWaitingRow(int row) const
{
    if (row < 0 || row >= _jobs.size()) {
        return false;
    }
    const QString state = _jobs[row].toObject()[QStringLiteral("state")].toString();
    return state == QStringLiteral("waiting");
}

void LasagnaBatchWindow::updateActionState()
{
    const int row = selectedRow();
    const bool hasSelection = row >= 0;
    const QJsonObject job = jobAtRow(row);
    const QString state = job[QStringLiteral("state")].toString();
    const bool waiting = state == QStringLiteral("waiting");
    const bool cancellable = state == QStringLiteral("upload")
        || state == QStringLiteral("waiting")
        || state == QStringLiteral("running");

    _upButton->setEnabled(hasSelection && waiting);
    _downButton->setEnabled(hasSelection && waiting);
    _cancelButton->setEnabled(hasSelection && cancellable);
}

void LasagnaBatchWindow::optimisticallyCancel(const QString& jobId)
{
    QJsonArray updated;
    for (const QJsonValue& value : _jobs) {
        QJsonObject job = value.toObject();
        if (job[QStringLiteral("job_id")].toString() == jobId) {
            job[QStringLiteral("state")] = QStringLiteral("cancelled");
            job[QStringLiteral("error")] = tr("Cancellation requested");
        }
        updated.append(job);
    }
    applyJobs(updated, jobId);
}

void LasagnaBatchWindow::optimisticallyMoveBefore(const QString& jobId, const QString& beforeJobId)
{
    if (jobId.isEmpty() || jobId == beforeJobId) {
        return;
    }

    QJsonObject moving;
    QJsonArray withoutMoving;
    for (const QJsonValue& value : _jobs) {
        const QJsonObject job = value.toObject();
        if (job[QStringLiteral("job_id")].toString() == jobId) {
            moving = job;
        } else {
            withoutMoving.append(job);
        }
    }
    if (moving.isEmpty()) {
        return;
    }

    QJsonArray updated;
    bool inserted = false;
    for (const QJsonValue& value : withoutMoving) {
        const QJsonObject job = value.toObject();
        if (!beforeJobId.isEmpty()
            && job[QStringLiteral("job_id")].toString() == beforeJobId) {
            updated.append(moving);
            inserted = true;
        }
        updated.append(job);
    }
    if (!inserted) {
        updated.append(moving);
    }

    int pos = 1;
    for (int i = 0; i < updated.size(); ++i) {
        QJsonObject job = updated[i].toObject();
        if (job[QStringLiteral("state")].toString() == QStringLiteral("waiting")) {
            job[QStringLiteral("queue_position")] = pos++;
            updated[i] = job;
        }
    }

    applyJobs(updated, jobId);
}
