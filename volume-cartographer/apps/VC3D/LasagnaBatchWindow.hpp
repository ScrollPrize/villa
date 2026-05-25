#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>

class QPushButton;
class QStandardItemModel;
class QTableView;

class LasagnaBatchWindow : public QWidget
{
    Q_OBJECT

public:
    explicit LasagnaBatchWindow(QWidget* parent = nullptr);

private:
    void setJobs(const QJsonArray& jobs);
    void applyJobs(const QJsonArray& jobs, const QString& preferredSelection = QString());
    void updateRow(int row, const QJsonObject& job);
    QString selectedJobId() const;
    int selectedRow() const;
    int rowForJobId(const QString& jobId) const;
    QJsonObject jobAtRow(int row) const;
    bool isWaitingRow(int row) const;
    void updateActionState();
    void optimisticallyCancel(const QString& jobId);
    void optimisticallyMoveBefore(const QString& jobId, const QString& beforeJobId);

    QTableView* _table{nullptr};
    QStandardItemModel* _model{nullptr};
    QPushButton* _upButton{nullptr};
    QPushButton* _downButton{nullptr};
    QPushButton* _cancelButton{nullptr};
    QPushButton* _downloadButton{nullptr};
    QJsonArray _jobs;
};
