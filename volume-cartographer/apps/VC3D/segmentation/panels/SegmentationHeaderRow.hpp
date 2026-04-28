#pragma once

#include <QWidget>

class QCheckBox;
class QLabel;
class QString;

class SegmentationHeaderRow : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationHeaderRow(QWidget* parent = nullptr);

    void setEditingChecked(bool checked);
    [[nodiscard]] bool isEditingChecked() const;
    void setAnnotateChecked(bool checked);
    [[nodiscard]] bool isAnnotateChecked() const;
    void setStatusText(const QString& text);

signals:
    void editingToggled(bool enabled);
    void annotateToggled(bool enabled);

private:
    QCheckBox* _chkEditing{nullptr};
    QCheckBox* _chkAnnotate{nullptr};
    QLabel* _lblStatus{nullptr};
};
