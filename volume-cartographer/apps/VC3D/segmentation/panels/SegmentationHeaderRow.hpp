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
    void setStatusText(const QString& text);

signals:
    void editingToggled(bool enabled);

private:
    QCheckBox* _chkEditing{nullptr};
    QLabel* _lblStatus{nullptr};
};
