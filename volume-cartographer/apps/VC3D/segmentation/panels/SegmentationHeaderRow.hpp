#pragma once

#include <QWidget>

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QString;

class SegmentationHeaderRow : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationHeaderRow(QWidget* parent = nullptr);

    void setEditingChecked(bool checked);
    void setAnnotateChecked(bool checked);
    void setDrawMaskChecked(bool checked);
    void setMaskBrushRadius(float radius);
    void setStatusText(const QString& text);

signals:
    void editingToggled(bool enabled);
    void annotateToggled(bool enabled);
    void drawMaskToggled(bool enabled);
    void maskBrushRadiusChanged(float radius);

private:
    QCheckBox* _chkEditing{nullptr};
    QCheckBox* _chkAnnotate{nullptr};
    QCheckBox* _chkDrawMask{nullptr};
    QDoubleSpinBox* _spinMaskBrushRadius{nullptr};
    QLabel* _lblStatus{nullptr};
};
