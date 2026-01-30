#pragma once

#include <QWidget>

class JsonProfileEditor;

class SegmentationCustomParamsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCustomParamsPanel(QWidget* parent = nullptr);

    JsonProfileEditor* editor() const { return _customParamsEditor; }

private:
    JsonProfileEditor* _customParamsEditor{nullptr};
};
