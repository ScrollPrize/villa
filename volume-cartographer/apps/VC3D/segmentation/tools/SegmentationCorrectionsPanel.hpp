#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QPushButton;

class SegmentationCorrectionsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit SegmentationCorrectionsPanel(QWidget* parent = nullptr);

    QGroupBox* correctionsGroup() const { return _groupCorrections; }
    QComboBox* correctionsCombo() const { return _comboCorrections; }
    QPushButton* correctionsNewButton() const { return _btnCorrectionsNew; }
    QCheckBox* correctionsAnnotateCheck() const { return _chkCorrectionsAnnotate; }

private:
    QGroupBox* _groupCorrections{nullptr};
    QComboBox* _comboCorrections{nullptr};
    QPushButton* _btnCorrectionsNew{nullptr};
    QCheckBox* _chkCorrectionsAnnotate{nullptr};
};
