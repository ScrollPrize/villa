#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <memory>

class QCheckBox;
class QSpinBox;

class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        explicit SettingsDialog(QWidget* parent = nullptr);

    protected slots:
        void accept() override;

    private:
        void setupCacheActionControls();

        QCheckBox* _remoteCacheDelta3dCheckBox{nullptr};
        QCheckBox* _renderCudaCheckBox{nullptr};
};
