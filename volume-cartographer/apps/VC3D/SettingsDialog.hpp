#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <memory>
#include <vector>

class QCheckBox;
class QSpinBox;

class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        explicit SettingsDialog(QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);

    protected slots:
        void accept() override;

    private:
        void setupCacheActionControls();

        QCheckBox* _remoteCacheDelta3dCheckBox{nullptr};
};
