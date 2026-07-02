#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <memory>

class QComboBox;
class VolumePkg;


class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        SettingsDialog(std::shared_ptr<VolumePkg> volumePackage = {}, QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);
        bool outputSegmentsChanged() const { return _outputSegmentsChanged; }

    protected slots:
        void accept() override;

    private:
        void setupOutputSegmentsControl();

        std::shared_ptr<VolumePkg> _volumePackage;
        QComboBox* _outputSegmentsCombo{nullptr};
        bool _outputSegmentsChanged{false};
};
