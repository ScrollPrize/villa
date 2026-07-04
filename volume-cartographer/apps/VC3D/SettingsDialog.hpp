#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <filesystem>
#include <memory>

class QComboBox;
class VolumePkg;


class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        SettingsDialog(std::shared_ptr<VolumePkg> volumePackage = {},
                       std::filesystem::path currentVolumeCacheDir = {},
                       QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);
        bool outputSegmentsChanged() const { return _outputSegmentsChanged; }

    protected slots:
        void accept() override;

    private:
        void setupOutputSegmentsControl();
        void compressExistingCache();

        std::shared_ptr<VolumePkg> _volumePackage;
        std::filesystem::path _currentVolumeCacheDir;
        QComboBox* _outputSegmentsCombo{nullptr};
        bool _outputSegmentsChanged{false};
};
