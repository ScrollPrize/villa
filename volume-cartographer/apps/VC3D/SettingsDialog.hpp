#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <filesystem>
#include <memory>
#include <vector>

class QComboBox;
class QCheckBox;
class QSpinBox;
class VolumePkg;

class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        SettingsDialog(std::shared_ptr<VolumePkg> volumePackage = {},
                       QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);
        bool outputSegmentsChanged() const { return _outputSegmentsChanged; }

    protected slots:
        void accept() override;

    private:
        void setupOutputSegmentsControl();
        void setupCacheActionControls();

        std::shared_ptr<VolumePkg> _volumePackage;
        std::filesystem::path _activeRemoteCacheRoot;
        QComboBox* _outputSegmentsCombo{nullptr};
        QCheckBox* _remoteCacheDelta3dCheckBox{nullptr};
        bool _outputSegmentsChanged{false};
};
