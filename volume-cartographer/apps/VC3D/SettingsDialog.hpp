#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

class QComboBox;
class QPushButton;
class QSpinBox;
class VolumePkg;
class Volume;


// Chunk geometry of the currently shown volume, needed to apply the
// delta-zyx compression filter when compacting its disk cache.
// levelChunkShapes is indexed by pyramid level ({0,0,0} = unknown).
struct CacheChunkLayout {
    std::vector<std::array<int, 3>> levelChunkShapes;
    std::size_t elemSize = 1;
};

class SettingsDialog : public QDialog, private Ui_VCSettingsDlg
{
    Q_OBJECT

    public:
        SettingsDialog(std::shared_ptr<VolumePkg> volumePackage = {},
                       std::shared_ptr<Volume> currentVolume = {},
                       std::filesystem::path currentVolumeCacheDir = {},
                       CacheChunkLayout currentVolumeChunkLayout = {},
                       QWidget* parent = nullptr);

        static std::vector<int> expandSettingToIntRange(const QString& setting);
        bool outputSegmentsChanged() const { return _outputSegmentsChanged; }

    protected slots:
        void accept() override;

    private:
        void setupOutputSegmentsControl();
        void setupCacheActionControls();
        void compressExistingCache();
        void redownloadExistingCache();

        std::shared_ptr<VolumePkg> _volumePackage;
        std::shared_ptr<Volume> _currentVolume;
        std::filesystem::path _currentVolumeCacheDir;
        CacheChunkLayout _currentVolumeChunkLayout;
        QComboBox* _outputSegmentsCombo{nullptr};
        QPushButton* _redownloadCacheButton{nullptr};
        QSpinBox* _cacheActionWorkersSpin{nullptr};
        bool _outputSegmentsChanged{false};
};
