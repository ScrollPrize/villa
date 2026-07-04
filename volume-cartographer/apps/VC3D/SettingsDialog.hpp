#pragma once

#include "ui_VCSettings.h"
#include <QStringList>
#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

class QComboBox;
class VolumePkg;


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
                       std::filesystem::path currentVolumeCacheDir = {},
                       CacheChunkLayout currentVolumeChunkLayout = {},
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
        CacheChunkLayout _currentVolumeChunkLayout;
        QComboBox* _outputSegmentsCombo{nullptr};
        bool _outputSegmentsChanged{false};
};
