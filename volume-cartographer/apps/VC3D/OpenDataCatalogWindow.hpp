#pragma once

#include "OpenDataManifest.hpp"

#include <QDialog>
#include <QFutureWatcher>
#include <QImage>
#include <QString>

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

class QCheckBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTabWidget;

namespace vc3d::opendata {

class OpenDataCatalogWindow : public QDialog
{
    Q_OBJECT

public:
    explicit OpenDataCatalogWindow(QWidget* parent = nullptr);
    ~OpenDataCatalogWindow() override;

    void setOpenSampleHandler(std::function<bool(const OpenDataSample&)> handler);

private slots:
    void reloadManifest();
    void onFetchFinished();
    void updateSampleFilter();
    void updateSelectedSample();
    void copySelectedVolumeUrl();
    void openSelectedVolumeUrl();
    void downloadSelectedVolumeNormalGrids();
    void copySelectedSegmentUrl();
    void openSelectedSegmentUrl();
    void cacheSelectedSegment();
    void openSelectedSegmentCacheFolder();
    void syncSelectedSampleCache();
    void openSelectedSample();

private:
    struct ManifestLoadResult {
        OpenDataManifest manifest;
        QString jsonText;
        QString sourceLabel;
        QString error;
    };

    struct PhotoLoadResult {
        QString sampleId;
        QString url;
        QImage image;
        QString error;
    };

    void buildUi();
    void loadCachedManifestIfAvailable();
    void applyManifest(OpenDataManifest manifest, QString sourceLabel, bool fromCache);
    void setStatus(const QString& text);
    void setLoading(bool loading);
    void persistFetchedManifest(const ManifestLoadResult& result) const;
    void populateSamples();
    void populateDetails(const OpenDataSample* sample);
    void clearDetails();
    void refreshSelectedSampleCacheStatus();
    void updateActionButtons();
    void loadOverviewPhoto(const OpenDataSample& sample);
    void onOverviewPhotoFinished(QFutureWatcher<PhotoLoadResult>* watcher);
    void setOverviewPhotoText(const QString& text);
    void setOverviewPhotoImage(const QImage& image);

    [[nodiscard]] const OpenDataSample* selectedSample() const;
    [[nodiscard]] const OpenDataVolume* selectedVolume() const;
    [[nodiscard]] const OpenDataSegment* selectedSegment() const;
    [[nodiscard]] QString selectedVolumeUrl() const;
    [[nodiscard]] QString selectedSegmentUrl() const;
    [[nodiscard]] std::filesystem::path selectedSegmentCacheDir() const;
    [[nodiscard]] std::filesystem::path cacheRoot() const;
    [[nodiscard]] std::filesystem::path cachedManifestPath() const;
    [[nodiscard]] std::filesystem::path cacheMetadataPath() const;

    QLineEdit* _searchEdit{nullptr};
    QCheckBox* _segmentsOnlyCheck{nullptr};
    QCheckBox* _inkOnlyCheck{nullptr};
    QTableWidget* _sampleTable{nullptr};
    QTabWidget* _tabs{nullptr};
    QLabel* _overviewLabel{nullptr};
    QLabel* _overviewPhotoLabel{nullptr};
    QTableWidget* _scansTable{nullptr};
    QTableWidget* _volumesTable{nullptr};
    QTableWidget* _segmentsTable{nullptr};
    QLabel* _statusLabel{nullptr};
    QPushButton* _refreshButton{nullptr};
    QPushButton* _openSampleButton{nullptr};
    QPushButton* _copyVolumeUrlButton{nullptr};
    QPushButton* _openVolumeUrlButton{nullptr};
    QPushButton* _downloadNormalGridsButton{nullptr};
    QPushButton* _copySegmentUrlButton{nullptr};
    QPushButton* _openSegmentUrlButton{nullptr};
    QPushButton* _cacheSegmentButton{nullptr};
    QPushButton* _openSegmentCacheFolderButton{nullptr};
    QPushButton* _syncSampleCacheButton{nullptr};
    QCheckBox* _doNotShowOnNextOpenCheck{nullptr};

    std::optional<OpenDataManifest> _manifest;
    std::vector<std::size_t> _visibleSampleIndexes;
    QFutureWatcher<ManifestLoadResult>* _fetchWatcher{nullptr};
    QFutureWatcher<PhotoLoadResult>* _photoWatcher{nullptr};
    std::vector<QFutureWatcher<PhotoLoadResult>*> _photoWatchers;
    QString _pendingPhotoSampleId;
    std::function<bool(const OpenDataSample&)> _openSampleHandler;
};

} // namespace vc3d::opendata
