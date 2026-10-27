#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"

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
    explicit OpenDataCatalogWindow(const vcr::catalog::OpenDataManifest& manifest, QWidget* parent = nullptr);
    ~OpenDataCatalogWindow() override;

    void setOpenSampleHandler(std::function<bool(const vcr::catalog::OpenDataSample&)> handler);
    using CreateProjectHandler = std::function<bool(const vcr::catalog::OpenDataSample&,
        const OpenDataResourceSelection&, const OpenDataNewProject&)>;
    void setCreateProjectHandler(CreateProjectHandler handler);
    using AttachPredictionHandler = std::function<void(const vcr::catalog::OpenDataSample&,
        const OpenDataResourceSelection&)>;
    void setAttachPredictionHandler(AttachPredictionHandler handler);
    // Refreshes the sample's surface store from the published catalog; returns the status line.
    using RefreshSampleHandler = std::function<QString(const vcr::catalog::OpenDataSample&)>;
    void setRefreshSampleHandler(RefreshSampleHandler handler);

signals:
    void attachVolumeRequested(const QString& url);
    void attachLasagnaRequested(const QString& url, bool fiber);
    void attachNormalGridsRequested(const QString& sampleId, const QString& volumeId,
                                   const QString& url, int level, bool explicitLevel);

private slots:
    void reloadManifest();
    void onFetchFinished();
    void updateSampleFilter();
    void updateSelectedSample();
    void copySelectedVolumeUrl();
    void openSelectedVolumeUrl();
    void copySelectedRepresentationUrl();
    void openSelectedRepresentationUrl();
    void downloadSelectedVolumeNormalGrids();
    void copySelectedSegmentUrl();
    void openSelectedSegmentUrl();
    void refreshSelectedSample();
    void openSelectedSample();
    void createSelectedProject();

private:
    struct ManifestLoadResult {
        vcr::catalog::OpenDataManifest manifest;
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
    void applyManifest(vcr::catalog::OpenDataManifest manifest, QString sourceLabel, bool fromCache);
    void setStatus(const QString& text);
    void setLoading(bool loading);
    void populateSamples();
    void populateDetails(const vcr::catalog::OpenDataSample* sample);
    void clearDetails();
    void updateActionButtons();
    void loadOverviewPhoto(const vcr::catalog::OpenDataSample& sample);
    void onOverviewPhotoFinished(QFutureWatcher<PhotoLoadResult>* watcher);
    void setOverviewPhotoText(const QString& text);
    void setOverviewPhotoImage(const QImage& image);

    [[nodiscard]] const vcr::catalog::OpenDataSample* selectedSample() const;
    [[nodiscard]] const vcr::catalog::OpenDataVolume* selectedVolume() const;
    [[nodiscard]] const vcr::catalog::OpenDataArtifact* selectedRepresentationArtifact() const;
    [[nodiscard]] const vcr::catalog::OpenDataSegment* selectedSegment() const;
    [[nodiscard]] QString selectedVolumeUrl() const;
    [[nodiscard]] QString selectedRepresentationUrl() const;
    [[nodiscard]] QString selectedSegmentUrl() const;
    [[nodiscard]] std::filesystem::path cacheRoot() const;
    [[nodiscard]] std::filesystem::path cachedManifestPath() const;

    QLineEdit* _searchEdit{nullptr};
    QCheckBox* _segmentsOnlyCheck{nullptr};
    QCheckBox* _inkOnlyCheck{nullptr};
    QTableWidget* _sampleTable{nullptr};
    QTabWidget* _tabs{nullptr};
    QLabel* _overviewLabel{nullptr};
    QLabel* _overviewPhotoLabel{nullptr};
    QTableWidget* _scansTable{nullptr};
    QTableWidget* _volumesTable{nullptr};
    QTableWidget* _representationsTable{nullptr};
    QTableWidget* _segmentsTable{nullptr};
    QLabel* _statusLabel{nullptr};
    QPushButton* _refreshButton{nullptr};
    QPushButton* _openSampleButton{nullptr};
    QPushButton* _createProjectButton{nullptr};
    QPushButton* _copyVolumeUrlButton{nullptr};
    QPushButton* _attachVolumeButton{nullptr};
    QPushButton* _openVolumeUrlButton{nullptr};
    QPushButton* _downloadNormalGridsButton{nullptr};
    QPushButton* _copyRepresentationUrlButton{nullptr};
    QPushButton* _attachRepresentationButton{nullptr};
    QPushButton* _openRepresentationUrlButton{nullptr};
    QPushButton* _copySegmentUrlButton{nullptr};
    QPushButton* _openSegmentUrlButton{nullptr};
    QPushButton* _refreshSampleButton{nullptr};
    QCheckBox* _doNotShowOnNextOpenCheck{nullptr};

    std::optional<vcr::catalog::OpenDataManifest> _manifest;
    std::vector<std::size_t> _visibleSampleIndexes;
    QFutureWatcher<ManifestLoadResult>* _fetchWatcher{nullptr};
    bool _manifestRefreshPending{false};
    QFutureWatcher<PhotoLoadResult>* _photoWatcher{nullptr};
    std::vector<QFutureWatcher<PhotoLoadResult>*> _photoWatchers;
    QString _pendingPhotoSampleId;
    std::function<bool(const vcr::catalog::OpenDataSample&)> _openSampleHandler;
    CreateProjectHandler _createProjectHandler;
    AttachPredictionHandler _attachPredictionHandler;
    RefreshSampleHandler _refreshSampleHandler;
};

} // namespace vc3d::opendata
