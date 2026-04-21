#pragma once

#include <QObject>
#include <QPointer>
#include <array>
#include <string>

#include "vc/core/cache/HttpMetadataFetcher.hpp"

class QAction;
class QDialog;
class QMenu;
class QMenuBar;
class CWindow;
class Volume;

namespace vc {
class Project;
struct DataSource;
}

class MenuActionController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kMaxRecentVolpkg = 10;
    static constexpr int kMaxRecentRemote = 10;
    static constexpr int kMaxRecentProject = 10;

    explicit MenuActionController(CWindow* window);

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void removeRecentVolpkgEntry(const QString& path);
    void refreshRecentMenu();
    void openVolpkgAt(const QString& path);
    // Try to restore ~/.VC3D/current_project.json from the previous session.
    // Returns true if a project was applied.
    bool tryRestoreAutosavedProject();
    void loadAttachedRemoteVolumesForCurrentPackage();
    void triggerTeleaInpaint();
    void openRemoteUrl(const QString& url, bool isRetry = false);

    // Realize a Project DataSource into the active VolumePkg. Local sources
    // load synchronously and return the count of items loaded. Remote sources
    // dispatch to a background thread and return 0 — the apply-and-refresh
    // step happens on the main thread when the worker completes.
    int loadSource(const vc::Project& proj, const vc::DataSource& ds);
    // Remove anything contributed by a DataSource from the active VolumePkg.
    void unloadSource(const vc::Project& proj, const vc::DataSource& ds);

    // Id-addressed variants, for use from the Project dock widget where
    // the user has already picked a specific source. Public so Qt::connect
    // from CWindow / the dock can bind to them.
    void reloadSourceById(const QString& sourceId);
    void removeSourceById(const QString& sourceId);
    void renameSourceById(const QString& sourceId);
    void setSourceEnabled(const QString& sourceId, bool enabled);
    void editSourceTags(const QString& sourceId);
    void revealSourceLocation(const QString& sourceId);

private slots:
    void openVolpkg();
    void openRecentVolpkg();
    void openLocalZarr();
    void openRemoteVolume();
    void browseS3();
    void attachRemoteZarr();
    void attachRemoteSegments();
    void openRecentRemoteVolume();
    void showSettingsDialog();
    // Project / Data menu actions (new JSON-backed project system).
    void openProject();
    void openRecentProject();
    void saveProjectAs();
    void dataAddFileDir();
    void dataAddRemote();
    void dataAddFromProject();
    void dataRemoveSource();
    void dataRenameSource();
    void dataReloadSource();
    void showAboutDialog();
    void showKeybindings();
    void resetSegmentationViews();
    void toggleConsoleOutput();
    void generateReviewReport();
    void toggleDrawBBox(bool enabled);
    void toggleCursorMirroring(bool enabled);
    void surfaceFromSelection();
    void clearSelection();
    void runTeleaInpaint();
    void importObjAsPatch();
    void exitApplication();

private:
    QStringList loadRecentPaths() const;
    void saveRecentPaths(const QStringList& paths);
    void rebuildRecentMenu();
    void ensureRecentActions();

    QStringList loadRecentRemoteUrls() const;
    void saveRecentRemoteUrls(const QStringList& urls);
    void updateRecentRemoteList(const QString& url);
    void refreshRecentRemoteMenu();
    void ensureRecentRemoteActions();

    QStringList loadRecentProjects() const;
    void saveRecentProjects(const QStringList& paths);
    void updateRecentProjectList(const QString& path);
    void refreshRecentProjectMenu();
    void ensureRecentProjectActions();
    void attachRemoteZarrUrl(const QString& url, bool persistEntry = true);
    void openRemoteZarr(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);
    void openRemoteScroll(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);
    void promptAndLoadRemoteSegments(const vc::cache::HttpAuth& auth, const std::string& cachePath);
    bool tryResolveRemoteAuth(const QString& url,
                              vc::cache::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Helpers used by loadSource for the local and remote branches.
    int loadSourceLocal(const vc::Project& proj, const vc::DataSource& ds);
    void loadSourceRemoteAsync(const vc::Project& proj, const vc::DataSource& ds);
    QString remoteCacheDirectory() const;
    QString remoteVolumeRegistryPath() const;
    void persistAttachedRemoteVolume(const QString& url, const std::shared_ptr<Volume>& volume);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};
    QMenu* _recentRemoteMenu{nullptr};
    QMenu* _projectMenu{nullptr};
    QMenu* _recentProjectMenu{nullptr};
    QMenu* _dataMenu{nullptr};

    QAction* _openAct{nullptr};
    QAction* _openLocalZarrAct{nullptr};
    QAction* _openRemoteAct{nullptr};
    QAction* _attachRemoteZarrAct{nullptr};
    QAction* _attachRemoteSegmentsAct{nullptr};
    QAction* _browseS3Act{nullptr};
    std::array<QAction*, kMaxRecentVolpkg> _recentActs{};
    std::array<QAction*, kMaxRecentRemote> _recentRemoteActs{};
    std::array<QAction*, kMaxRecentProject> _recentProjectActs{};
    QAction* _settingsAct{nullptr};
    QAction* _exitAct{nullptr};
    QAction* _keybindsAct{nullptr};
    QAction* _aboutAct{nullptr};
    QAction* _resetViewsAct{nullptr};
    QAction* _showConsoleAct{nullptr};
    QAction* _reportingAct{nullptr};
    QAction* _drawBBoxAct{nullptr};
    QAction* _mirrorCursorAct{nullptr};
    QAction* _surfaceFromSelectionAct{nullptr};
    QAction* _selectionClearAct{nullptr};
    QAction* _teleaAct{nullptr};
    QAction* _importObjAct{nullptr};
    QAction* _projectOpenAct{nullptr};
    QAction* _projectSaveAsAct{nullptr};
    QAction* _dataAddFileDirAct{nullptr};
    QAction* _dataAddRemoteAct{nullptr};
    QAction* _dataAddFromProjectAct{nullptr};
    QAction* _dataRemoveSourceAct{nullptr};
    QAction* _dataRenameSourceAct{nullptr};
    QAction* _dataReloadSourceAct{nullptr};
    int _remoteOpenAuthRetries{0};
    int _remoteScrollAuthRetries{0};

    QPointer<QDialog> _keybindsDialog;
};
