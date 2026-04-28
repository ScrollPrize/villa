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
class Volpkg;
struct DataSource;
}

class MenuActionController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kMaxRecentVolpkg = 10;
    static constexpr int kMaxRecentRemote = 10;

    explicit MenuActionController(CWindow* window);

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void removeRecentVolpkgEntry(const QString& path);
    void refreshRecentMenu();
    void openVolpkgAt(const QString& path);
    // Unified dispatcher: routes any path/URL to the right open handler.
    // Extension-sniffs .volpkg.json / .volpkg / scroll URLs / local zarr dirs.
    void openFile(const QString& pathOrUrl);
    bool tryRestoreAutosavedProject();
    void loadEmptyVolumePackage();
    void triggerTeleaInpaint();
    void openRemoteUrl(const QString& url, bool isRetry = false);

    // Realize a Project DataSource into the active VolumePkg. Local sources
    // load synchronously and return the count of items loaded. Remote sources
    // dispatch to a background thread and return 0 — the apply-and-refresh
    // step happens on the main thread when the worker completes.
    int loadSource(const vc::Volpkg& proj, const vc::DataSource& ds);
    int attachDataSource(vc::DataSource ds);
    // Remove anything contributed by a DataSource from the active VolumePkg.
    void unloadSource(const vc::Volpkg& proj, const vc::DataSource& ds);

    // Id-addressed variants, for use from the Project dock widget where
    // the user has already picked a specific source. Public so Qt::connect
    // from CWindow / the dock can bind to them.
    void reloadSourceById(const QString& sourceId);
    void removeSourceById(const QString& sourceId);
    void renameSourceById(const QString& sourceId);
    void setSourceEnabled(const QString& sourceId, bool enabled);
    void editSourceTags(const QString& sourceId);
    void revealSourceLocation(const QString& sourceId);

    // Auth resolver exposed for the unified browser dialog. Returns false if
    // auth resolution failed (and writes a message into err); on success,
    // stores credentials in *out.
    bool resolveAuthForBrowser(const QString& url,
                               vc::cache::HttpAuth* out,
                               QString* err);

private slots:
    void newVolpkg();
    void openVolpkg();
    void openRecentVolpkg();
    void openRecentRemoteVolume();
    void saveVolpkgAs();
    void attachVolume();
    void attachSegments();
    void attachOther();
    void attachFromVolpkg();
    void showSettingsDialog();
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

    void attachRemoteZarrUrl(const QString& url, bool persistEntry = true);
    void openRemoteZarr(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);
    void openRemoteScroll(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);
    void promptAndLoadRemoteSegments(const vc::cache::HttpAuth& auth, const std::string& cachePath);
    // Non-prompting variant: attaches the supplied segments URL and triggers
    // the same discovery + surface-caching pipeline as the prompting path.
    // On success, persists (segUrl, zarrUrl) in QSettings so auto-open of
    // the same remote zarr can skip the prompt next time. Pass the current
    // remote volume's URL as zarrUrl; empty disables persistence.
    void loadRemoteSegmentsWithUrl(const QString& segUrl,
                                   const vc::cache::HttpAuth& auth,
                                   const std::string& cachePath,
                                   const QString& zarrUrl);
    bool tryResolveRemoteAuth(const QString& url,
                              vc::cache::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Helpers used by loadSource for the local and remote branches.
    int loadSourceLocal(const vc::Volpkg& proj, const vc::DataSource& ds);
    void loadSourceRemoteAsync(const vc::Volpkg& proj, const vc::DataSource& ds);
    QString remoteCacheDirectory() const;

    CWindow* _window{nullptr};
    bool _inAutosaveRestore{false};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};
    QMenu* _recentRemoteMenu{nullptr};

    QAction* _newVolpkgAct{nullptr};
    QAction* _openAct{nullptr};
    QAction* _saveAsAct{nullptr};
    QAction* _attachVolumeAct{nullptr};
    QAction* _attachSegmentsAct{nullptr};
    QAction* _attachOtherAct{nullptr};
    QAction* _attachFromVolpkgAct{nullptr};
    std::array<QAction*, kMaxRecentVolpkg> _recentActs{};
    std::array<QAction*, kMaxRecentRemote> _recentRemoteActs{};
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
    int _remoteOpenAuthRetries{0};
    int _remoteScrollAuthRetries{0};

    QPointer<QDialog> _keybindsDialog;
};
