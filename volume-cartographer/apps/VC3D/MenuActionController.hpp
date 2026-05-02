#pragma once

#include <QObject>
#include <QPointer>
#include <array>
#include <string>

#include "vc/core/util/RemoteAuth.hpp"

class QAction;
class QDialog;
class QMenu;
class QMenuBar;
class CWindow;
class Volume;

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
    void loadAttachedRemoteVolumesForCurrentPackage();
    void openRemoteUrl(const QString& url, bool isRetry = false);

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
    void showAboutDialog();
    void showKeybindings();
    void resetSegmentationViews();
    void toggleConsoleOutput();
    void generateReviewReport();
    void toggleDrawBBox(bool enabled);
    void toggleCursorMirroring(bool enabled);
    void surfaceFromSelection();
    void clearSelection();
    void importObjAsPatch();
    void beginRotateSurfaceTransform();
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
    void openRemoteZarr(const std::string& httpsUrl, const vc::HttpAuth& auth, const std::string& cachePath);
    void openRemoteScroll(const std::string& httpsUrl, const vc::HttpAuth& auth, const std::string& cachePath);
    void promptAndLoadRemoteSegments(const vc::HttpAuth& auth, const std::string& cachePath);
    // Non-prompting variant: attaches the supplied segments URL and triggers
    // the same discovery + surface-caching pipeline as the prompting path.
    // On success, persists (segUrl, zarrUrl) in QSettings so auto-open of
    // the same remote zarr can skip the prompt next time. Pass the current
    // remote volume's URL as zarrUrl; empty disables persistence.
    void loadRemoteSegmentsWithUrl(const QString& segUrl,
                                   const vc::HttpAuth& auth,
                                   const std::string& cachePath,
                                   const QString& zarrUrl);
    bool tryResolveRemoteAuth(const QString& url,
                              vc::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    QString remoteCacheDirectory() const;
    QString remoteVolumeRegistryPath() const;
    void persistAttachedRemoteVolume(const QString& url, const std::shared_ptr<Volume>& volume);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _transformsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};
    QMenu* _recentRemoteMenu{nullptr};

    QAction* _openAct{nullptr};
    QAction* _openLocalZarrAct{nullptr};
    QAction* _openRemoteAct{nullptr};
    QAction* _attachRemoteZarrAct{nullptr};
    QAction* _attachRemoteSegmentsAct{nullptr};
    QAction* _browseS3Act{nullptr};
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
    QAction* _importObjAct{nullptr};
    QAction* _rotateSurfaceAct{nullptr};
    int _remoteOpenAuthRetries{0};
    int _remoteScrollAuthRetries{0};

    QPointer<QDialog> _keybindsDialog;
};
