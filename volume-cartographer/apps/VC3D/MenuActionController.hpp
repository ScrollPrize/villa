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
    void triggerTeleaInpaint();

private slots:
    void openVolpkg();
    void openRecentVolpkg();
    void openLocalZarr();
    void openRemoteVolume();
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
    void openRemoteUrl(const QString& url);
    void openRemoteZarr(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);
    void openRemoteScroll(const std::string& httpsUrl, const vc::cache::HttpAuth& auth, const std::string& cachePath);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};
    QMenu* _recentRemoteMenu{nullptr};

    QAction* _openAct{nullptr};
    QAction* _openLocalZarrAct{nullptr};
    QAction* _openRemoteAct{nullptr};
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

    QPointer<QDialog> _keybindsDialog;
};
