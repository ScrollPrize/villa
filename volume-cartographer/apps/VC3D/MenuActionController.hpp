#pragma once

#include <QObject>
#include <QPointer>
#include <array>
#include <memory>
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

    explicit MenuActionController(CWindow* window);

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void removeRecentVolpkgEntry(const QString& path);
    void refreshRecentMenu();
    void openVolpkgAt(const QString& path);
    void loadAttachedRemoteVolumesForCurrentPackage();

private slots:
    void newProject();
    void saveProjectAs();
    void attachVolume();
    void attachSegments();
    void attachNormalGrid();
    void detachEntry();
    void setOutputSegments();
    void convertLegacyVolpkg();
    void openVolpkg();
    void openRecentVolpkg();
    void openLocalZarr();
    void attachRemoteZarr();
    void showSettingsDialog();
    void showAboutDialog();
    void showKeybindings();
    void resetSegmentationViews();
    void toggleConsoleOutput();
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
    void ensureRecentActions();

    QStringList loadRecentRemoteUrls() const;
    void saveRecentRemoteUrls(const QStringList& urls);
    void updateRecentRemoteList(const QString& url);
    void attachRemoteZarrUrl(const QString& url, bool persistEntry = true);
    bool tryResolveRemoteAuth(const QString& url,
                              vc::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Runs vc_volpkg_convert against `inputLocation` (legacy folder or remote URL),
    // prompts the user for an output .volpkg.json, and returns the written path
    // via `convertedOut` on success.
    bool runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut);
    QString remoteCacheDirectory() const;
    QString remoteVolumeRegistryPath() const;
    void persistAttachedRemoteVolume(const QString& url, const std::shared_ptr<Volume>& volume);
    QString promptLocation(const QString& title,
                           const QString& hint,
                           const QString& defaultDir,
                           const QStringList& localFilters,
                           bool acceptFiles,
                           bool acceptDirs);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _transformsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};

    QAction* _newProjectAct{nullptr};
    QAction* _saveProjectAsAct{nullptr};
    QAction* _attachVolumeAct{nullptr};
    QAction* _attachSegmentsAct{nullptr};
    QAction* _attachNormalGridAct{nullptr};
    QAction* _detachEntryAct{nullptr};
    QAction* _setOutputSegmentsAct{nullptr};
    QAction* _convertLegacyAct{nullptr};
    QAction* _openAct{nullptr};
    QAction* _openLocalZarrAct{nullptr};
    QAction* _attachRemoteZarrAct{nullptr};
    std::array<QAction*, kMaxRecentVolpkg> _recentActs{};
    QAction* _settingsAct{nullptr};
    QAction* _exitAct{nullptr};
    QAction* _keybindsAct{nullptr};
    QAction* _aboutAct{nullptr};
    QAction* _resetViewsAct{nullptr};
    QAction* _showConsoleAct{nullptr};
    QAction* _drawBBoxAct{nullptr};
    QAction* _mirrorCursorAct{nullptr};
    QAction* _surfaceFromSelectionAct{nullptr};
    QAction* _selectionClearAct{nullptr};
    QAction* _importObjAct{nullptr};
    QAction* _rotateSurfaceAct{nullptr};

    QPointer<QDialog> _keybindsDialog;
};
