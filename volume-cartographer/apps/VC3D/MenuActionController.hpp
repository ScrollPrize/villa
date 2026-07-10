#pragma once

#include <QObject>
#include <QFutureWatcher>
#include <QPointer>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "vc/core/util/RemoteAuth.hpp"
#include "OpenDataVolumePrefill.hpp"

class QAction;
class QDialog;
class QMenu;
class QMenuBar;
class CWindow;
class Volume;

namespace vc3d::opendata {
struct OpenDataSample;
class OpenDataCatalogWindow;
}

class MenuActionController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kMaxRecentVolpkg = 10;

    explicit MenuActionController(CWindow* window);
    ~MenuActionController() override;

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void removeRecentVolpkgEntry(const QString& path);
    void refreshRecentMenu();
    void openVolpkgAt(const QString& path);
    void showOpenDataCatalog();
    bool isOpenDataCatalogVisible() const;

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

signals:
    // Emitted when the user picks Actions -> Merge tifxyz... CWindow
    // wires this to SegmentationCommandHandler::onMergeTifxyz with an
    // empty seed list so the dialog opens with an empty grid.
    void mergeTifxyzFromMenuRequested();
    // Emitted when the user picks Actions -> Patch tifxyz... CWindow
    // wires this to SegmentationCommandHandler::onMergePatch with an
    // empty seed list so the dialog opens with empty combo boxes.
    void mergePatchFromMenuRequested();
    void openDataCatalogVisibilityChanged(bool visible);

private:
    QStringList loadRecentPaths() const;
    void saveRecentPaths(const QStringList& paths);
    void ensureRecentActions();

    QStringList loadRecentRemoteUrls() const;
    void saveRecentRemoteUrls(const QStringList& urls);
    void updateRecentRemoteList(const QString& url);
    void attachRemoteZarrUrl(const QString& url);
    bool openOpenDataSample(const vc3d::opendata::OpenDataSample& sample);
    void startOpenDataVolumePrefill(const std::shared_ptr<Volume>& volume);
    void cancelOpenDataVolumePrefills();
    bool tryResolveRemoteAuth(const QString& url,
                              vc::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Runs vc_volpkg_convert against `inputLocation` (legacy folder or remote URL),
    // prompts the user for an output .volpkg.json, and returns the written path
    // via `convertedOut` on success.
    bool runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut);
    QString remoteCacheDirectory(bool allowPrompt);
    QString configuredRemoteCacheDirectory() const;
    QString suggestedRemoteCacheDirectory() const;
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
    QAction* _attachRemoteZarrAct{nullptr};
    QAction* _openDataCatalogAct{nullptr};
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
    QAction* _mergeTifxyzAct{nullptr};
    QAction* _mergePatchAct{nullptr};
    QAction* _recalculateFiberScoresAct{nullptr};

    QPointer<QDialog> _keybindsDialog;
    QPointer<vc3d::opendata::OpenDataCatalogWindow> _openDataCatalogDialog;
    std::vector<QFutureWatcher<vc3d::opendata::OpenDataVolumePrefillResult>*> _openDataPrefillWatchers;
    std::vector<std::shared_ptr<std::atomic<bool>>> _openDataPrefillCancelFlags;
    std::shared_ptr<std::atomic<bool>> _openDataPrefillCancelFlag;
};
