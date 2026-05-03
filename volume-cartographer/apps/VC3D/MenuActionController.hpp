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
    void triggerTeleaInpaint();

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
    void ensureRecentActions();

    bool tryResolveRemoteAuth(const QString& url,
                              vc::cache::HttpAuth* authOut,
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Runs vc_volpkg_convert against `inputLocation` (legacy folder or remote URL),
    // prompts the user for an output .volpkg.json, and returns the written path
    // via `convertedOut` on success.
    bool runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut);
    QString remoteCacheDirectory() const;
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
    std::array<QAction*, kMaxRecentVolpkg> _recentActs{};
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
