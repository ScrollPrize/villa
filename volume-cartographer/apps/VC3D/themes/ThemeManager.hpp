#pragma once

// Applies VC3D themes to the running application and keeps track of the
// installed ones. One instance serves the whole application (instance()); the
// menu and the agent bridge both go through it.

#include <QByteArray>
#include <QFont>
#include <QList>
#include <QMap>
#include <QObject>
#include <QPair>
#include <QPalette>
#include <QString>
#include <QStringList>
#include <QTimer>

#include "themes/ThemeEngine.hpp"

class QFileSystemWatcher;

namespace vc3d::theme {

class ThemeManager : public QObject
{
    Q_OBJECT

public:
    struct ThemeInfo
    {
        QString id;
        QString name;
        QString author;
        QString description;
        QString source;       // "builtin", "user" or "path"
        QString path;         // theme file; empty for classic
        QString colorScheme;  // "dark" or "light"
        bool active{false};
    };

    struct ImportOutcome
    {
        QString id;
        QString installedPath;
        ThemeProblems problems;
        bool alreadyExists{false};
        bool ok{false};
    };

    // Captures the application's current style, palette, font and style sheet
    // as the classic look to return to.
    explicit ThemeManager(QObject* parent = nullptr);
    ~ThemeManager() override;

    static ThemeManager& instance();

    // Folder overrides (tests); empty uses the standard locations.
    void setUserThemesDirectory(const QString& path);
    QString userThemesDirectory() const;
    void setIconCacheDirectory(const QString& path);
    QString iconCacheDirectory() const;
    // INI file where appearance/theme is saved; empty disables saving.
    void setSettingsFile(const QString& path);
    // The --theme value, used by an initialize() call that names no override.
    void setStartupOverride(const QString& theme);

    // Scans the theme folders and applies the first theme that works of:
    // `overrideTheme` (or the startup override), VC3D_THEME, the saved
    // setting, classic. Each choice that fails is reported on the console and
    // through statusMessage. Overrides are not saved. Emits themeChanged.
    void initialize(const QString& settingsFile, const QString& overrideTheme = {});

    QList<ThemeInfo> themes() const;
    QString currentThemeId() const { return _current.id; }
    const ResolvedTheme& currentTheme() const { return _current; }
    const ThemeRegistry& registry() const { return _registry; }
    // Problems found by the last scan of the theme folders.
    ThemeProblems discoveryProblems() const { return _discoveryProblems; }

    // Applies a theme by id, or loads and applies a theme file or folder.
    // `persist` saves the choice as appearance/theme. Warnings about a theme
    // file are appended to `warnings`.
    bool setTheme(const QString& idOrPath, bool persist, QString* error = nullptr,
                  ThemeProblems* warnings = nullptr);
    // Validates and installs a theme into the user folder, then rescans.
    ImportOutcome importTheme(const QString& location, bool overwrite);
    // Writes `id` (the current theme when empty) as an editable template.
    bool exportTheme(const QString& path, const QString& id = {}, QString* error = nullptr,
                     ThemeProblems* warnings = nullptr);
    // Rescans the theme folders and re-applies the current theme if it changed.
    // If the current theme is gone, classic is applied (not saved).
    ThemeProblems reloadThemes();

    // Style sheet for the main window: classic's gradient menu bar, or empty.
    QString windowStyleSheet() const { return _windowStyleSheet; }

signals:
    // The main window's style sheet for the theme being applied. Emitted
    // during each switch, before the application is restyled when the sheet
    // is empty and after it otherwise, so Qt never restyles every widget
    // against both sheets.
    void windowStyleSheetChanged(const QString& styleSheet);
    void themeChanged(const QString& id);
    void themesChanged();
    // A live reload of the theme being edited failed; the last good version
    // stays applied. Emitted once per distinct error.
    void liveReloadFailed(const QString& message);
    // A one-line note for the main window's status bar.
    void statusMessage(const QString& text, int timeoutMs);

private:
    QStringList pathFolders() const;
    ThemeProblems rescan(ThemeRegistry* registry) const;
    bool addPathTheme(const QString& location, QString* id, QString* error, ThemeProblems* warnings);
    bool apply(const ResolvedTheme& theme, QString* error);
    void applyToApplication(const ResolvedTheme& theme, bool dark, const QString& styleSheet);
    QString render(const ResolvedTheme& theme, bool dark, QString* iconKey, ThemeProblems* problems) const;
    // Re-applies a freshly resolved current theme if its look changed, and
    // otherwise only updates its name, author and watched files.
    void refresh(const ResolvedTheme& theme, const QString& styleSheet);
    bool systemPrefersDark() const;
    bool prefersDark(const ResolvedTheme& theme) const;
    void persist(const QString& value) const;
    void watch(const ResolvedTheme& theme);
    QString watchSignature() const;
    void liveReload();

    ThemeRegistry _registry;
    ThemeProblems _discoveryProblems;
    ResolvedTheme _current;
    QString _currentStyleSheet;  // as rendered, before the startup style sheet
    QString _windowStyleSheet;
    QString _settingsFile;
    QString _userThemesDirectory;
    QString _iconCacheDirectory;
    QString _startupOverride;
    // Themes loaded from a path (--theme, VC3D_THEME, a saved path), by id.
    QMap<QString, QString> _pathThemes;

    // The look to restore for classic.
    QString _startupStyle;
    QFont _startupFont;
    QString _startupStyleSheet;
    QList<QPair<QByteArray, QPalette>> _startupClassPalettes;
    QString _appliedStyle;
    bool _systemDark{false};   // the platform's preference, as last known
    bool _classicDark{false};  // the classic variant on screen
    bool _lookTouched{false};  // a theme has set the palette or color scheme
    bool _paletteTouched{false};
    bool _fontTouched{false};
    bool _colorSchemeSet{false};
    bool _awaitingSystemScheme{false};

    QFileSystemWatcher* _watcher{nullptr};
    QStringList _watchFiles;
    QString _watchSignature;
    QTimer _reloadTimer;
    QString _lastLiveReloadError;
};

}  // namespace vc3d::theme
