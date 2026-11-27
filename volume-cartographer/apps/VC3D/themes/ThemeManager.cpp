#include "themes/ThemeManager.hpp"

#include "VCSettings.hpp"
#include "themes/ClassicTheme.hpp"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QFontDatabase>
#include <QGuiApplication>
#include <QPointer>
#include <QSaveFile>
#include <QSettings>
#include <QStandardPaths>
#include <QStyle>
#include <QStyleHints>

namespace vc3d::theme {
namespace {

constexpr int kLiveReloadDelayMs = 200;
constexpr int kStatusMessageMs = 15000;
#if QT_VERSION >= QT_VERSION_CHECK(6, 8, 0)
// How long after a theme stops forcing its color scheme the platform's next
// report is taken as the system preference.
constexpr int kSchemeSettleMs = 2000;
#endif

// Widget classes that platform themes (macOS, Windows, KDE) give palettes of
// their own. Setting the application palette clears those, so classic puts
// them back when it returns to the platform palette.
constexpr const char* kPlatformPaletteClasses[] = {
    "QToolButton", "QAbstractButton", "QCheckBox", "QRadioButton", "QHeaderView",
    "QAbstractItemView", "QMessageBoxLabel", "QTabBar", "QLabel", "QGroupBox",
    "QMenu", "QMenuBar", "QTextEdit", "QTextControl", "QLineEdit",
};

// The application style's name. While a style sheet is set, the style is a
// style sheet wrapper whose own name is empty; the real style is its child.
QString applicationStyleName()
{
    const QStyle* style = QApplication::style();
    if (!style)
        return {};
    if (style->name().isEmpty()) {
        if (const auto* inner = style->findChild<QStyle*>())
            return inner->name();
    }
    return style->name();
}

QString firstError(const ThemeProblems& problems)
{
    ThemeProblems errors;
    for (const auto& problem : problems) {
        if (!problem.warning)
            errors.append(problem);
    }
    return formatProblems(errors);
}

QString firstLine(const QString& text)
{
    return text.section(QLatin1Char('\n'), 0, 0);
}

}  // namespace

ThemeManager::ThemeManager(QObject* parent)
    : QObject(parent)
    , _watcher(new QFileSystemWatcher(this))
{
    _startupStyle = applicationStyleName();
    _appliedStyle = _startupStyle;
    _startupFont = QApplication::font();
    _startupStyleSheet = qApp->styleSheet();
    for (const char* className : kPlatformPaletteClasses) {
        const QPalette palette = QApplication::palette(className);
        if (palette != QApplication::palette())
            _startupClassPalettes.append({QByteArray(className), palette});
    }
    _systemDark = classicPrefersDark();
    _classicDark = _systemDark;
    _windowStyleSheet = classicWindowStyleSheet(_systemDark);
    ThemeProblems ignored;
    if (const auto classic = _registry.resolve(QString::fromLatin1(kClassicId), &ignored))
        _current = *classic;

#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    connect(QGuiApplication::styleHints(), &QStyleHints::colorSchemeChanged, this,
            [this](Qt::ColorScheme scheme) {
                if (_colorSchemeSet || scheme == Qt::ColorScheme::Unknown)
                    return;
                _systemDark = scheme == Qt::ColorScheme::Dark;
                // Right after a theme stops forcing its scheme, the platform
                // reports the real one, possibly while the switch is still
                // under way. Classic then redoes its look once if it guessed
                // wrong; otherwise it keeps the look it started with, as it
                // always has.
                if (_awaitingSystemScheme) {
                    _awaitingSystemScheme = false;
                    QTimer::singleShot(0, this, [this] {
                        if (_current.isClassic() && _current.colorScheme.isEmpty() &&
                            _systemDark != _classicDark) {
                            QString error;
                            apply(_current, &error);
                        }
                    });
                }
            });
#endif

    _reloadTimer.setSingleShot(true);
    _reloadTimer.setInterval(kLiveReloadDelayMs);
    connect(&_reloadTimer, &QTimer::timeout, this, &ThemeManager::liveReload);
    connect(_watcher, &QFileSystemWatcher::fileChanged, this, [this] { _reloadTimer.start(); });
    // The folders are watched to notice editors that replace a file; any
    // other change in them is ignored.
    connect(_watcher, &QFileSystemWatcher::directoryChanged, this, [this] {
        if (watchSignature() != _watchSignature)
            _reloadTimer.start();
    });
}

ThemeManager::~ThemeManager() = default;

ThemeManager& ThemeManager::instance()
{
    static QPointer<ThemeManager> manager;
    if (!manager)
        manager = new ThemeManager(QCoreApplication::instance());
    return *manager;
}

void ThemeManager::setUserThemesDirectory(const QString& path)
{
    _userThemesDirectory = path;
}

QString ThemeManager::userThemesDirectory() const
{
    if (!_userThemesDirectory.isEmpty())
        return _userThemesDirectory;
    return QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation))
        .filePath(QStringLiteral("themes"));
}

void ThemeManager::setIconCacheDirectory(const QString& path)
{
    _iconCacheDirectory = path;
}

QString ThemeManager::iconCacheDirectory() const
{
    if (!_iconCacheDirectory.isEmpty())
        return _iconCacheDirectory;
    return QDir(QStandardPaths::writableLocation(QStandardPaths::CacheLocation))
        .filePath(QStringLiteral("themes"));
}

void ThemeManager::setSettingsFile(const QString& path)
{
    _settingsFile = path;
}

void ThemeManager::setStartupOverride(const QString& theme)
{
    _startupOverride = theme.trimmed();
}

QStringList ThemeManager::pathFolders() const
{
    return qEnvironmentVariable("VC3D_THEME_PATH").split(QDir::listSeparator(), Qt::SkipEmptyParts);
}

ThemeProblems ThemeManager::rescan(ThemeRegistry* registry) const
{
    ThemeProblems problems = registry->scan(userThemesDirectory(), pathFolders());
    // Themes loaded by path get the same checks as the scanned ones; a broken
    // one is left out until it is fixed.
    for (auto it = _pathThemes.begin(); it != _pathThemes.end(); ++it) {
        ThemeProblems themeProblems;
        auto definition = loadTheme(it.value(), ThemeSource::Path, &themeProblems);
        if (definition && !isBuiltinThemeId(definition->id)) {
            const auto resolved = registry->resolve(*definition, &themeProblems);
            if (resolved && checkRendering(*resolved, &themeProblems))
                registry->add(std::move(*definition), &themeProblems);
        }
        problems.append(themeProblems);
    }
    return problems;
}

void ThemeManager::initialize(const QString& settingsFile, const QString& overrideTheme)
{
    _settingsFile = settingsFile;
    _discoveryProblems = rescan(&_registry);
    for (const auto& problem : _discoveryProblems)
        qWarning("VC3D theme: %s", qPrintable(formatProblems({problem})));

    QList<QPair<QString, QString>> choices;
    const QString override = overrideTheme.trimmed().isEmpty() ? _startupOverride : overrideTheme.trimmed();
    if (!override.isEmpty())
        choices.append({QStringLiteral("--theme"), override});
    const QString environment = qEnvironmentVariable("VC3D_THEME").trimmed();
    if (!environment.isEmpty())
        choices.append({QStringLiteral("VC3D_THEME"), environment});
    if (!_settingsFile.isEmpty()) {
        QSettings settings(_settingsFile, QSettings::IniFormat);
        const QString saved = settings.value(vc3d::settings::appearance::THEME,
                                             vc3d::settings::appearance::THEME_DEFAULT)
                                  .toString()
                                  .trimmed();
        if (!saved.isEmpty())
            choices.append({QStringLiteral("saved theme"), saved});
    }
    choices.append({QStringLiteral("classic theme"), QString::fromLatin1(kClassicId)});

    QStringList failed;
    QString firstFailure;
    for (const auto& [origin, requested] : choices) {
        QString error;
        if (setTheme(requested, false, &error)) {
            // A launch from Finder or a desktop menu has no console to show the warnings.
            if (!failed.isEmpty()) {
                emit statusMessage(tr("Theme %1 could not be used (%2); showing %3")
                                       .arg(failed.join(QStringLiteral(", ")), firstLine(firstFailure),
                                            _current.name),
                                   kStatusMessageMs);
            }
            return;
        }
        qWarning("VC3D theme: cannot use %s \"%s\": %s", qPrintable(origin), qPrintable(requested),
                 qPrintable(error));
        failed << QStringLiteral("\"%1\"").arg(requested);
        if (firstFailure.isEmpty())
            firstFailure = error;
    }
}

QList<ThemeManager::ThemeInfo> ThemeManager::themes() const
{
    QList<ThemeInfo> list;
    for (const ThemeDefinition* theme : _registry.themes()) {
        ThemeInfo info;
        info.id = theme->id;
        info.name = theme->name;
        info.author = theme->author;
        info.description = theme->description;
        info.source = themeSourceName(theme->source);
        info.path = theme->filePath;
        info.active = theme->id == _current.id;
        ThemeProblems ignored;
        const auto resolved = _registry.resolve(*theme, &ignored);
        if (resolved && !resolved->colorScheme.isEmpty())
            info.colorScheme = resolved->colorScheme;
        else
            info.colorScheme = systemPrefersDark() ? QStringLiteral("dark") : QStringLiteral("light");
        list.append(info);
    }
    return list;
}

bool ThemeManager::addPathTheme(const QString& location, QString* id, QString* error,
                                ThemeProblems* warnings)
{
    ThemeProblems problems;
    auto definition = loadTheme(location, ThemeSource::Path, &problems);
    if (definition && isBuiltinThemeId(definition->id)) {
        problems.append({definition->filePath,
                         QStringLiteral("the id \"%1\" belongs to a built-in theme").arg(definition->id)});
    } else if (definition) {
        // Checked before it is listed, so a broken file never shows in the menu.
        const auto resolved = _registry.resolve(*definition, &problems);
        if (resolved)
            checkRendering(*resolved, &problems);
    }
    for (const auto& problem : problems) {
        if (!problem.warning)
            continue;
        qWarning("VC3D theme: %s", qPrintable(formatProblems({problem})));
        if (warnings)
            warnings->append(problem);
    }
    if (!definition || hasErrors(problems)) {
        *error = firstError(problems);
        return false;
    }
    if (const auto* existing = _registry.find(definition->id);
        existing && existing->filePath != definition->filePath) {
        qWarning("VC3D theme: %s replaces the installed theme \"%s\" for this session",
                 qPrintable(definition->filePath), qPrintable(definition->id));
    }
    *id = definition->id;
    const QString absolute = QFileInfo(location).absoluteFilePath();
    _registry.add(std::move(*definition), &problems);
    _pathThemes.insert(*id, absolute);
    return true;
}

bool ThemeManager::setTheme(const QString& idOrPath, bool persist, QString* error,
                            ThemeProblems* warnings)
{
    QString scratch;
    QString* message = error ? error : &scratch;
    const QString requested = idOrPath.trimmed();
    QString id = normalizeThemeId(requested);
    QString saved = id;

    const bool knownId = !looksLikeThemePath(requested) && _registry.contains(id);
    if (!knownId) {
        if (!QFileInfo::exists(requested)) {
            *message = QStringLiteral("unknown theme \"%1\"").arg(requested);
            return false;
        }
        if (!addPathTheme(requested, &id, message, warnings))
            return false;
        emit themesChanged();
    }
    // A theme loaded from a path is saved as that path, so it loads again
    // on the next start.
    if (_pathThemes.contains(id))
        saved = _pathThemes.value(id);

    ThemeProblems problems;
    const auto resolved = _registry.resolve(id, &problems);
    if (!resolved || hasErrors(problems)) {
        *message = firstError(problems);
        return false;
    }
    if (!apply(*resolved, message))
        return false;
    if (persist)
        this->persist(saved);
    return true;
}

QString ThemeManager::render(const ResolvedTheme& theme, bool dark, QString* iconKey,
                             ThemeProblems* problems) const
{
    if (theme.usesFusion()) {
        *iconKey = themeIconKey(theme);
        QString iconError;
        if (ensureThemeIcons(theme.tokens, iconCacheDirectory(), *iconKey, &iconError).isEmpty()) {
            problems->append({iconCacheDirectory(), iconError});
            return {};
        }
    }
    return renderStyleSheet(theme, *iconKey, dark ? QStringLiteral("dark") : QStringLiteral("light"),
                            problems);
}

// Until a theme sets the palette or the color scheme, Qt's own report is
// current. Afterwards it can describe the theme rather than the system, so
// the last report from before (or a later platform notice) stands in.
bool ThemeManager::systemPrefersDark() const
{
    return _lookTouched ? _systemDark : classicPrefersDark();
}

bool ThemeManager::prefersDark(const ResolvedTheme& theme) const
{
    if (!theme.colorScheme.isEmpty())
        return theme.colorScheme == QLatin1String("dark");
    return systemPrefersDark();
}

bool ThemeManager::apply(const ResolvedTheme& theme, QString* error)
{
    const bool dark = prefersDark(theme);
    ThemeProblems problems;
    QString iconKey;
    const QString styleSheet = render(theme, dark, &iconKey, &problems);
    if (hasErrors(problems)) {
        *error = firstError(problems);
        return false;
    }
    if (theme.usesFusion())
        QDir::setSearchPaths(QLatin1String(kIconSearchPrefix), {iconCacheDirectory()});

    applyToApplication(theme, dark, styleSheet);
    _current = theme;
    _lastLiveReloadError.clear();
    watch(theme);
    emit themeChanged(theme.id);
    return true;
}

void ThemeManager::applyToApplication(const ResolvedTheme& theme, bool dark, const QString& styleSheet)
{
    if (!_lookTouched)
        _systemDark = classicPrefersDark();

    // While the main window has a style sheet of its own, every restyle below
    // also matches each widget against it. Leaving Classic, the window drops
    // its sheet first; entering Classic, it takes its sheet last. With a
    // project open this halves the switch from Classic to Modern Dark.
    const QString windowSheet =
        theme.isClassic() && !theme.replaced ? classicWindowStyleSheet(dark) : QString();
    _windowStyleSheet = windowSheet;
    if (windowSheet.isEmpty())
        emit windowStyleSheetChanged(windowSheet);

#if QT_VERSION >= QT_VERSION_CHECK(6, 8, 0)
    // The flags change first: the platform may report the new scheme before
    // these calls return.
    if (theme.usesFusion() && !theme.colorScheme.isEmpty()) {
        _colorSchemeSet = true;
        _lookTouched = true;
        QGuiApplication::styleHints()->setColorScheme(theme.colorScheme == QLatin1String("dark")
                                                          ? Qt::ColorScheme::Dark
                                                          : Qt::ColorScheme::Light);
    } else if (_colorSchemeSet) {
        _colorSchemeSet = false;
        _awaitingSystemScheme = true;
        QTimer::singleShot(kSchemeSettleMs, this, [this] { _awaitingSystemScheme = false; });
        QGuiApplication::styleHints()->unsetColorScheme();
    }
#endif

    const QString style = theme.usesFusion() ? QStringLiteral("Fusion") : _startupStyle;
    if (!style.isEmpty() && _appliedStyle.compare(style, Qt::CaseInsensitive) != 0) {
        if (QApplication::setStyle(style))
            _appliedStyle = style;
    }

    QString styleSheetText = styleSheet;
    if (theme.usesFusion()) {
        QApplication::setPalette(theme.palette.applyTo(QPalette()));
        _paletteTouched = true;
        _lookTouched = true;
    } else {
        // Back to the style's own palette and the platform's class palettes,
        // then classic's dark detection.
        const bool wasTouched = _paletteTouched;
        if (_paletteTouched) {
            QApplication::setPalette(QPalette());
            _paletteTouched = false;
        }
        if (dark)
            applyClassicDarkPalette();
        if (!theme.palette.isEmpty())
            QApplication::setPalette(theme.palette.applyTo(QApplication::palette()));
        if (dark || !theme.palette.isEmpty()) {
            _paletteTouched = true;
            _lookTouched = true;
        } else if (wasTouched) {
            for (const auto& [className, palette] : _startupClassPalettes)
                QApplication::setPalette(palette, className.constData());
        }
        _classicDark = dark;
        // A style sheet given with Qt's -stylesheet option stays under classic.
        if (!_startupStyleSheet.isEmpty()) {
            styleSheetText = styleSheet.isEmpty() ? _startupStyleSheet
                                                  : _startupStyleSheet + QStringLiteral("\n\n") + styleSheet;
        }
    }

    QFont font = _startupFont;
    bool customFont = false;
    if (!theme.font.family.isEmpty() &&
        QFontDatabase::families().contains(theme.font.family, Qt::CaseInsensitive)) {
        font.setFamily(theme.font.family);
        customFont = true;
    }
    if (theme.font.pointSize > 0.0) {
        font.setPointSizeF(theme.font.pointSize);
        customFont = true;
    }
    if (customFont || _fontTouched) {
        QApplication::setFont(font);
        _fontTouched = customFont;
    }

    if (qApp->styleSheet() != styleSheetText)
        qApp->setStyleSheet(styleSheetText);
    _currentStyleSheet = styleSheet;
    if (!windowSheet.isEmpty())
        emit windowStyleSheetChanged(windowSheet);
}

void ThemeManager::persist(const QString& value) const
{
    if (_settingsFile.isEmpty())
        return;
    QSettings settings(_settingsFile, QSettings::IniFormat);
    settings.setValue(vc3d::settings::appearance::THEME, value);
    settings.sync();
}

ThemeManager::ImportOutcome ThemeManager::importTheme(const QString& location, bool overwrite)
{
    const ImportResult result =
        vc3d::theme::importTheme(location, userThemesDirectory(), overwrite, _registry);
    ImportOutcome outcome{result.id, result.installedPath, result.problems, result.alreadyExists, result.ok};
    if (!result.ok)
        return outcome;
    _pathThemes.remove(result.id);
    reloadThemes();
    return outcome;
}

bool ThemeManager::exportTheme(const QString& path, const QString& id, QString* error,
                               ThemeProblems* warnings)
{
    QString scratch;
    QString* message = error ? error : &scratch;
    ThemeProblems problems;
    std::optional<ResolvedTheme> theme;
    if (id.isEmpty())
        theme = _current;
    else
        theme = _registry.resolve(normalizeThemeId(id), &problems);
    if (!theme) {
        *message = firstError(problems);
        return false;
    }
    const QByteArray json = exportThemeJson(*theme, _registry, &problems);
    if (json.isEmpty()) {
        *message = firstError(problems);
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(json) != json.size() || !file.commit()) {
        *message = QStringLiteral("cannot write %1: %2").arg(path, file.errorString());
        return false;
    }
    if (warnings) {
        for (const auto& problem : problems) {
            if (problem.warning)
                warnings->append(problem);
        }
    }
    return true;
}

ThemeProblems ThemeManager::reloadThemes()
{
    _discoveryProblems = rescan(&_registry);
    ThemeProblems problems;
    const auto resolved = _registry.resolve(_current.id, &problems);
    QString styleSheet;
    if (resolved && !hasErrors(problems)) {
        QString iconKey;
        styleSheet = render(*resolved, prefersDark(*resolved), &iconKey, &problems);
    }
    if (resolved && !hasErrors(problems)) {
        refresh(*resolved, styleSheet);
    } else {
        const QString note =
            QStringLiteral("the active theme \"%1\" is gone or broken; switched to Classic").arg(_current.id);
        qWarning("VC3D theme: %s", qPrintable(note));
        _discoveryProblems.append({_current.filePath, note, true});
        setTheme(QString::fromLatin1(kClassicId), false);
    }
    emit themesChanged();
    return _discoveryProblems;
}

void ThemeManager::refresh(const ResolvedTheme& theme, const QString& styleSheet)
{
    if (styleSheet != _currentStyleSheet || !(theme.palette == _current.palette) ||
        !(theme.font == _current.font) || theme.colorScheme != _current.colorScheme) {
        QString error;
        apply(theme, &error);
        return;
    }
    _current = theme;
    watch(theme);
}

void ThemeManager::watch(const ResolvedTheme& theme)
{
    const QStringList watched = _watcher->files() + _watcher->directories();
    if (!watched.isEmpty())
        _watcher->removePaths(watched);
    _watchFiles = theme.watchFiles;
    QStringList folders;
    for (const QString& file : _watchFiles) {
        if (QFileInfo::exists(file))
            _watcher->addPath(file);
        folders << QFileInfo(file).absolutePath();
    }
    folders.removeDuplicates();
    if (!folders.isEmpty())
        _watcher->addPaths(folders);
    _watchSignature = watchSignature();
}

// Size, modification time and existence of every watched file, to tell
// whether a change in a watched folder touched one of them.
QString ThemeManager::watchSignature() const
{
    QStringList parts;
    for (const QString& file : _watchFiles) {
        const QFileInfo info(file);
        parts << QStringLiteral("%1|%2|%3|%4")
                     .arg(file)
                     .arg(info.exists() ? 1 : 0)
                     .arg(info.size())
                     .arg(info.lastModified().toMSecsSinceEpoch());
    }
    return parts.join(QLatin1Char('\n'));
}

void ThemeManager::liveReload()
{
    if (_watchFiles.isEmpty())
        return;
    // Editors that save by replacing the file drop it from the watcher.
    const QStringList watchedFiles = _watcher->files();
    for (const QString& file : _watchFiles) {
        if (!watchedFiles.contains(file) && QFileInfo::exists(file))
            _watcher->addPath(file);
    }
    _watchSignature = watchSignature();

    ThemeRegistry fresh;
    ThemeProblems scanProblems = rescan(&fresh);
    ThemeProblems problems;
    const auto resolved = fresh.resolve(_current.id, &problems);
    QString styleSheet;
    QString iconKey;
    if (resolved)
        styleSheet = render(*resolved, prefersDark(*resolved), &iconKey, &problems);
    if (!resolved || hasErrors(problems)) {
        // The broken file is left out of the fresh scan; its own problems say why.
        ThemeProblems reported;
        for (const auto& problem : scanProblems + problems) {
            if (!problem.warning && _watchFiles.contains(problem.file))
                reported.append(problem);
        }
        const QString message = reported.isEmpty() ? firstError(problems) : formatProblems(reported);
        if (message != _lastLiveReloadError) {
            _lastLiveReloadError = message;
            qWarning("VC3D theme: keeping the last good version of \"%s\":\n%s",
                     qPrintable(_current.id), qPrintable(message));
            emit liveReloadFailed(message);
            emit statusMessage(tr("Theme not reloaded: %1").arg(firstLine(message)), kStatusMessageMs);
        }
        return;
    }

    _registry = fresh;
    _discoveryProblems = scanProblems;
    _lastLiveReloadError.clear();
    // Warnings about the files being edited, such as a misspelled token,
    // would otherwise go unseen.
    ThemeProblems warnings;
    for (const auto& problem : scanProblems + problems) {
        if (problem.warning && _watchFiles.contains(problem.file))
            warnings.append(problem);
    }
    refresh(*resolved, styleSheet);
    if (!warnings.isEmpty()) {
        qWarning("VC3D theme: \"%s\" reloaded with warnings:\n%s", qPrintable(_current.id),
                 qPrintable(formatProblems(warnings)));
        emit statusMessage(tr("Theme reloaded with %n warning(s): %1", nullptr, static_cast<int>(warnings.size()))
                               .arg(warnings.first().message),
                           kStatusMessageMs);
    }
    emit themesChanged();
}

}  // namespace vc3d::theme
