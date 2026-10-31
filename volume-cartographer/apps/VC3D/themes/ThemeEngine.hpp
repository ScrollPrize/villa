#pragma once

// VC3D themes: parsing, validation, base resolution and rendering of
// .vc3dtheme files. Nothing here touches the running application; applying a
// theme is ThemeManager's job.
//
// A theme is a JSON file (or a folder holding theme.vc3dtheme and its assets)
// that names a base theme and overrides any of its tokens, palette roles,
// font and extra style sheet. Three themes are built in: "classic" (VC3D's
// original native look, the default), and "modern-dark" / "modern-light",
// which render the shared style sheet template :/themes/theme.qss.in from
// their tokens. The format is documented in volume-cartographer/docs/vc3d_themes.md.

#include <QByteArray>
#include <QColor>
#include <QJsonObject>
#include <QList>
#include <QMap>
#include <QPalette>
#include <QString>
#include <QStringList>

#include <optional>
#include <vector>

namespace vc3d::theme {

inline constexpr int kFormatVersion = 1;
inline constexpr int kMaxBaseDepth = 8;
// Limits that keep a broken or hostile theme from stalling VC3D: the size of
// a theme file or extra style sheet, of an image a style sheet loads, of the
// rendered style sheet, of one token value, of the name and author, and of
// the description, and the number of problems reported per file.
inline constexpr qint64 kMaxThemeFileBytes = 64 * 1024;
inline constexpr qint64 kMaxThemeImageBytes = 4 * 1024 * 1024;
inline constexpr int kMaxStyleSheetChars = 256 * 1024;
inline constexpr int kMaxTokenValueChars = 256;
inline constexpr int kMaxNameChars = 80;
inline constexpr int kMaxDescriptionChars = 500;
inline constexpr int kMaxProblemsPerFile = 20;
inline constexpr auto kClassicId = "classic";
inline constexpr auto kModernDarkId = "modern-dark";
inline constexpr auto kModernLightId = "modern-light";
inline constexpr auto kThemeFileSuffix = ".vc3dtheme";
inline constexpr auto kFolderThemeFileName = "theme.vc3dtheme";
inline constexpr auto kTemplateResource = ":/themes/theme.qss.in";
inline constexpr auto kBuiltinResourceDir = ":/themes";
// Prefix registered with QDir::setSearchPaths for the rendered theme icons, so
// the template can say url(vc3dtheme:@name@/check.png).
inline constexpr auto kIconSearchPrefix = "vc3dtheme";
// Used for @mono@ when the resolved tokens do not define "mono".
inline constexpr auto kDefaultMonospaceStack =
    "\"JetBrains Mono\", \"SF Mono\", Menlo, Consolas, \"DejaVu Sans Mono\", monospace";

// A problem found while loading, resolving, rendering or importing a theme.
// Warnings are reported but do not stop the theme from being used.
struct ThemeProblem
{
    QString file;
    QString message;
    bool warning{false};
};
using ThemeProblems = QList<ThemeProblem>;

bool hasErrors(const ThemeProblems& problems);
// One problem per line, "file: message", warnings marked as such.
QString formatProblems(const ThemeProblems& problems);

enum class ThemeSource {
    Builtin,  // compiled into VC3D
    User,     // the per-user themes folder
    Path,     // a VC3D_THEME_PATH folder, or a file given with --theme
};
QString themeSourceName(ThemeSource source);

// Explicit or derived QPalette roles. Roles in `active` apply to the Active
// and Inactive groups; `disabled` overrides the Disabled group.
struct ThemePalette
{
    QMap<QPalette::ColorRole, QColor> active;
    QMap<QPalette::ColorRole, QColor> disabled;

    bool isEmpty() const { return active.isEmpty() && disabled.isEmpty(); }
    bool operator==(const ThemePalette& other) const = default;
    // Applies the roles on top of `base`.
    QPalette applyTo(QPalette base) const;
};

struct ThemeFont
{
    QString family;
    double pointSize{0.0};  // 0 keeps the current size

    bool operator==(const ThemeFont& other) const = default;
};

// One theme as written. Unset values are inherited from the base theme.
struct ThemeDefinition
{
    QString id;
    QString name;
    QString author;
    QString description;
    QString base;          // empty only for built-in themes
    QString colorScheme;   // "dark", "light", or empty to inherit
    QMap<QString, QString> tokens;
    ThemePalette palette;
    ThemeFont font;
    QString stylesheet;      // extra QSS text, from "stylesheet" or "qss"
    QString stylesheetFile;  // absolute path when the QSS came from a file
    bool replaceStylesheet{false};
    ThemeSource source{ThemeSource::User};
    QString filePath;   // the .vc3dtheme file
    QString directory;  // folder that relative paths and @themedir@ refer to
    bool folderTheme{false};
};

// Extra QSS of one theme in a resolved chain.
struct ThemeStylesheetPart
{
    QString text;
    QString directory;
    QString file;  // the QSS file, or the theme file for inline "qss"
    // Built-in style sheets skip the checks that keep user style sheets to
    // styling: no qproperty-*, url() only inside the theme's folder.
    bool trusted{false};
};

// A theme with its whole base chain applied.
struct ResolvedTheme
{
    QString id;
    QString name;
    QString author;
    QString description;
    QString rootId;       // classic, modern-dark or modern-light
    QString colorScheme;  // "dark" or "light"; empty when following classic's detection
    QStringList chain;    // ids from the root to this theme
    QMap<QString, QString> tokens;
    ThemePalette palette;  // explicit roles plus, for template themes, derived ones
    ThemeFont font;
    std::vector<ThemeStylesheetPart> stylesheets;  // root first
    bool replaced{false};  // a theme in the chain used stylesheetMode "replace"
    ThemeSource source{ThemeSource::Builtin};
    QString filePath;
    QStringList watchFiles;  // user files whose edits change this theme

    bool isClassic() const { return rootId == QLatin1String(kClassicId); }
    // Fusion plus the rendered template (unless a theme replaced it).
    bool usesFusion() const { return !isClassic(); }
    bool usesTemplate() const { return !isClassic() && !replaced; }
};

QString normalizeThemeId(const QString& raw);
QStringList builtinThemeIds();
bool isBuiltinThemeId(const QString& id);
// Whether a --theme value or a theme.set id names a theme file or folder
// rather than an installed theme: it contains a slash or ends in .vc3dtheme.
bool looksLikeThemePath(const QString& value);

// Token names and values of the built-in modern-dark theme. They define which
// token names are known and which ones hold colors.
const QMap<QString, QString>& builtinTokens();

// Loads a theme from a .vc3dtheme file or from a folder holding
// theme.vc3dtheme. The id comes from the file or folder name. Problems are
// appended; the result is empty when any of them is an error.
std::optional<ThemeDefinition> loadTheme(
    const QString& location, ThemeSource source, ThemeProblems* problems);
// Same, from JSON already in memory.
std::optional<ThemeDefinition> parseTheme(
    const QByteArray& json, const QString& filePath, const QString& id,
    ThemeSource source, ThemeProblems* problems);

// The palette roles a template theme gets from its tokens.
ThemePalette derivePalette(const QMap<QString, QString>& tokens);

// Replaces @name@ placeholders with values. Placeholders without a value are
// left in place and listed in `unknown` (without the @ signs).
QString substituteTokens(const QString& text, const QMap<QString, QString>& values,
                         QStringList* unknown = nullptr);
// Placeholder names used in `text`, sorted and unique.
QStringList placeholdersIn(const QString& text);
// The built-in style sheet template.
QString templateText();

// Renders the application style sheet: the template (for template themes)
// followed by each extra style sheet. `iconKey` is the value of @name@.
// `classicScheme` stands in for @scheme@ when the theme follows classic's
// detection. Unknown placeholders are errors, and so are user style sheets
// that do not parse, set widget properties (qproperty-*), or load images from
// outside their theme's folder.
QString renderStyleSheet(const ResolvedTheme& theme, const QString& iconKey,
                         const QString& classicScheme, ThemeProblems* problems);
// Renders the style sheet the way ThemeManager applies the theme and appends
// the problems; true when none is an error.
bool checkRendering(const ResolvedTheme& theme, ThemeProblems* problems);

struct ThemeIconSpec
{
    QString name;
    int size;  // logical pixels; an @2x file twice that size is written too
};
const std::vector<ThemeIconSpec>& themeIconSpecs();
// Icon folder name: the theme id plus a short hash of its tokens.
QString themeIconKey(const ResolvedTheme& theme);
// Renders the icons into cacheRoot/key/ unless a complete folder is already
// there, then prunes stale folders (pruneThemeIcons with a week). Returns the
// folder, or an empty string with `error` set.
QString ensureThemeIcons(const QMap<QString, QString>& tokens, const QString& cacheRoot,
                         const QString& key, QString* error);
// Removes icon folders of the same theme id other than `keepKey`, and
// unfinished folders of any theme, last modified at least `minAgeSeconds` ago.
void pruneThemeIcons(const QString& cacheRoot, const QString& keepKey, qint64 minAgeSeconds);

// Role name <-> enum, case-insensitive.
std::optional<QPalette::ColorRole> paletteRoleFromName(const QString& name);
QString paletteRoleName(QPalette::ColorRole role);
// {"Window": "#rrggbb", ..., "disabled": {...}} in file format.
QJsonObject paletteToJson(const ThemePalette& palette);

// The themes VC3D knows about: built-ins first, then the folders in search
// order. Later duplicates of an id are skipped with a warning.
class ThemeRegistry
{
public:
    ThemeRegistry();

    // Replaces every non-built-in theme with the themes found in `folders`
    // (searched in order). Returns the problems found.
    ThemeProblems scan(const QString& userFolder, const QStringList& pathFolders);
    // Adds a theme; an existing non-built-in theme with that id is replaced.
    // Built-in ids are refused.
    bool add(ThemeDefinition definition, ThemeProblems* problems);

    const ThemeDefinition* find(const QString& id) const;
    bool contains(const QString& id) const { return find(id) != nullptr; }
    // Built-ins in their fixed order, then the rest sorted by name.
    QList<const ThemeDefinition*> themes() const;

    std::optional<ResolvedTheme> resolve(const QString& id, ThemeProblems* problems) const;
    // Resolves a definition that is not (or not yet) in the registry.
    std::optional<ResolvedTheme> resolve(const ThemeDefinition& definition,
                                         ThemeProblems* problems) const;

private:
    void addBuiltins();
    void scanFolder(const QString& folder, ThemeSource source, ThemeProblems* problems);

    QList<ThemeDefinition> _themes;
};

// Writes the resolved theme as a complete .vc3dtheme: every token, the palette
// roles that do not simply follow the tokens, the font, and the extra style
// sheets of the whole chain as inline "qss". The base is the built-in the
// theme came from; classic itself is exported as a modern-dark template.
// Style sheets that use @themedir@ files are exported with a warning, since
// those files are not copied.
QByteArray exportThemeJson(const ResolvedTheme& theme, const ThemeRegistry& registry,
                           ThemeProblems* problems);

struct ImportResult
{
    QString id;
    QString installedPath;  // with alreadyExists, the installed theme in the way
    ThemeProblems problems;
    bool alreadyExists{false};  // refused because overwrite was false
    bool ok{false};
};

// Copies a theme file or folder into a staging folder inside `userFolder`,
// validates the copy, and installs it (a folder theme as a whole folder). A
// file named theme.vc3dtheme is refused (its folder is the theme), as are
// links and special files, built-in ids, and single-file themes whose style
// sheet uses @themedir@ files. An installed theme with the same id is only
// replaced when `overwrite`, and is kept if installing the new one fails.
ImportResult importTheme(const QString& location, const QString& userFolder,
                         bool overwrite, const ThemeRegistry& registry);

}  // namespace vc3d::theme
