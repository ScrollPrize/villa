#include "themes/ThemeEngine.hpp"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QJsonValue>
#include <QMetaEnum>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QPolygonF>
#include <QRegularExpression>
#include <QScopeGuard>
#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vc3d::theme {
namespace {

constexpr int kMaxImportFiles = 200;
constexpr qint64 kMaxImportBytes = 32LL * 1024 * 1024;
// Icon folders for a theme's earlier colors are kept this long, in case
// another running VC3D still shows them.
constexpr qint64 kIconFolderMaxAgeSeconds = 7LL * 24 * 60 * 60;
// Part of the icon folder hash: change it when the icon drawing changes so
// cached folders are drawn again.
constexpr auto kIconRenderVersion = "icons-1";
constexpr auto kCommentKey = "$comment";

const QRegularExpression& placeholderPattern()
{
    static const QRegularExpression pattern(QStringLiteral("@([a-z][a-z0-9_]*)@"));
    return pattern;
}

const QRegularExpression& tokenNamePattern()
{
    static const QRegularExpression pattern(QStringLiteral("^[a-z][a-z0-9_]*$"));
    return pattern;
}

// Placeholders the engine fills in; themes cannot define tokens with these names.
const QStringList& engineTokenNames()
{
    static const QStringList names{
        QStringLiteral("name"),
        QStringLiteral("scheme"),
        QStringLiteral("themedir"),
    };
    return names;
}

bool isColor(const QString& value)
{
    return !value.isEmpty() && QColor::fromString(value).isValid();
}

// "#aarrggbb", which Qt reads with the alpha first, unlike CSS.
bool isAlphaFirstColor(const QString& value)
{
    static const QRegularExpression pattern(QStringLiteral("^#[0-9a-fA-F]{8}$"));
    return pattern.match(value).hasMatch();
}

// A built-in length token such as "6px"; a theme's value may leave out the unit.
bool isLength(const QString& value, bool unitRequired)
{
    static const QRegularExpression withUnit(QStringLiteral("^\\d+(\\.\\d+)?(px|pt|em|ex)$"));
    static const QRegularExpression unitOptional(QStringLiteral("^\\d+(\\.\\d+)?(px|pt|em|ex)?$"));
    return (unitRequired ? withUnit : unitOptional).match(value).hasMatch();
}

// Token values are pasted into QSS, so they must not end a declaration, open or
// close a block, or start a comment.
bool isSafeQssValue(const QString& value)
{
    return !value.isEmpty() && !value.contains(QLatin1Char(';')) &&
           !value.contains(QLatin1Char('{')) && !value.contains(QLatin1Char('}')) &&
           !value.contains(QLatin1String("/*")) && !value.contains(QLatin1String("*/")) &&
           !value.contains(QLatin1Char('\n'));
}

// Line breaks, other control characters and bidirectional overrides, which
// would let a theme's name or description disguise itself in the menu.
bool hasControlCharacters(const QString& text)
{
    for (const QChar ch : text) {
        const char16_t c = ch.unicode();
        if (ch.category() == QChar::Other_Control || c == 0x061c || c == 0x200e || c == 0x200f ||
            c == 0x2028 || c == 0x2029 || (c >= 0x202a && c <= 0x202e) || (c >= 0x2066 && c <= 0x2069)) {
            return true;
        }
    }
    return false;
}

QString colorText(const QColor& color)
{
    return color.alpha() == 255 ? color.name(QColor::HexRgb) : color.name(QColor::HexArgb);
}

QString quoted(const QString& text)
{
    return QLatin1Char('"') + text + QLatin1Char('"');
}

QString alphaFirstWarning(const QString& where, const QString& value)
{
    const QColor color = QColor::fromString(value);
    return QStringLiteral("%1 %2 is read as %3 at %4% opacity: Qt puts the alpha first (#aarrggbb)")
        .arg(where, quoted(value), color.name(QColor::HexRgb))
        .arg(qRound(color.alphaF() * 100.0));
}

int editDistance(const QString& a, const QString& b)
{
    std::vector<int> row(static_cast<size_t>(b.size()) + 1);
    std::iota(row.begin(), row.end(), 0);
    for (qsizetype i = 1; i <= a.size(); ++i) {
        int diagonal = row[0];
        row[0] = static_cast<int>(i);
        for (qsizetype j = 1; j <= b.size(); ++j) {
            const int above = row[static_cast<size_t>(j)];
            const int cost = a.at(i - 1) == b.at(j - 1) ? 0 : 1;
            row[static_cast<size_t>(j)] =
                std::min({above + 1, row[static_cast<size_t>(j - 1)] + 1, diagonal + cost});
            diagonal = above;
        }
    }
    return row.back();
}

// " (did you mean "x"?)" for the closest candidate within two edits, ignoring case.
QString suggestion(const QString& name, const QStringList& candidates)
{
    QString best;
    int bestDistance = 3;
    for (const QString& candidate : candidates) {
        const int distance = editDistance(name.toLower(), candidate.toLower());
        if (distance < bestDistance) {
            best = candidate;
            bestDistance = distance;
        }
    }
    return best.isEmpty() ? QString() : QStringLiteral(" (did you mean %1?)").arg(quoted(best));
}

// Only regular files are read, and only up to the size limit: a FIFO or a
// device would block or never end.
bool readFile(const QString& path, QByteArray* contents, QString* error)
{
    const QFileInfo info(path);
    if (!info.isFile()) {
        *error = info.exists() ? QStringLiteral("not a regular file")
                               : QStringLiteral("the file does not exist");
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = QStringLiteral("cannot read the file: %1").arg(file.errorString());
        return false;
    }
    *contents = file.read(kMaxThemeFileBytes + 1);
    if (contents->size() > kMaxThemeFileBytes) {
        *error = QStringLiteral("the file is larger than %1 KiB").arg(kMaxThemeFileBytes / 1024);
        return false;
    }
    return true;
}

int lineOfOffset(const QByteArray& json, int offset)
{
    return 1 + static_cast<int>(std::count(json.cbegin(),
                                           json.cbegin() + std::clamp<qsizetype>(offset, 0, json.size()),
                                           '\n'));
}

int columnOfOffset(const QByteArray& json, int offset)
{
    const qsizetype end = std::clamp<qsizetype>(offset, 0, json.size());
    if (end == 0)
        return 1;
    return static_cast<int>(end - json.lastIndexOf('\n', end - 1));
}

QString idFromLocation(const QFileInfo& file, bool folderTheme)
{
    if (folderTheme)
        return file.absoluteDir().dirName();
    QString name = file.fileName();
    if (name.endsWith(QLatin1String(kThemeFileSuffix), Qt::CaseInsensitive))
        name.chop(static_cast<int>(qstrlen(kThemeFileSuffix)));
    return name;
}

bool isInside(const QString& path, const QString& folder)
{
    if (path.isEmpty() || folder.isEmpty())
        return false;
    const QString cleanFolder = QDir::cleanPath(folder);
    return QDir::cleanPath(path).startsWith(cleanFolder + QLatin1Char('/'));
}

QStringList paletteRoleNames()
{
    QStringList names;
    const QMetaEnum roles = QMetaEnum::fromType<QPalette::ColorRole>();
    for (int i = 0; i < roles.keyCount(); ++i) {
        const QString key = QString::fromLatin1(roles.key(i));
        if (key != QLatin1String("NoRole") && key != QLatin1String("NColorRoles"))
            names << key;
    }
    return names;
}

void parsePaletteGroup(const QJsonObject& object, QMap<QPalette::ColorRole, QColor>* roles,
                       const QString& group, const QString& file, ThemeProblems* problems)
{
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (group.isEmpty() && it.key() == QLatin1String("disabled"))
            continue;
        const auto role = paletteRoleFromName(it.key());
        const QString where = group.isEmpty() ? QStringLiteral("palette")
                                              : QStringLiteral("palette.%1").arg(group);
        if (!role) {
            problems->append({file, QStringLiteral("unknown palette role %1 in %2 is ignored%3")
                                        .arg(quoted(it.key()), where,
                                             suggestion(it.key(), paletteRoleNames())),
                              true});
            continue;
        }
        const QString value = it.value().toString().trimmed();
        if (!it.value().isString() || !isColor(value)) {
            problems->append({file, QStringLiteral("%1.%2 must be a color such as \"#1f2631\"")
                                        .arg(where, it.key())});
            continue;
        }
        if (isAlphaFirstColor(value))
            problems->append({file, alphaFirstWarning(QStringLiteral("%1.%2").arg(where, it.key()), value), true});
        roles->insert(*role, QColor::fromString(value));
    }
}

enum class IconShape { Check, Dash, ChevronDown, ChevronUp, ChevronRight };

struct IconRecipe
{
    const char* name;
    int size;
    IconShape shape;
    const char* token;
    double weight;
    double span;
};

// The icons the template refers to as url(vc3dtheme:@name@/<name>.png), each
// drawn in one token's color. `weight` is the stroke width and `span` the
// chevron's half width, as fractions of the icon size.
constexpr IconRecipe kIconRecipes[] = {
    {"check", 14, IconShape::Check, "on_accent", 0.15, 0.00},
    {"check-disabled", 14, IconShape::Check, "on_accent_disabled", 0.15, 0.00},
    {"dash", 14, IconShape::Dash, "on_accent", 0.00, 0.00},
    {"dash-disabled", 14, IconShape::Dash, "on_accent_disabled", 0.00, 0.00},
    {"menu-check", 14, IconShape::Check, "accent", 0.15, 0.00},
    {"chevron-down", 12, IconShape::ChevronDown, "text_secondary", 0.12, 0.30},
    {"chevron-down-disabled", 12, IconShape::ChevronDown, "text_muted", 0.12, 0.30},
    {"chevron-down-hover", 12, IconShape::ChevronDown, "text", 0.12, 0.30},
    {"chevron-right", 12, IconShape::ChevronRight, "text_secondary", 0.12, 0.30},
    {"chevron-right-selected", 12, IconShape::ChevronRight, "accent_soft_text", 0.12, 0.30},
    {"spin-up", 10, IconShape::ChevronUp, "text_secondary", 0.14, 0.38},
    {"spin-down", 10, IconShape::ChevronDown, "text_secondary", 0.14, 0.38},
    {"spin-up-disabled", 10, IconShape::ChevronUp, "text_muted", 0.14, 0.38},
    {"spin-down-disabled", 10, IconShape::ChevronDown, "text_muted", 0.14, 0.38},
    {"spin-up-hover", 10, IconShape::ChevronUp, "accent", 0.14, 0.38},
    {"spin-down-hover", 10, IconShape::ChevronDown, "accent", 0.14, 0.38},
};

// Draws at four times the target size and scales down, which keeps the thin
// strokes smooth at 10 to 14 pixels.
QImage drawIcon(const IconRecipe& recipe, const QColor& color, int scale)
{
    const int target = recipe.size * scale;
    const int big = target * 4;
    QImage image(big, big, QImage::Format_ARGB32_Premultiplied);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    const double s = big;

    if (recipe.shape == IconShape::Dash) {
        const double h = s * 0.075;
        painter.setPen(Qt::NoPen);
        painter.setBrush(color);
        painter.drawRoundedRect(QRectF(s * 0.24, s * 0.5 - h, s * 0.52, 2.0 * h), h, h);
    } else {
        const double width = std::max(1.0, static_cast<double>(std::lround(s * recipe.weight)));
        painter.setPen(QPen(color, width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.setBrush(Qt::NoBrush);
        QPolygonF points;
        const double c = s / 2.0;
        const double a = s * recipe.span;
        switch (recipe.shape) {
        case IconShape::Check:
            points << QPointF(s * 0.20, s * 0.52) << QPointF(s * 0.42, s * 0.73)
                   << QPointF(s * 0.80, s * 0.30);
            break;
        case IconShape::ChevronDown:
            points << QPointF(c - a, c - a / 2) << QPointF(c, c + a / 2) << QPointF(c + a, c - a / 2);
            break;
        case IconShape::ChevronUp:
            points << QPointF(c - a, c + a / 2) << QPointF(c, c - a / 2) << QPointF(c + a, c + a / 2);
            break;
        case IconShape::ChevronRight:
            points << QPointF(c - a / 2, c - a) << QPointF(c + a / 2, c) << QPointF(c - a / 2, c + a);
            break;
        case IconShape::Dash:
            break;
        }
        painter.drawPolyline(points);
    }
    painter.end();
    return image.scaled(target, target, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

QStringList iconFileNames()
{
    QStringList names;
    for (const auto& recipe : kIconRecipes) {
        names << QStringLiteral("%1.png").arg(QLatin1String(recipe.name))
              << QStringLiteral("%1@2x.png").arg(QLatin1String(recipe.name));
    }
    return names;
}

bool iconFolderComplete(const QString& folder)
{
    const QDir dir(folder);
    if (!dir.exists())
        return false;
    for (const QString& name : iconFileNames()) {
        if (!dir.exists(name))
            return false;
    }
    return true;
}

ThemeDefinition classicDefinition()
{
    ThemeDefinition classic;
    classic.id = QString::fromLatin1(kClassicId);
    classic.name = QStringLiteral("Classic");
    classic.author = QStringLiteral("VC3D");
    classic.description = QStringLiteral(
        "VC3D's original look: the native style, a dark palette when the system "
        "prefers dark, and the gradient menu bar.");
    classic.source = ThemeSource::Builtin;
    return classic;
}

// Removes the comments of a style sheet, outside of strings, keeping line
// breaks, and checks that comments and strings end and that braces balance.
// Qt drops a whole style sheet that does not parse, so this names the line.
QString stripStyleSheetComments(const QString& qss, QString* error)
{
    enum class State { Code, Comment, String };
    State state = State::Code;
    QChar quote;
    int line = 1;
    int startLine = 1;
    std::vector<int> openBraces;
    QString out;
    out.reserve(qss.size());
    for (qsizetype i = 0; i < qss.size(); ++i) {
        const QChar ch = qss.at(i);
        const QChar next = i + 1 < qss.size() ? qss.at(i + 1) : QChar();
        if (ch == QLatin1Char('\n'))
            ++line;
        switch (state) {
        case State::Comment:
            if (ch == QLatin1Char('*') && next == QLatin1Char('/')) {
                state = State::Code;
                ++i;
            } else if (ch == QLatin1Char('\n')) {
                out += ch;
            }
            break;
        case State::String:
            if (ch == QLatin1Char('\n')) {
                *error = QStringLiteral("line %1: a string is not closed").arg(startLine);
                return {};
            }
            out += ch;
            if (ch == QLatin1Char('\\') && i + 1 < qss.size()) {
                out += next;
                ++i;
            } else if (ch == quote) {
                state = State::Code;
            }
            break;
        case State::Code:
            if (ch == QLatin1Char('/') && next == QLatin1Char('*')) {
                state = State::Comment;
                startLine = line;
                ++i;
                break;
            }
            out += ch;
            if (ch == QLatin1Char('\\') && i + 1 < qss.size()) {
                out += next;
                if (next == QLatin1Char('\n'))
                    ++line;
                ++i;
            } else if (ch == QLatin1Char('"') || ch == QLatin1Char('\'')) {
                state = State::String;
                quote = ch;
                startLine = line;
            } else if (ch == QLatin1Char('{')) {
                openBraces.push_back(line);
            } else if (ch == QLatin1Char('}')) {
                if (openBraces.empty()) {
                    *error = QStringLiteral("line %1: \"}\" has no matching \"{\"").arg(line);
                    return {};
                }
                openBraces.pop_back();
            }
            break;
        }
    }
    if (state == State::Comment)
        *error = QStringLiteral("line %1: a comment is never closed").arg(startLine);
    else if (state == State::String)
        *error = QStringLiteral("line %1: a string is never closed").arg(startLine);
    else if (!openBraces.empty())
        *error = QStringLiteral("line %1: \"{\" is never closed").arg(openBraces.back());
    return error->isEmpty() ? out : QString();
}

// The style sheet as Qt's scanner reads it: "\71" and other hexadecimal
// escapes decoded, and the backslash of any other escape dropped.
QString unescapeStyleSheet(const QString& qss)
{
    QString out;
    out.reserve(qss.size());
    for (qsizetype i = 0; i < qss.size(); ++i) {
        const QChar ch = qss.at(i);
        if (ch != QLatin1Char('\\') || i + 1 >= qss.size()) {
            out += ch;
            continue;
        }
        const auto isHexDigit = [](QChar c) {
            return (c >= QLatin1Char('0') && c <= QLatin1Char('9')) ||
                   (c >= QLatin1Char('a') && c <= QLatin1Char('f')) ||
                   (c >= QLatin1Char('A') && c <= QLatin1Char('F'));
        };
        qsizetype end = i + 1;
        while (end < qss.size() && end - (i + 1) < 6 && isHexDigit(qss.at(end)))
            ++end;
        if (end == i + 1) {
            out += qss.at(i + 1);
            ++i;
            continue;
        }
        bool ok = false;
        const uint code = QStringView(qss).mid(i + 1, end - i - 1).toUInt(&ok, 16);
        const char32_t character = ok && code > 0 && code <= 0x10ffff ? code : 0xfffd;
        out += QString::fromUcs4(&character, 1);
        // CSS ends an escape with one optional space; counting it in reads
        // more names as properties, never fewer.
        if (end < qss.size() && qss.at(end).isSpace())
            ++end;
        i = end - 1;
    }
    return out;
}

// A user style sheet may only style widgets: no qproperty-*, and url()
// only for the drawn icons, Qt resources, and regular files of limited
// size inside the theme's own folder.
void checkUserStyleSheet(const QString& qss, const ThemeStylesheetPart& part, ThemeProblems* problems)
{
    const QString plain = unescapeStyleSheet(qss);
    static const QRegularExpression property(QStringLiteral("qproperty-([A-Za-z0-9_]*)"),
                                             QRegularExpression::CaseInsensitiveOption);
    QStringList reported;
    for (auto it = property.globalMatch(plain); it.hasNext();) {
        const QString name = it.next().captured(1);
        if (reported.contains(name))
            continue;
        reported << name;
        problems->append({part.file, QStringLiteral("\"qproperty-%1\" sets a widget property; a theme "
                                                    "style sheet may only style widgets")
                                         .arg(name)});
    }

    static const QRegularExpression url(QStringLiteral(R"re(url\(\s*(?:"([^"]*)"\s*|'([^']*)'\s*|([^)]*))\))re"),
                                        QRegularExpression::CaseInsensitiveOption);
    const QString folder = QFileInfo(part.directory).canonicalFilePath();
    for (auto it = url.globalMatch(plain); it.hasNext();) {
        const auto match = it.next();
        const QString target = (match.captured(1) + match.captured(2) + match.captured(3)).trimmed();
        if (target.startsWith(QLatin1String(kIconSearchPrefix) + QLatin1Char(':')) ||
            target.startsWith(QLatin1String(":/"))) {
            if (target.contains(QLatin1String("..")))
                problems->append({part.file, QStringLiteral("url(%1) must not contain \"..\"").arg(target)});
            continue;
        }
        const QFileInfo image(target);
        const QString canonical = image.canonicalFilePath();
        const bool inside = QDir::isAbsolutePath(target) &&
                            (canonical.isEmpty() ? isInside(target, part.directory) || isInside(target, folder)
                                                 : isInside(canonical, folder));
        if (!inside) {
            problems->append({part.file, QStringLiteral("url(%1) must name an image inside the theme's "
                                                        "folder, written as url(@themedir@/name.png)")
                                             .arg(target)});
        } else if (canonical.isEmpty()) {
            problems->append({part.file, QStringLiteral("url(%1): the file does not exist").arg(target)});
        } else if (!image.isFile() || image.size() > kMaxThemeImageBytes) {
            problems->append({part.file, QStringLiteral("url(%1) must name a regular file of at most %2 MiB")
                                             .arg(target)
                                             .arg(kMaxThemeImageBytes / (1024 * 1024))});
        }
    }
}

// Copies a theme folder file by file. Links and special files are refused:
// a link could pull private files into the themes folder, and reading a FIFO
// would block.
bool copyThemeFolder(const QString& source, const QString& destination, QString* error)
{
    QStringList files;
    qint64 bytes = 0;
    QDirIterator it(source, QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        const QFileInfo info = it.fileInfo();
        const QString relative = QDir(source).relativeFilePath(info.filePath());
        if (info.isSymLink()) {
            *error = QStringLiteral("%1 is a link; a theme folder may only hold regular files").arg(relative);
            return false;
        }
        if (info.isDir())
            continue;
        if (!info.isFile()) {
            *error = QStringLiteral("%1 is not a regular file").arg(relative);
            return false;
        }
        files << relative;
        bytes += info.size();
        if (files.size() > kMaxImportFiles || bytes > kMaxImportBytes) {
            *error = QStringLiteral("the theme folder holds more than %1 files or %2 MiB")
                         .arg(kMaxImportFiles)
                         .arg(kMaxImportBytes / (1024 * 1024));
            return false;
        }
    }
    if (!QDir().mkpath(destination)) {
        *error = QStringLiteral("cannot create %1").arg(destination);
        return false;
    }
    for (const QString& relative : files) {
        const QString target = QDir(destination).filePath(relative);
        if (!QDir().mkpath(QFileInfo(target).absolutePath()) ||
            !QFile::copy(QDir(source).filePath(relative), target)) {
            *error = QStringLiteral("cannot copy %1").arg(relative);
            return false;
        }
    }
    return true;
}

void removeInstalled(const QString& path)
{
    const QFileInfo info(path);
    if (info.isDir() && !info.isSymLink())
        QDir(path).removeRecursively();
    else if (info.exists() || info.isSymLink())
        QFile::remove(path);
}

std::optional<ThemeDefinition> loadThemeFile(const QString& filePath, const QString& id,
                                             ThemeSource source, bool folderTheme,
                                             ThemeProblems* problems)
{
    QByteArray json;
    QString error;
    if (!readFile(filePath, &json, &error)) {
        problems->append({filePath, error});
        return std::nullopt;
    }
    auto definition = parseTheme(json, filePath, id, source, problems);
    if (definition)
        definition->folderTheme = folderTheme;
    return definition;
}

// Rewrites staging paths in problems so they name the theme being imported.
void relabelProblems(ThemeProblems* problems, const QString& from, const QString& to)
{
    for (auto& problem : *problems) {
        if (problem.file.startsWith(from))
            problem.file = to + problem.file.mid(from.size());
        problem.message.replace(from, to);
    }
}

}  // namespace

// ---------------------------------------------------------------------------

bool hasErrors(const ThemeProblems& problems)
{
    return std::any_of(problems.cbegin(), problems.cend(),
                       [](const ThemeProblem& problem) { return !problem.warning; });
}

QString formatProblems(const ThemeProblems& problems)
{
    QStringList lines;
    for (const auto& problem : problems) {
        QString line = problem.file.isEmpty() ? problem.message
                                              : problem.file + QStringLiteral(": ") + problem.message;
        if (problem.warning)
            line = QStringLiteral("warning: ") + line;
        lines << line;
    }
    return lines.join(QLatin1Char('\n'));
}

QString themeSourceName(ThemeSource source)
{
    switch (source) {
    case ThemeSource::Builtin: return QStringLiteral("builtin");
    case ThemeSource::User:    return QStringLiteral("user");
    case ThemeSource::Path:    return QStringLiteral("path");
    }
    return QStringLiteral("user");
}

QPalette ThemePalette::applyTo(QPalette base) const
{
    for (auto it = active.begin(); it != active.end(); ++it)
        base.setColor(QPalette::All, it.key(), it.value());
    for (auto it = disabled.begin(); it != disabled.end(); ++it)
        base.setColor(QPalette::Disabled, it.key(), it.value());
    return base;
}

QString normalizeThemeId(const QString& raw)
{
    QString id;
    bool dash = false;
    for (const QChar ch : raw.trimmed().toLower()) {
        if ((ch >= QLatin1Char('a') && ch <= QLatin1Char('z')) ||
            (ch >= QLatin1Char('0') && ch <= QLatin1Char('9'))) {
            if (dash && !id.isEmpty())
                id += QLatin1Char('-');
            id += ch;
            dash = false;
        } else {
            dash = true;
        }
    }
    return id;
}

QStringList builtinThemeIds()
{
    return {QString::fromLatin1(kClassicId), QString::fromLatin1(kModernDarkId),
            QString::fromLatin1(kModernLightId)};
}

bool isBuiltinThemeId(const QString& id)
{
    return builtinThemeIds().contains(id);
}

bool looksLikeThemePath(const QString& value)
{
    return value.contains(QLatin1Char('/')) || value.contains(QLatin1Char('\\')) ||
           value.endsWith(QLatin1String(kThemeFileSuffix), Qt::CaseInsensitive);
}

const QMap<QString, QString>& builtinTokens()
{
    static const QMap<QString, QString> tokens = [] {
        QMap<QString, QString> result;
        QFile file(QStringLiteral("%1/%2%3").arg(QLatin1String(kBuiltinResourceDir),
                                                 QLatin1String(kModernDarkId),
                                                 QLatin1String(kThemeFileSuffix)));
        if (!file.open(QIODevice::ReadOnly)) {
            qWarning("VC3D theme: the built-in modern-dark theme resource is missing");
            return result;
        }
        const QJsonObject object =
            QJsonDocument::fromJson(file.readAll()).object().value(QStringLiteral("tokens")).toObject();
        for (auto it = object.begin(); it != object.end(); ++it)
            result.insert(it.key(), it.value().toString().trimmed());
        return result;
    }();
    return tokens;
}

std::optional<ThemeDefinition> loadTheme(
    const QString& location, ThemeSource source, ThemeProblems* problems)
{
    const QFileInfo info(location);
    bool folderTheme = false;
    QString filePath;
    if (info.isDir()) {
        folderTheme = true;
        filePath = QDir(info.absoluteFilePath()).filePath(QLatin1String(kFolderThemeFileName));
    } else {
        // A theme folder's own theme.vc3dtheme stands for the folder.
        filePath = info.absoluteFilePath();
        folderTheme = info.fileName() == QLatin1String(kFolderThemeFileName);
    }
    const QFileInfo fileInfo(filePath);
    if (!fileInfo.isFile()) {
        problems->append({location, folderTheme
                                        ? QStringLiteral("the folder has no %1").arg(QLatin1String(kFolderThemeFileName))
                                        : QStringLiteral("the theme file does not exist")});
        return std::nullopt;
    }

    const QString rawId = idFromLocation(fileInfo, folderTheme);
    const QString id = normalizeThemeId(rawId);
    if (id.isEmpty()) {
        problems->append({filePath, QStringLiteral("cannot make a theme id from the name %1; use letters or digits")
                                        .arg(quoted(rawId))});
        return std::nullopt;
    }
    return loadThemeFile(filePath, id, source, folderTheme, problems);
}

std::optional<ThemeDefinition> parseTheme(
    const QByteArray& json, const QString& filePath, const QString& id,
    ThemeSource source, ThemeProblems* problems)
{
    ThemeProblems local;
    const auto error = [&](const QString& message) { local.append({filePath, message, false}); };
    const auto warn = [&](const QString& message) { local.append({filePath, message, true}); };
    const auto finish = [&]() {
        if (local.size() > kMaxProblemsPerFile) {
            const ThemeProblems dropped = local.mid(kMaxProblemsPerFile);
            local = local.first(kMaxProblemsPerFile);
            local.append({filePath, QStringLiteral("%1 more problems are not shown").arg(dropped.size()),
                          !hasErrors(dropped)});
        }
        problems->append(local);
    };

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(json, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        error(QStringLiteral("not valid JSON (line %1, column %2): %3. JSON allows no comments, "
                             "trailing commas or single quotes; check for a missing comma before "
                             "this point")
                  .arg(lineOfOffset(json, parseError.offset))
                  .arg(columnOfOffset(json, parseError.offset))
                  .arg(parseError.errorString()));
        finish();
        return std::nullopt;
    }
    if (!document.isObject()) {
        error(QStringLiteral("a theme file must hold one JSON object"));
        finish();
        return std::nullopt;
    }
    const QJsonObject object = document.object();

    ThemeDefinition definition;
    definition.id = id;
    definition.source = source;
    definition.filePath = filePath;
    definition.directory = QFileInfo(filePath).absolutePath();
    if (filePath.startsWith(QLatin1Char(':')))
        definition.directory = QFileInfo(filePath).path();

    static const QStringList knownKeys{
        QStringLiteral("vc3dTheme"), QStringLiteral("name"), QStringLiteral("author"),
        QStringLiteral("description"), QStringLiteral("base"), QStringLiteral("colorScheme"),
        QStringLiteral("tokens"), QStringLiteral("palette"), QStringLiteral("font"),
        QStringLiteral("stylesheet"), QStringLiteral("qss"), QStringLiteral("stylesheetMode"),
    };
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!knownKeys.contains(it.key()) && it.key() != QLatin1String(kCommentKey))
            warn(QStringLiteral("unknown key %1 is ignored%2").arg(quoted(it.key()), suggestion(it.key(), knownKeys)));
    }

    const QJsonValue version = object.value(QStringLiteral("vc3dTheme"));
    if (version.isUndefined()) {
        error(QStringLiteral("missing \"vc3dTheme\": the format version, 1"));
    } else if (!version.isDouble() || version.toDouble() != std::floor(version.toDouble())) {
        error(QStringLiteral("\"vc3dTheme\" must be the integer 1"));
    } else if (version.toDouble() > kFormatVersion) {
        error(QStringLiteral("this theme uses format version %1, but this VC3D reads version %2; "
                             "update VC3D to use it")
                  .arg(version.toInt())
                  .arg(kFormatVersion));
    } else if (version.toDouble() < 1) {
        error(QStringLiteral("\"vc3dTheme\" must be the integer 1"));
    }

    const auto optionalString = [&](const char* key, QString* out) {
        const QJsonValue value = object.value(QLatin1String(key));
        if (value.isUndefined())
            return;
        if (!value.isString()) {
            error(QStringLiteral("\"%1\" must be a string").arg(QLatin1String(key)));
            return;
        }
        *out = value.toString().trimmed();
    };
    // Shown in the menu and its tool tips, so one line of plain text.
    const auto plainText = [&](const char* key, QString* out, int maxLength) {
        optionalString(key, out);
        if (out->size() > maxLength) {
            error(QStringLiteral("\"%1\" is longer than %2 characters").arg(QLatin1String(key)).arg(maxLength));
            out->clear();
        } else if (hasControlCharacters(*out)) {
            error(QStringLiteral("\"%1\" must be plain text without line breaks or control characters")
                      .arg(QLatin1String(key)));
            out->clear();
        }
    };

    const qsizetype problemsBeforeName = local.size();
    plainText("name", &definition.name, kMaxNameChars);
    if (definition.name.isEmpty() && local.size() == problemsBeforeName)
        error(QStringLiteral("missing \"name\": the name shown in the Theme menu"));
    plainText("author", &definition.author, kMaxNameChars);
    plainText("description", &definition.description, kMaxDescriptionChars);

    QString base;
    optionalString("base", &base);
    if (source == ThemeSource::Builtin) {
        definition.base.clear();
    } else if (object.contains(QStringLiteral("base"))) {
        definition.base = normalizeThemeId(base);
        if (definition.base.isEmpty())
            error(QStringLiteral("\"base\" must name a theme, such as \"modern-dark\""));
        else if (definition.base == id)
            error(QStringLiteral("a theme cannot be its own base"));
    } else {
        definition.base = QString::fromLatin1(kModernDarkId);
    }

    QString scheme;
    optionalString("colorScheme", &scheme);
    definition.colorScheme = scheme.toLower();
    if (!definition.colorScheme.isEmpty() && definition.colorScheme != QLatin1String("dark") &&
        definition.colorScheme != QLatin1String("light")) {
        error(QStringLiteral("\"colorScheme\" must be \"dark\" or \"light\", not %1").arg(quoted(scheme)));
    }

    // Unknown tokens are reported after the style sheet is read: a theme may
    // define tokens of its own for its own style sheet.
    QStringList unknownTokens;
    const QJsonValue tokens = object.value(QStringLiteral("tokens"));
    if (!tokens.isUndefined() && !tokens.isObject()) {
        error(QStringLiteral("\"tokens\" must be an object of token names and values"));
    } else {
        const QJsonObject tokenObject = tokens.toObject();
        const auto& known = builtinTokens();
        for (auto it = tokenObject.begin(); it != tokenObject.end(); ++it) {
            const QString name = it.key();
            const QString value = it.value().toString().trimmed();
            if (!tokenNamePattern().match(name).hasMatch()) {
                QString hint;
                const QString spelled = name.toLower().replace(QLatin1Char('-'), QLatin1Char('_'));
                if (known.contains(spelled))
                    hint = QStringLiteral(" (did you mean %1?)").arg(quoted(spelled));
                error(QStringLiteral("token name %1 must use lowercase letters, digits and underscores%2")
                          .arg(quoted(name), hint));
                continue;
            }
            if (engineTokenNames().contains(name)) {
                error(QStringLiteral("token name %1 is reserved; VC3D fills in @%2@ itself").arg(quoted(name), name));
                continue;
            }
            if (!it.value().isString()) {
                error(QStringLiteral("token %1 must be a string").arg(quoted(name)));
                continue;
            }
            const bool knownToken = known.contains(name);
            if (source != ThemeSource::Builtin && !knownToken)
                unknownTokens << name;
            if (knownToken && isColor(known.value(name)) && !isColor(value)) {
                error(QStringLiteral("token %1 must be a color such as \"#22d3ee\", not %2")
                          .arg(quoted(name), quoted(value)));
                continue;
            }
            if (knownToken && isLength(known.value(name), true) && !isLength(value, false)) {
                error(QStringLiteral("token %1 must be a length such as %2, not %3")
                          .arg(quoted(name), quoted(known.value(name)), quoted(value)));
                continue;
            }
            if (value.size() > kMaxTokenValueChars) {
                error(QStringLiteral("token %1 is longer than %2 characters").arg(quoted(name)).arg(kMaxTokenValueChars));
                continue;
            }
            if (!isSafeQssValue(value)) {
                error(QStringLiteral("token %1 must not be empty or contain ; { } /* or */").arg(quoted(name)));
                continue;
            }
            if (isColor(value) && isAlphaFirstColor(value))
                warn(alphaFirstWarning(QStringLiteral("token %1:").arg(quoted(name)), value));
            definition.tokens.insert(name, value);
        }
    }

    const QJsonValue palette = object.value(QStringLiteral("palette"));
    if (!palette.isUndefined() && !palette.isObject()) {
        error(QStringLiteral("\"palette\" must be an object of palette roles and colors"));
    } else if (palette.isObject()) {
        const QJsonObject paletteObject = palette.toObject();
        parsePaletteGroup(paletteObject, &definition.palette.active, QString(), filePath, &local);
        const QJsonValue disabled = paletteObject.value(QStringLiteral("disabled"));
        if (!disabled.isUndefined() && !disabled.isObject())
            error(QStringLiteral("\"palette.disabled\" must be an object of palette roles and colors"));
        else if (disabled.isObject())
            parsePaletteGroup(disabled.toObject(), &definition.palette.disabled,
                              QStringLiteral("disabled"), filePath, &local);
    }

    const QJsonValue font = object.value(QStringLiteral("font"));
    if (!font.isUndefined() && !font.isObject()) {
        error(QStringLiteral("\"font\" must be an object with \"family\" and \"pointSize\""));
    } else if (font.isObject()) {
        const QJsonObject fontObject = font.toObject();
        for (auto it = fontObject.begin(); it != fontObject.end(); ++it) {
            if (it.key() != QLatin1String("family") && it.key() != QLatin1String("pointSize"))
                warn(QStringLiteral("unknown key \"font.%1\" is ignored").arg(it.key()));
        }
        const QJsonValue family = fontObject.value(QStringLiteral("family"));
        if (!family.isUndefined() && !family.isString())
            error(QStringLiteral("\"font.family\" must be a string"));
        else if (family.toString().size() > kMaxNameChars || hasControlCharacters(family.toString()))
            error(QStringLiteral("\"font.family\" must be a font name"));
        else
            definition.font.family = family.toString().trimmed();
        const QJsonValue size = fontObject.value(QStringLiteral("pointSize"));
        if (!size.isUndefined()) {
            if (!size.isDouble() || size.toDouble() <= 0.0 || size.toDouble() > 96.0)
                error(QStringLiteral("\"font.pointSize\" must be a number above 0 and at most 96"));
            else
                definition.font.pointSize = size.toDouble();
        }
    }

    const bool hasFile = object.contains(QStringLiteral("stylesheet"));
    const bool hasInline = object.contains(QStringLiteral("qss"));
    if (hasFile && hasInline) {
        error(QStringLiteral("use either \"stylesheet\" (a file) or \"qss\" (inline text), not both"));
    } else if (hasFile) {
        QString relative;
        optionalString("stylesheet", &relative);
        const QString path = QDir::cleanPath(QDir(definition.directory).filePath(relative));
        const QFileInfo sheet(path);
        if (relative.isEmpty()) {
            error(QStringLiteral("\"stylesheet\" must be a file name next to the theme file"));
        } else if (QDir::isAbsolutePath(relative) || !isInside(path, definition.directory)) {
            error(QStringLiteral("\"stylesheet\" must be a relative path inside the theme's folder"));
        } else if (source != ThemeSource::Builtin && sheet.exists() &&
                   !isInside(sheet.canonicalFilePath(), QFileInfo(definition.directory).canonicalFilePath())) {
            error(QStringLiteral("\"stylesheet\": %1 is a link that leads outside the theme's folder")
                      .arg(quoted(relative)));
        } else {
            QByteArray qss;
            QString readError;
            if (!readFile(path, &qss, &readError)) {
                error(QStringLiteral("\"stylesheet\": cannot read %1 next to the theme file: %2")
                          .arg(quoted(relative), readError));
            } else {
                definition.stylesheet = QString::fromUtf8(qss);
                definition.stylesheetFile = path;
            }
        }
    } else if (hasInline) {
        optionalString("qss", &definition.stylesheet);
        if (definition.stylesheet.size() > kMaxThemeFileBytes)
            error(QStringLiteral("\"qss\" is longer than %1 KiB").arg(kMaxThemeFileBytes / 1024));
    }

    const QStringList usedPlaceholders = placeholdersIn(definition.stylesheet);
    for (const QString& name : unknownTokens) {
        if (!usedPlaceholders.contains(name)) {
            warn(QStringLiteral("unknown token %1; the built-in template does not use it%2")
                     .arg(quoted(name), suggestion(name, builtinTokens().keys())));
        }
    }

    QString mode;
    optionalString("stylesheetMode", &mode);
    if (!mode.isEmpty() && mode != QLatin1String("append") && mode != QLatin1String("replace"))
        error(QStringLiteral("\"stylesheetMode\" must be \"append\" or \"replace\""));
    definition.replaceStylesheet = mode == QLatin1String("replace");

    const bool failed = hasErrors(local);
    finish();
    if (failed)
        return std::nullopt;
    return definition;
}

ThemePalette derivePalette(const QMap<QString, QString>& tokens)
{
    ThemePalette palette;
    const auto color = [&](const char* token) -> std::optional<QColor> {
        const QString value = tokens.value(QLatin1String(token));
        if (!isColor(value))
            return std::nullopt;
        return QColor::fromString(value);
    };
    const auto set = [&](QMap<QPalette::ColorRole, QColor>& group, QPalette::ColorRole role,
                         const char* token) {
        if (const auto value = color(token))
            group.insert(role, *value);
    };

    auto& active = palette.active;
    set(active, QPalette::Window, "window");
    set(active, QPalette::WindowText, "text");
    set(active, QPalette::Base, "input");
    set(active, QPalette::AlternateBase, "alt_row");
    set(active, QPalette::ToolTipBase, "raised");
    set(active, QPalette::ToolTipText, "text");
    set(active, QPalette::PlaceholderText, "text_muted");
    set(active, QPalette::Text, "text");
    set(active, QPalette::Button, "button");
    set(active, QPalette::ButtonText, "text");
    set(active, QPalette::Highlight, "selection");
    set(active, QPalette::HighlightedText, "selection_text");
    set(active, QPalette::Link, "accent");
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
    set(active, QPalette::Accent, "accent");
#endif
    const auto window = color("window");
    const auto border = color("border");
    if (window && border) {
        active.insert(QPalette::Light, border->lighter(130));
        active.insert(QPalette::Midlight, border->lighter(115));
        active.insert(QPalette::Mid, *border);
        active.insert(QPalette::Dark, window->darker(150));
        active.insert(QPalette::Shadow, window->darker(300));
    }

    auto& disabled = palette.disabled;
    set(disabled, QPalette::Text, "text_muted");
    set(disabled, QPalette::WindowText, "text_muted");
    set(disabled, QPalette::ButtonText, "text_muted");
    set(disabled, QPalette::Base, "input_disabled");
    set(disabled, QPalette::Button, "button_disabled");
    return palette;
}

QString substituteTokens(const QString& text, const QMap<QString, QString>& values,
                         QStringList* unknown)
{
    QString result;
    result.reserve(text.size());
    qsizetype last = 0;
    auto it = placeholderPattern().globalMatch(text);
    while (it.hasNext()) {
        const auto match = it.next();
        const QString name = match.captured(1);
        result += QStringView(text).mid(last, match.capturedStart() - last);
        const auto value = values.constFind(name);
        if (value == values.cend()) {
            result += match.captured(0);
            if (unknown && !unknown->contains(name))
                unknown->append(name);
        } else {
            result += value.value();
        }
        last = match.capturedEnd();
    }
    result += QStringView(text).mid(last);
    return result;
}

QStringList placeholdersIn(const QString& text)
{
    QStringList names;
    auto it = placeholderPattern().globalMatch(text);
    while (it.hasNext()) {
        const QString name = it.next().captured(1);
        if (!names.contains(name))
            names.append(name);
    }
    names.sort();
    return names;
}

QString templateText()
{
    QFile file(QString::fromLatin1(kTemplateResource));
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return QString::fromUtf8(file.readAll());
}

QString renderStyleSheet(const ResolvedTheme& theme, const QString& iconKey,
                         const QString& classicScheme, ThemeProblems* problems)
{
    QMap<QString, QString> values = theme.tokens;
    if (!values.contains(QStringLiteral("mono")))
        values.insert(QStringLiteral("mono"), QString::fromLatin1(kDefaultMonospaceStack));
    if (!iconKey.isEmpty())
        values.insert(QStringLiteral("name"), iconKey);
    values.insert(QStringLiteral("scheme"),
                  theme.colorScheme.isEmpty() ? classicScheme : theme.colorScheme);

    QStringList parts;
    if (theme.usesTemplate()) {
        const QString text = templateText();
        if (text.isEmpty()) {
            problems->append({QString::fromLatin1(kTemplateResource),
                              QStringLiteral("the built-in style sheet template is missing")});
            return {};
        }
        values.insert(QStringLiteral("themedir"), QString::fromLatin1(kBuiltinResourceDir));
        QStringList unknown;
        parts << substituteTokens(text, values, &unknown);
        if (!unknown.isEmpty()) {
            problems->append({QString::fromLatin1(kTemplateResource),
                              QStringLiteral("the built-in template uses unknown tokens: @%1@")
                                  .arg(unknown.join(QStringLiteral("@, @")))});
        }
    }
    for (const auto& part : theme.stylesheets) {
        QString text = part.text;
        if (!part.trusted) {
            QString syntaxError;
            text = stripStyleSheetComments(part.text, &syntaxError);
            if (!syntaxError.isEmpty()) {
                problems->append({part.file, QStringLiteral("the style sheet does not parse, %1").arg(syntaxError)});
                continue;
            }
        }
        values.insert(QStringLiteral("themedir"), QDir::fromNativeSeparators(part.directory));
        QStringList unknown;
        const QString rendered = substituteTokens(text, values, &unknown);
        if (!unknown.isEmpty()) {
            problems->append({part.file, QStringLiteral("the style sheet uses unknown tokens: @%1@")
                                             .arg(unknown.join(QStringLiteral("@, @")))});
        }
        if (!part.trusted)
            checkUserStyleSheet(rendered, part, problems);
        parts << rendered;
    }
    const QString styleSheet = parts.join(QStringLiteral("\n\n"));
    if (styleSheet.size() > kMaxStyleSheetChars) {
        problems->append({theme.filePath, QStringLiteral("the rendered style sheet is larger than %1 KiB")
                                              .arg(kMaxStyleSheetChars / 1024)});
    }
    return styleSheet;
}

bool checkRendering(const ResolvedTheme& theme, ThemeProblems* problems)
{
    ThemeProblems local;
    renderStyleSheet(theme, theme.usesFusion() ? QStringLiteral("icons") : QString(),
                     QStringLiteral("dark"), &local);
    problems->append(local);
    return !hasErrors(local);
}

const std::vector<ThemeIconSpec>& themeIconSpecs()
{
    static const std::vector<ThemeIconSpec> specs = [] {
        std::vector<ThemeIconSpec> result;
        for (const auto& recipe : kIconRecipes)
            result.push_back({QString::fromLatin1(recipe.name), recipe.size});
        return result;
    }();
    return specs;
}

QString themeIconKey(const ResolvedTheme& theme)
{
    QCryptographicHash hash(QCryptographicHash::Sha1);
    hash.addData(QByteArrayView(kIconRenderVersion));
    for (auto it = theme.tokens.begin(); it != theme.tokens.end(); ++it)
        hash.addData(QStringLiteral("\n%1=%2").arg(it.key(), it.value()).toUtf8());
    return theme.id + QLatin1Char('-') + QString::fromLatin1(hash.result().toHex().left(8));
}

QString ensureThemeIcons(const QMap<QString, QString>& tokens, const QString& cacheRoot,
                         const QString& key, QString* error)
{
    QDir root(cacheRoot);
    if (cacheRoot.isEmpty() || !root.mkpath(QStringLiteral("."))) {
        *error = QStringLiteral("cannot create the theme icon folder %1").arg(cacheRoot);
        return {};
    }
    const QString target = root.filePath(key);
    if (iconFolderComplete(target))
        return target;

    const QString staging = root.filePath(QStringLiteral(".%1.tmp-%2")
                                              .arg(key)
                                              .arg(QCoreApplication::applicationPid()));
    QDir(staging).removeRecursively();
    if (!QDir().mkpath(staging)) {
        *error = QStringLiteral("cannot create %1").arg(staging);
        return {};
    }
    for (const auto& recipe : kIconRecipes) {
        const QString value = tokens.value(QLatin1String(recipe.token));
        if (!isColor(value)) {
            *error = QStringLiteral("token \"%1\" is not a color").arg(QLatin1String(recipe.token));
            QDir(staging).removeRecursively();
            return {};
        }
        const QColor color = QColor::fromString(value);
        for (const int scale : {1, 2}) {
            const QString file = QDir(staging).filePath(
                scale == 1 ? QStringLiteral("%1.png").arg(QLatin1String(recipe.name))
                           : QStringLiteral("%1@2x.png").arg(QLatin1String(recipe.name)));
            if (!drawIcon(recipe, color, scale).save(file, "PNG")) {
                *error = QStringLiteral("cannot write %1").arg(file);
                QDir(staging).removeRecursively();
                return {};
            }
        }
    }
    // Another VC3D may have finished the same folder meanwhile; a complete
    // folder is never replaced, only an unfinished one.
    if (iconFolderComplete(target)) {
        QDir(staging).removeRecursively();
        return target;
    }
    if (QFileInfo::exists(target))
        QDir(target).removeRecursively();
    if (!QDir().rename(staging, target)) {
        QDir(staging).removeRecursively();
        if (iconFolderComplete(target))
            return target;
        *error = QStringLiteral("cannot create %1").arg(target);
        return {};
    }
    pruneThemeIcons(cacheRoot, key, kIconFolderMaxAgeSeconds);
    return target;
}

void pruneThemeIcons(const QString& cacheRoot, const QString& keepKey, qint64 minAgeSeconds)
{
    static const QRegularExpression keyPattern(QStringLiteral("^(.+)-[0-9a-f]{8}$"));
    static const QRegularExpression stagingPattern(QStringLiteral("^\\..+\\.tmp-\\d+$"));
    const auto keep = keyPattern.match(keepKey);
    if (!keep.hasMatch())
        return;
    const QRegularExpression sameTheme(
        QStringLiteral("^%1-[0-9a-f]{8}$").arg(QRegularExpression::escape(keep.captured(1))));
    const QDateTime cutoff = QDateTime::currentDateTime().addSecs(-minAgeSeconds);
    const QFileInfoList entries =
        QDir(cacheRoot).entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::Hidden);
    for (const QFileInfo& entry : entries) {
        const QString name = entry.fileName();
        if (name == keepKey || entry.isSymLink() || entry.lastModified() > cutoff)
            continue;
        if (sameTheme.match(name).hasMatch() || stagingPattern.match(name).hasMatch())
            QDir(entry.absoluteFilePath()).removeRecursively();
    }
}

std::optional<QPalette::ColorRole> paletteRoleFromName(const QString& name)
{
    const QMetaEnum roles = QMetaEnum::fromType<QPalette::ColorRole>();
    for (int i = 0; i < roles.keyCount(); ++i) {
        const QString key = QString::fromLatin1(roles.key(i));
        if (key == QLatin1String("NoRole") || key == QLatin1String("NColorRoles"))
            continue;
        if (key.compare(name, Qt::CaseInsensitive) == 0)
            return static_cast<QPalette::ColorRole>(roles.value(i));
    }
    return std::nullopt;
}

QString paletteRoleName(QPalette::ColorRole role)
{
    return QString::fromLatin1(QMetaEnum::fromType<QPalette::ColorRole>().valueToKey(role));
}

QJsonObject paletteToJson(const ThemePalette& palette)
{
    QJsonObject object;
    for (auto it = palette.active.begin(); it != palette.active.end(); ++it)
        object.insert(paletteRoleName(it.key()), colorText(it.value()));
    if (!palette.disabled.isEmpty()) {
        QJsonObject disabled;
        for (auto it = palette.disabled.begin(); it != palette.disabled.end(); ++it)
            disabled.insert(paletteRoleName(it.key()), colorText(it.value()));
        object.insert(QStringLiteral("disabled"), disabled);
    }
    return object;
}

// ---------------------------------------------------------------------------

ThemeRegistry::ThemeRegistry()
{
    addBuiltins();
}

void ThemeRegistry::addBuiltins()
{
    _themes.append(classicDefinition());
    for (const char* id : {kModernDarkId, kModernLightId}) {
        const QString path = QStringLiteral("%1/%2%3").arg(QLatin1String(kBuiltinResourceDir),
                                                           QLatin1String(id),
                                                           QLatin1String(kThemeFileSuffix));
        ThemeProblems problems;
        const auto definition =
            loadThemeFile(path, QString::fromLatin1(id), ThemeSource::Builtin, false, &problems);
        if (!definition) {
            qWarning("VC3D theme: built-in theme %s is unusable:\n%s", id,
                     qPrintable(formatProblems(problems)));
            continue;
        }
        _themes.append(*definition);
    }
}

ThemeProblems ThemeRegistry::scan(const QString& userFolder, const QStringList& pathFolders)
{
    _themes.erase(std::remove_if(_themes.begin(), _themes.end(),
                                 [](const ThemeDefinition& theme) {
                                     return theme.source != ThemeSource::Builtin;
                                 }),
                  _themes.end());
    ThemeProblems problems;
    if (!userFolder.isEmpty())
        scanFolder(userFolder, ThemeSource::User, &problems);
    for (const QString& folder : pathFolders) {
        if (folder.trimmed().isEmpty())
            continue;
        if (!QFileInfo(folder).isDir()) {
            problems.append({folder, QStringLiteral("VC3D_THEME_PATH folder does not exist"), true});
            continue;
        }
        scanFolder(folder, ThemeSource::Path, &problems);
    }

    // A theme whose base is missing, cyclic, or whose style sheet does not
    // render is reported and left out of the list.
    QStringList broken;
    for (const auto& theme : _themes) {
        if (theme.source == ThemeSource::Builtin)
            continue;
        ThemeProblems themeProblems;
        const auto resolved = resolve(theme, &themeProblems);
        if (resolved)
            checkRendering(*resolved, &themeProblems);
        if (hasErrors(themeProblems))
            broken << theme.id;
        for (const auto& problem : themeProblems) {
            if (!problem.warning)
                problems.append(problem);
        }
    }
    _themes.erase(std::remove_if(_themes.begin(), _themes.end(),
                                 [&](const ThemeDefinition& theme) {
                                     return theme.source != ThemeSource::Builtin &&
                                            broken.contains(theme.id);
                                 }),
                  _themes.end());
    return problems;
}

void ThemeRegistry::scanFolder(const QString& folder, ThemeSource source, ThemeProblems* problems)
{
    const QDir dir(folder);
    if (!dir.exists())
        return;
    const QFileInfoList entries =
        dir.entryInfoList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot | QDir::Readable,
                          QDir::Name | QDir::DirsLast);
    for (const QFileInfo& entry : entries) {
        if (entry.fileName().startsWith(QLatin1Char('.')))
            continue;
        if (entry.isDir()) {
            if (!QFileInfo(QDir(entry.absoluteFilePath()).filePath(QLatin1String(kFolderThemeFileName))).isFile())
                continue;
        } else if (!entry.fileName().endsWith(QLatin1String(kThemeFileSuffix), Qt::CaseInsensitive)) {
            continue;
        } else if (entry.fileName() == QLatin1String(kFolderThemeFileName)) {
            problems->append({entry.absoluteFilePath(),
                              QStringLiteral("%1 belongs inside a theme's own folder; skipped")
                                  .arg(quoted(QLatin1String(kFolderThemeFileName))),
                              true});
            continue;
        }
        auto definition = loadTheme(entry.absoluteFilePath(), source, problems);
        if (!definition)
            continue;
        if (isBuiltinThemeId(definition->id)) {
            problems->append({definition->filePath,
                              QStringLiteral("the id %1 belongs to a built-in theme; skipped")
                                  .arg(quoted(definition->id)), true});
            continue;
        }
        if (const ThemeDefinition* existing = find(definition->id)) {
            problems->append({definition->filePath,
                              QStringLiteral("another theme already uses the id %1 (%2); skipped")
                                  .arg(quoted(definition->id), existing->filePath), true});
            continue;
        }
        _themes.append(std::move(*definition));
    }
}

bool ThemeRegistry::add(ThemeDefinition definition, ThemeProblems* problems)
{
    if (isBuiltinThemeId(definition.id)) {
        problems->append({definition.filePath,
                          QStringLiteral("the id %1 belongs to a built-in theme").arg(quoted(definition.id))});
        return false;
    }
    for (auto& theme : _themes) {
        if (theme.id == definition.id) {
            theme = std::move(definition);
            return true;
        }
    }
    _themes.append(std::move(definition));
    return true;
}

const ThemeDefinition* ThemeRegistry::find(const QString& id) const
{
    for (const auto& theme : _themes) {
        if (theme.id == id)
            return &theme;
    }
    return nullptr;
}

QList<const ThemeDefinition*> ThemeRegistry::themes() const
{
    QList<const ThemeDefinition*> result;
    for (const QString& id : builtinThemeIds()) {
        if (const auto* theme = find(id))
            result.append(theme);
    }
    QList<const ThemeDefinition*> others;
    for (const auto& theme : _themes) {
        if (theme.source != ThemeSource::Builtin)
            others.append(&theme);
    }
    std::stable_sort(others.begin(), others.end(), [](const auto* a, const auto* b) {
        const int byName = a->name.compare(b->name, Qt::CaseInsensitive);
        return byName != 0 ? byName < 0 : a->id < b->id;
    });
    result.append(others);
    return result;
}

std::optional<ResolvedTheme> ThemeRegistry::resolve(const QString& id, ThemeProblems* problems) const
{
    const ThemeDefinition* definition = find(id);
    if (!definition) {
        problems->append({QString(), QStringLiteral("unknown theme %1").arg(quoted(id))});
        return std::nullopt;
    }
    return resolve(*definition, problems);
}

std::optional<ResolvedTheme> ThemeRegistry::resolve(const ThemeDefinition& definition,
                                                    ThemeProblems* problems) const
{
    std::vector<const ThemeDefinition*> chain{&definition};
    QStringList ids{definition.id};
    while (!chain.back()->base.isEmpty()) {
        const ThemeDefinition* current = chain.back();
        if (static_cast<int>(chain.size()) > kMaxBaseDepth) {
            problems->append({definition.filePath,
                              QStringLiteral("the chain of base themes is deeper than %1: %2")
                                  .arg(kMaxBaseDepth)
                                  .arg(ids.join(QStringLiteral(" -> ")))});
            return std::nullopt;
        }
        const ThemeDefinition* base = find(current->base);
        if (!base) {
            QStringList known;
            for (const auto& theme : _themes)
                known << theme.id;
            QStringList shown = known.mid(0, 12);
            if (known.size() > shown.size())
                shown << QStringLiteral("...");
            problems->append({current->filePath,
                              QStringLiteral("base theme %1 not found%2; the themes are %3")
                                  .arg(quoted(current->base), suggestion(current->base, known),
                                       shown.join(QStringLiteral(", ")))});
            return std::nullopt;
        }
        if (ids.contains(base->id)) {
            problems->append({definition.filePath,
                              QStringLiteral("base themes form a cycle: %1 -> %2")
                                  .arg(ids.join(QStringLiteral(" -> ")), base->id)});
            return std::nullopt;
        }
        ids << base->id;
        chain.push_back(base);
    }

    ResolvedTheme resolved;
    resolved.id = definition.id;
    resolved.name = definition.name;
    resolved.author = definition.author;
    resolved.description = definition.description;
    resolved.source = definition.source;
    resolved.filePath = definition.filePath;
    resolved.rootId = chain.back()->id;
    const bool derived = resolved.rootId != QLatin1String(kClassicId);

    QMap<QString, QString> tokens;
    ThemePalette previousDerived;
    bool root = true;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const ThemeDefinition& level = **it;
        resolved.chain << level.id;
        QMap<QString, QString> merged = tokens;
        for (auto token = level.tokens.begin(); token != level.tokens.end(); ++token)
            merged.insert(token.key(), token.value());

        // Roles that follow the tokens are derived again wherever a level
        // changes the tokens they come from, so a theme that recolors
        // "window" also recolors the Window role inherited from its base.
        if (derived) {
            const ThemePalette current = derivePalette(merged);
            const auto refresh = [&](const QMap<QPalette::ColorRole, QColor>& now,
                                     const QMap<QPalette::ColorRole, QColor>& before,
                                     QMap<QPalette::ColorRole, QColor>& out) {
                for (auto role = now.begin(); role != now.end(); ++role) {
                    if (root || before.value(role.key()) != role.value())
                        out.insert(role.key(), role.value());
                }
            };
            refresh(current.active, previousDerived.active, resolved.palette.active);
            refresh(current.disabled, previousDerived.disabled, resolved.palette.disabled);
            previousDerived = current;
        }
        for (auto role = level.palette.active.begin(); role != level.palette.active.end(); ++role)
            resolved.palette.active.insert(role.key(), role.value());
        for (auto role = level.palette.disabled.begin(); role != level.palette.disabled.end(); ++role)
            resolved.palette.disabled.insert(role.key(), role.value());
        tokens = merged;

        if (!level.colorScheme.isEmpty())
            resolved.colorScheme = level.colorScheme;
        if (!level.font.family.isEmpty())
            resolved.font.family = level.font.family;
        if (level.font.pointSize > 0.0)
            resolved.font.pointSize = level.font.pointSize;
        if (level.replaceStylesheet) {
            resolved.stylesheets.clear();
            resolved.replaced = true;
        }
        if (!level.stylesheet.isEmpty()) {
            resolved.stylesheets.push_back(
                {level.stylesheet, level.directory,
                 level.stylesheetFile.isEmpty() ? level.filePath : level.stylesheetFile,
                 level.source == ThemeSource::Builtin});
        }
        if (level.source != ThemeSource::Builtin) {
            resolved.watchFiles << level.filePath;
            if (!level.stylesheetFile.isEmpty())
                resolved.watchFiles << level.stylesheetFile;
        }
        root = false;
    }
    resolved.tokens = tokens;
    return resolved;
}

// ---------------------------------------------------------------------------

QByteArray exportThemeJson(const ResolvedTheme& theme, const ThemeRegistry& registry,
                           ThemeProblems* problems)
{
    ResolvedTheme source = theme;
    QString description = QStringLiteral("Based on %1. Describe your theme here.").arg(theme.name);
    if (theme.id == QLatin1String(kClassicId)) {
        const auto modern = registry.resolve(QString::fromLatin1(kModernDarkId), problems);
        if (!modern)
            return {};
        source = *modern;
        description = QStringLiteral(
            "Classic has no tokens, so this template starts from Modern Dark. "
            "Describe your theme here.");
    }

    // The palette lists every role the tokens do not produce on their own,
    // including roles no token covers, so a light theme made from a dark
    // template shows each dark value it has to change. Roles that follow the
    // tokens are left out, so editing a token still recolors them.
    ThemeDefinition probe;
    probe.id = QStringLiteral("export");
    probe.base = source.rootId;
    probe.tokens = source.tokens;
    probe.source = ThemeSource::User;
    const auto followed = registry.resolve(probe, problems);
    if (!followed)
        return {};
    const ThemePalette covered = source.isClassic() ? ThemePalette() : derivePalette(source.tokens);
    ThemePalette palette;
    const auto pick = [](const QMap<QPalette::ColorRole, QColor>& roles,
                         const QMap<QPalette::ColorRole, QColor>& follow,
                         const QMap<QPalette::ColorRole, QColor>& tokenRoles,
                         QMap<QPalette::ColorRole, QColor>& out) {
        for (auto it = roles.begin(); it != roles.end(); ++it) {
            if (follow.value(it.key()) != it.value() || !tokenRoles.contains(it.key()))
                out.insert(it.key(), it.value());
        }
    };
    pick(source.palette.active, followed->palette.active, covered.active, palette.active);
    pick(source.palette.disabled, followed->palette.disabled, covered.disabled, palette.disabled);

    QJsonObject object;
    object.insert(QStringLiteral("vc3dTheme"), kFormatVersion);
    object.insert(QLatin1String(kCommentKey),
                  QStringLiteral("A VC3D theme. Give it a name, change the values you want, and "
                                 "leave the rest; volume-cartographer/docs/vc3d_themes.md lists "
                                 "what each token colors."));
    object.insert(QStringLiteral("name"), source.name + QStringLiteral(" (copy)"));
    object.insert(QStringLiteral("author"), QString());
    object.insert(QStringLiteral("description"), description);
    object.insert(QStringLiteral("base"), source.rootId);
    if (!source.colorScheme.isEmpty())
        object.insert(QStringLiteral("colorScheme"), source.colorScheme);
    if (!source.tokens.isEmpty() || !source.isClassic()) {
        QJsonObject tokens;
        for (auto it = source.tokens.begin(); it != source.tokens.end(); ++it)
            tokens.insert(it.key(), it.value());
        object.insert(QStringLiteral("tokens"), tokens);
    }
    object.insert(QStringLiteral("palette"), paletteToJson(palette));
    if (!source.font.family.isEmpty() || source.font.pointSize > 0.0) {
        QJsonObject font;
        if (!source.font.family.isEmpty())
            font.insert(QStringLiteral("family"), source.font.family);
        if (source.font.pointSize > 0.0)
            font.insert(QStringLiteral("pointSize"), source.font.pointSize);
        object.insert(QStringLiteral("font"), font);
    }
    QStringList styleSheets;
    for (const auto& part : source.stylesheets) {
        styleSheets << part.text;
        if (part.text.contains(QStringLiteral("@themedir@"))) {
            problems->append({part.file, QStringLiteral("the style sheet uses files from its theme's folder "
                                                        "(@themedir@), which are not exported; copy them "
                                                        "into a theme folder next to the exported file"),
                              true});
        }
    }
    if (!styleSheets.isEmpty())
        object.insert(QStringLiteral("qss"), styleSheets.join(QStringLiteral("\n\n")));
    if (source.replaced)
        object.insert(QStringLiteral("stylesheetMode"), QStringLiteral("replace"));
    return QJsonDocument(object).toJson(QJsonDocument::Indented);
}

ImportResult importTheme(const QString& location, const QString& userFolder,
                         bool overwrite, const ThemeRegistry& registry)
{
    ImportResult result;
    const auto fail = [&](const QString& file, const QString& message) {
        result.problems.append({file, message});
        return result;
    };
    const QFileInfo sourceInfo(location);
    const QString sourcePath = QDir::cleanPath(sourceInfo.absoluteFilePath());
    const bool folderTheme = sourceInfo.isDir();
    if (!folderTheme) {
        if (!sourceInfo.exists())
            return fail(location, QStringLiteral("the theme file does not exist"));
        if (sourceInfo.fileName() == QLatin1String(kFolderThemeFileName)) {
            return fail(sourcePath, QStringLiteral("%1 is the main file of a folder theme; import its "
                                                   "folder, or rename the file to import it alone")
                                        .arg(quoted(QLatin1String(kFolderThemeFileName))));
        }
        if (!sourceInfo.isFile())
            return fail(sourcePath, QStringLiteral("not a regular file"));
    }
    const QString rawId = folderTheme ? QFileInfo(sourcePath).fileName() : idFromLocation(sourceInfo, false);
    const QString id = normalizeThemeId(rawId);
    if (id.isEmpty()) {
        return fail(sourcePath, QStringLiteral("cannot make a theme id from the name %1; use letters or digits")
                                    .arg(quoted(rawId)));
    }
    result.id = id;
    if (isBuiltinThemeId(id)) {
        return fail(sourcePath, QStringLiteral("the id %1 belongs to a built-in theme; rename the file "
                                               "or folder to import it")
                                    .arg(quoted(id)));
    }

    QDir user(userFolder);
    if (userFolder.isEmpty() || !user.mkpath(QStringLiteral(".")))
        return fail(userFolder, QStringLiteral("cannot create the user themes folder"));
    const QString fileTarget = user.filePath(id + QLatin1String(kThemeFileSuffix));
    const QString folderTarget = user.filePath(id);
    const QString canonicalSource = QFileInfo(sourcePath).canonicalFilePath();
    const QString userCanonical = QFileInfo(userFolder).canonicalFilePath();
    if (folderTheme && (canonicalSource == userCanonical || isInside(userCanonical, canonicalSource)))
        return fail(sourcePath, QStringLiteral("the theme folder contains the user themes folder"));
    // Importing an installed theme again only checks it.
    QString installedSelf;
    for (const QString& target : {fileTarget, folderTarget}) {
        if (!canonicalSource.isEmpty() && QFileInfo(target).canonicalFilePath() == canonicalSource)
            installedSelf = target;
    }
    if (installedSelf.isEmpty() && !overwrite &&
        (QFileInfo::exists(fileTarget) || QFileInfo::exists(folderTarget))) {
        result.alreadyExists = true;
        result.installedPath = QFileInfo::exists(folderTarget) ? folderTarget : fileTarget;
        return fail(result.installedPath,
                    QStringLiteral("a theme with the id %1 is already installed").arg(quoted(id)));
    }

    // Everything is checked on a private copy, so what is installed is what
    // was checked.
    const qint64 pid = QCoreApplication::applicationPid();
    const QString staging = user.filePath(QStringLiteral(".import-%1-%2").arg(id).arg(pid));
    const QString stagedFile =
        user.filePath(QStringLiteral(".import-%1-%2%3").arg(id).arg(pid).arg(QLatin1String(kThemeFileSuffix)));
    const auto cleanup = qScopeGuard([&] {
        removeInstalled(staging);
        removeInstalled(stagedFile);
    });
    removeInstalled(staging);
    removeInstalled(stagedFile);
    const QString stagedTheme = QDir(staging).filePath(QLatin1String(kFolderThemeFileName));
    const QString sourceFolder = folderTheme ? sourcePath : sourceInfo.absolutePath();
    QString error;
    if (folderTheme) {
        if (!copyThemeFolder(sourcePath, staging, &error))
            return fail(sourcePath, error);
    } else {
        if (!QDir().mkpath(staging) || !QFile::copy(sourcePath, stagedTheme))
            return fail(sourcePath, QStringLiteral("cannot copy the theme into %1").arg(userFolder));
        // A separate style sheet travels with the theme file. A bad path, a
        // missing file or broken JSON is reported by the check below.
        QByteArray json;
        if (readFile(stagedTheme, &json, &error)) {
            const QJsonValue sheet = QJsonDocument::fromJson(json).object().value(QStringLiteral("stylesheet"));
            const QString relative = sheet.toString().trimmed();
            const QString from = QDir::cleanPath(QDir(sourceFolder).filePath(relative));
            const QString to = QDir::cleanPath(QDir(staging).filePath(relative));
            const QFileInfo fromInfo(from);
            if (sheet.isString() && !relative.isEmpty() && !QDir::isAbsolutePath(relative) &&
                isInside(from, sourceFolder) && isInside(to, staging) &&
                (fromInfo.exists() || fromInfo.isSymLink())) {
                if (fromInfo.isSymLink() || !fromInfo.isFile())
                    return fail(from, QStringLiteral("the style sheet must be a regular file, not a link"));
                if (!QDir().mkpath(QFileInfo(to).absolutePath()) || !QFile::copy(from, to))
                    return fail(from, QStringLiteral("cannot copy the style sheet into %1").arg(userFolder));
            }
        }
    }

    const QString shownTheme = folderTheme ? QDir(sourcePath).filePath(QLatin1String(kFolderThemeFileName))
                                           : sourcePath;
    ThemeProblems checks;
    const auto definition = loadThemeFile(stagedTheme, id, ThemeSource::User, folderTheme, &checks);
    bool valid = definition.has_value();
    if (valid && !folderTheme && definition->stylesheet.contains(QStringLiteral("@themedir@"))) {
        checks.append({stagedTheme, QStringLiteral("the style sheet uses files from the theme's folder "
                                                   "(@themedir@), which only a folder theme carries; put "
                                                   "the theme (as theme.vc3dtheme), its style sheet and "
                                                   "its images in one folder and import that folder")});
        valid = false;
    }
    if (valid) {
        const auto resolved = registry.resolve(*definition, &checks);
        valid = resolved && checkRendering(*resolved, &checks) && !hasErrors(checks);
    }
    relabelProblems(&checks, stagedTheme, shownTheme);
    relabelProblems(&checks, staging, sourceFolder);
    result.problems.append(checks);
    if (!valid)
        return result;
    if (!installedSelf.isEmpty()) {
        result.installedPath = installedSelf;
        result.ok = true;
        return result;
    }

    const bool asFolder = folderTheme || !definition->stylesheetFile.isEmpty();
    const QString target = asFolder ? folderTarget : fileTarget;
    const QString staged = asFolder ? staging : stagedFile;
    if (!asFolder && !QFile::rename(stagedTheme, stagedFile))
        return fail(target, QStringLiteral("cannot install the theme"));

    // An installed theme with this id is moved aside first and put back if
    // the new one cannot be moved in.
    QList<QPair<QString, QString>> backups;
    const auto restore = [&]() {
        for (const auto& [installed, backup] : backups) {
            if (!QDir().rename(backup, installed)) {
                qWarning("VC3D theme: could not restore %s from %s", qPrintable(installed),
                         qPrintable(backup));
            }
        }
    };
    for (const QString& installed : {fileTarget, folderTarget}) {
        const QFileInfo info(installed);
        if (!info.exists() && !info.isSymLink())
            continue;
        const QString backup = user.filePath(QStringLiteral(".backup-%1-%2").arg(info.fileName()).arg(pid));
        removeInstalled(backup);
        if (!QDir().rename(installed, backup)) {
            restore();
            return fail(installed, QStringLiteral("cannot replace the installed theme; it was kept"));
        }
        backups.append({installed, backup});
    }
    if (!QDir().rename(staged, target)) {
        restore();
        return fail(target, QStringLiteral("cannot install the theme; the installed version was kept"));
    }
    for (const auto& backup : backups)
        removeInstalled(backup.second);
    result.installedPath = target;
    result.ok = true;
    return result;
}

}  // namespace vc3d::theme
