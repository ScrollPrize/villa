#include "themes/ClassicTheme.hpp"
#include "themes/ThemeEngine.hpp"
#include "themes/ThemeManager.hpp"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QStyle>
#include <QStyleFactory>
#include <QTemporaryDir>
#include <QtTest/QtTest>

#include <algorithm>
#include <cmath>

#ifdef Q_OS_UNIX
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace vc3d::theme;

namespace {

void writeFile(const QString& path, const QByteArray& contents)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    QVERIFY2(file.open(QIODevice::WriteOnly | QIODevice::Truncate), qPrintable(path));
    file.write(contents);
}

QByteArray themeJson(const QString& name, const QString& base = QString(),
                     const QString& extra = QString())
{
    QString json = QStringLiteral("{\"vc3dTheme\": 1, \"name\": \"%1\"").arg(name);
    if (!base.isEmpty())
        json += QStringLiteral(", \"base\": \"%1\"").arg(base);
    if (!extra.isEmpty())
        json += QStringLiteral(", ") + extra;
    // Tests write JSON with single quotes for readability.
    json.replace(QLatin1Char('\''), QLatin1Char('"'));
    json += QLatin1Char('}');
    return json.toUtf8();
}

bool containsMessage(const ThemeProblems& problems, const QString& text, bool warning)
{
    for (const auto& problem : problems) {
        if (problem.warning == warning && problem.message.contains(text))
            return true;
    }
    return false;
}

QString styleName()
{
    const QStyle* style = QApplication::style();
    if (style->name().isEmpty()) {
        if (const auto* inner = style->findChild<QStyle*>())
            return inner->name();
    }
    return style->name();
}

std::optional<ThemeDefinition> parse(QByteArray json, ThemeProblems* problems,
                                     const QString& id = QStringLiteral("test"))
{
    json.replace('\'', '"');
    return parseTheme(json, QStringLiteral("/tmp/%1.vc3dtheme").arg(id), id, ThemeSource::User,
                      problems);
}

// Problems of a folder theme whose extra style sheet is `qss`, rendered the
// way it would be applied.
ThemeProblems styleSheetProblems(const QString& folder, const QByteArray& qss)
{
    writeFile(QDir(folder).filePath("theme.vc3dtheme"), themeJson("Probe", {}, "'stylesheet': 'extra.qss'"));
    writeFile(QDir(folder).filePath("extra.qss"), qss);
    ThemeProblems problems;
    const auto definition = loadTheme(folder, ThemeSource::User, &problems);
    if (!definition)
        return problems;
    ThemeRegistry registry;
    if (const auto theme = registry.resolve(*definition, &problems))
        checkRendering(*theme, &problems);
    return problems;
}

QStringList entriesIn(const QString& folder)
{
    QStringList entries =
        QDir(folder).entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System);
    entries.sort();
    return entries;
}

}  // namespace

class ThemeEngineTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase()
    {
        QStandardPaths::setTestModeEnabled(true);
        qunsetenv("VC3D_THEME");
        qunsetenv("VC3D_THEME_PATH");
        // Start from a style other than Fusion, so switching back to classic
        // has to restore it.
        if (QStyleFactory::keys().contains(QStringLiteral("Windows"), Qt::CaseInsensitive))
            QApplication::setStyle(QStringLiteral("Windows"));
    }

    void builtinThemesResolve()
    {
        ThemeRegistry registry;
        QCOMPARE(registry.themes().size(), 3);
        QCOMPARE(registry.themes().at(0)->id, QStringLiteral("classic"));
        for (const QString& id : builtinThemeIds()) {
            ThemeProblems problems;
            const auto theme = registry.resolve(id, &problems);
            QVERIFY2(theme && problems.isEmpty(), qPrintable(formatProblems(problems)));
            QCOMPARE(theme->rootId, id);
        }
        QVERIFY(!builtinTokens().isEmpty());
        ThemeProblems problems;
        QCOMPARE(registry.resolve(QStringLiteral("modern-dark"), &problems)->colorScheme,
                 QStringLiteral("dark"));
        QCOMPARE(registry.resolve(QStringLiteral("modern-light"), &problems)->colorScheme,
                 QStringLiteral("light"));
        QVERIFY(registry.resolve(QStringLiteral("classic"), &problems)->colorScheme.isEmpty());
    }

    void templatePlaceholdersAreKnown()
    {
        const QString text = templateText();
        QVERIFY(!text.isEmpty());
        ThemeRegistry registry;
        const QStringList engine{QStringLiteral("name"), QStringLiteral("mono"),
                                 QStringLiteral("scheme"), QStringLiteral("themedir")};
        for (const QString& id : {QStringLiteral("modern-dark"), QStringLiteral("modern-light")}) {
            ThemeProblems problems;
            const auto theme = registry.resolve(id, &problems);
            QVERIFY(theme);
            for (const QString& name : placeholdersIn(text)) {
                QVERIFY2(theme->tokens.contains(name) || engine.contains(name),
                         qPrintable(QStringLiteral("%1 lacks token @%2@").arg(id, name)));
            }
            const QString sheet = renderStyleSheet(*theme, QStringLiteral("key"),
                                                   QStringLiteral("dark"), &problems);
            QVERIFY2(problems.isEmpty(), qPrintable(formatProblems(problems)));
            QVERIFY(placeholdersIn(sheet).isEmpty());
            QVERIFY(sheet.contains(theme->tokens.value(QStringLiteral("accent"))));
        }
    }

    void templateIconsAreRendered()
    {
        QStringList rendered;
        for (const auto& spec : themeIconSpecs())
            rendered << spec.name + QStringLiteral(".png");
        const QRegularExpression url(QStringLiteral("vc3dtheme:@name@/([A-Za-z0-9@_.-]+)"));
        auto it = url.globalMatch(templateText());
        int references = 0;
        while (it.hasNext()) {
            const QString file = it.next().captured(1);
            QVERIFY2(rendered.contains(file), qPrintable(file));
            ++references;
        }
        QVERIFY(references > 0);
    }

    void parseValidation()
    {
        {
            ThemeProblems problems;
            QVERIFY(!parse("{'vc3dTheme': 1}", &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("missing \"name\""), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse("{'vc3dTheme': 2, 'name': 'Future'}", &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("format version 2"), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse("{'name': 'No version'}", &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("vc3dTheme"), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse(themeJson("Bad", {}, "'tokens': {'accent': 'not-a-color'}"), &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("must be a color"), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse(themeJson("Bad", {}, "'palette': {'Window': '#12'}"), &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("palette.Window"), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse(themeJson("Bad", {}, "'colorScheme': 'purple'"), &problems));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse(themeJson("Bad", {}, "'qss': 'a', 'stylesheet': 'b.qss'"), &problems));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse(themeJson("Bad", {}, "'tokens': {'name': '#ffffff'}"), &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("reserved"), false));
        }
        {
            ThemeProblems problems;
            QVERIFY(!parse("{'vc3dTheme': 1,\n 'name': }", &problems));
            QVERIFY(containsMessage(problems, QStringLiteral("line 2"), false));
        }
        {
            ThemeProblems problems;
            const auto theme = parse(
                themeJson("Warned", {}, "'tokens': {'my_glow': '#123456'}, 'flavor': 1"), &problems);
            QVERIFY(theme);
            QVERIFY(containsMessage(problems, QStringLiteral("unknown token \"my_glow\""), true));
            QVERIFY(containsMessage(problems, QStringLiteral("unknown key \"flavor\""), true));
            QVERIFY(!hasErrors(problems));
            QCOMPARE(theme->base, QStringLiteral("modern-dark"));
            QCOMPARE(theme->tokens.value(QStringLiteral("my_glow")), QStringLiteral("#123456"));
        }
    }

    void idsAreNormalized()
    {
        QCOMPARE(normalizeThemeId(QStringLiteral("Ocean Night")), QStringLiteral("ocean-night"));
        QCOMPARE(normalizeThemeId(QStringLiteral("  __High_Contrast!! ")), QStringLiteral("high-contrast"));
        QCOMPARE(normalizeThemeId(QStringLiteral("%%%")), QString());
    }

    void baseResolution()
    {
        QTemporaryDir dir;
        writeFile(dir.filePath("parent.vc3dtheme"),
                  themeJson("Parent", "modern-dark",
                            "'tokens': {'window': '#102030', 'accent': '#ff8800'}"));
        writeFile(dir.filePath("child.vc3dtheme"),
                  themeJson("Child", "parent", "'tokens': {'accent': '#00ff00'}"));
        writeFile(dir.filePath("loop-a.vc3dtheme"), themeJson("A", "loop-b"));
        writeFile(dir.filePath("loop-b.vc3dtheme"), themeJson("B", "loop-a"));
        writeFile(dir.filePath("orphan.vc3dtheme"), themeJson("Orphan", "nowhere"));
        writeFile(dir.filePath("deep-0.vc3dtheme"), themeJson("Deep 0", "modern-dark"));
        for (int i = 1; i <= 9; ++i) {
            writeFile(dir.filePath(QStringLiteral("deep-%1.vc3dtheme").arg(i)),
                      themeJson(QStringLiteral("Deep %1").arg(i), QStringLiteral("deep-%1").arg(i - 1)));
        }

        ThemeRegistry registry;
        const ThemeProblems problems = registry.scan(dir.path(), {});
        QVERIFY(containsMessage(problems, QStringLiteral("cycle"), false));
        QVERIFY(containsMessage(problems, QStringLiteral("base theme \"nowhere\" not found"), false));
        QVERIFY(containsMessage(problems, QStringLiteral("deeper than 8"), false));
        QVERIFY(!registry.contains(QStringLiteral("loop-a")));
        QVERIFY(!registry.contains(QStringLiteral("orphan")));
        QVERIFY(registry.contains(QStringLiteral("deep-6")));
        QVERIFY(!registry.contains(QStringLiteral("deep-8")));

        ThemeProblems resolveProblems;
        const auto child = registry.resolve(QStringLiteral("child"), &resolveProblems);
        QVERIFY(child);
        QCOMPARE(child->chain, QStringList({"modern-dark", "parent", "child"}));
        QCOMPARE(child->rootId, QStringLiteral("modern-dark"));
        QCOMPARE(child->tokens.value("accent"), QStringLiteral("#00ff00"));
        QCOMPARE(child->tokens.value("window"), QStringLiteral("#102030"));
        QCOMPARE(child->tokens.value("text"), builtinTokens().value("text"));
        QCOMPARE(child->colorScheme, QStringLiteral("dark"));
        // Roles that follow a changed token are derived again; the rest keep
        // the base's explicit values.
        QCOMPARE(child->palette.active.value(QPalette::Window), QColor("#102030"));
        QCOMPARE(child->palette.active.value(QPalette::Link), QColor("#00ff00"));
        QCOMPARE(child->palette.active.value(QPalette::LinkVisited), QColor("#a78bfa"));
        QCOMPARE(child->watchFiles.size(), 2);
    }

    void builtinIdsAreProtected()
    {
        QTemporaryDir dir;
        writeFile(dir.filePath("modern-dark.vc3dtheme"), themeJson("Impostor"));
        ThemeRegistry registry;
        const ThemeProblems problems = registry.scan(dir.path(), {});
        QVERIFY(containsMessage(problems, QStringLiteral("built-in theme"), true));
        QCOMPARE(registry.find(QStringLiteral("modern-dark"))->source, ThemeSource::Builtin);

        ThemeDefinition shadow;
        shadow.id = QStringLiteral("classic");
        ThemeProblems addProblems;
        QVERIFY(!registry.add(shadow, &addProblems));

        QTemporaryDir user;
        const ImportResult result =
            importTheme(dir.filePath("modern-dark.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(containsMessage(result.problems, QStringLiteral("built-in theme"), false));
    }

    void duplicateIdsKeepTheFirst()
    {
        QTemporaryDir user;
        QTemporaryDir shared;
        writeFile(user.filePath("ocean.vc3dtheme"), themeJson("Ocean (user)"));
        writeFile(user.filePath("ocean/theme.vc3dtheme"), themeJson("Ocean (folder)"));
        writeFile(shared.filePath("ocean.vc3dtheme"), themeJson("Ocean (shared)"));
        writeFile(shared.filePath("reef.vc3dtheme"), themeJson("Reef"));
        ThemeRegistry registry;
        const ThemeProblems problems = registry.scan(user.path(), {shared.path()});
        QCOMPARE(registry.find("ocean")->name, QStringLiteral("Ocean (user)"));
        QCOMPARE(registry.find("ocean")->source, ThemeSource::User);
        QCOMPARE(registry.find("reef")->source, ThemeSource::Path);
        int duplicates = 0;
        for (const auto& problem : problems)
            duplicates += problem.message.contains(QStringLiteral("already uses the id")) ? 1 : 0;
        QCOMPARE(duplicates, 2);
    }

    void tokenSubstitution()
    {
        QMap<QString, QString> values{{"accent", "#123456"}, {"name", "key"}};
        QStringList unknown;
        const QString out = substituteTokens(
            QStringLiteral("a { color: @accent@; image: url(vc3dtheme:@name@/x@2x.png); b: @nope@ }"),
            values, &unknown);
        QCOMPARE(out, QStringLiteral("a { color: #123456; image: url(vc3dtheme:key/x@2x.png); b: @nope@ }"));
        QCOMPARE(unknown, QStringList{"nope"});

        ThemeRegistry registry;
        ThemeProblems problems;
        auto definition = parse(themeJson("Extra", {}, "'qss': 'QLabel { color: @missing@; }'"),
                                &problems);
        QVERIFY(definition);
        const auto theme = registry.resolve(*definition, &problems);
        QVERIFY(theme);
        renderStyleSheet(*theme, QStringLiteral("key"), QStringLiteral("dark"), &problems);
        QVERIFY(containsMessage(problems, QStringLiteral("@missing@"), false));
    }

    void extraStylesheetAppendAndReplace()
    {
        QTemporaryDir dir;
        writeFile(dir.filePath("grain/theme.vc3dtheme"),
                  themeJson("Grain", {}, "'stylesheet': 'extra.qss'"));
        writeFile(dir.filePath("grain/extra.qss"),
                  "QLabel { color: @accent@; background: url(@themedir@/grain.png); }");
        writeFile(dir.filePath("grain/grain.png"), "png");
        writeFile(dir.filePath("bare.vc3dtheme"),
                  themeJson("Bare", {},
                            "'qss': 'QWidget { color: @text@; }', 'stylesheetMode': 'replace'"));
        ThemeRegistry registry;
        QVERIFY(registry.scan(dir.path(), {}).isEmpty());

        ThemeProblems problems;
        const auto grain = registry.resolve(QStringLiteral("grain"), &problems);
        QVERIFY(grain && grain->usesTemplate());
        const QString sheet = renderStyleSheet(*grain, QStringLiteral("k"), QStringLiteral("dark"), &problems);
        QVERIFY2(problems.isEmpty(), qPrintable(formatProblems(problems)));
        QVERIFY(sheet.startsWith(substituteTokens(templateText().left(40), {})));
        QVERIFY(sheet.endsWith(QStringLiteral("QLabel { color: %1; background: url(%2/grain.png); }")
                                   .arg(builtinTokens().value("accent"),
                                        QDir(dir.filePath("grain")).absolutePath())));
        QVERIFY(grain->watchFiles.contains(QDir::cleanPath(dir.filePath("grain/extra.qss"))));

        const auto bare = registry.resolve(QStringLiteral("bare"), &problems);
        QVERIFY(bare && bare->replaced && !bare->usesTemplate() && bare->usesFusion());
        QCOMPARE(renderStyleSheet(*bare, QStringLiteral("k"), QStringLiteral("dark"), &problems),
                 QStringLiteral("QWidget { color: %1; }").arg(builtinTokens().value("text")));
    }

    void derivedPalette()
    {
        const auto& tokens = builtinTokens();
        const ThemePalette palette = derivePalette(tokens);
        const auto color = [&](const char* token) { return QColor(tokens.value(QLatin1String(token))); };
        QCOMPARE(palette.active.value(QPalette::Window), color("window"));
        QCOMPARE(palette.active.value(QPalette::WindowText), color("text"));
        QCOMPARE(palette.active.value(QPalette::Base), color("input"));
        QCOMPARE(palette.active.value(QPalette::AlternateBase), color("alt_row"));
        QCOMPARE(palette.active.value(QPalette::ToolTipBase), color("raised"));
        QCOMPARE(palette.active.value(QPalette::PlaceholderText), color("text_muted"));
        QCOMPARE(palette.active.value(QPalette::Button), color("button"));
        QCOMPARE(palette.active.value(QPalette::Highlight), color("selection"));
        QCOMPARE(palette.active.value(QPalette::HighlightedText), color("selection_text"));
        QCOMPARE(palette.active.value(QPalette::Link), color("accent"));
        QCOMPARE(palette.active.value(QPalette::Mid), color("border"));
        QVERIFY(palette.active.contains(QPalette::Light));
        QVERIFY(palette.active.contains(QPalette::Shadow));
        QCOMPARE(palette.disabled.value(QPalette::Text), color("text_muted"));
        QCOMPARE(palette.disabled.value(QPalette::Base), color("input_disabled"));
        QCOMPARE(palette.disabled.value(QPalette::Button), color("button_disabled"));
    }

    void explicitPaletteOverrides()
    {
        ThemeRegistry registry;
        ThemeProblems problems;
        auto definition = parse(themeJson("Explicit", {},
                                          "'tokens': {'selection': '#111111'}, "
                                          "'palette': {'highlight': '#b45309', "
                                          "'disabled': {'Text': '#6b7280'}}"),
                                &problems);
        QVERIFY(definition);
        const auto theme = registry.resolve(*definition, &problems);
        QVERIFY(theme);
        QCOMPARE(theme->palette.active.value(QPalette::Highlight), QColor("#b45309"));
        QCOMPARE(theme->palette.disabled.value(QPalette::Text), QColor("#6b7280"));
        const QPalette applied = theme->palette.applyTo(QPalette());
        QCOMPARE(applied.color(QPalette::Active, QPalette::Highlight), QColor("#b45309"));
        QCOMPARE(applied.color(QPalette::Inactive, QPalette::Highlight), QColor("#b45309"));
        QCOMPARE(applied.color(QPalette::Disabled, QPalette::Text), QColor("#6b7280"));
    }

    void iconsAreRenderedOnce()
    {
        QTemporaryDir cache;
        ThemeRegistry registry;
        ThemeProblems problems;
        const auto theme = registry.resolve(QStringLiteral("modern-light"), &problems);
        QVERIFY(theme);
        const QString key = themeIconKey(*theme);
        QVERIFY(key.startsWith(QStringLiteral("modern-light-")));
        QString error;
        const QString folder = ensureThemeIcons(theme->tokens, cache.path(), key, &error);
        QVERIFY2(!folder.isEmpty(), qPrintable(error));
        for (const auto& spec : themeIconSpecs()) {
            const QImage one(QDir(folder).filePath(spec.name + QStringLiteral(".png")));
            const QImage two(QDir(folder).filePath(spec.name + QStringLiteral("@2x.png")));
            QCOMPARE(one.size(), QSize(spec.size, spec.size));
            QCOMPARE(two.size(), QSize(spec.size * 2, spec.size * 2));
        }
        // An existing complete folder with the same hash is reused as is.
        const QString marker = QDir(folder).filePath(QStringLiteral("marker"));
        writeFile(marker, "x");
        QCOMPARE(ensureThemeIcons(theme->tokens, cache.path(), key, &error), folder);
        QVERIFY(QFile::exists(marker));

        ResolvedTheme changed = *theme;
        changed.tokens.insert(QStringLiteral("accent"), QStringLiteral("#ff0000"));
        QVERIFY(themeIconKey(changed) != key);
    }

    void importCopiesFilesAndFolders()
    {
        QTemporaryDir source;
        QTemporaryDir user;
        ThemeRegistry registry;

        writeFile(source.filePath("Ocean Night.vc3dtheme"), themeJson("Ocean Night"));
        ImportResult result = importTheme(source.filePath("Ocean Night.vc3dtheme"), user.path(), false, registry);
        QVERIFY2(result.ok, qPrintable(formatProblems(result.problems)));
        QCOMPARE(result.id, QStringLiteral("ocean-night"));
        QVERIFY(QFileInfo(user.filePath("ocean-night.vc3dtheme")).isFile());

        result = importTheme(source.filePath("Ocean Night.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(result.alreadyExists);
        result = importTheme(source.filePath("Ocean Night.vc3dtheme"), user.path(), true, registry);
        QVERIFY(result.ok);

        writeFile(source.filePath("grain/theme.vc3dtheme"),
                  themeJson("Grain", {}, "'stylesheet': 'qss/extra.qss'"));
        writeFile(source.filePath("grain/qss/extra.qss"),
                  "QLabel { background: url(@themedir@/grain.png); }");
        writeFile(source.filePath("grain/grain.png"), "png");
        result = importTheme(source.filePath("grain"), user.path(), false, registry);
        QVERIFY2(result.ok, qPrintable(formatProblems(result.problems)));
        QVERIFY(QFileInfo(user.filePath("grain/theme.vc3dtheme")).isFile());
        QVERIFY(QFileInfo(user.filePath("grain/qss/extra.qss")).isFile());
        QVERIFY(QFileInfo(user.filePath("grain/grain.png")).isFile());

        // A single file with its own style sheet is installed as a folder.
        writeFile(source.filePath("loose.vc3dtheme"), themeJson("Loose", {}, "'stylesheet': 'loose.qss'"));
        writeFile(source.filePath("loose.qss"), "QLabel { color: @text@; }");
        result = importTheme(source.filePath("loose.vc3dtheme"), user.path(), false, registry);
        QVERIFY2(result.ok, qPrintable(formatProblems(result.problems)));
        QVERIFY(QFileInfo(user.filePath("loose/theme.vc3dtheme")).isFile());
        QVERIFY(QFileInfo(user.filePath("loose/loose.qss")).isFile());

        writeFile(source.filePath("broken.vc3dtheme"), themeJson("Broken", "missing-base"));
        result = importTheme(source.filePath("broken.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(!QFileInfo::exists(user.filePath("broken.vc3dtheme")));

        ThemeRegistry installed;
        QVERIFY(installed.scan(user.path(), {}).isEmpty());
        QVERIFY(installed.contains("ocean-night") && installed.contains("grain") && installed.contains("loose"));
    }

    void exportRoundTrip()
    {
        QTemporaryDir dir;
        QTemporaryDir out;
        writeFile(dir.filePath("parent.vc3dtheme"),
                  themeJson("Parent", "modern-light",
                            "'tokens': {'window': '#fafafa', 'accent': '#7c3aed'}, "
                            "'palette': {'Highlight': '#b45309'}, 'font': {'pointSize': 11}"));
        writeFile(dir.filePath("child.vc3dtheme"),
                  themeJson("Child", "parent", "'tokens': {'border': '#999999'}"));
        ThemeRegistry registry;
        QVERIFY(registry.scan(dir.path(), {}).isEmpty());

        ThemeProblems problems;
        const auto original = registry.resolve(QStringLiteral("child"), &problems);
        QVERIFY(original);
        const QByteArray json = exportThemeJson(*original, registry, &problems);
        QVERIFY2(!json.isEmpty(), qPrintable(formatProblems(problems)));
        const QJsonObject object = QJsonDocument::fromJson(json).object();
        QCOMPARE(object.value("vc3dTheme").toInt(), 1);
        QVERIFY(object.value("$comment").toString().contains(QStringLiteral("vc3d_themes.md")));
        QCOMPARE(object.value("description").toString(), QStringLiteral("Based on Child. Describe your theme here."));
        // Roles no token covers are written out, so they can be edited.
        QVERIFY(object.value("palette").toObject().contains(QStringLiteral("LinkVisited")));
        QCOMPARE(object.value("base").toString(), QStringLiteral("modern-light"));
        QCOMPARE(object.value("tokens").toObject().size(), original->tokens.size());

        writeFile(out.filePath("copy.vc3dtheme"), json);
        ThemeRegistry fresh;
        QVERIFY(fresh.scan(out.path(), {}).isEmpty());
        const auto copy = fresh.resolve(QStringLiteral("copy"), &problems);
        QVERIFY2(copy, qPrintable(formatProblems(problems)));
        QCOMPARE(copy->tokens, original->tokens);
        QVERIFY(copy->palette == original->palette);
        QVERIFY(copy->font == original->font);
        QCOMPARE(copy->colorScheme, original->colorScheme);

        const auto classic = registry.resolve(QStringLiteral("classic"), &problems);
        const QJsonObject classicExport =
            QJsonDocument::fromJson(exportThemeJson(*classic, registry, &problems)).object();
        QCOMPARE(classicExport.value("base").toString(), QStringLiteral("modern-dark"));
        QVERIFY(classicExport.value("description").toString().contains(QStringLiteral("Classic")));
        QCOMPARE(classicExport.value("tokens").toObject().size(), builtinTokens().size());
    }

    void themePathFoldersAreDiscovered()
    {
        QTemporaryDir user;
        QTemporaryDir first;
        QTemporaryDir second;
        writeFile(first.filePath("lab.vc3dtheme"), themeJson("Lab"));
        writeFile(second.filePath("night/theme.vc3dtheme"), themeJson("Night", "modern-light"));
        qputenv("VC3D_THEME_PATH",
                (first.path() + QDir::listSeparator() + second.path()).toUtf8());
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.reloadThemes();
        qunsetenv("VC3D_THEME_PATH");
        QStringList ids;
        for (const auto& info : manager.themes()) {
            ids << info.id;
            if (info.id == QLatin1String("lab") || info.id == QLatin1String("night"))
                QCOMPARE(info.source, QStringLiteral("path"));
        }
        QCOMPARE(ids, QStringList({"classic", "modern-dark", "modern-light", "lab", "night"}));
    }

    void applyAndSwitch()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        QTemporaryDir config;
        writeFile(user.filePath("ocean.vc3dtheme"),
                  themeJson("Ocean", "modern-dark",
                            "'tokens': {'window': '#0a1a2f'}, 'font': {'pointSize': 13}"));

        const QString startupStyle = styleName();
        if (QStyleFactory::keys().contains(QStringLiteral("Windows"), Qt::CaseInsensitive))
            QCOMPARE(startupStyle.toLower(), QStringLiteral("windows"));
        const QPalette startupPalette = QApplication::palette();
        const QFont startupFont = QApplication::font();
        QVERIFY(qApp->styleSheet().isEmpty());

        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        const QString settingsFile = config.filePath("VC3D.ini");
        manager.initialize(settingsFile);
        QCOMPARE(manager.currentThemeId(), QStringLiteral("classic"));
        QCOMPARE(QApplication::palette(), startupPalette);
        QVERIFY(!manager.windowStyleSheet().isEmpty());
        QSignalSpy changed(&manager, &ThemeManager::themeChanged);
        // Each window style sheet, and whether the application had a style
        // sheet when it came.
        QList<QPair<QString, bool>> windowSheets;
        connect(&manager, &ThemeManager::windowStyleSheetChanged, &manager,
                [&windowSheets](const QString& sheet) {
                    windowSheets.append({sheet, !qApp->styleSheet().isEmpty()});
                });

        QString error;
        QVERIFY2(manager.setTheme(QStringLiteral("modern-dark"), true, &error), qPrintable(error));
        QCOMPARE(changed.count(), 1);
        // Leaving Classic, the window drops its sheet before the restyle.
        QCOMPARE(windowSheets.size(), 1);
        QVERIFY(windowSheets.last().first.isEmpty());
        QVERIFY(!windowSheets.last().second);
        QCOMPARE(styleName().toLower(), QStringLiteral("fusion"));
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#161b22"));
        QVERIFY(qApp->styleSheet().contains(QStringLiteral("QMenuBar")));
        QVERIFY(manager.windowStyleSheet().isEmpty());
        const QString key = themeIconKey(manager.currentTheme());
        QVERIFY(qApp->styleSheet().contains(QStringLiteral("vc3dtheme:%1/check.png").arg(key)));
        QVERIFY(QFile::exists(QStringLiteral("vc3dtheme:%1/check.png").arg(key)));
        QCOMPARE(QSettings(settingsFile, QSettings::IniFormat).value("appearance/theme").toString(),
                 QStringLiteral("modern-dark"));

        QVERIFY2(manager.setTheme(QStringLiteral("ocean"), false, &error), qPrintable(error));
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#0a1a2f"));
        QCOMPARE(QApplication::font().pointSizeF(), 13.0);
        QCOMPARE(QSettings(settingsFile, QSettings::IniFormat).value("appearance/theme").toString(),
                 QStringLiteral("modern-dark"));

        QVERIFY(!manager.setTheme(QStringLiteral("no-such-theme"), false, &error));
        QVERIFY(error.contains(QStringLiteral("no-such-theme")));
        QCOMPARE(manager.currentThemeId(), QStringLiteral("ocean"));

        QVERIFY2(manager.setTheme(QStringLiteral("classic"), true, &error), qPrintable(error));
        QCOMPARE(styleName(), startupStyle);
        QCOMPARE(QApplication::palette(), startupPalette);
        QCOMPARE(QApplication::font(), startupFont);
        QVERIFY(qApp->styleSheet().isEmpty());
        QVERIFY(!manager.windowStyleSheet().isEmpty());
        QCOMPARE(changed.count(), 3);
        // Entering Classic, the window takes its sheet after the restyle.
        QCOMPARE(windowSheets.last().first, manager.windowStyleSheet());
        QVERIFY(!windowSheets.last().second);
    }

    void startupChoiceAndFallback()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        QTemporaryDir config;
        const QString settingsFile = config.filePath("VC3D.ini");
        {
            QSettings settings(settingsFile, QSettings::IniFormat);
            settings.setValue("appearance/theme", "gone-theme");
        }
        {
            ThemeManager manager;
            manager.setUserThemesDirectory(user.path());
            manager.setIconCacheDirectory(cache.path());
            QTest::ignoreMessage(QtWarningMsg, QRegularExpression("gone-theme"));
            manager.initialize(settingsFile);
            QCOMPARE(manager.currentThemeId(), QStringLiteral("classic"));
        }
        {
            qputenv("VC3D_THEME", "modern-light");
            ThemeManager manager;
            manager.setUserThemesDirectory(user.path());
            manager.setIconCacheDirectory(cache.path());
            manager.initialize(settingsFile);
            QCOMPARE(manager.currentThemeId(), QStringLiteral("modern-light"));
            manager.initialize(settingsFile, QStringLiteral("modern-dark"));
            QCOMPARE(manager.currentThemeId(), QStringLiteral("modern-dark"));
            qunsetenv("VC3D_THEME");
            manager.setTheme(QStringLiteral("classic"), false);
        }
        // Overrides are never saved.
        QCOMPARE(QSettings(settingsFile, QSettings::IniFormat).value("appearance/theme").toString(),
                 QStringLiteral("gone-theme"));

        // A theme file given by path is used as is and saved as that path.
        writeFile(config.filePath("mine/Forest.vc3dtheme"), themeJson("Forest", "modern-light"));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.initialize(settingsFile, config.filePath("mine/Forest.vc3dtheme"));
        QCOMPARE(manager.currentThemeId(), QStringLiteral("forest"));
        QCOMPARE(manager.currentTheme().source, ThemeSource::Path);
        QVERIFY(manager.setTheme(QStringLiteral("forest"), true));
        QCOMPARE(QSettings(settingsFile, QSettings::IniFormat).value("appearance/theme").toString(),
                 QFileInfo(config.filePath("mine/Forest.vc3dtheme")).absoluteFilePath());
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void liveReloadKeepsLastGoodVersion()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        const QString file = user.filePath("live.vc3dtheme");
        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': '#202020'}"));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QString error;
        QVERIFY2(manager.setTheme(QStringLiteral("live"), false, &error), qPrintable(error));
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#202020"));

        QSignalSpy changed(&manager, &ThemeManager::themeChanged);
        QSignalSpy failed(&manager, &ThemeManager::liveReloadFailed);
        // Replace the file the way editors do: write elsewhere, then rename.
        writeFile(user.filePath("live.tmp"),
                  themeJson("Live", "modern-dark", "'tokens': {'window': '#303030'}"));
        QFile::remove(file);
        QFile::rename(user.filePath("live.tmp"), file);
        QTRY_COMPARE_WITH_TIMEOUT(changed.count(), 1, 5000);
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#303030"));

        QTest::ignoreMessage(QtWarningMsg, QRegularExpression("keeping the last good version"));
        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': 'nope'}"));
        QTRY_COMPARE_WITH_TIMEOUT(failed.count(), 1, 5000);
        QCOMPARE(manager.currentThemeId(), QStringLiteral("live"));
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#303030"));
        QVERIFY(failed.at(0).at(0).toString().contains(QStringLiteral("window")));
        // The same broken file again is not reported twice.
        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': 'nope'}"));
        QTest::qWait(600);
        QCOMPARE(failed.count(), 1);

        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': '#404040'}"));
        QTRY_COMPARE_WITH_TIMEOUT(QApplication::palette().color(QPalette::Window), QColor("#404040"), 5000);
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void folderThemeFilesAreNotImportedAlone()
    {
        QTemporaryDir downloads;
        QTemporaryDir user;
        writeFile(downloads.filePath("theme.vc3dtheme"), themeJson("Stray"));
        writeFile(downloads.filePath("notes/todo.txt"), "private");
        ThemeRegistry registry;
        const ImportResult result =
            importTheme(downloads.filePath("theme.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(containsMessage(result.problems, QStringLiteral("import its folder"), false));
        QVERIFY(entriesIn(user.path()).isEmpty());

        // A stray theme.vc3dtheme in a themes folder is skipped rather than
        // named after the folder.
        writeFile(user.filePath("theme.vc3dtheme"), themeJson("Stray"));
        const ThemeProblems problems = registry.scan(user.path(), {});
        QVERIFY(containsMessage(problems, QStringLiteral("belongs inside a theme's own folder"), true));
        QCOMPARE(registry.themes().size(), 3);
    }

    void userStyleSheetsOnlyStyle()
    {
        QTemporaryDir dir;
        const QString folder = dir.filePath("probe");
        const auto errors = [&](const QByteArray& qss) {
            ThemeProblems errorsOnly;
            for (const auto& problem : styleSheetProblems(folder, qss)) {
                if (!problem.warning)
                    errorsOnly.append(problem);
            }
            return errorsOnly;
        };
        // Widget properties change what controls do, not how they look.
        QVERIFY(containsMessage(errors("QPushButton#btnDelete { qproperty-text: \"Save\"; }"),
                                QStringLiteral("\"qproperty-text\""), false));
        QVERIFY(containsMessage(errors("QPushButton { q\\property-text: \"Save\"; }"),
                                QStringLiteral("\"qproperty-text\""), false));
        QVERIFY(containsMessage(errors("QCheckBox { qpropert\\79-checked: false; }"),
                                QStringLiteral("\"qproperty-checked\""), false));
        QVERIFY(errors("/* QPushButton { qproperty-text: \"x\"; } */ QLabel { color: @text@; }").isEmpty());
        // Not even drawBase: Qt keeps it after the theme is gone, so it
        // would outlive a return to Classic.
        QVERIFY(containsMessage(errors("QTabBar { qproperty-drawBase: 0; }"),
                                QStringLiteral("\"qproperty-drawBase\""), false));

        // Qt drops a whole style sheet that does not parse.
        QVERIFY(containsMessage(errors("QPushButton { color: red; }\nQLabel { color: blue"),
                                QStringLiteral("line 2: \"{\" is never closed"), false));
        QVERIFY(containsMessage(errors("QLabel { color: blue; }\n}"),
                                QStringLiteral("line 2: \"}\" has no matching \"{\""), false));
        QVERIFY(containsMessage(errors("QLabel { color: blue; } /* open"),
                                QStringLiteral("comment is never closed"), false));
        QVERIFY(containsMessage(errors("QLabel { font-family: \"Inter; }"),
                                QStringLiteral("line 1: a string is never closed"), false));
        // Text in a comment is not a placeholder.
        QVERIFY(errors("/* mail me@host@ */ QLabel { color: @text@; }").isEmpty());

        // Images come from the theme's own folder.
        writeFile(QDir(folder).filePath("grain.png"), "png");
        writeFile(dir.filePath("outside.png"), "png");
        QVERIFY(errors("QLabel { background: url(@themedir@/grain.png); }").isEmpty());
        QVERIFY(errors("QLabel { background: url(\"@themedir@/grain.png\" ); }").isEmpty());
        QVERIFY(errors("QLabel { image: url(vc3dtheme:key/check.png); }").isEmpty());
        QVERIFY(containsMessage(errors("QLabel { background: url(@themedir@/missing.png); }"),
                                QStringLiteral("does not exist"), false));
        QVERIFY(containsMessage(errors("QLabel { background: url(@themedir@/../outside.png); }"),
                                QStringLiteral("inside the theme's folder"), false));
        QVERIFY(containsMessage(errors(QStringLiteral("QLabel { background: url(%1); }")
                                           .arg(dir.filePath("outside.png")).toUtf8()),
                                QStringLiteral("inside the theme's folder"), false));
        QVERIFY(containsMessage(errors("QLabel { background: url(grain.png); }"),
                                QStringLiteral("inside the theme's folder"), false));
#ifdef Q_OS_UNIX
        QCOMPARE(mkfifo(QFile::encodeName(QDir(folder).filePath("pipe.png")).constData(), 0600), 0);
        QVERIFY(containsMessage(errors("QLabel { background: url(@themedir@/pipe.png); }"),
                                QStringLiteral("regular file"), false));
#endif

        // Such a theme is not listed at all. The scan renders each theme the
        // way it is applied, so a Classic-based theme, which has no drawn
        // icons, cannot use @name@.
        QTemporaryDir themes;
        writeFile(themes.filePath("sneaky.vc3dtheme"),
                  themeJson("Sneaky", {}, "'qss': 'QCheckBox { qproperty-checked: false; }'"));
        writeFile(themes.filePath("plain.vc3dtheme"),
                  themeJson("Plain", "classic", "'qss': 'QCheckBox::indicator { image: url(vc3dtheme:@name@/check.png); }'"));
        ThemeRegistry registry;
        const ThemeProblems scanProblems = registry.scan(themes.path(), {});
        QVERIFY(containsMessage(scanProblems, QStringLiteral("qproperty-checked"), false));
        QVERIFY(containsMessage(scanProblems, QStringLiteral("unknown tokens: @name@"), false));
        QVERIFY(!registry.contains(QStringLiteral("sneaky")));
        QVERIFY(!registry.contains(QStringLiteral("plain")));
    }

    void specialFilesAreNotRead()
    {
#ifndef Q_OS_UNIX
        QSKIP("FIFOs, symbolic links and /dev/zero are tested on Unix");
#else
        QTemporaryDir dir;
        ThemeProblems problems;
        // Reading a FIFO would block.
        writeFile(dir.filePath("fifo/theme.vc3dtheme"), themeJson("Fifo", {}, "'stylesheet': 'pipe.qss'"));
        QCOMPARE(mkfifo(QFile::encodeName(dir.filePath("fifo/pipe.qss")).constData(), 0600), 0);
        QVERIFY(!loadTheme(dir.filePath("fifo"), ThemeSource::User, &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("not a regular file"), false));

        // Reading /dev/zero would never end.
        problems.clear();
        writeFile(dir.filePath("zero.vc3dtheme"), themeJson("Zero"));
        QVERIFY(QFile::remove(dir.filePath("zero.vc3dtheme")));
        QVERIFY(QFile::link(QStringLiteral("/dev/zero"), dir.filePath("zero.vc3dtheme")));
        QVERIFY(!loadTheme(dir.filePath("zero.vc3dtheme"), ThemeSource::User, &problems));

        // A style sheet may not lead out of the theme's folder.
        problems.clear();
        writeFile(dir.filePath("secret.txt"), "SECRET");
        writeFile(dir.filePath("escape/theme.vc3dtheme"), themeJson("Escape", {}, "'stylesheet': 'extra.qss'"));
        QVERIFY(QFile::link(dir.filePath("secret.txt"), dir.filePath("escape/extra.qss")));
        QVERIFY(!loadTheme(dir.filePath("escape"), ThemeSource::User, &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("leads outside"), false));
#endif
    }

    void importInstallsWhatItChecked()
    {
        QTemporaryDir source;
        QTemporaryDir user;
        ThemeRegistry registry;
        ImportResult result;
#ifdef Q_OS_UNIX
        // A linked style sheet would copy a private file into the themes folder.
        writeFile(source.filePath("secret.txt"), "SECRET-KEY-MATERIAL");
        writeFile(source.filePath("evil.vc3dtheme"), themeJson("Evil", {}, "'stylesheet': 'extra.qss'"));
        QVERIFY(QFile::link(source.filePath("secret.txt"), source.filePath("extra.qss")));
        result = importTheme(source.filePath("evil.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(entriesIn(user.path()).isEmpty());

        writeFile(source.filePath("linked/real.json"), themeJson("Linked"));
        QVERIFY(QFile::link(source.filePath("linked/real.json"), source.filePath("linked/theme.vc3dtheme")));
        result = importTheme(source.filePath("linked"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(containsMessage(result.problems, QStringLiteral("is a link"), false));
        QVERIFY(entriesIn(user.path()).isEmpty());
#endif
        // A lone theme file cannot bring the images its style sheet names.
        writeFile(source.filePath("grainy.vc3dtheme"),
                  themeJson("Grainy", {}, "'qss': 'QLabel { background: url(@themedir@/grain.png); }'"));
        writeFile(source.filePath("grain.png"), "png");
        result = importTheme(source.filePath("grainy.vc3dtheme"), user.path(), false, registry);
        QVERIFY(!result.ok);
        QVERIFY(containsMessage(result.problems, QStringLiteral("only a folder theme carries"), false));
        QVERIFY(!formatProblems(result.problems).contains(QStringLiteral(".import-")));
        QVERIFY(formatProblems(result.problems).contains(source.filePath("grainy.vc3dtheme")));
        QVERIFY(entriesIn(user.path()).isEmpty());

        // A broken new version leaves the installed one alone; a good one
        // replaces it and leaves nothing else behind.
        writeFile(source.filePath("ocean/theme.vc3dtheme"), themeJson("Ocean folder"));
        QVERIFY(importTheme(source.filePath("ocean"), user.path(), false, registry).ok);
        writeFile(source.filePath("ocean.vc3dtheme"), themeJson("Ocean broken", "missing-base"));
        result = importTheme(source.filePath("ocean.vc3dtheme"), user.path(), true, registry);
        QVERIFY(!result.ok);
        QCOMPARE(entriesIn(user.path()), QStringList{"ocean"});
        writeFile(source.filePath("ocean.vc3dtheme"), themeJson("Ocean file"));
        result = importTheme(source.filePath("ocean.vc3dtheme"), user.path(), true, registry);
        QVERIFY2(result.ok, qPrintable(formatProblems(result.problems)));
        QCOMPARE(result.installedPath, user.filePath("ocean.vc3dtheme"));
        QCOMPARE(entriesIn(user.path()), QStringList{"ocean.vc3dtheme"});
    }

    void limitsKeepThemesSmall()
    {
        ThemeProblems problems;
        QVERIFY(!parse(themeJson(QString(81, QLatin1Char('n'))), &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("longer than 80"), false));
        problems.clear();
        QVERIFY(!parse("{\"vc3dTheme\": 1, \"name\": \"Two\\nlines\"}", &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("plain text"), false));
        problems.clear();
        QVERIFY(!parse(QStringLiteral("{\"vc3dTheme\": 1, \"name\": \"abc\\u202edef\"}").toUtf8(), &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("plain text"), false));
        problems.clear();
        QVERIFY(!parse(themeJson("Long", {}, QStringLiteral("'tokens': {'mono': '%1'}").arg(QString(300, QLatin1Char('a')))),
                       &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("longer than 256"), false));

        // Problems are capped per file.
        problems.clear();
        QStringList keys;
        for (int i = 0; i < 500; ++i)
            keys << QStringLiteral("'extra%1': 1").arg(i);
        QVERIFY(parse(themeJson("Many", {}, keys.join(QStringLiteral(", "))), &problems));
        QCOMPARE(problems.size(), kMaxProblemsPerFile + 1);
        QVERIFY(problems.last().message.contains(QStringLiteral("more problems")));

        QTemporaryDir dir;
        writeFile(dir.filePath("big.vc3dtheme"),
                  themeJson("Big", {}, QStringLiteral("'qss': '%1'").arg(QString(70 * 1024, QLatin1Char(' ')))));
        problems.clear();
        QVERIFY(!loadTheme(dir.filePath("big.vc3dtheme"), ThemeSource::User, &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("larger than 64 KiB"), false));
    }

    void messagesPointAtTheFix()
    {
        ThemeProblems problems;
        auto theme = parse(themeJson("Typo", {},
                                     "'tokens': {'acent': '#ff0000', 'glow': '#00ff00'}, "
                                     "'qss': 'QLabel { color: @glow@; }', "
                                     "'palette': {'Hightlight': '#ffffff'}, 'Name2': 1, '$comment': 'notes'"),
                           &problems);
        QVERIFY(theme);
        QVERIFY(containsMessage(problems, QStringLiteral("unknown token \"acent\"; the built-in template does "
                                                         "not use it (did you mean \"accent\"?)"),
                                true));
        // A token the theme's own style sheet uses is not reported.
        QVERIFY(!containsMessage(problems, QStringLiteral("\"glow\""), true));
        QVERIFY(containsMessage(problems, QStringLiteral("(did you mean \"Highlight\"?)"), true));
        QVERIFY(containsMessage(problems, QStringLiteral("unknown key \"Name2\" is ignored (did you mean \"name\"?)"), true));
        QVERIFY(!containsMessage(problems, QStringLiteral("$comment"), true));

        problems.clear();
        QVERIFY(!parse(themeJson("Case", {}, "'tokens': {'Accent': '#ff0000'}"), &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("(did you mean \"accent\"?)"), false));
        problems.clear();
        QVERIFY(!parse("{'vc3dTheme': 1, 'name': 'x',}", &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("column"), false));
        problems.clear();
        QVERIFY(parse(themeJson("Alpha", {}, "'tokens': {'accent': '#ff8a3d80'}"), &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("Qt puts the alpha first"), true));
        problems.clear();
        QVERIFY(!parse(themeJson("Big radius", {}, "'tokens': {'radius': 'big'}"), &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("must be a length"), false));
        problems.clear();
        QVERIFY(parse(themeJson("Unitless", {}, "'tokens': {'radius': '7'}"), &problems));
        problems.clear();
        theme = parse(themeJson("Shout", {}, "'colorScheme': 'Dark'"), &problems);
        QVERIFY(theme);
        QCOMPARE(theme->colorScheme, QStringLiteral("dark"));

        problems.clear();
        ThemeRegistry registry;
        theme = parse(themeJson("Drak", "modern-drak"), &problems);
        QVERIFY(theme);
        QVERIFY(!registry.resolve(*theme, &problems));
        QVERIFY(containsMessage(problems, QStringLiteral("(did you mean \"modern-dark\"?)"), false));
        QVERIFY(containsMessage(problems, QStringLiteral("classic, modern-dark, modern-light"), false));
    }

    void exportCarriesStyleSheets()
    {
        QTemporaryDir dir;
        QTemporaryDir out;
        writeFile(dir.filePath("framed.vc3dtheme"),
                  themeJson("Framed", {}, "'qss': 'QPushButton:focus { border: 2px solid @focus@; }'"));
        writeFile(dir.filePath("bare.vc3dtheme"),
                  themeJson("Bare", {}, "'qss': 'QWidget { color: @text@; }', 'stylesheetMode': 'replace'"));
        writeFile(dir.filePath("old-school.vc3dtheme"),
                  themeJson("Old School", "classic",
                            "'colorScheme': 'light', 'palette': {'Window': '#553322'}, "
                            "'qss': 'QLabel { font-weight: bold; }'"));
        writeFile(dir.filePath("grain/theme.vc3dtheme"),
                  themeJson("Grain", {}, "'qss': 'QLabel { background: url(@themedir@/grain.png); }'"));
        writeFile(dir.filePath("grain/grain.png"), "png");
        ThemeRegistry registry;
        QVERIFY(registry.scan(dir.path(), {}).isEmpty());

        for (const QString& id : {QStringLiteral("framed"), QStringLiteral("bare"), QStringLiteral("old-school")}) {
            ThemeProblems problems;
            const auto original = registry.resolve(id, &problems);
            QVERIFY(original);
            const QByteArray json = exportThemeJson(*original, registry, &problems);
            QVERIFY2(problems.isEmpty(), qPrintable(formatProblems(problems)));
            const QString copyId = id + QStringLiteral("-copy");
            writeFile(out.filePath(copyId + QStringLiteral(".vc3dtheme")), json);
            ThemeRegistry fresh;
            QVERIFY(fresh.scan(out.path(), {}).isEmpty());
            const auto copy = fresh.resolve(copyId, &problems);
            QVERIFY2(copy, qPrintable(formatProblems(problems)));
            QCOMPARE(copy->rootId, original->rootId);
            QCOMPARE(copy->replaced, original->replaced);
            QVERIFY(copy->palette == original->palette);
            QCOMPARE(renderStyleSheet(*copy, QStringLiteral("k"), QStringLiteral("dark"), &problems),
                     renderStyleSheet(*original, QStringLiteral("k"), QStringLiteral("dark"), &problems));
            QVERIFY(problems.isEmpty());
        }

        ThemeProblems problems;
        const QJsonObject classicBased = QJsonDocument::fromJson(
            exportThemeJson(*registry.resolve(QStringLiteral("old-school"), &problems), registry, &problems)).object();
        QCOMPARE(classicBased.value("base").toString(), QStringLiteral("classic"));
        QCOMPARE(classicBased.value("palette").toObject().value("Window").toString(), QStringLiteral("#553322"));
        QVERIFY(!classicBased.contains(QStringLiteral("tokens")));

        // Files a style sheet loads from its folder are not exported, and the
        // export says so.
        exportThemeJson(*registry.resolve(QStringLiteral("grain"), &problems), registry, &problems);
        QVERIFY(containsMessage(problems, QStringLiteral("@themedir@"), true));
    }

    void iconCacheKeepsCurrentFolders()
    {
        QTemporaryDir cache;
        for (const char* name : {"ocean-00000001", "ocean-00000002", "ocean-night-00000003",
                                 "reef-00000004", ".ocean-00000005.tmp-123"}) {
            QVERIFY(QDir().mkpath(cache.filePath(QLatin1String(name))));
        }
        pruneThemeIcons(cache.path(), QStringLiteral("ocean-00000002"), 0);
        QCOMPARE(entriesIn(cache.path()),
                 QStringList({"ocean-00000002", "ocean-night-00000003", "reef-00000004"}));
        // Recent folders may still be on screen in another VC3D.
        QVERIFY(QDir().mkpath(cache.filePath("ocean-00000009")));
        pruneThemeIcons(cache.path(), QStringLiteral("ocean-00000002"), 3600);
        QVERIFY(QFileInfo::exists(cache.filePath("ocean-00000009")));
    }

    void classicRestoresTheStartupLook()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        // Qt's -stylesheet option, and a class palette as macOS sets them.
        const QString startupSheet = QStringLiteral("QLabel { color: #123456; }");
        qApp->setStyleSheet(startupSheet);
        QPalette labelPalette = QApplication::palette();
        labelPalette.setColor(QPalette::WindowText, QColor("#654321"));
        QApplication::setPalette(labelPalette, "QLabel");
        {
            ThemeManager manager;
            manager.setUserThemesDirectory(user.path());
            manager.setIconCacheDirectory(cache.path());
            manager.initialize(QString());
            QCOMPARE(manager.currentThemeId(), QStringLiteral("classic"));
            QCOMPARE(qApp->styleSheet(), startupSheet);
            QVERIFY(manager.setTheme(QStringLiteral("modern-dark"), false));
            QVERIFY(!qApp->styleSheet().contains(startupSheet));
            QVERIFY(QApplication::palette("QLabel").color(QPalette::WindowText) != QColor("#654321"));
            QVERIFY(manager.setTheme(QStringLiteral("classic"), false));
            QCOMPARE(qApp->styleSheet(), startupSheet);
            QCOMPARE(QApplication::palette("QLabel").color(QPalette::WindowText), QColor("#654321"));
        }
        qApp->setStyleSheet(QString());
        QApplication::setPalette(QPalette());
    }

    void classicReportsTheSystemScheme()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        const QString system = classicPrefersDark() ? QStringLiteral("dark") : QStringLiteral("light");
        const QString other = system == QLatin1String("dark") ? QStringLiteral("light") : QStringLiteral("dark");
        writeFile(user.filePath("other-classic.vc3dtheme"),
                  themeJson("Other Classic", "classic", QStringLiteral("'colorScheme': '%1'").arg(other)));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QVERIFY(manager.setTheme(QStringLiteral("other-classic"), false));
        for (const auto& info : manager.themes()) {
            if (info.id == QLatin1String("classic"))
                QCOMPARE(info.colorScheme, system);
            if (info.id == QLatin1String("other-classic"))
                QCOMPARE(info.colorScheme, other);
        }
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void startupTriesEachChoiceInTurn()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        QTemporaryDir config;
        const QString settingsFile = config.filePath("VC3D.ini");
        {
            QSettings settings(settingsFile, QSettings::IniFormat);
            settings.setValue("appearance/theme", "modern-light");
        }
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        QSignalSpy status(&manager, &ThemeManager::statusMessage);
        QTest::ignoreMessage(QtWarningMsg, QRegularExpression("no-such"));
        manager.setStartupOverride(QStringLiteral("no-such"));
        manager.initialize(settingsFile);
        QCOMPARE(manager.currentThemeId(), QStringLiteral("modern-light"));
        QCOMPARE(status.count(), 1);
        QVERIFY(status.at(0).at(0).toString().contains(QStringLiteral("no-such")));
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void reloadKeepsTheActiveThemeCurrent()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        writeFile(user.filePath("named.vc3dtheme"), themeJson("Old Name"));
        writeFile(user.filePath("gone.vc3dtheme"), themeJson("Gone"));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QVERIFY(manager.setTheme(QStringLiteral("named"), false));
        writeFile(user.filePath("named.vc3dtheme"), themeJson("New Name"));
        manager.reloadThemes();
        QCOMPARE(manager.currentTheme().name, QStringLiteral("New Name"));

        QVERIFY(manager.setTheme(QStringLiteral("gone"), false));
        QVERIFY(QFile::remove(user.filePath("gone.vc3dtheme")));
        QTest::ignoreMessage(QtWarningMsg, QRegularExpression("gone or broken"));
        manager.reloadThemes();
        QCOMPARE(manager.currentThemeId(), QStringLiteral("classic"));
        const auto list = manager.themes();
        QVERIFY(std::any_of(list.cbegin(), list.cend(), [](const auto& info) { return info.active; }));
    }

    void pathThemesAreCheckedBeforeListing()
    {
        QTemporaryDir dir;
        QTemporaryDir user;
        QTemporaryDir cache;
        writeFile(dir.filePath("lost.vc3dtheme"), themeJson("Lost", "no-such-base"));
        writeFile(dir.filePath("typo.vc3dtheme"), themeJson("Typo", {}, "'tokens': {'acent': '#ff0000'}"));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QString error;
        QVERIFY(!manager.setTheme(dir.filePath("lost.vc3dtheme"), false, &error));
        QVERIFY(error.contains(QStringLiteral("no-such-base")));
        QVERIFY(!manager.registry().contains(QStringLiteral("lost")));

        ThemeProblems warnings;
        QTest::ignoreMessage(QtWarningMsg, QRegularExpression("acent"));
        QVERIFY2(manager.setTheme(dir.filePath("typo.vc3dtheme"), false, &error, &warnings), qPrintable(error));
        QVERIFY(containsMessage(warnings, QStringLiteral("(did you mean \"accent\"?)"), true));
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void liveReloadWatchesOnlyTheThemeFiles()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        const QString file = user.filePath("live.vc3dtheme");
        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': '#202020'}"));
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QVERIFY(manager.setTheme(QStringLiteral("live"), false));
        QSignalSpy listed(&manager, &ThemeManager::themesChanged);
        QSignalSpy status(&manager, &ThemeManager::statusMessage);
        // A download landing in the same folder does not rescan anything.
        writeFile(user.filePath("unrelated.txt"), "download");
        QTest::qWait(600);
        QCOMPARE(listed.count(), 0);

        QTest::ignoreMessage(QtWarningMsg, QRegularExpression("reloaded with warnings"));
        writeFile(file, themeJson("Live", "modern-dark", "'tokens': {'window': '#303030', 'windwo': '#000000'}"));
        QTRY_COMPARE_WITH_TIMEOUT(listed.count(), 1, 5000);
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#303030"));
        QCOMPARE(status.count(), 1);
        QVERIFY(status.at(0).at(0).toString().contains(QStringLiteral("windwo")));
        manager.setTheme(QStringLiteral("classic"), false);
    }

    void docsMatchTheEngine()
    {
        QFile file(QStringLiteral(VC3D_THEME_DOCS));
        QVERIFY2(file.open(QIODevice::ReadOnly), VC3D_THEME_DOCS);
        const QString docs = QString::fromUtf8(file.readAll());
        const auto& tokens = builtinTokens();
        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            QVERIFY2(docs.contains(QStringLiteral("| `%1` |").arg(it.key())),
                     qPrintable(QStringLiteral("token %1 is missing from the docs").arg(it.key())));
        }

        // The accent recipe works when copied as is, and sets every accent token.
        const qsizetype section = docs.indexOf(QStringLiteral("### Change the accent"));
        QVERIFY(section >= 0);
        const qsizetype start = docs.indexOf(QStringLiteral("```json\n"), section) + 8;
        const qsizetype end = docs.indexOf(QStringLiteral("```"), start);
        ThemeProblems problems;
        const auto theme = parseTheme(docs.mid(start, end - start).toUtf8(),
                                      QStringLiteral("/tmp/ocean-night.vc3dtheme"),
                                      QStringLiteral("ocean-night"), ThemeSource::User, &problems);
        QVERIFY2(theme && problems.isEmpty(), qPrintable(formatProblems(problems)));
        ThemeRegistry registry;
        const auto resolved = registry.resolve(*theme, &problems);
        QVERIFY(resolved && checkRendering(*resolved, &problems) && problems.isEmpty());
        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            if (it.key().contains(QStringLiteral("accent")) || it.key().startsWith(QStringLiteral("focus")) ||
                it.key().startsWith(QStringLiteral("selection"))) {
                QVERIFY2(theme->tokens.contains(it.key()), qPrintable(it.key()));
            }
        }
    }

    void importAndExportThroughManager()
    {
        QTemporaryDir user;
        QTemporaryDir cache;
        QTemporaryDir out;
        ThemeManager manager;
        manager.setUserThemesDirectory(user.path());
        manager.setIconCacheDirectory(cache.path());
        manager.reloadThemes();
        QSignalSpy listChanged(&manager, &ThemeManager::themesChanged);

        const QString exported = out.filePath("My Dark.vc3dtheme");
        QString error;
        QVERIFY2(manager.exportTheme(exported, QStringLiteral("modern-dark"), &error), qPrintable(error));
        const auto outcome = manager.importTheme(exported, false);
        QVERIFY2(outcome.ok, qPrintable(formatProblems(outcome.problems)));
        QCOMPARE(outcome.id, QStringLiteral("my-dark"));
        QVERIFY(listChanged.count() >= 1);
        QVERIFY(manager.registry().contains(QStringLiteral("my-dark")));
        QVERIFY(manager.setTheme(QStringLiteral("my-dark"), false));
        QCOMPARE(QApplication::palette().color(QPalette::Window), QColor("#161b22"));
        manager.setTheme(QStringLiteral("classic"), false);
    }
};

QTEST_MAIN(ThemeEngineTest)
#include "test_theme_engine.moc"
