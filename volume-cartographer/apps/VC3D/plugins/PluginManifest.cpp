#include "plugins/PluginManifest.hpp"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>

namespace vc3d::plugins {
namespace {

const QStringList& contextPlaceholders()
{
    static const QStringList names{
        QStringLiteral("volume"),     QStringLiteral("volume_id"),
        QStringLiteral("segment"),    QStringLiteral("segment_id"),
        QStringLiteral("project"),    QStringLiteral("ink_overlay"),
        QStringLiteral("umbilicus"),  QStringLiteral("point"),
        QStringLiteral("point_x"),    QStringLiteral("point_y"),
        QStringLiteral("point_z"),    QStringLiteral("region"),
        QStringLiteral("output"),     QStringLiteral("plugin_dir"),
    };
    return names;
}

const QString kInputPrefix = QStringLiteral("input:");

// Splits a template into literal text and {placeholder} names. "{{" and "}}"
// are literal braces. Returns false with an error for an unbalanced brace.
struct Token {
    bool placeholder{false};
    QString text;
};

bool tokenize(const QString& in, QList<Token>& out, QString& error)
{
    QString literal;
    for (qsizetype i = 0; i < in.size(); ++i) {
        const QChar c = in.at(i);
        if (c == QLatin1Char('{')) {
            if (i + 1 < in.size() && in.at(i + 1) == QLatin1Char('{')) {
                literal += c;
                ++i;
                continue;
            }
            const qsizetype close = in.indexOf(QLatin1Char('}'), i + 1);
            if (close < 0) {
                error = QStringLiteral("unclosed '{' in \"%1\"").arg(in);
                return false;
            }
            if (!literal.isEmpty()) {
                out.push_back({false, literal});
                literal.clear();
            }
            out.push_back({true, in.mid(i + 1, close - i - 1)});
            i = close;
        } else if (c == QLatin1Char('}')) {
            if (i + 1 < in.size() && in.at(i + 1) == QLatin1Char('}')) {
                literal += c;
                ++i;
                continue;
            }
            error = QStringLiteral("unmatched '}' in \"%1\" (write '}}' for a literal brace)").arg(in);
            return false;
        } else {
            literal += c;
        }
    }
    if (!literal.isEmpty()) {
        out.push_back({false, literal});
    }
    return true;
}

bool checkTemplate(const QString& in, const QSet<QString>& inputs, QString& error)
{
    QList<Token> tokens;
    if (!tokenize(in, tokens, error)) {
        return false;
    }
    for (const auto& token : tokens) {
        if (!token.placeholder) continue;
        if (token.text.startsWith(kInputPrefix)) {
            if (!inputs.contains(token.text.mid(kInputPrefix.size()))) {
                error = QStringLiteral("{%1} names an input the manifest does not declare").arg(token.text);
                return false;
            }
        } else if (!contextPlaceholders().contains(token.text)) {
            error = QStringLiteral("unknown placeholder {%1}").arg(token.text);
            return false;
        }
    }
    return true;
}

QString formatNumber(double value)
{
    return QString::number(value, 'g', 10);
}

std::optional<ResultKind> parseKind(const QString& name)
{
    if (name == QLatin1String("none")) return ResultKind::None;
    if (name == QLatin1String("file")) return ResultKind::File;
    if (name == QLatin1String("segment")) return ResultKind::Segment;
    if (name == QLatin1String("ink_overlay")) return ResultKind::InkOverlay;
    if (name == QLatin1String("points")) return ResultKind::Points;
    if (name == QLatin1String("umbilicus")) return ResultKind::Umbilicus;
    return std::nullopt;
}

QString valueFor(const QString& name, const PluginManifest& manifest, const PluginContext& ctx)
{
    if (name.startsWith(kInputPrefix)) return ctx.inputs.value(name.mid(kInputPrefix.size()));
    if (name == QLatin1String("volume")) return ctx.volume;
    if (name == QLatin1String("volume_id")) return ctx.volumeId;
    if (name == QLatin1String("segment")) return ctx.segment;
    if (name == QLatin1String("segment_id")) return ctx.segmentId;
    if (name == QLatin1String("project")) return ctx.project;
    if (name == QLatin1String("ink_overlay")) return ctx.inkOverlay;
    if (name == QLatin1String("umbilicus")) return ctx.umbilicus;
    if (name == QLatin1String("output")) return ctx.output;
    if (name == QLatin1String("plugin_dir")) return manifest.pluginDir;
    if (ctx.point) {
        const auto& p = *ctx.point;
        if (name == QLatin1String("point")) {
            return QStringLiteral("%1,%2,%3").arg(formatNumber(p[0]), formatNumber(p[1]), formatNumber(p[2]));
        }
        if (name == QLatin1String("point_x")) return formatNumber(p[0]);
        if (name == QLatin1String("point_y")) return formatNumber(p[1]);
        if (name == QLatin1String("point_z")) return formatNumber(p[2]);
    }
    if (ctx.region && name == QLatin1String("region")) {
        const auto& r = *ctx.region;
        QStringList parts;
        for (double v : r) parts << formatNumber(v);
        return parts.join(QLatin1Char(','));
    }
    return {};
}

bool expandOne(const QString& in, const PluginManifest& manifest, const PluginContext& ctx,
               QString& out, QString& error)
{
    QList<Token> tokens;
    if (!tokenize(in, tokens, error)) {
        return false;
    }
    out.clear();
    for (const auto& token : tokens) {
        if (!token.placeholder) {
            out += token.text;
            continue;
        }
        const QString value = valueFor(token.text, manifest, ctx);
        if (value.isEmpty()) {
            error = QStringLiteral("%1 needs %2").arg(manifest.name, requirementFor(token.text));
            return false;
        }
        out += value;
    }
    return true;
}

} // namespace

QString requirementFor(const QString& placeholder)
{
    if (placeholder == QLatin1String("volume") || placeholder == QLatin1String("volume_id"))
        return QStringLiteral("an open volume");
    if (placeholder == QLatin1String("segment") || placeholder == QLatin1String("segment_id"))
        return QStringLiteral("an active segment");
    if (placeholder == QLatin1String("project")) return QStringLiteral("a project saved on disk");
    if (placeholder == QLatin1String("ink_overlay")) return QStringLiteral("a selected ink detection overlay");
    if (placeholder == QLatin1String("umbilicus")) return QStringLiteral("an attached umbilicus");
    if (placeholder.startsWith(QLatin1String("point"))) return QStringLiteral("a focus point");
    if (placeholder == QLatin1String("region")) return QStringLiteral("an enabled focus region");
    if (placeholder.startsWith(kInputPrefix))
        return QStringLiteral("a value for \"%1\"").arg(placeholder.mid(kInputPrefix.size()));
    return QStringLiteral("{%1}").arg(placeholder);
}

QString resultKindName(ResultKind kind)
{
    switch (kind) {
        case ResultKind::None: return QStringLiteral("none");
        case ResultKind::File: return QStringLiteral("file");
        case ResultKind::Segment: return QStringLiteral("segment");
        case ResultKind::InkOverlay: return QStringLiteral("ink_overlay");
        case ResultKind::Points: return QStringLiteral("points");
        case ResultKind::Umbilicus: return QStringLiteral("umbilicus");
    }
    return QStringLiteral("none");
}

ParseResult parseManifest(const QByteArray& json, const QString& manifestPath)
{
    ParseResult result;
    const QString where = QFileInfo(manifestPath).fileName();
    auto fail = [&](const QString& why) {
        result.error = QStringLiteral("%1: %2").arg(where, why);
        return result;
    };

    QJsonParseError parseError{};
    const auto doc = QJsonDocument::fromJson(json, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        return fail(QStringLiteral("not a JSON object (%1)").arg(parseError.errorString()));
    }
    const QJsonObject root = doc.object();

    static const QSet<QString> knownKeys{
        QStringLiteral("schema"), QStringLiteral("id"), QStringLiteral("name"),
        QStringLiteral("description"), QStringLiteral("menu"), QStringLiteral("program"),
        QStringLiteral("args"), QStringLiteral("inputs"), QStringLiteral("result"),
        QStringLiteral("homepage"), QStringLiteral("license")};
    for (auto it = root.begin(); it != root.end(); ++it) {
        if (!knownKeys.contains(it.key())) {
            return fail(QStringLiteral("unknown key \"%1\"").arg(it.key()));
        }
    }

    if (root.value(QStringLiteral("schema")).toString() != QLatin1String(kSchema)) {
        return fail(QStringLiteral("\"schema\" must be \"%1\"").arg(QLatin1String(kSchema)));
    }

    PluginManifest m;
    m.manifestPath = QFileInfo(manifestPath).absoluteFilePath();
    m.pluginDir = QFileInfo(manifestPath).absolutePath();
    m.sha256Hex = QCryptographicHash::hash(json, QCryptographicHash::Sha256).toHex();
    m.id = root.value(QStringLiteral("id")).toString();
    m.name = root.value(QStringLiteral("name")).toString().trimmed();
    m.description = root.value(QStringLiteral("description")).toString();
    m.menu = root.value(QStringLiteral("menu")).toString().trimmed();
    m.program = root.value(QStringLiteral("program")).toString();
    m.homepage = root.value(QStringLiteral("homepage")).toString();
    m.license = root.value(QStringLiteral("license")).toString();

    static const QRegularExpression idPattern(QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"));
    if (!idPattern.match(m.id).hasMatch()) {
        return fail(QStringLiteral("\"id\" must be 1 to 64 letters, digits, '.', '_' or '-'"));
    }
    if (m.name.isEmpty()) {
        return fail(QStringLiteral("\"name\" is required"));
    }
    if (m.program.trimmed().isEmpty()) {
        return fail(QStringLiteral("\"program\" is required"));
    }
    if (m.program.contains(QLatin1Char('{')) || m.program.contains(QLatin1Char('}'))) {
        return fail(QStringLiteral("\"program\" cannot contain placeholders; it is fixed by the manifest"));
    }

    QSet<QString> inputNames;
    const auto inputsValue = root.value(QStringLiteral("inputs"));
    if (!inputsValue.isUndefined()) {
        if (!inputsValue.isArray()) return fail(QStringLiteral("\"inputs\" must be an array"));
        static const QRegularExpression inputPattern(QStringLiteral("^[a-z][a-z0-9_]{0,31}$"));
        for (const auto& item : inputsValue.toArray()) {
            const auto obj = item.toObject();
            PluginInput input;
            input.name = obj.value(QStringLiteral("name")).toString();
            input.label = obj.value(QStringLiteral("label")).toString(input.name);
            input.defaultValue = obj.value(QStringLiteral("default")).toString();
            if (!item.isObject() || !inputPattern.match(input.name).hasMatch()) {
                return fail(QStringLiteral("each input needs a \"name\" of lowercase letters, digits or '_'"));
            }
            if (inputNames.contains(input.name)) {
                return fail(QStringLiteral("input \"%1\" is declared twice").arg(input.name));
            }
            inputNames.insert(input.name);
            m.inputs.push_back(input);
        }
    }

    const auto argsValue = root.value(QStringLiteral("args"));
    if (!argsValue.isUndefined()) {
        if (!argsValue.isArray()) return fail(QStringLiteral("\"args\" must be an array of strings"));
        for (const auto& item : argsValue.toArray()) {
            if (!item.isString()) return fail(QStringLiteral("\"args\" must be an array of strings"));
            QString error;
            if (!checkTemplate(item.toString(), inputNames, error)) return fail(error);
            m.args << item.toString();
        }
    }

    const auto resultValue = root.value(QStringLiteral("result"));
    if (!resultValue.isUndefined()) {
        if (!resultValue.isObject()) return fail(QStringLiteral("\"result\" must be an object"));
        const auto resultObj = resultValue.toObject();
        const auto kind = parseKind(resultObj.value(QStringLiteral("kind")).toString());
        if (!kind) {
            return fail(QStringLiteral("\"result.kind\" must be none, file, segment, ink_overlay, points or umbilicus"));
        }
        m.resultKind = *kind;
        m.resultPath = resultObj.value(QStringLiteral("path")).toString();
        if (m.resultKind != ResultKind::None) {
            // VC3D only loads what the run wrote into its own fresh folder.
            if (!m.resultPath.startsWith(QLatin1String("{output}/")) ||
                m.resultPath.split(QLatin1Char('/')).contains(QStringLiteral(".."))) {
                return fail(QStringLiteral("\"result.path\" must start with {output}/ and stay inside it"));
            }
            QString error;
            if (!checkTemplate(m.resultPath, inputNames, error)) return fail(error);
        }
    }

    result.manifest = std::move(m);
    return result;
}

QStringList pluginDirectories(const QString& homeDir, const QString& envPath)
{
    QStringList dirs;
    if (!homeDir.isEmpty()) {
        dirs << QDir(homeDir).filePath(QStringLiteral(".vc3d/plugins"));
    }
    for (const auto& entry : envPath.split(QDir::listSeparator(), Qt::SkipEmptyParts)) {
        const QString dir = QDir::cleanPath(entry.trimmed());
        if (!dir.isEmpty() && !dirs.contains(dir)) dirs << dir;
    }
    return dirs;
}

ScanResult scanPluginDirectories(const QStringList& dirs)
{
    ScanResult scan;
    QSet<QString> seenIds;
    for (const auto& dirPath : dirs) {
        QDir dir(dirPath);
        if (!dir.exists()) continue;
        const auto files = dir.entryInfoList({QStringLiteral("*.json")}, QDir::Files, QDir::Name);
        for (const auto& info : files) {
            QFile file(info.absoluteFilePath());
            if (!file.open(QIODevice::ReadOnly)) {
                scan.problems << QStringLiteral("%1: cannot be read").arg(info.absoluteFilePath());
                continue;
            }
            auto parsed = parseManifest(file.readAll(), info.absoluteFilePath());
            if (!parsed.manifest) {
                scan.problems << QStringLiteral("%1 (%2)").arg(parsed.error, info.absolutePath());
                continue;
            }
            if (seenIds.contains(parsed.manifest->id)) {
                scan.problems << QStringLiteral("%1: id \"%2\" is already used by an earlier plugin")
                                     .arg(info.absoluteFilePath(), parsed.manifest->id);
                continue;
            }
            seenIds.insert(parsed.manifest->id);
            scan.plugins.push_back(std::move(*parsed.manifest));
        }
    }
    return scan;
}

Expansion expand(const PluginManifest& manifest, const PluginContext& context)
{
    Expansion out;
    for (const auto& arg : manifest.args) {
        QString value;
        if (!expandOne(arg, manifest, context, value, out.error)) {
            out.args.clear();
            return out;
        }
        out.args << value;
    }
    if (manifest.resultKind != ResultKind::None) {
        if (!expandOne(manifest.resultPath, manifest, context, out.resultPath, out.error)) {
            out.args.clear();
            return out;
        }
        out.resultPath = QDir::cleanPath(out.resultPath);
        const QString root = QDir::cleanPath(context.output) + QLatin1Char('/');
        if (!out.resultPath.startsWith(root)) {
            out.error = QStringLiteral("%1: the result path leaves the run folder").arg(manifest.name);
            out.args.clear();
            out.resultPath.clear();
        }
    }
    return out;
}

QString resolveProgram(const PluginManifest& manifest, const QString& appDir)
{
    auto executable = [](const QString& path) {
        const QFileInfo info(path);
        return info.exists() && info.isFile() && info.isExecutable() ? info.absoluteFilePath() : QString();
    };
    const QString program = manifest.program.trimmed();
    if (QFileInfo(program).isAbsolute()) {
        return executable(program);
    }
    if (program.contains(QLatin1Char('/')) || program.contains(QLatin1Char('\\'))) {
        return executable(QDir(manifest.pluginDir).filePath(program));
    }
    if (!appDir.isEmpty()) {
        if (auto local = executable(QDir(appDir).filePath(program)); !local.isEmpty()) {
            return local;
        }
    }
    return QStandardPaths::findExecutable(program);
}

QString displayCommand(const QString& program, const QStringList& args)
{
    QStringList parts{program};
    for (const auto& arg : args) {
        const bool plain = !arg.isEmpty() &&
            !arg.contains(QRegularExpression(QStringLiteral("[\\s'\"\\\\$`;&|<>()*?!#~]")));
        parts << (plain ? arg : QStringLiteral("'%1'").arg(QString(arg).replace(QLatin1Char('\''), QStringLiteral("'\\''"))));
    }
    return parts.join(QLatin1Char(' '));
}

} // namespace vc3d::plugins
