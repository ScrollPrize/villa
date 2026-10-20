#pragma once

// Manifest-declared external tools ("plugins") for VC3D.
//
// A plugin is a JSON file that declares one program, its arguments with
// placeholders for what VC3D currently has open, and how VC3D should load the
// file the program writes. No code is loaded into VC3D: the declared program
// runs as a separate process with an argument list (never through a shell),
// and the result is read with VC3D's existing loaders. See
// docs/plugins.md for the format and the trust model.
//
// This header is Qt Core only so it can be tested without a GUI.

#include <QByteArray>
#include <QList>
#include <QMap>
#include <QString>
#include <QStringList>

#include <array>
#include <optional>

namespace vc3d::plugins {

inline constexpr char kSchema[] = "vc3d-plugin/1";

enum class ResultKind {
    None,       // nothing to load; VC3D reports the run folder
    File,       // a report or image; VC3D shows the path and offers to open it
    Segment,    // a tifxyz segment folder; attached to the project
    InkOverlay, // an image on the active segment's grid; added to its ink detections
    Points,     // a VC3D point collection JSON; its collections are added
    Umbilicus,  // an umbilicus.json; set as the project's umbilicus
};

struct PluginInput {
    QString name;
    QString label;
    QString defaultValue;
};

struct PluginManifest {
    QString id;
    QString name;
    QString description;
    QString menu;
    QString program;
    QStringList args;
    QList<PluginInput> inputs;
    ResultKind resultKind{ResultKind::None};
    QString resultPath;
    QString homepage;
    QString license;
    QString manifestPath;
    QString pluginDir;
    QByteArray sha256Hex;
};

// What VC3D knows when a plugin starts. An empty value means "not available";
// a plugin that uses its placeholder refuses to start and says why.
struct PluginContext {
    QString volume;
    QString volumeId;
    QString segment;
    QString segmentId;
    QString project;
    QString inkOverlay;
    QString umbilicus;
    std::optional<std::array<double, 3>> point;
    std::optional<std::array<double, 6>> region;
    QString output;
    QMap<QString, QString> inputs;
};

struct ParseResult {
    std::optional<PluginManifest> manifest;
    QString error;
};

struct ScanResult {
    QList<PluginManifest> plugins;
    QStringList problems;
};

struct Expansion {
    QStringList args;
    QString resultPath;
    QString error;
};

// Parses and validates one manifest. manifestPath is used for messages, for
// {plugin_dir} and for resolving a relative program path.
ParseResult parseManifest(const QByteArray& json, const QString& manifestPath);

// ~/.vc3d/plugins first, then each entry of VC3D_PLUGIN_PATH (list separator
// ':' on Unix, ';' on Windows). Missing folders are kept; the scan skips them.
QStringList pluginDirectories(const QString& homeDir, const QString& envPath);

// Reads every *.json in each folder (not recursive), sorted by file name.
// Invalid manifests and repeated ids become problems, never plugins.
ScanResult scanPluginDirectories(const QStringList& dirs);

// Substitutes placeholders. Each template stays exactly one argument, so a
// value with spaces or shell characters can never split or run anything.
Expansion expand(const PluginManifest& manifest, const PluginContext& context);

// An absolute program is used as is. A bare name is looked up next to the
// VC3D executable (so villa's vc_* tools resolve first), then on PATH. A
// relative path with a separator is resolved against the manifest's folder.
// Returns an empty string when nothing executable is found.
QString resolveProgram(const PluginManifest& manifest, const QString& appDir);

QString resultKindName(ResultKind kind);

// Human wording for an unavailable placeholder, e.g. "an active segment".
QString requirementFor(const QString& placeholder);

// Arguments joined for display, each quoted when it contains spaces.
QString displayCommand(const QString& program, const QStringList& args);

} // namespace vc3d::plugins
