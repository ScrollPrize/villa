#include "plugins/PluginManifest.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <iostream>

using namespace vc3d::plugins;

// Plain main() with a counting check: no moc, Qt Core only.
namespace {

int failures = 0;

void check(bool ok, const char* expr, int line, const QString& detail = {})
{
    if (!ok) {
        ++failures;
        std::cerr << "line " << line << ": check failed: " << expr;
        if (!detail.isEmpty()) std::cerr << " (" << detail.toStdString() << ")";
        std::cerr << '\n';
    }
}

#define CHECK(expr) check(static_cast<bool>(expr), #expr, __LINE__)
#define CHECK_MSG(expr, detail) check(static_cast<bool>(expr), #expr, __LINE__, (detail))
#define CHECK_EQ(a, b) check((a) == (b), #a " == " #b, __LINE__)

QByteArray manifestJson(const QString& extra = {})
{
    return QStringLiteral(R"({
        "schema": "vc3d-plugin/1",
        "id": "example.render",
        "name": "Render active segment",
        "program": "vc_render_tifxyz",
        "args": ["--volume", "{volume}", "--segmentation", "{segment}",
                 "--tif-output", "{output}/layers/%02d.tif", "--label", "{input:label}"],
        "inputs": [{"name": "label", "label": "Label", "default": "run"}],
        "result": {"kind": "file", "path": "{output}/layers/00.tif"}%1
    })").arg(extra).toUtf8();
}

QString parseError(const QString& json)
{
    return parseManifest(json.toUtf8(), QStringLiteral("/plugins/bad.json")).error;
}

void writeFile(const QString& path, const QByteArray& bytes)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    CHECK(file.open(QIODevice::WriteOnly));
    file.write(bytes);
}

void parsesValidManifest()
{
    const auto parsed = parseManifest(manifestJson(), QStringLiteral("/plugins/render.json"));
    CHECK_MSG(parsed.manifest.has_value(), parsed.error);
    const auto& m = *parsed.manifest;
    CHECK_EQ(m.id, QStringLiteral("example.render"));
    CHECK_EQ(m.resultKind, ResultKind::File);
    CHECK_EQ(m.inputs.size(), 1);
    CHECK_EQ(m.inputs.front().defaultValue, QStringLiteral("run"));
    CHECK_EQ(m.pluginDir, QFileInfo(QStringLiteral("/plugins/render.json")).absolutePath());
    CHECK_EQ(m.sha256Hex.size(), 64);
}

void rejectsUnsafeOrUnclearManifests()
{
    const QString base = QStringLiteral(R"({"schema":"vc3d-plugin/1","id":"a","name":"A","program":"p")");
    CHECK(parseError(QStringLiteral("[]")).contains(QStringLiteral("not a JSON object")));
    CHECK(parseError(QStringLiteral(R"({"schema":"vc3d-plugin/2","id":"a","name":"A","program":"p"})"))
                .contains(QStringLiteral("schema")));
    CHECK(parseError(base + QStringLiteral(R"(,"run":"rm -rf /"})")).contains(QStringLiteral("unknown key")));
    CHECK(parseError(QStringLiteral(R"({"schema":"vc3d-plugin/1","id":"../x","name":"A","program":"p"})"))
                .contains(QStringLiteral("\"id\"")));
    CHECK(parseError(QStringLiteral(R"({"schema":"vc3d-plugin/1","id":"a","name":"A","program":"{volume}/bin"})"))
                .contains(QStringLiteral("cannot contain placeholders")));
    CHECK(parseError(base + QStringLiteral(R"(,"args":["{volumes}"]})")).contains(QStringLiteral("unknown placeholder")));
    CHECK(parseError(base + QStringLiteral(R"(,"args":["{input:scroll}"]})")).contains(QStringLiteral("does not declare")));
    CHECK(parseError(base + QStringLiteral(R"(,"args":["{volume"]})")).contains(QStringLiteral("unclosed")));
    CHECK(parseError(base + QStringLiteral(R"(,"args":["a}"]})")).contains(QStringLiteral("unmatched")));
    CHECK(parseError(base + QStringLiteral(R"(,"args":[3]})")).contains(QStringLiteral("strings")));
    CHECK(parseError(base + QStringLiteral(R"(,"result":{"kind":"script","path":"{output}/a"}})"))
                .contains(QStringLiteral("result.kind")));
    CHECK(parseError(base + QStringLiteral(R"(,"result":{"kind":"points","path":"/tmp/a.json"}})"))
                .contains(QStringLiteral("{output}/")));
    CHECK(parseError(base + QStringLiteral(R"(,"result":{"kind":"points","path":"{output}/../a.json"}})"))
                .contains(QStringLiteral("{output}/")));
    CHECK(parseError(base + QStringLiteral(R"(,"inputs":[{"name":"x"},{"name":"x"}]})")).contains(QStringLiteral("twice")));
}

void expandsEachTemplateToExactlyOneArgument()
{
    auto m = *parseManifest(manifestJson(), QStringLiteral("/plugins/render.json")).manifest;
    PluginContext ctx;
    ctx.volume = QStringLiteral("/data/scroll vol.zarr");
    ctx.segment = QStringLiteral("/data/segs/w035; rm -rf ~");
    ctx.output = QStringLiteral("/runs/example.render/1");
    ctx.inputs.insert(QStringLiteral("label"), QStringLiteral("a b"));
    const auto e = expand(m, ctx);
    CHECK_MSG(e.error.isEmpty(), e.error);
    CHECK_EQ(e.args, (QStringList{QStringLiteral("--volume"), QStringLiteral("/data/scroll vol.zarr"),
                                  QStringLiteral("--segmentation"), QStringLiteral("/data/segs/w035; rm -rf ~"),
                                  QStringLiteral("--tif-output"), QStringLiteral("/runs/example.render/1/layers/%02d.tif"),
                                  QStringLiteral("--label"), QStringLiteral("a b")}));
    CHECK_EQ(e.resultPath, QStringLiteral("/runs/example.render/1/layers/00.tif"));
}

void refusesWhenContextIsMissing()
{
    auto m = *parseManifest(manifestJson(), QStringLiteral("/plugins/render.json")).manifest;
    PluginContext ctx;
    ctx.volume = QStringLiteral("/data/v.zarr");
    ctx.output = QStringLiteral("/runs/x");
    ctx.inputs.insert(QStringLiteral("label"), QStringLiteral("x"));
    const auto e = expand(m, ctx);
    CHECK(e.args.isEmpty());
    CHECK(e.error.contains(QStringLiteral("an active segment")));
}

void formatsPointRegionAndBraces()
{
    const auto parsed = parseManifest(QByteArrayLiteral(R"({
        "schema": "vc3d-plugin/1", "id": "p", "name": "P", "program": "tool",
        "args": ["{point}", "{point_z}", "{region}", "{{\"k\": 1}}", "{plugin_dir}/model.pt"],
        "result": {"kind": "points", "path": "{output}/points.json"}
    })"), QStringLiteral("/opt/plugins/p.json"));
    CHECK_MSG(parsed.manifest.has_value(), parsed.error);
    PluginContext ctx;
    ctx.point = std::array<double, 3>{1.5, 2.0, 3000.25};
    ctx.region = std::array<double, 6>{0, 1, 2, 10, 11, 12};
    ctx.output = QStringLiteral("/runs/p/1/");
    const auto e = expand(*parsed.manifest, ctx);
    CHECK_MSG(e.error.isEmpty(), e.error);
    CHECK_EQ(e.args.at(0), QStringLiteral("1.5,2,3000.25"));
    CHECK_EQ(e.args.at(1), QStringLiteral("3000.25"));
    CHECK_EQ(e.args.at(2), QStringLiteral("0,1,2,10,11,12"));
    CHECK_EQ(e.args.at(3), QStringLiteral("{\"k\": 1}"));
    CHECK_EQ(e.args.at(4), QFileInfo(QStringLiteral("/opt/plugins/p.json")).absolutePath() + QStringLiteral("/model.pt"));
    CHECK_EQ(e.resultPath, QStringLiteral("/runs/p/1/points.json"));
}

void resultPathCannotEscapeThroughAnInput()
{
    const auto parsed = parseManifest(QByteArrayLiteral(R"({
        "schema": "vc3d-plugin/1", "id": "p", "name": "P", "program": "tool",
        "inputs": [{"name": "name"}],
        "result": {"kind": "segment", "path": "{output}/{input:name}"}
    })"), QStringLiteral("/opt/plugins/p.json"));
    CHECK(parsed.manifest.has_value());
    PluginContext ctx;
    ctx.output = QStringLiteral("/runs/p/1");
    ctx.inputs.insert(QStringLiteral("name"), QStringLiteral("../../../home/user/segs"));
    const auto e = expand(*parsed.manifest, ctx);
    CHECK(e.error.contains(QStringLiteral("leaves the run folder")));
    CHECK(e.resultPath.isEmpty());
}

void scansFoldersAndReportsProblems()
{
    QTemporaryDir first;
    QTemporaryDir second;
    CHECK(first.isValid() && second.isValid());
    writeFile(first.filePath(QStringLiteral("a.json")), manifestJson());
    writeFile(first.filePath(QStringLiteral("b.json")), QByteArrayLiteral("{not json"));
    writeFile(first.filePath(QStringLiteral("notes.txt")), QByteArrayLiteral("ignored"));
    writeFile(second.filePath(QStringLiteral("a.json")), manifestJson());
    const auto scan = scanPluginDirectories({first.path(), second.path(), QStringLiteral("/no/such/dir")});
    CHECK_EQ(scan.plugins.size(), 1);
    CHECK_EQ(scan.problems.size(), 2);
    CHECK(scan.problems.join(QLatin1Char('\n')).contains(QStringLiteral("already used")));

    const auto dirs = pluginDirectories(QStringLiteral("/home/u"),
                                        QStringLiteral("/a") + QDir::listSeparator() + QStringLiteral("/b/"));
    CHECK_EQ(dirs, (QStringList{QStringLiteral("/home/u/.vc3d/plugins"), QStringLiteral("/a"), QStringLiteral("/b")}));
}

void resolvesPrograms()
{
    QTemporaryDir dir;
    CHECK(dir.isValid());
    const QString tool = dir.filePath(QStringLiteral("bin/tool.sh"));
    writeFile(tool, QByteArrayLiteral("#!/bin/sh\n"));
    QFile::setPermissions(tool, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
    PluginManifest m;
    m.pluginDir = dir.path();
    m.program = QStringLiteral("bin/tool.sh");
    CHECK_EQ(resolveProgram(m, {}), QFileInfo(tool).absoluteFilePath());
    m.program = QStringLiteral("tool.sh");
    CHECK_EQ(resolveProgram(m, dir.filePath(QStringLiteral("bin"))), QFileInfo(tool).absoluteFilePath());
    m.program = QStringLiteral("definitely-not-a-real-program-vc3d");
    CHECK(resolveProgram(m, {}).isEmpty());
}

void quotesDisplayedCommands()
{
    CHECK_EQ(displayCommand(QStringLiteral("/bin/tool"), {QStringLiteral("a"), QStringLiteral("b c"), QStringLiteral("it's")}),
             QStringLiteral("/bin/tool a 'b c' 'it'\\''s'"));
}

} // namespace

int main()
{
    parsesValidManifest();
    rejectsUnsafeOrUnclearManifests();
    expandsEachTemplateToExactlyOneArgument();
    refusesWhenContextIsMissing();
    formatsPointRegionAndBraces();
    resultPathCannotEscapeThroughAnInput();
    scansFoldersAndReportsProblems();
    resolvesPrograms();
    quotesDisplayedCommands();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "plugin manifest checks passed\n";
    return 0;
}
