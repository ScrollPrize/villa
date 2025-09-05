#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"

#include <QSettings>
#include <QMessageBox>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QInputDialog>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QFile>
#include <QTextStream>

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "ToolDialogs.hpp"





// --------- local helpers for running external tools -------------------------
static bool runProcessBlocking(const QString& program,
                               const QStringList& args,
                               const QString& workDir,
                               QString* out=nullptr,
                               QString* err=nullptr)
{
    QProcess p;
    if (!workDir.isEmpty()) p.setWorkingDirectory(workDir);
    p.setProcessChannelMode(QProcess::SeparateChannels);
    p.start(program, args);
    if (!p.waitForStarted()) { if (err) *err = QObject::tr("Failed to start %1").arg(program); return false; }
    if (!p.waitForFinished(-1)) { if (err) *err = QObject::tr("Timeout running %1").arg(program); return false; }
    if (out) *out = QString::fromLocal8Bit(p.readAllStandardOutput());
    if (err) *err = QString::fromLocal8Bit(p.readAllStandardError());
    return (p.exitStatus()==QProcess::NormalExit && p.exitCode()==0);
}

static QString resolvePythonPath()
{
    QSettings s("VC.ini", QSettings::IniFormat);
    const QString ini = s.value("python/path").toString();
    if (!ini.isEmpty() && QFileInfo::exists(ini)) return ini;

    const QString env = QString::fromLocal8Bit(qgetenv("VC_PYTHON"));
    if (!env.isEmpty() && QFileInfo::exists(env)) return env;

    // Prefer micromamba env you mentioned
    if (QFileInfo::exists("/opt/micromamba/envs/py310/bin/python"))
        return "/opt/micromamba/envs/py310/bin/python";

    // Reasonable fallbacks
    if (QFileInfo::exists("/opt/venv/bin/python3"))  return "/opt/venv/bin/python3";
    if (QFileInfo::exists("/usr/local/bin/python3")) return "/usr/local/bin/python3";
    if (QFileInfo::exists("/usr/bin/python3"))       return "/usr/bin/python3";
    return "python3";
}

static QString resolveFlatboiScript()
{
    QSettings s("VC.ini", QSettings::IniFormat);
    const QString ini = s.value("scripts/flatboi_path").toString();
    if (!ini.isEmpty()) return ini;

    const QString envDir = QString::fromLocal8Bit(qgetenv("VC_SCRIPTS_DIR"));
    if (!envDir.isEmpty()) return QDir(envDir).filePath("flatboi.py");

    // Default to the repo path you requested
    if (QFileInfo::exists("/src/scripts/flatboi.py"))
        return "/src/scripts/flatboi.py";

    if (QFileInfo::exists("/usr/bin/flatboi.py"))
        return "/usr/bin/flatboi.py";

    // Last resort: relative to binary
    QDir bin(QCoreApplication::applicationDirPath());

    if (QFileInfo::exists(bin.filePath("flatboi.py"))) {
        return bin.filePath("flatboi.py");
    }

    return QDir(bin.filePath("../scripts")).filePath("flatboi.py");
}
// ---------------------------------------------------------------------------


void CWindow::onRenderSegment(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot render segment: No volume or invalid segment selected"));
        return;
    }

    QSettings settings("VC.ini", QSettings::IniFormat);

    const QString volumePath = getCurrentVolumePath();
    const QString segmentPath = QString::fromStdString(surfMeta->path.string());
    const QString segmentOutDir = QString::fromStdString(surfMeta->path.string());
    const QString outputFormat = "%s/layers/%02d.tif";
    const float scale = 1.0f;
    const int resolution = 0;
    const int layers = 31;
    const QString outputPattern = QString(outputFormat).replace("%s", segmentOutDir);

    // Prompt user for parameters
    RenderParamsDialog dlg(this, volumePath, segmentPath, outputPattern, scale, resolution, layers);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Render cancelled"), 3000);
        return;
    }

    // Initialize command line tool runner if needed
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);
        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool /*tool*/, bool success, const QString& message,
                       const QString& /*outputPath*/, bool copyToClipboard) {
                    if (success) {
                        QString displayMsg = message;
                        if (copyToClipboard) {
                            displayMsg += tr(" - Path copied to clipboard");
                        }
                        statusBar()->showMessage(displayMsg, 5000);
                        QMessageBox::information(this, tr("Rendering Complete"), displayMsg);
                    } else {
                        statusBar()->showMessage(tr("Rendering failed"), 5000);
                        QMessageBox::critical(this, tr("Rendering Error"), message);
                    }
                });
    }

    // Check if a tool is already running
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Set up parameters and execute the render tool
    _cmdRunner->setSegmentPath(dlg.segmentPath());
    _cmdRunner->setOutputPattern(dlg.outputPattern());
    _cmdRunner->setRenderParams(static_cast<float>(dlg.scale()), dlg.groupIdx(), dlg.numSlices());
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->setVolumePath(dlg.volumePath());
    _cmdRunner->setRenderAdvanced(
        dlg.cropX(), dlg.cropY(), dlg.cropWidth(), dlg.cropHeight(),
        dlg.affinePath(), dlg.invertAffine(),
        static_cast<float>(dlg.scaleSegmentation()), dlg.rotateDegrees(), dlg.flipAxis());
    _cmdRunner->setIncludeTifs(dlg.includeTifs());

    _cmdRunner->execute(CommandLineToolRunner::Tool::RenderTifXYZ);

    statusBar()->showMessage(tr("Rendering segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onSlimFlattenAndRender(const std::string& segmentId)
{
    auto surfMeta = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot SLIM-flatten: No volume or invalid segment selected"));
        return;
    }
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Paths
    const std::filesystem::path segDirFs = surfMeta->path;           // tifxyz folder
    const QString  segDir   = QString::fromStdString(segDirFs.string());
    const QString  objPath  = QDir(segDir).filePath(QString::fromStdString(segmentId) + ".obj");
    const QString  flatObj  = QDir(segDir).filePath(QString::fromStdString(segmentId) + "_flatboi.obj");
    QString        outTifxyz= segDir + "_flatboi";

    // If the output dir already exists, offer to delete it (vc_obj2tifxyz requires a non-existent target)
    if (QFileInfo::exists(outTifxyz)) {
        const auto ans = QMessageBox::question(
            this, tr("Output Exists"),
            tr("The output directory already exists:\n%1\n\nDelete it and recreate?").arg(outTifxyz),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (ans == QMessageBox::No) {
            statusBar()->showMessage(tr("SLIM-flatten cancelled by user (existing output)."), 5000);
            return;
        }
        QDir dir(outTifxyz);
        if (!dir.removeRecursively()) {
            QMessageBox::critical(this, tr("Error"),
                                  tr("Failed to remove existing output directory:\n%1").arg(outTifxyz));
            return;
        }
    }

    // 1) tifxyz -> obj
    statusBar()->showMessage(tr("Converting TIFXYZ to OBJ…"), 0);
    {
        QString err;
        if (!runProcessBlocking("vc_tifxyz2obj", QStringList() << segDir << objPath, segDir, nullptr, &err)) {
            QMessageBox::critical(this, tr("Error"), tr("vc_tifxyz2obj failed.\n\n%1").arg(err));
            statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
            return;
        }
    }

    // 2) SLIM via python: python /src/scripts/flatboi.py <obj> 60
    statusBar()->showMessage(tr("Running SLIM (flatboi.py)…"), 0);
    {
        const QString py = resolvePythonPath();
        const QString script = resolveFlatboiScript();
        if (!QFileInfo::exists(script)) {
            QMessageBox::critical(this, tr("Error"), tr("flatboi.py not found at:\n%1").arg(script));
            statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
            return;
        }
        QString err;
        if (!runProcessBlocking(py, QStringList() << script << objPath << "60", segDir, nullptr, &err)) {
            QMessageBox::critical(this, tr("Error"), tr("flatboi.py failed.\n\n%1").arg(err));
            statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
            return;
        }
        if (!QFileInfo::exists(flatObj)) {
            // flatboi writes <basename>_flatboi.obj next to the input .obj
            QMessageBox::critical(this, tr("Error"),
                                  tr("Flattened OBJ was not created:\n%1").arg(flatObj));
            statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
            return;
        }
    }

    // 3) flattened obj -> tifxyz  (IMPORTANT: do NOT pre-create the directory)
    statusBar()->showMessage(tr("Converting flattened OBJ back to TIFXYZ…"), 0);
    {
        QString err;
        if (!runProcessBlocking("vc_obj2tifxyz", QStringList() << flatObj << outTifxyz, segDir, nullptr, &err)) {
            QMessageBox::critical(this, tr("Error"), tr("vc_obj2tifxyz failed.\n\n%1").arg(err));
            statusBar()->showMessage(tr("SLIM-flatten failed"), 5000);
            return;
        }
    }

    // 4) render the *_flatboi folder
    if (!initializeCommandLineRunner()) {
        QMessageBox::critical(this, tr("Error"), tr("Failed to initialize command runner."));
        return;
    }
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }
    {
        QString outputFormat = "%s/layers/%02d.tif";
        float scale = 1.0f;
        int resolution = 0;
        int layers = 31;
        const QString outPattern = outputFormat.replace("%s", outTifxyz);

        _cmdRunner->setSegmentPath(outTifxyz);
        _cmdRunner->setOutputPattern(outPattern);
        _cmdRunner->setRenderParams(scale, resolution, layers);
        _cmdRunner->execute(CommandLineToolRunner::Tool::RenderTifXYZ);
        statusBar()->showMessage(tr("Rendering flattened segment…"), 0);
    }
}


void CWindow::onGrowSegmentFromSegment(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow segment: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow segment: Invalid segment or segment not loaded"));
        return;
    }

    // Initialize command line tool runner if needed
    if (!initializeCommandLineRunner()) {
        return;
    }

    // Check if a tool is already running
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Get paths
    QString srcSegment = QString::fromStdString(surfMeta->path.string());

    // Get the volpkg path and create traces directory if it doesn't exist
    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path tracesDir = volpkgPath / "traces";
    std::filesystem::path jsonParamsPath = volpkgPath / "trace_params.json";
    std::filesystem::path pathsDir = volpkgPath / "paths";

    statusBar()->showMessage(tr("Preparing to run grow_seg_from_segment..."), 2000);

    // Create traces directory if it doesn't exist
    if (!std::filesystem::exists(tracesDir)) {
        try {
            std::filesystem::create_directory(tracesDir);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to create traces directory: %1").arg(e.what()));
            return;
        }
    }

    // Check if trace_params.json exists
    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(this, tr("Error"), tr("trace_params.json not found in the volpkg"));
        return;
    }

    // Prompt user for parameters
    TraceParamsDialog dlg(this,
                          getCurrentVolumePath(),
                          QString::fromStdString(pathsDir.string()),
                          QString::fromStdString(tracesDir.string()),
                          QString::fromStdString(jsonParamsPath.string()),
                          srcSegment);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Run trace cancelled"), 3000);
        return;
    }

    // Merge JSON from disk with UI overrides, write to a temp file inside target dir
    QJsonObject base;
    {
        QFile f(dlg.jsonParams());
        if (f.open(QIODevice::ReadOnly)) {
            const auto doc = QJsonDocument::fromJson(f.readAll());
            f.close();
            if (doc.isObject()) base = doc.object();
        }
    }
    const QJsonObject ui = dlg.makeParamsJson();
    for (auto it = ui.begin(); it != ui.end(); ++it) base[it.key()] = it.value();

    const QString mergedJsonPath = QDir(dlg.tgtDir()).filePath(QString("trace_params_ui.json"));
    {
        QFile f(mergedJsonPath);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to write params JSON: %1").arg(mergedJsonPath));
            return;
        }
        f.write(QJsonDocument(base).toJson(QJsonDocument::Indented));
        f.close();
    }

    // Set up parameters and execute the tool
    _cmdRunner->setTraceParams(
        dlg.volumePath(),
        dlg.srcDir(),
        dlg.tgtDir(),
        mergedJsonPath,
        dlg.srcSegment());
    _cmdRunner->setOmpThreads(dlg.ompThreads());

    // Show console before executing to see any debug output
    _cmdRunner->showConsoleOutput();
    _cmdRunner->execute(CommandLineToolRunner::Tool::GrowSegFromSegment);
    statusBar()->showMessage(tr("Growing segment from: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onAddOverlap(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot add overlap: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot add overlap: Invalid segment or segment not loaded"));
        return;
    }

    // Initialize command line tool runner if needed
    if (!initializeCommandLineRunner()) {
        return;
    }

    // Check if a tool is already running
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Get paths
    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";
    QString tifxyzPath = QString::fromStdString(surfMeta->path.string());

    // Set up parameters and execute the tool
    _cmdRunner->setAddOverlapParams(
        QString::fromStdString(pathsDir.string()),
        tifxyzPath
    );

    _cmdRunner->execute(CommandLineToolRunner::Tool::SegAddOverlap);

    statusBar()->showMessage(tr("Adding overlap for segment: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onConvertToObj(const std::string& segmentId)
{
    if (currentVolume == nullptr || !fVpkg) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot convert to OBJ: No volume package loaded"));
        return;
    }

    auto surfMeta = fVpkg->getSurface(segmentId);
    if (!surfMeta) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot convert to OBJ: Invalid segment or segment not loaded"));
        return;
    }

    // Initialize command line tool runner if needed
    if (!initializeCommandLineRunner()) {
        return;
    }

    // Check if a tool is already running
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Get source tifxyz path (this is a directory containing the TIFXYZ files)
    std::filesystem::path tifxyzPath = surfMeta->path;

    // Generate output OBJ path inside the TIFXYZ directory with segment ID as filename
    std::filesystem::path objPath = tifxyzPath / (segmentId + ".obj");

    // Prompt for parameters
    ConvertToObjDialog dlg(this,
                           QString::fromStdString(tifxyzPath.string()),
                           QString::fromStdString(objPath.string()));
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Convert to OBJ cancelled"), 3000);
        return;
    }

    // Set up parameters and execute the tool
    _cmdRunner->setToObjParams(dlg.tifxyzPath(), dlg.objPath());
    _cmdRunner->setOmpThreads(dlg.ompThreads());
    _cmdRunner->setToObjOptions(dlg.normalizeUV(), dlg.alignGrid(), dlg.decimateIterations(), dlg.cleanSurface(), static_cast<float>(dlg.cleanK()));
    _cmdRunner->execute(CommandLineToolRunner::Tool::tifxyz2obj);
    statusBar()->showMessage(tr("Converting segment to OBJ: %1").arg(QString::fromStdString(segmentId)), 5000);
}

void CWindow::onGrowSeeds(const std::string& segmentId, bool isExpand, bool isRandomSeed)
{
    if (currentVolume == nullptr) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot grow seeds: No volume loaded"));
        return;
    }

    // Initialize command line tool runner if needed
    if (!initializeCommandLineRunner()) {
        return;
    }

    // Check if a tool is already running
    if (_cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    // Get paths
    std::filesystem::path volpkgPath = std::filesystem::path(fVpkgPath.toStdString());
    std::filesystem::path pathsDir = volpkgPath / "paths";

    // Create traces directory if it doesn't exist
    if (!std::filesystem::exists(pathsDir)) {
        QMessageBox::warning(this, tr("Error"), tr("Paths directory not found in the volpkg"));
        return;
    }

    // Get JSON parameters file
    QString jsonFileName = isExpand ? "expand.json" : "seed.json";
    std::filesystem::path jsonParamsPath = volpkgPath / jsonFileName.toStdString();

    // Check if JSON file exists
    if (!std::filesystem::exists(jsonParamsPath)) {
        QMessageBox::warning(this, tr("Error"), tr("%1 not found in the volpkg").arg(jsonFileName));
        return;
    }

    // Get current POI (focus point) for seed coordinates if needed
    int seedX = 0, seedY = 0, seedZ = 0;
    if (!isExpand && !isRandomSeed) {
        POI *poi = _surf_col->poi("focus");
        if (!poi) {
            QMessageBox::warning(this, tr("Error"), tr("No focus point selected. Click on a volume with Ctrl key to set a seed point."));
            return;
        }
        seedX = static_cast<int>(poi->p[0]);
        seedY = static_cast<int>(poi->p[1]);
        seedZ = static_cast<int>(poi->p[2]);
    }

    // Set up parameters and execute the tool
    _cmdRunner->setGrowParams(
        QString(),  // Volume path will be set automatically in execute()
        QString::fromStdString(pathsDir.string()),
        QString::fromStdString(jsonParamsPath.string()),
        seedX,
        seedY,
        seedZ,
        isExpand,
        isRandomSeed
    );

    _cmdRunner->execute(CommandLineToolRunner::Tool::GrowSegFromSeeds);

    QString modeDesc = isExpand ? "expand mode" :
                      (isRandomSeed ? "random seed mode" : "seed mode");
    statusBar()->showMessage(tr("Growing segment using %1 in %2").arg(jsonFileName).arg(modeDesc), 5000);
}

// Helper method to initialize command line runner
bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);

        // Read parallel processes and iteration count settings from INI file
        QSettings settings("VC.ini", QSettings::IniFormat);
        int parallelProcesses = settings.value("perf/parallel_processes", 8).toInt();
        int iterationCount = settings.value("perf/iteration_count", 1000).toInt();

        // Apply the settings
        _cmdRunner->setParallelProcesses(parallelProcesses);
        _cmdRunner->setIterationCount(iterationCount);

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool /*tool*/, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    if (success) {
                        QString displayMsg = message;
                        if (copyToClipboard) {
                            displayMsg += tr(" - Path copied to clipboard");
                        }
                        statusBar()->showMessage(displayMsg, 5000);
                        QMessageBox::information(this, tr("Operation Complete"), displayMsg);
                    } else {
                        statusBar()->showMessage(tr("Operation failed"), 5000);
                        QMessageBox::critical(this, tr("Error"), message);
                    }
                });
    }
    return true;
}

void CWindow::onDeleteSegments(const std::vector<std::string>& segmentIds)
{
    if (segmentIds.empty()) {
        return;
    }

    // Create confirmation message
    QString message;
    if (segmentIds.size() == 1) {
        message = tr("Are you sure you want to delete segment '%1'?\n\nThis action cannot be undone.")
                    .arg(QString::fromStdString(segmentIds[0]));
    } else {
        message = tr("Are you sure you want to delete %1 segments?\n\nThis action cannot be undone.")
                    .arg(segmentIds.size());
    }

    // Show confirmation dialog
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, tr("Confirm Deletion"), message,
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    if (reply != QMessageBox::Yes) {
        return;
    }

    // Delete each segment
    int successCount = 0;
    QStringList failedSegments;
    bool needsReload = false;

    for (const auto& segmentId : segmentIds) {
        try {
            // Use the VolumePkg's removeSegmentation method
            fVpkg->removeSegmentation(segmentId);
            successCount++;
            needsReload = true;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to delete segment " << segmentId << ": " << e.what() << std::endl;

            // Check if it's a permission error
            if (e.code() == std::errc::permission_denied) {
                failedSegments << QString::fromStdString(segmentId) + " (permission denied)";
            } else {
                failedSegments << QString::fromStdString(segmentId) + " (filesystem error)";
            }
        } catch (const std::exception& e) {
            failedSegments << QString::fromStdString(segmentId);
            std::cerr << "Failed to delete segment " << segmentId << ": " << e.what() << std::endl;
        }
    }

    // Only update UI if we successfully deleted something
    if (needsReload) {
        try {
            // Use incremental removal to update the UI for each successfully deleted segment
            for (const auto& segmentId : segmentIds) {
                // Only remove from UI if it was successfully deleted from disk
                if (std::find(failedSegments.begin(), failedSegments.end(),
                            QString::fromStdString(segmentId)) == failedSegments.end() &&
                    std::find(failedSegments.begin(), failedSegments.end(),
                            QString::fromStdString(segmentId) + " (permission denied)") == failedSegments.end() &&
                    std::find(failedSegments.begin(), failedSegments.end(),
                            QString::fromStdString(segmentId) + " (filesystem error)") == failedSegments.end()) {
                    RemoveSingleSegmentation(segmentId);
                }
            }

            // Update the volpkg label and filters
            UpdateVolpkgLabel(0);
            onSegFilterChanged(0);
        } catch (const std::exception& e) {
            std::cerr << "Error updating UI after deletion: " << e.what() << std::endl;
            QMessageBox::warning(this, tr("Warning"),
                               tr("Segments were deleted but there was an error refreshing the list. "
                                  "Please reload surfaces manually."));
        }
    }

    // Show result message
    if (successCount == segmentIds.size()) {
        statusBar()->showMessage(tr("Successfully deleted %1 segment(s)").arg(successCount), 5000);
    } else if (successCount > 0) {
        QMessageBox::warning(this, tr("Partial Success"),
            tr("Deleted %1 segment(s), but failed to delete: %2\n\n"
               "Note: Permission errors may require manual deletion or running with elevated privileges.")
            .arg(successCount)
            .arg(failedSegments.join(", ")));
    } else {
        QMessageBox::critical(this, tr("Deletion Failed"),
            tr("Failed to delete any segments.\n\n"
               "Failed segments: %1\n\n"
               "This may be due to insufficient permissions. "
               "Try running the application with elevated privileges or manually delete the folders.")
            .arg(failedSegments.join(", ")));
    }
}
