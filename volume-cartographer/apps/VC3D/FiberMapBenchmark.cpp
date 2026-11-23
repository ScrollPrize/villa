// Headless timing of the fiber map's Update, so the stages can be measured
// and re-measured without anyone clicking a button.
//
// `vc3d --bench-fiber-map <project.vcproj> [--follow-solve] [--runs N]`
// opens the project exactly as the GUI does, shows the fiber map, optionally
// selects the newest saved solve and follows it, then presses Update and
// waits for the rebuild to publish. Every run prints one line; the map's own
// per-stage log line ("fiber map rebuild: ...") is on the same output.
#include "CWindow.hpp"
#include "FiberMapScaffoldSolve.hpp"

#include <set>
#include "FiberMapWorkspace.hpp"
#include "MenuActionController.hpp"
#include "VCSettings.hpp"

#include "vc/core/util/Logging.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QElapsedTimer>
#include <QPushButton>
#include <QSettings>
#include <QTabWidget>
#include <QTreeWidget>

#include <functional>
#include <iostream>

namespace
{
// Spin the event loop until `ready` answers true, or the budget runs out.
bool waitFor(const std::function<bool()>& ready, int milliseconds)
{
    QElapsedTimer timer;
    timer.start();
    while (!ready()) {
        if (timer.elapsed() > milliseconds) return false;
        QApplication::processEvents(QEventLoop::AllEvents, 20);
    }
    return true;
}

QPushButton* buttonNamed(QWidget* parent, const QString& text)
{
    for (auto* button : parent->findChildren<QPushButton*>())
        if (button->text() == text) return button;
    return nullptr;
}
} // namespace

int runFiberMapBenchmark(const QString& project, bool followSolve, int runs, const QString& solveNormals)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const auto cacheGB = settings.value(vc3d::settings::perf::RAM_CACHE_SIZE_GB,
                                        vc3d::settings::perf::RAM_CACHE_SIZE_GB_DEFAULT).toULongLong();
    CWindow window(static_cast<std::size_t>(cacheGB));
    window.resize(1600, 1000);
    window.show();
    QApplication::processEvents();

    auto* menu = window.findChild<MenuActionController*>();
    if (menu == nullptr) {
        std::cerr << "bench: no menu controller" << std::endl;
        return 2;
    }
    QString error;
    if (!menu->openVolpkgAt(project, /*interactive=*/false, &error)) {
        std::cerr << "bench: the project could not be opened: " << error.toStdString() << std::endl;
        return 2;
    }
    auto* map = window.findChild<FiberMapWorkspace*>();
    if (map == nullptr) {
        std::cerr << "bench: no fiber map workspace" << std::endl;
        return 2;
    }
    // The fiber map tab, so the workspace is shown and its docks exist.
    for (auto* tabs : window.findChildren<QTabWidget*>()) {
        for (int index = 0; index < tabs->count(); ++index) {
            if (tabs->widget(index)->property("workspaceId").toString() == QLatin1String("fiber-map")) {
                tabs->setCurrentIndex(index);
            }
        }
    }
    // Let the project settle: the stores attach and the catalog lists.
    waitFor([] { return false; }, 3000);

    auto* update = buttonNamed(map, QStringLiteral("Update"));
    if (update == nullptr) {
        std::cerr << "bench: no Update button" << std::endl;
        return 2;
    }
    if (followSolve) {
        auto* solves = map->findChild<QTreeWidget*>(QStringLiteral("fiberMapSolvesTree"));
        auto* follow = map->findChild<QCheckBox*>(QStringLiteral("fiberMapFollowSolve"));
        if (solves == nullptr || follow == nullptr || solves->topLevelItemCount() == 0) {
            std::cerr << "bench: no saved solve to follow" << std::endl;
            return 2;
        }
        solves->setCurrentItem(solves->topLevelItem(0));
        QApplication::processEvents();
        follow->setChecked(true);
        // Following a solve asks for a rebuild of its own; let it finish.
        waitFor([update] { return update->isEnabled(); }, 600000);
    }

    // BENCH_DUMP_APPROVAL="<text>;<text>": for every saved solve, print the
    // approval list's rows whose columns contain every given text - what the
    // dock shows, for checking it against the store.
    if (const auto wanted = qEnvironmentVariable("BENCH_DUMP_APPROVAL"); !wanted.isEmpty()) {
        auto* solves = map->findChild<QTreeWidget*>(QStringLiteral("fiberMapSolvesTree"));
        auto* panel = map->findChild<vc3d::fiber_map::ScaffoldApprovalPanel*>();
        auto* list = panel ? panel->findChild<QTreeWidget*>() : nullptr;
        if (solves == nullptr || list == nullptr) {
            std::cerr << "bench: no solves or approval list" << std::endl;
            return 2;
        }
        const auto needles = wanted.split(QLatin1Char(';'), Qt::SkipEmptyParts);
        for (int run = 0; run < solves->topLevelItemCount(); ++run) {
            solves->setCurrentItem(solves->topLevelItem(run));
            waitFor([] { return false; }, 1500);
            std::cout << "approval list of solve " << run << " (" << solves->topLevelItem(run)->text(0).toStdString()
                      << "): " << list->topLevelItemCount() << " top-level rows" << std::endl;
            const std::function<void(QTreeWidgetItem*, int)> walk = [&](QTreeWidgetItem* item, int depth) {
                QStringList columns;
                for (int column = 0; column < list->columnCount(); ++column) columns << item->text(column);
                const auto line = columns.join(QStringLiteral(" | "));
                bool all = true;
                for (const auto& needle : needles) all = all && line.contains(needle);
                if (all) {
                    std::cout << "  " << (depth ? "    child: " : "row: ") << line.toStdString() << std::endl;
                    // What selecting the row asks the map to draw.
                    const auto highlight = panel->highlightOf(item);
                    for (const auto& span : highlight.spans)
                        std::cout << "        span " << span.fiberUuid.substr(0, 8) << " controls " << span.controlFrom
                                  << "-" << span.controlTo << " positions " << span.positionFrom << "-"
                                  << span.positionTo << std::endl;
                    for (const auto& error : highlight.errors) std::cout << "        ERROR " << error << std::endl;
                }
                for (int child = 0; child < item->childCount(); ++child) walk(item->child(child), depth + 1);
            };
            for (int row = 0; row < list->topLevelItemCount(); ++row) walk(list->topLevelItem(row), 0);
            // The normals a stretch of this solve would be opened with: read
            // through the run's own reference, for every fiber the list names.
            if (auto* controller = map->findChild<vc3d::fiber_map::ScaffoldSolveController*>()) {
                std::set<std::string> fibers;
                const std::function<void(QTreeWidgetItem*)> collect = [&](QTreeWidgetItem* item) {
                    for (const auto& span : panel->highlightOf(item).spans) fibers.insert(span.fiberUuid);
                    for (int child = 0; child < item->childCount(); ++child) collect(item->child(child));
                };
                for (int row = 0; row < list->topLevelItemCount(); ++row) collect(list->topLevelItem(row));
                std::size_t resolved = 0;
                QString firstError;
                for (const auto& uuid : fibers) {
                    QString why;
                    if (controller->runFiberNormals(uuid, &why)) ++resolved;
                    else if (firstError.isEmpty()) firstError = why;
                }
                std::cout << "  solve normals resolved for " << resolved << " of " << fibers.size() << " fibers"
                          << (firstError.isEmpty() ? std::string() : "; first error: " + firstError.toStdString())
                          << std::endl;
            }
        }
        return 0;
    }

    // `--solve geometry|fused`: time a measured scaffold solve instead, as
    // the Solve button starts it, reporting each stage as it is entered.
    if (!solveNormals.isEmpty()) {
        auto* controller = map->findChild<vc3d::fiber_map::ScaffoldSolveController*>();
        auto* panel = map->findChild<vc3d::fiber_map::ScaffoldSolverPanel*>();
        if (controller == nullptr || panel == nullptr) {
            std::cerr << "bench: no solve controller" << std::endl;
            return 2;
        }
        QObject::connect(controller, &vc3d::fiber_map::ScaffoldSolveController::statusMessage, controller,
                         [](const QString& message) {
                             std::cout << "bench solve status: " << message.toStdString() << std::endl;
                         });
        for (int run = 1; run <= runs; ++run) {
            auto request = panel->request();
            request.measureCrossings = true;
            request.normalSource = solveNormals.toStdString();
            QElapsedTimer timer, stageTimer;
            timer.start();
            controller->startSolve(request);
            std::cout << "bench solve: startSolve returned after " << timer.elapsed() << " ms" << std::endl;
            std::string stage;
            stageTimer.start();
            const bool finished = waitFor([&] {
                const auto progress = controller->progress();
                if (progress.stage != stage) {
                    if (!stage.empty())
                        std::cout << "bench solve stage: " << stage << " " << stageTimer.elapsed() << " ms" << std::endl;
                    stage = progress.stage;
                    stageTimer.restart();
                }
                return !controller->busy();
            }, 1800000);
            std::cout << "bench solve run " << run << " of " << runs << " (" << solveNormals.toStdString()
                      << " normals): " << (finished ? "" : "DID NOT FINISH, last stage " + stage + ", ")
                      << timer.elapsed() << " ms wall; " << controller->lastError().toStdString() << std::endl;
            if (!finished) return 3;
        }
        return 0;
    }

    for (int run = 1; run <= runs; ++run) {
        QElapsedTimer timer;
        timer.start();
        update->click();
        // The button is disabled while a rebuild is in flight.
        if (!waitFor([update] { return !update->isEnabled(); }, 5000)) {
            std::cerr << "bench: the rebuild never started" << std::endl;
            return 3;
        }
        if (!waitFor([update] { return update->isEnabled(); }, 1800000)) {
            std::cerr << "bench: the rebuild did not finish within 30 minutes" << std::endl;
            return 3;
        }
        std::cout << "bench run " << run << " of " << runs << ": "
                  << (followSolve ? "following a solve" : "the map's own evidence") << ", "
                  << timer.elapsed() << " ms wall" << std::endl;
    }
    return 0;
}
