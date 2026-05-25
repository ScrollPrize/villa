#include <QApplication>
#include <QJsonArray>
#include <QJsonObject>
#include <QPushButton>
#include <QStandardItemModel>
#include <QStringList>

#define private public
#include "LasagnaBatchWindow.hpp"
#undef private

#include <memory>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

void ensureApplication(int& argc, char** argv, std::unique_ptr<QApplication>& app)
{
    if (!QApplication::instance()) {
        app = std::make_unique<QApplication>(argc, argv);
    }
}

QJsonObject job(const QString& id,
                const QString& state,
                int queuePosition,
                const QString& source,
                const QString& config,
                const QString& outputName = {},
                const QString& outputDir = {},
                double progress = 0.0,
                double submittedAt = 0.0,
                const QString& error = {})
{
    QJsonObject object;
    object[QStringLiteral("job_id")] = id;
    object[QStringLiteral("state")] = state;
    object[QStringLiteral("queue_position")] = queuePosition;
    object[QStringLiteral("source")] = source;
    object[QStringLiteral("config_name")] = config;
    object[QStringLiteral("output_name")] = outputName;
    object[QStringLiteral("output_dir")] = outputDir;
    object[QStringLiteral("overall_progress")] = progress;
    object[QStringLiteral("submitted_at")] = submittedAt;
    object[QStringLiteral("error")] = error;
    return object;
}

QString cell(const LasagnaBatchWindow& window, int row, int col)
{
    const auto* item = window._model->item(row, col);
    return item ? item->text() : QString();
}

QStringList rowTexts(const LasagnaBatchWindow& window, int row)
{
    QStringList values;
    for (int col = 0; col < window._model->columnCount(); ++col) {
        values << cell(window, row, col);
    }
    return values;
}

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    std::unique_ptr<QApplication> app;
    ensureApplication(argc, argv, app);

    LasagnaBatchWindow window;

    QStringList headers;
    for (int col = 0; col < window._model->columnCount(); ++col) {
        headers << window._model->headerData(col, Qt::Horizontal).toString();
    }
    require(headers.contains(QStringLiteral("Output")),
            "Lasagna batch table must expose the Output column");

    const QJsonArray jobs{
        job(QStringLiteral("waiting-1"), QStringLiteral("waiting"), 1,
            QStringLiteral("local"), QStringLiteral("cfg-a"),
            QStringLiteral("sheet_v001.tifxyz"), QStringLiteral("/tmp/ignored"),
            0.0, 1.0),
        job(QStringLiteral("running-1"), QStringLiteral("running"), 0,
            QStringLiteral("remote"), QStringLiteral("cfg-b"),
            {}, QStringLiteral("/tmp/from-dir.tifxyz"),
            0.426, 2.0),
        job(QStringLiteral("error-1"), QStringLiteral("error"), 0,
            QStringLiteral("local"), QStringLiteral("cfg-c"),
            {}, {}, 0.0, 0.0, QStringLiteral("boom")),
    };
    window.applyJobs(jobs);

    require(window._model->rowCount() == 3, "Unexpected rendered job count");
    require(rowTexts(window, 0).join('|') ==
                QStringLiteral("1|local|cfg-a|sheet_v001.tifxyz|Waiting #1|1970-01-01T01:00:01"),
            "Waiting row text did not match expected order/source/config/output/progress/submitted values");
    require(cell(window, 1, 3) == QStringLiteral("from-dir.tifxyz"),
            "output_dir basename should be shown when output_name is missing");
    require(cell(window, 1, 4) == QStringLiteral("Running 42.6%"),
            "Running progress text did not include one decimal percent");
    require(cell(window, 2, 4) == QStringLiteral("Error: boom"),
            "Error row did not include service error text");

    window.optimisticallyCancel(QStringLiteral("running-1"));
    require(cell(window, 1, 4) == QStringLiteral("Cancelled"),
            "Optimistic cancel should render cancelled state immediately");
    require(!window._cancelButton->isEnabled(),
            "Cancelled row should no longer be cancellable after optimistic cancel");

    const QJsonArray reorderJobs{
        job(QStringLiteral("w1"), QStringLiteral("waiting"), 1, QStringLiteral("s1"), QStringLiteral("cfg")),
        job(QStringLiteral("run"), QStringLiteral("running"), 0, QStringLiteral("sr"), QStringLiteral("cfg")),
        job(QStringLiteral("w2"), QStringLiteral("waiting"), 2, QStringLiteral("s2"), QStringLiteral("cfg")),
        job(QStringLiteral("w3"), QStringLiteral("waiting"), 3, QStringLiteral("s3"), QStringLiteral("cfg")),
    };
    window.applyJobs(reorderJobs);
    window.optimisticallyMoveBefore(QStringLiteral("w3"), QStringLiteral("w1"));
    require(window.jobAtRow(0)[QStringLiteral("job_id")].toString() == QStringLiteral("w3"),
            "Optimistic move-up should move a waiting row before the requested waiting row");
    require(window.jobAtRow(1)[QStringLiteral("job_id")].toString() == QStringLiteral("w1") &&
                window.jobAtRow(3)[QStringLiteral("job_id")].toString() == QStringLiteral("w2"),
            "Non-moving waiting row should stay stable across optimistic move-up");
    require(window.jobAtRow(2)[QStringLiteral("job_id")].toString() == QStringLiteral("run"),
            "Non-waiting rows should keep their relative order when waiting rows move around them");

    window.optimisticallyMoveBefore(QStringLiteral("w3"), QString());
    require(window.jobAtRow(3)[QStringLiteral("job_id")].toString() == QStringLiteral("w3"),
            "Optimistic move-down-to-end should append the waiting row");
    require(cell(window, 0, 0) == QStringLiteral("1") &&
                cell(window, 2, 0) == QStringLiteral("2") &&
                cell(window, 3, 0) == QStringLiteral("3"),
            "Waiting queue positions should be renumbered after optimistic reorder");

    return 0;
}
