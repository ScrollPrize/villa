#include <QtTest/QtTest>

#include "SegmentFiles.hpp"

#include <QTemporaryDir>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
namespace sf = vc3d::segment_files;

namespace {

void writeFile(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

}  // namespace

class SegmentFilesTest : public QObject
{
    Q_OBJECT

private slots:
    void backupsDirectoryFollowsRoot();
    void listBackupsKeepsNumberedSlots();
};

void SegmentFilesTest::backupsDirectoryFollowsRoot()
{
    QCOMPARE(sf::backupsDirectory("/p/paths/seg", "/p"), fs::path("/p/backups/seg"));
    QCOMPARE(sf::backupsDirectory("/p/paths/seg", {}), fs::path("/p/paths/backups/seg"));
}

void SegmentFilesTest::listBackupsKeepsNumberedSlots()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path dir = fs::path(root.path().toStdString()) / "backups" / "seg";
    writeFile(dir / "3" / "meta.json", R"({"backup_timestamp": "2026-10-18 10:00"})");
    writeFile(dir / "0" / "x.tif", "x");
    writeFile(dir / "11" / "x.tif", "x");
    writeFile(dir / "notes" / "x.tif", "x");
    writeFile(dir / "5", "a file, not a slot");

    const auto backups = sf::listBackups(dir);
    QCOMPARE(int(backups.size()), 2);
    QCOMPARE(backups[0].index, 0);
    QVERIFY(backups[0].timestamp.isEmpty());
    QCOMPARE(backups[1].index, 3);
    QCOMPARE(backups[1].timestamp, QStringLiteral("2026-10-18 10:00"));
    QCOMPARE(backups[1].path, dir / "3");

    QVERIFY(sf::listBackups(dir / "missing").empty());
}

QTEST_GUILESS_MAIN(SegmentFilesTest)
#include "test_segment_files.moc"
