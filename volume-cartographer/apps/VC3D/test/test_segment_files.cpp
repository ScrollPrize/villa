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

std::string readFile(const fs::path& path)
{
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Names in `dir`, for checking that no staging or aside folder is left.
QStringList entries(const fs::path& dir)
{
    QStringList names;
    for (const auto& entry : fs::directory_iterator(dir)) {
        names << QString::fromStdString(entry.path().filename().string());
    }
    names.sort();
    return names;
}

}  // namespace

class SegmentFilesTest : public QObject
{
    Q_OBJECT

private slots:
    void backupsDirectoryFollowsRoot();
    void listBackupsKeepsNumberedSlots();
    void restoreReplacesEveryBackedUpFile();
    void restoreRefusalChangesNothing();
    void moveRenamesIntoPlace();
    void moveReplacesOnlyWhenAsked();
    void failedMoveKeepsReplacedSegment();
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

void SegmentFilesTest::restoreReplacesEveryBackedUpFile()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path base(root.path().toStdString());
    const fs::path seg = base / "paths" / "seg";
    const fs::path slot = base / "backups" / "seg" / "2";
    for (const char* name : {"x.tif", "y.tif", "z.tif", "mask.tif", "meta.json"}) {
        writeFile(seg / name, std::string("current ") + name);
        writeFile(slot / name, std::string("backup ") + name);
    }
    writeFile(seg / "approval.tif", "kept");
    writeFile(seg / "layers" / "00.tif", "render");

    bool ranBeforeReplace = false;
    QString error;
    const int restored = sf::restoreBackupFiles(
        slot, seg,
        [&](QString*) {
            ranBeforeReplace = true;
            // Nothing is replaced before this runs.
            return readFile(seg / "x.tif") == "current x.tif";
        },
        &error);
    QCOMPARE(restored, 5);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QVERIFY(ranBeforeReplace);
    for (const char* name : {"x.tif", "y.tif", "z.tif", "mask.tif", "meta.json"}) {
        QCOMPARE(readFile(seg / name), std::string("backup ") + name);
    }
    QCOMPARE(readFile(seg / "approval.tif"), std::string("kept"));
    QCOMPARE(readFile(seg / "layers" / "00.tif"), std::string("render"));
    QCOMPARE(entries(base / "paths"), QStringList{QStringLiteral("seg")});
    // The slot itself is left as it was.
    QCOMPARE(readFile(slot / "x.tif"), std::string("backup x.tif"));
}

void SegmentFilesTest::restoreRefusalChangesNothing()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path base(root.path().toStdString());
    const fs::path seg = base / "paths" / "seg";
    const fs::path slot = base / "backups" / "seg" / "0";
    writeFile(seg / "x.tif", "current");
    writeFile(slot / "x.tif", "backup");

    QString error;
    QCOMPARE(sf::restoreBackupFiles(slot, seg,
                                    [](QString* e) {
                                        *e = QStringLiteral("no snapshot");
                                        return false;
                                    },
                                    &error),
             -1);
    QCOMPARE(error, QStringLiteral("no snapshot"));
    QCOMPARE(readFile(seg / "x.tif"), std::string("current"));
    QCOMPARE(entries(base / "paths"), QStringList{QStringLiteral("seg")});

    QCOMPARE(sf::restoreBackupFiles(base / "backups" / "seg" / "7", seg, {}, &error), -1);
    QVERIFY(error.contains(QStringLiteral("does not exist")));
    fs::create_directories(base / "backups" / "seg" / "5");
    QCOMPARE(sf::restoreBackupFiles(base / "backups" / "seg" / "5", seg, {}, &error), 0);
}

void SegmentFilesTest::moveRenamesIntoPlace()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path base(root.path().toStdString());
    writeFile(base / "traces" / "seg" / "x.tif", "trace");
    fs::create_directories(base / "paths");

    QString error;
    QVERIFY(sf::moveSegmentDirectory(base / "traces" / "seg", base / "paths" / "seg",
                                     false, &error));
    QVERIFY(!fs::exists(base / "traces" / "seg"));
    QCOMPARE(readFile(base / "paths" / "seg" / "x.tif"), std::string("trace"));
}

void SegmentFilesTest::moveReplacesOnlyWhenAsked()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path base(root.path().toStdString());
    writeFile(base / "traces" / "seg" / "x.tif", "trace");
    writeFile(base / "paths" / "seg" / "x.tif", "path");

    QString error;
    QVERIFY(!sf::moveSegmentDirectory(base / "traces" / "seg", base / "paths" / "seg",
                                      false, &error));
    QVERIFY(error.contains(QStringLiteral("already exists")));
    QCOMPARE(readFile(base / "traces" / "seg" / "x.tif"), std::string("trace"));
    QCOMPARE(readFile(base / "paths" / "seg" / "x.tif"), std::string("path"));

    QVERIFY(sf::moveSegmentDirectory(base / "traces" / "seg", base / "paths" / "seg",
                                     true, &error));
    QCOMPARE(readFile(base / "paths" / "seg" / "x.tif"), std::string("trace"));
    QCOMPARE(entries(base / "paths"), QStringList{QStringLiteral("seg")});
}

void SegmentFilesTest::failedMoveKeepsReplacedSegment()
{
    QTemporaryDir root;
    QVERIFY(root.isValid());
    const fs::path base(root.path().toStdString());
    writeFile(base / "paths" / "seg" / "x.tif", "path");

    // A missing source fails before the destination is touched.
    QString error;
    QVERIFY(!sf::moveSegmentDirectory(base / "traces" / "seg", base / "paths" / "seg",
                                      true, &error));
    QCOMPARE(readFile(base / "paths" / "seg" / "x.tif"), std::string("path"));

    // A rename that fails after the destination moved aside puts it back: a
    // source inside the destination cannot be renamed over it.
    writeFile(base / "paths" / "seg" / "inner" / "x.tif", "inner");
    QVERIFY(!sf::moveSegmentDirectory(base / "paths" / "seg" / "inner",
                                      base / "paths" / "seg", true, &error));
    QCOMPARE(readFile(base / "paths" / "seg" / "x.tif"), std::string("path"));
    QCOMPARE(entries(base / "paths"), QStringList{QStringLiteral("seg")});
}

QTEST_GUILESS_MAIN(SegmentFilesTest)
#include "test_segment_files.moc"
