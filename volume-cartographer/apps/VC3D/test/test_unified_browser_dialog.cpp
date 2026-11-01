#include "UnifiedBrowserDialog.hpp"
#include "support/LoopbackHttpServer.hpp"
#include <QRegularExpression>

#include <QDir>
#include <QCheckBox>
#include <QFile>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QUrl>
#include <QtTest/QtTest>

#ifdef Q_OS_WIN
#include <qt_windows.h>
#endif

namespace
{

QLineEdit* pathBar(UnifiedBrowserDialog& dialog)
{
    auto* edit = dialog.findChild<QLineEdit*>();
    Q_ASSERT(edit);
    return edit;
}

QPushButton* openButton(UnifiedBrowserDialog& dialog)
{
    const auto buttons = dialog.findChildren<QPushButton*>();
    for (auto* button : buttons) {
        if (button->text() == QStringLiteral("Open"))
            return button;
    }
    return nullptr;
}

void typePath(UnifiedBrowserDialog& dialog, const QString& path)
{
    auto* edit = pathBar(dialog);
    edit->setFocus();
    edit->selectAll();
    QTest::keyClicks(edit, path);
}

void clickOpen(UnifiedBrowserDialog& dialog)
{
    auto* button = openButton(dialog);
    QVERIFY(button);
    QTest::mouseClick(button, Qt::LeftButton);
}

void configureRemoteDialog(UnifiedBrowserDialog& dialog, bool files, bool dirs)
{
    dialog.setStartUri(QStringLiteral("s3://"));
    dialog.setAcceptsFiles(files);
    dialog.setAcceptsDirs(dirs);
}

}  // namespace

class UnifiedBrowserDialogTest : public QObject
{
    Q_OBJECT

private slots:
    void hiddenFilesToggle()
    {
        QTemporaryDir temporary(QDir::current().filePath("browser-hidden-XXXXXX"));
        QVERIFY(temporary.isValid());
#ifdef Q_OS_WIN
        const auto hiddenFile = temporary.filePath("hidden.volpkg.json");
        const auto hiddenDir = temporary.filePath("hidden-dir");
#else
        const auto hiddenFile = temporary.filePath(".hidden.volpkg.json");
        const auto hiddenDir = temporary.filePath(".hidden-dir");
#endif
        QFile hidden(hiddenFile);
        QVERIFY(hidden.open(QIODevice::WriteOnly));
        hidden.close();
        QVERIFY(QDir().mkdir(hiddenDir));
#ifdef Q_OS_WIN
        for (const auto& path : {hiddenFile, hiddenDir}) {
            const auto native = QDir::toNativeSeparators(path).toStdWString();
            const DWORD attributes = GetFileAttributesW(native.c_str());
            QVERIFY(attributes != INVALID_FILE_ATTRIBUTES);
            QVERIFY(SetFileAttributesW(native.c_str(), attributes | FILE_ATTRIBUTE_HIDDEN));
        }
#endif
        QVERIFY(QFileInfo(hiddenFile).isHidden());
        QVERIFY(QFileInfo(hiddenDir).isHidden());
        QFile visible(temporary.filePath("visible.volpkg.json"));
        QVERIFY(visible.open(QIODevice::WriteOnly));
        visible.close();
        UnifiedBrowserDialog dialog;
        dialog.setStartUri(temporary.path());
        auto* list = dialog.findChild<QListWidget*>();
        QCOMPARE(list->count(), 1);
        QCOMPARE(list->item(0)->text(), QStringLiteral("visible.volpkg.json"));
        auto* toggle = dialog.findChild<QCheckBox*>();
        QVERIFY(toggle);
        toggle->setChecked(true);
        QCOMPARE(list->count(), 3);
        toggle->setChecked(false);
        QCOMPARE(list->count(), 1);
        QCOMPARE(list->item(0)->text(), QStringLiteral("visible.volpkg.json"));
#ifdef Q_OS_WIN
        QFile dotted(temporary.filePath(".ordinary.volpkg.json"));
        QVERIFY(dotted.open(QIODevice::WriteOnly));
        dotted.close();
        QVERIFY(!QFileInfo(dotted).isHidden());
        dialog.setStartUri(temporary.path());
        QCOMPARE(list->count(), 2);
        QCOMPARE(list->findItems(".ordinary.volpkg.json", Qt::MatchExactly).size(), 1);
#endif
    }

    void directoryEnterOnlyNavigates()
    {
        QTemporaryDir temporary(QDir::current().filePath("browser-enter-XXXXXX"));
        QVERIFY(temporary.isValid());
        for (const auto suffix : {QString{}, QStringLiteral("/")}) {
            UnifiedBrowserDialog dialog;
            dialog.setAcceptsFiles(true);
            dialog.setAcceptsDirs(true);
            dialog.show();
            QSignalSpy accepted(&dialog, &QDialog::accepted);
            typePath(dialog, temporary.path() + suffix);
            QTest::keyClick(pathBar(dialog), Qt::Key_Return);
            QCOMPARE(accepted.count(), 0);
            QCOMPARE(QDir::cleanPath(pathBar(dialog)->text()), temporary.path());
        }
    }

    void typedRemoteFileOpen_data()
    {
        QTest::addColumn<QString>("uri");
        QTest::newRow("s3") << QStringLiteral("s3://bucket/path/data.lasagna.json");
        QTest::newRow("sftp") << QStringLiteral("sftp://user@alias:2222/path/data.lasagna.json");
        QTest::newRow("ssh") << QStringLiteral("ssh://alias/path/data.lasagna.json");
        QTest::newRow("http") << QStringLiteral("http://example.com/path/data.lasagna.json");
        QTest::newRow("https") << QStringLiteral("https://example.com/path/data.lasagna.json?token=abc");
    }

    void typedRemoteFileOpen()
    {
        QFETCH(QString, uri);
        UnifiedBrowserDialog dialog;
        configureRemoteDialog(dialog, true, false);

        typePath(dialog, uri);
        clickOpen(dialog);

        QCOMPARE(dialog.result(), int(QDialog::Accepted));
        QCOMPARE(dialog.selectedUri(), uri);
    }

    void typedRemoteFileEnter()
    {
        const QString uri = QStringLiteral("s3://bucket/path/data.lasagna.json");
        UnifiedBrowserDialog dialog;
        configureRemoteDialog(dialog, true, false);

        typePath(dialog, uri);
        QTest::keyClick(pathBar(dialog), Qt::Key_Return);

        QCOMPARE(dialog.result(), int(QDialog::Accepted));
        QCOMPARE(dialog.selectedUri(), uri);
    }

    void typedPathOverridesStaleSelection()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString first = temporary.filePath(QStringLiteral("first.json"));
        const QString second = temporary.filePath(QStringLiteral("second.json"));
        for (const QString& path : {first, second}) {
            QFile file(path);
            QVERIFY(file.open(QIODevice::WriteOnly));
        }

        UnifiedBrowserDialog dialog;
        dialog.setStartUri(temporary.path());
        dialog.setAcceptsFiles(true);
        dialog.setAcceptsDirs(false);
        auto* list = dialog.findChild<QListWidget*>();
        QVERIFY(list);
        QCOMPARE(list->count(), 2);
        list->setCurrentRow(0);

        typePath(dialog, second);
        clickOpen(dialog);

        QCOMPARE(dialog.result(), int(QDialog::Accepted));
#ifdef Q_OS_WIN
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file:///") + second);
#else
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file://") + second);
#endif
    }

    void typedRemoteDirectoryAndDualMode()
    {
        for (const auto& uri : {
                 QStringLiteral("sftp://alias/data.zarr"),
                 QStringLiteral("https://example.com/data.zarr?token=a%2Fb"),
                 QStringLiteral("sftp://alias/data.zarr#vc-base-scale=2")}) {
            UnifiedBrowserDialog dialog;
            configureRemoteDialog(dialog, false, true);
            typePath(dialog, uri);
            clickOpen(dialog);
            QCOMPARE(dialog.result(), int(QDialog::Accepted));
            const auto suffix = uri.indexOf(QRegularExpression("[?#]"));
            auto expected = uri;
            expected.insert(suffix < 0 ? expected.size() : suffix, '/');
            QCOMPARE(dialog.selectedUri(), expected);
        }
        {
            UnifiedBrowserDialog dialog;
            configureRemoteDialog(dialog, false, true);
            typePath(dialog, QStringLiteral("s3://bucket/prefix"));
            clickOpen(dialog);
            QCOMPARE(dialog.result(), int(QDialog::Accepted));
            QCOMPARE(dialog.selectedUri(), QStringLiteral("s3://bucket/prefix/"));
        }
        {
            UnifiedBrowserDialog dialog;
            configureRemoteDialog(dialog, true, true);
            const QString uri = QStringLiteral("https://example.com/project.volpkg.json");
            typePath(dialog, uri);
            clickOpen(dialog);
            QCOMPARE(dialog.result(), int(QDialog::Accepted));
            QCOMPARE(dialog.selectedUri(), uri);
        }
    }

    void remoteListingThroughTheReader()
    {
        // A signed loopback bucket, reached with the environment's keys and endpoint; the
        // dialog lists one level at a time through vc_reader and keeps the location's URIs.
        QTemporaryDir temporary(QDir::current().filePath("browser-remote-XXXXXX"));
        QVERIFY(temporary.isValid());
        const QDir root(temporary.path());
        const auto write = [&](const QString& relative, const QByteArray& bytes) {
            QVERIFY(root.mkpath(QFileInfo(root.filePath(relative)).path()));
            QFile file(root.filePath(relative));
            QVERIFY(file.open(QIODevice::WriteOnly));
            file.write(bytes);
        };
        for (const char* plane : {"xy", "xz", "yz"})
            write(QStringLiteral("bucket/ng/%1/000000.grid").arg(plane), "g");
        write(QStringLiteral("bucket/ng/metadata.json"), "{}");
        vc::test::LoopbackHttpServer server;
        server.serveBucket("grids", root.filePath("bucket").toStdString(), true);
        qputenv("AWS_ACCESS_KEY_ID", "key");
        qputenv("AWS_SECRET_ACCESS_KEY", "secret");
        qputenv("AWS_REGION", "us-east-1");
        qputenv("AWS_ENDPOINT_URL", server.url().c_str());

        UnifiedBrowserDialog dialog;
        configureRemoteDialog(dialog, true, true);
        dialog.setStartUri(QStringLiteral("s3://grids/ng/"));
        auto* list = dialog.findChild<QListWidget*>();
        QVERIFY(list);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 4, 10000);
        QCOMPARE(list->item(0)->text(), QStringLiteral("xy/"));
        QCOMPARE(list->item(0)->data(Qt::UserRole).toString(), QStringLiteral("s3://grids/ng/xy/"));
        QVERIFY(list->item(0)->data(Qt::UserRole + 1).toBool());
        QCOMPARE(list->item(2)->text(), QStringLiteral("yz/"));
        QCOMPARE(list->item(3)->text(), QStringLiteral("metadata.json"));
        QCOMPARE(list->item(3)->data(Qt::UserRole).toString(), QStringLiteral("s3://grids/ng/metadata.json"));
        QVERIFY(!list->item(3)->data(Qt::UserRole + 1).toBool());
        for (const auto& request : server.requests())
            QVERIFY(request.signedV4());

        // Into a subdirectory from the path bar.
        typePath(dialog, QStringLiteral("s3://grids/ng/xz/"));
        QTest::keyClick(pathBar(dialog), Qt::Key_Return);
        QTRY_COMPARE_WITH_TIMEOUT(list->count(), 1, 10000);
        QCOMPARE(list->item(0)->text(), QStringLiteral("000000.grid"));
        QCOMPARE(list->item(0)->data(Qt::UserRole).toString(), QStringLiteral("s3://grids/ng/xz/000000.grid"));

        // A location with no listing says so instead of failing.
        typePath(dialog, QStringLiteral("http://127.0.0.1:1/plain/"));
        QTest::keyClick(pathBar(dialog), Qt::Key_Return);
        auto* status = dialog.findChild<QLabel*>(QStringLiteral("browserStatus"));
        if (status) QTRY_VERIFY_WITH_TIMEOUT(status->text().contains(QStringLiteral("Browsing is available")), 10000);
        for (const char* name : {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_ENDPOINT_URL"})
            qunsetenv(name);
    }

    void typedLocalDirectoryOpen()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        UnifiedBrowserDialog dialog;
        dialog.setAcceptsFiles(false);
        dialog.setAcceptsDirs(true);

        typePath(dialog, temporary.path());
        clickOpen(dialog);

        QCOMPARE(dialog.result(), int(QDialog::Accepted));
#ifdef Q_OS_WIN
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file:///") + temporary.path() + QStringLiteral("/"));
#else
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file://") + temporary.path() + QStringLiteral("/"));
#endif
    }

    void fileUriPreservesUncAuthority()
    {
        UnifiedBrowserDialog dialog;
        dialog.setStartUri(
            QStringLiteral("file://wsl.localhost/Ubuntu/home/user/segments"));

        QCOMPARE(pathBar(dialog)->text(),
                 QStringLiteral("//wsl.localhost/Ubuntu/home/user/segments"));
    }

#ifdef Q_OS_WIN
    void typedWindowsNativeSeparators()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());

        UnifiedBrowserDialog dialog;
        dialog.setAcceptsFiles(false);
        dialog.setAcceptsDirs(true);

        const QString nativePath = QDir::toNativeSeparators(temporary.path());
        typePath(dialog, nativePath);
        clickOpen(dialog);

        QCOMPARE(dialog.result(), int(QDialog::Accepted));
        QCOMPARE(dialog.selectedUri(),
                 QUrl::fromLocalFile(QDir::fromNativeSeparators(nativePath))
                         .toString() +
                     QStringLiteral("/"));
    }
#endif

    void rejectsHostlessRemoteUrls_data()
    {
        QTest::addColumn<QString>("uri");
        QTest::newRow("bare-s3") << QStringLiteral("s3://");
        QTest::newRow("hostless-s3") << QStringLiteral("s3:///data.lasagna.json");
        QTest::newRow("hostless-https") << QStringLiteral("https:///data.lasagna.json");
    }

    void rejectsHostlessRemoteUrls()
    {
        QFETCH(QString, uri);
        UnifiedBrowserDialog dialog;
        configureRemoteDialog(dialog, true, false);

        typePath(dialog, uri);
        clickOpen(dialog);

        QCOMPARE(dialog.result(), 0);
        QVERIFY(dialog.selectedUri().isEmpty());
    }
};

QTEST_MAIN(UnifiedBrowserDialogTest)
#include "test_unified_browser_dialog.moc"
