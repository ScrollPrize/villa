#include "UnifiedBrowserDialog.hpp"

#include <QFile>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QtTest/QtTest>

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
    void typedRemoteFileOpen_data()
    {
        QTest::addColumn<QString>("uri");
        QTest::newRow("s3") << QStringLiteral("s3://bucket/path/data.lasagna.json");
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
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file://") + second);
    }

    void typedRemoteDirectoryAndDualMode()
    {
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
        QCOMPARE(dialog.selectedUri(), QStringLiteral("file://") + temporary.path() + QStringLiteral("/"));
    }

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
