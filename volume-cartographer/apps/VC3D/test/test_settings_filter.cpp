#include "SettingsFilter.hpp"

#include "ui_VCSettings.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QtTest/QtTest>

namespace {

// A small settings form with the row shapes VCSettings.ui uses: grid rows of
// label, editor and help button, spanning check boxes, a form layout, and an
// OK/Cancel button box.
struct Form {
    QDialog dialog;
    QLabel* noMatch{nullptr};
    QLineEdit* filter{nullptr};
    QGroupBox* packages{nullptr};
    QGroupBox* cache{nullptr};
    QGroupBox* performance{nullptr};
    QLabel* pathLabel{nullptr};
    QCheckBox* autoOpen{nullptr};
    QLabel* ramLabel{nullptr};
    QPushButton* ramHelp{nullptr};
    QLabel* remoteLabel{nullptr};
    QLabel* maximumLabel{nullptr};
    QLabel* appHiddenLabel{nullptr};
    QCheckBox* appHiddenBox{nullptr};
    QLabel* interpolationLabel{nullptr};

    Form()
    {
        auto* main = new QVBoxLayout(&dialog);

        packages = new QGroupBox(QStringLiteral("Volume Packages"), &dialog);
        auto* packagesGrid = new QGridLayout(packages);
        pathLabel = new QLabel(QStringLiteral("Default *.volpkg path"), packages);
        packagesGrid->addWidget(pathLabel, 0, 0);
        packagesGrid->addWidget(new QLineEdit(packages), 0, 1);
        autoOpen = new QCheckBox(QStringLiteral("Auto open last used *.volpkg upon app start"), packages);
        packagesGrid->addWidget(autoOpen, 1, 0, 1, 2);
        main->addWidget(packages);

        cache = new QGroupBox(QStringLiteral("Cache"), &dialog);
        auto* cacheGrid = new QGridLayout(cache);
        ramLabel = new QLabel(QStringLiteral("RAM cache size (GB)"), cache);
        auto* ramSpin = new QSpinBox(cache);
        ramSpin->setSuffix(QStringLiteral(" GB"));
        ramHelp = new QPushButton(cache);
        ramHelp->setToolTip(QStringLiteral(
            "<html><head/><body><p>Total RAM budget for the in-memory chunk cache.</p></body></html>"));
        cacheGrid->addWidget(ramLabel, 0, 0);
        cacheGrid->addWidget(ramSpin, 0, 1);
        cacheGrid->addWidget(ramHelp, 0, 2);
        remoteLabel = new QLabel(QStringLiteral("Remote cache directory"), cache);
        auto* remoteEdit = new QLineEdit(cache);
        remoteEdit->setPlaceholderText(QStringLiteral("~/.VC3D/remote_cache"));
        cacheGrid->addWidget(remoteLabel, 1, 0);
        cacheGrid->addWidget(remoteEdit, 1, 1);
        cacheGrid->addWidget(new QPushButton(QStringLiteral("Browse..."), cache), 1, 2);
        maximumLabel = new QLabel(QStringLiteral("Remote Zarr cache maximum (GiB)"), cache);
        auto* maximumSpin = new QSpinBox(cache);
        maximumSpin->setSpecialValueText(QStringLiteral("Unlimited"));
        cacheGrid->addWidget(maximumLabel, 2, 0);
        cacheGrid->addWidget(maximumSpin, 2, 1);
        appHiddenLabel = new QLabel(QStringLiteral("Cache mode the app hides"), cache);
        appHiddenBox = new QCheckBox(cache);
        cacheGrid->addWidget(appHiddenLabel, 3, 0);
        cacheGrid->addWidget(appHiddenBox, 3, 1);
        appHiddenLabel->hide();
        appHiddenBox->hide();
        main->addWidget(cache);

        performance = new QGroupBox(QStringLiteral("Performance"), &dialog);
        auto* performanceForm = new QFormLayout(performance);
        auto* interpolation = new QComboBox(performance);
        interpolation->addItems({QStringLiteral("Nearest"), QStringLiteral("Linear")});
        interpolationLabel = new QLabel(QStringLiteral("Interpolation"), performance);
        performanceForm->addRow(interpolationLabel, interpolation);
        main->addWidget(performance);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
        QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        main->addWidget(buttons);

        noMatch = new QLabel(QStringLiteral("No settings match the filter."), &dialog);
        filter = vc3d::settings_filter::createFilterField(&dialog, noMatch);
        main->insertWidget(0, filter);
        main->insertWidget(1, noMatch);
        dialog.show();
    }
};

} // namespace

class SettingsFilterTest : public QObject
{
    Q_OBJECT

private slots:
    void matchesLabelsAndHidesOtherGroups()
    {
        Form form;
        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("ram")), 1);
        QVERIFY(form.cache->isVisible());
        QVERIFY(form.ramLabel->isVisible());
        QVERIFY(form.ramHelp->isVisible());
        QVERIFY(!form.remoteLabel->isVisible());
        QVERIFY(!form.maximumLabel->isVisible());
        QVERIFY(!form.packages->isVisible());
        QVERIFY(!form.performance->isVisible());
    }

    void groupTitleKeepsEveryRowOfItsGroup()
    {
        Form form;
        // Three cache rows are visible; the fourth stays hidden by the app.
        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("CACHE")), 3);
        QVERIFY(form.ramLabel->isVisible());
        QVERIFY(form.remoteLabel->isVisible());
        QVERIFY(form.maximumLabel->isVisible());
        QVERIFY(form.appHiddenLabel->isHidden());
        QVERIFY(!form.packages->isVisible());
    }

    void matchesTooltipsPlaceholdersSpecialValuesAndItems()
    {
        Form form;
        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("in-memory budget")), 1);
        QVERIFY(form.ramLabel->isVisible());

        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("remote_cache")), 1);
        QVERIFY(form.remoteLabel->isVisible());
        QVERIFY(!form.ramLabel->isVisible());

        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("unlimited")), 1);
        QVERIFY(form.maximumLabel->isVisible());

        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("linear")), 1);
        QVERIFY(form.performance->isVisible());
        QVERIFY(form.interpolationLabel->isVisible());
        QVERIFY(!form.cache->isVisible());

        // Words may come from the group title and the row.
        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("packages auto")), 1);
        QVERIFY(form.autoOpen->isVisible());
        QVERIFY(!form.pathLabel->isVisible());
    }

    void clearingRestoresOnlyWhatTheFilterHid()
    {
        Form form;
        QCOMPARE(vc3d::settings_filter::apply(&form.dialog, QStringLiteral("zzz")), 0);
        QVERIFY(!form.packages->isVisible());
        QVERIFY(!form.cache->isVisible());
        QVERIFY(!form.performance->isVisible());

        vc3d::settings_filter::apply(&form.dialog, QString());
        QVERIFY(form.packages->isVisible());
        QVERIFY(form.cache->isVisible());
        QVERIFY(form.performance->isVisible());
        QVERIFY(form.pathLabel->isVisible());
        QVERIFY(form.ramLabel->isVisible());
        QVERIFY(form.appHiddenLabel->isHidden());
        QVERIFY(form.appHiddenBox->isHidden());
    }

    void filterFieldDrivesTheFormFromTheKeyboard()
    {
        Form form;
        QSignalSpy accepted(&form.dialog, &QDialog::accepted);
        QSignalSpy rejected(&form.dialog, &QDialog::rejected);
        QVERIFY(form.noMatch->isHidden());

        QTest::keyClicks(form.filter, QStringLiteral("zzz"));
        QVERIFY(form.noMatch->isVisible());
        QVERIFY(!form.cache->isVisible());

        // Escape clears a non-empty filter instead of closing the dialog.
        QTest::keyClick(form.filter, Qt::Key_Escape);
        QVERIFY(form.filter->text().isEmpty());
        QVERIFY(form.noMatch->isHidden());
        QVERIFY(form.cache->isVisible());
        QVERIFY(form.dialog.isVisible());

        // Enter moves to the first matching control and does not accept.
        QTest::keyClicks(form.filter, QStringLiteral("remote cache directory"));
        QTest::keyClick(form.filter, Qt::Key_Return);
        QCOMPARE(accepted.count(), 0);
        QVERIFY(form.dialog.isVisible());
        QWidget* focused = form.dialog.focusWidget();
        QVERIFY(focused);
        QVERIFY(focused != form.filter);
        QCOMPARE(focused->parentWidget(), form.cache);

        // With an empty filter Escape still closes the dialog.
        form.filter->setFocus();
        form.filter->clear();
        QTest::keyClick(form.filter, Qt::Key_Escape);
        QCOMPARE(rejected.count(), 1);
    }

    void emptyFilterLeavesEnterToTheDialog()
    {
        Form form;
        QSignalSpy accepted(&form.dialog, &QDialog::accepted);
        form.filter->setFocus();
        QTest::keyClick(form.filter, Qt::Key_Return);
        QCOMPARE(accepted.count(), 1);
    }

    void installFilterKeysCallsEnterOnlyWithText()
    {
        QDialog dialog;
        auto* layout = new QVBoxLayout(&dialog);
        auto* field = new QLineEdit(&dialog);
        layout->addWidget(field);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
        QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        layout->addWidget(buttons);
        int entered = 0;
        vc3d::settings_filter::installFilterKeys(field, [&entered]() { ++entered; });
        dialog.show();
        QSignalSpy rejected(&dialog, &QDialog::rejected);

        QTest::keyClicks(field, QStringLiteral("zoom"));
        QTest::keyClick(field, Qt::Key_Enter);
        QCOMPARE(entered, 1);
        QVERIFY(dialog.isVisible());
        QTest::keyClick(field, Qt::Key_Escape);
        QVERIFY(field->text().isEmpty());
        QCOMPARE(rejected.count(), 0);

        // The empty field passes Enter on to the dialog's default button.
        QTest::keyClick(field, Qt::Key_Return);
        QCOMPARE(entered, 1);
        QCOMPARE(rejected.count(), 1);
    }

    void filtersTheRealSettingsForm()
    {
        QDialog dialog;
        Ui_VCSettingsDlg ui;
        ui.setupUi(&dialog);
        dialog.show();

        QVERIFY(vc3d::settings_filter::apply(&dialog, QStringLiteral("zoom")) >= 1);
        QVERIFY(ui.spinZoomSensitivity->isVisible());
        QVERIFY(!ui.edtDefaultPathVolpkg->isVisible());

        QVERIFY(vc3d::settings_filter::apply(&dialog, QStringLiteral("ram cache")) >= 1);
        QVERIFY(ui.groupBox_5->isVisible());
        QVERIFY(ui.spinRamCacheSizeGB->isVisible());
        QVERIFY(!ui.groupBox_2->isVisible());
        QVERIFY(!ui.spinZoomSensitivity->isVisible());

        vc3d::settings_filter::apply(&dialog, QString());
        for (QGroupBox* group : {ui.groupBox_2, ui.groupBox_3, ui.groupBox_4, ui.groupBox_5}) {
            QVERIFY(group->isVisible());
        }
        QVERIFY(ui.edtDefaultPathVolpkg->isVisible());
        QVERIFY(ui.spinZoomSensitivity->isVisible());

        // As SettingsDialog places it: on top, with the focus. Enter in the
        // empty field still presses OK.
        QLineEdit* filter = vc3d::settings_filter::createFilterField(&dialog);
        ui.mainVBox->insertWidget(0, filter);
        filter->setFocus();
        QSignalSpy accepted(&dialog, &QDialog::accepted);
        QTest::keyClick(filter, Qt::Key_Return);
        QCOMPARE(accepted.count(), 1);
    }
};

QTEST_MAIN(SettingsFilterTest)
#include "test_settings_filter.moc"
