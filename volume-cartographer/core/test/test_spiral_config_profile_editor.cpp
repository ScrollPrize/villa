#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QScrollArea>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTest>
#include <QToolButton>

#define private public
#include "elements/SpiralConfigProfileEditor.hpp"
#undef private

#include "elements/CollapsibleSettingsGroup.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

QJsonObject field(const QString& label,
                  const QString& type,
                  const QString& impact)
{
    return {
        {QStringLiteral("label"), label},
        {QStringLiteral("type"), type},
        {QStringLiteral("runtime_impact"), impact},
    };
}

QJsonObject syntheticCatalog()
{
    QJsonObject fields;
    QJsonObject defaults;

    QJsonObject optimizer = field(
        QStringLiteral("Learning Rate"), QStringLiteral("number"),
        QStringLiteral("run_boundary"));
    optimizer[QStringLiteral("minimum")] = 0.0;
    optimizer[QStringLiteral("maximum")] = 10.0;
    optimizer[QStringLiteral("precision")] = 3;
    optimizer[QStringLiteral("step")] = 0.1;
    fields[QStringLiteral("optimizer_learning_rate")] = optimizer;
    defaults[QStringLiteral("optimizer_learning_rate")] = 0.5;

    QJsonObject model = field(
        QStringLiteral("Iterations"), QStringLiteral("integer"),
        QStringLiteral("new_fit"));
    model[QStringLiteral("minimum")] = 1;
    model[QStringLiteral("maximum")] = 1000;
    fields[QStringLiteral("model_iterations")] = model;
    defaults[QStringLiteral("model_iterations")] = 25;

    fields[QStringLiteral("patch_enabled")] = field(
        QStringLiteral("Use Patches"), QStringLiteral("boolean"),
        QStringLiteral("run_boundary"));
    defaults[QStringLiteral("patch_enabled")] = true;

    QJsonObject sample = field(
        QStringLiteral("Sampling Mode"), QStringLiteral("enum"),
        QStringLiteral("prepared_input_rebuild"));
    sample[QStringLiteral("values")] =
        QJsonArray{QStringLiteral("fast"), QStringLiteral("accurate")};
    fields[QStringLiteral("sample_count_mode")] = sample;
    defaults[QStringLiteral("sample_count_mode")] = QStringLiteral("fast");

    QJsonObject input = field(
        QStringLiteral("Input Channels"), QStringLiteral("array"),
        QStringLiteral("shell_reload"));
    input[QStringLiteral("nullable")] = true;
    fields[QStringLiteral("input_channels")] = input;
    defaults[QStringLiteral("input_channels")] = QJsonValue::Null;

    const QList<QPair<QString, QString>> remaining{
        {QStringLiteral("pcl_radius"), QStringLiteral("PCL Radius")},
        {QStringLiteral("tracks_count"), QStringLiteral("Track Count")},
        {QStringLiteral("dense_weight"), QStringLiteral("Dense Weight")},
        {QStringLiteral("loss_scale"), QStringLiteral("Loss Scale")},
        {QStringLiteral("dt_limit"), QStringLiteral("DT Limit")},
        {QStringLiteral("output_stride"), QStringLiteral("Output Stride")},
        {QStringLiteral("shell_width"), QStringLiteral("Shell Width")},
        {QStringLiteral("influence_radius"), QStringLiteral("Influence Radius")},
    };
    for (const auto& item : remaining) {
        QJsonObject spec = field(
            item.second, QStringLiteral("integer"),
            QStringLiteral("run_boundary"));
        spec[QStringLiteral("minimum")] = 0;
        spec[QStringLiteral("maximum")] = 100;
        fields[item.first] = spec;
        defaults[item.first] = 1;
    }

    return {
        {QStringLiteral("defaults"), defaults},
        {QStringLiteral("schema"), QJsonObject{
             {QStringLiteral("fields"), fields},
         }},
    };
}

void processLayout()
{
    QApplication::sendPostedEvents();
    QApplication::processEvents();
    QTest::qWait(20);
}

} // namespace

int main(int argc, char** argv)
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "offscreen");

    QTemporaryDir configDir;
    require(configDir.isValid(), "Failed to create temporary config directory");
    qputenv("VC3D_CONFIG_DIR", configDir.path().toUtf8());

    std::unique_ptr<QApplication> app;
    if (!QApplication::instance())
        app = std::make_unique<QApplication>(argc, argv);

    SpiralConfigProfileEditor editor;
    editor.setCatalog(syntheticCatalog());

    const QStringList expectedPrefixes{
        QStringLiteral("optimizer"), QStringLiteral("model"),
        QStringLiteral("patch"), QStringLiteral("sample"),
        QStringLiteral("input"), QStringLiteral("pcl"),
        QStringLiteral("tracks"), QStringLiteral("dense"),
        QStringLiteral("loss"), QStringLiteral("dt"),
        QStringLiteral("output"), QStringLiteral("shell"),
        QStringLiteral("influence"),
    };
    const QStringList expectedTitles{
        QStringLiteral("Optimizer"), QStringLiteral("Model"),
        QStringLiteral("Patch"), QStringLiteral("Sample Count"),
        QStringLiteral("Input"), QStringLiteral("PCL"),
        QStringLiteral("Tracks"), QStringLiteral("Dense"),
        QStringLiteral("Loss"), QStringLiteral("DT"),
        QStringLiteral("Output"), QStringLiteral("Shell"),
        QStringLiteral("Influence"),
    };
    require(editor._controlGroups.size() == expectedPrefixes.size(),
            "Catalog fields were not split into the expected groups");
    for (int index = 0; index < expectedPrefixes.size(); ++index) {
        const auto& group = editor._controlGroups[index];
        require(group.prefix == expectedPrefixes[index],
                "Groups are not in stable prefix order");
        require(group.widget->isExpanded(),
                "Configuration groups should initially be expanded");
        auto* button = group.widget->findChild<QToolButton*>();
        require(button && button->text() == expectedTitles[index],
                "A configuration group has the wrong human-friendly title");
    }

    require(qobject_cast<QDoubleSpinBox*>(
                editor._fieldEditors.value(QStringLiteral("optimizer_learning_rate"))),
            "Number fields should use QDoubleSpinBox");
    require(qobject_cast<QSpinBox*>(
                editor._fieldEditors.value(QStringLiteral("model_iterations"))),
            "Integer fields should use QSpinBox");
    require(qobject_cast<QCheckBox*>(
                editor._fieldEditors.value(QStringLiteral("patch_enabled"))),
            "Boolean fields should use QCheckBox");
    require(qobject_cast<QComboBox*>(
                editor._fieldEditors.value(QStringLiteral("sample_count_mode"))),
            "Enum fields should use QComboBox");
    require(qobject_cast<QLineEdit*>(
                editor._fieldEditors.value(QStringLiteral("input_channels"))),
            "Array fields should use QLineEdit");
    auto* impact = editor.findChild<QLabel*>(
        QStringLiteral("spiralConfigImpact_optimizer_learning_rate"));
    require(impact && impact->text() == QStringLiteral("Next Run"),
            "Runtime impact should use its compact display label");

    const QString beforeCollapse = editor.currentText();
    auto* optimizerGroup = editor._controlGroups.front().widget;
    optimizerGroup->setExpanded(false);
    require(!optimizerGroup->isExpanded(),
            "Configuration groups should be collapsible");
    require(editor.currentText() == beforeCollapse,
            "Collapsing a group must not change configuration values");

    editor._search->setText(QStringLiteral("Learning Rate"));
    processLayout();
    require(optimizerGroup->isExpanded(),
            "Search should temporarily expand a matching group");
    require(!optimizerGroup->isHidden(),
            "Search should retain groups containing matching rows");
    require(editor._controlGroups[1].widget->isHidden(),
            "Search should hide groups with no matching rows");
    editor._search->clear();
    processLayout();
    require(!optimizerGroup->isExpanded(),
            "Clearing search should restore the prior collapse state");
    require(!editor._controlGroups[1].widget->isHidden(),
            "Clearing search should restore hidden groups");

    editor.showWindow();
    QDialog* dialog = editor.findChild<QDialog*>(
        QStringLiteral("spiralAdvancedConfigDialog"));
    QWidget* grid = editor.findChild<QWidget*>(
        QStringLiteral("spiralConfigGroupGrid"));
    require(dialog && grid, "Configuration dialog grid was not created");
    const QList<QPair<int, int>> widths{{520, 1}, {900, 2}, {1320, 3}};
    for (const auto& width : widths) {
        dialog->resize(width.first, 760);
        processLayout();
        require(grid->property("spiralColumnCount").toInt() == width.second,
                "Responsive group grid selected the wrong column count");
        require(editor._controlsScroll->horizontalScrollBarPolicy()
                    == Qt::ScrollBarAlwaysOff,
                "Controls area should never show a horizontal scrollbar");
    }

    auto* iterations = qobject_cast<QSpinBox*>(
        editor._fieldEditors.value(QStringLiteral("model_iterations")));
    QSignalSpy textChanged(&editor, &SpiralConfigProfileEditor::textChanged);
    iterations->setValue(31);
    processLayout();
    const QJsonObject edited = QJsonDocument::fromJson(
        editor.currentText().toUtf8()).object();
    require(edited.value(QStringLiteral("model_iterations")).toInt() == 31,
            "Editing a grouped control did not update Expert JSON");
    require(editor._dirty && textChanged.count() > 0,
            "Editing a grouped control did not update profile dirty state");

    QJsonObject external = edited;
    external[QStringLiteral("model_iterations")] = 47;
    editor.setCurrentText(QString::fromUtf8(
        QJsonDocument(external).toJson(QJsonDocument::Compact)));
    processLayout();
    require(iterations->value() == 47,
            "Expert JSON changes did not update grouped controls");

    return 0;
}
