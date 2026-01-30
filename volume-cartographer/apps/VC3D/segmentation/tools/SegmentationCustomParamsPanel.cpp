#include "SegmentationCustomParamsPanel.hpp"

#include "elements/JsonProfileEditor.hpp"
#include "elements/JsonProfilePresets.hpp"

#include <QVBoxLayout>

SegmentationCustomParamsPanel::SegmentationCustomParamsPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* panelLayout = new QVBoxLayout(this);
    panelLayout->setContentsMargins(0, 0, 0, 0);
    panelLayout->setSpacing(0);

    _customParamsEditor = new JsonProfileEditor(tr("Custom Params"), this);
    _customParamsEditor->setDescription(
        tr("Additional JSON fields merge into the tracer params. Leave empty for defaults."));
    _customParamsEditor->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));
    _customParamsEditor->setTextToolTip(
        tr("Optional JSON that merges into tracer parameters before growth."));

    const auto profiles = vc3d::json_profiles::tracerParamProfiles(
        [this](const char* text) { return tr(text); });
    _customParamsEditor->setProfiles(profiles, QStringLiteral("custom"));

    panelLayout->addWidget(_customParamsEditor);
}
