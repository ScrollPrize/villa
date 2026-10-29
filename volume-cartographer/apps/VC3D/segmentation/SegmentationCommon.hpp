#pragma once

#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(lcSegWidget)

// Values are persisted in settings.
enum class NeuralTracerModelType {
    Heatmap = 0,
    DisplacementCopy = 2,
};

enum class TtaMode {
    Mirror = 0,
    Rotate3 = 1,
    None = 2,
};
