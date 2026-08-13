#pragma once

#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(lcSegWidget)

enum class NeuralTracerModelType {
    Heatmap = 0,
    DenseDisplacement = 1,
    DisplacementCopy = 2,
};

enum class NeuralTracerOutputMode {
    OverwriteCurrentSegment = 0,
    CreateNewSegment = 1,
};

enum class DenseTtaMode {
    Mirror = 0,
    Rotate3 = 1,
    None = 2,
};
