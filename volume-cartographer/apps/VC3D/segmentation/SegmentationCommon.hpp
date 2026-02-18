#pragma once

#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(lcSegWidget)

enum class NeuralTracerModelType {
    Heatmap = 0,
    DenseDisplacement = 1,
};

enum class NeuralTracerOutputMode {
    OverwriteCurrentSegment = 0,
    CreateNewSegment = 1,
};
