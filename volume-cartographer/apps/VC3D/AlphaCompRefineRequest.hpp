#pragma once

#include <QString>

struct AlphaCompRefineRequest {
    bool refine{true};
    double start{-6.0};
    double stop{30.0};
    double step{2.0};
    int low{26};
    int high{255};
    double borderOff{1.0};
    int radius{3};
    bool genVertexColor{false};
    bool overwrite{true};
    double readerScale{0.5};
    QString scaleGroup{QStringLiteral("1")};
    int ompThreads{-1};

    // Empty paths are resolved from the current volume and surface.
    QString volumePath;
    QString sourcePath;
    QString outputDir;
};
