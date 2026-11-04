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
    double readerScale{0.5};
    QString scaleGroup{QStringLiteral("1")};
    int ompThreads{-1};

    // The volume of the open project it samples; empty => the current volume.
    QString volumeId;
    // The refined surface's name; empty => <source name>_refined.
    QString outputName;
    // Refine the source surface itself instead of creating a new one.
    bool inPlace{false};
};
