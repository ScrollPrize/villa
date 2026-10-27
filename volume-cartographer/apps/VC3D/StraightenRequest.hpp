#pragma once

#include <QString>
#include <QStringList>

// A vc_straighten run, from its dialog or the agent bridge.
struct StraightenRequest {
    bool    unbend{true};
    double  unbendSmoothCols{300.0};  // only emitted when unbend is true
    int     overlapPasses{2};         // always emitted as --overlap-pairs
    bool    orthogonalize{true};
    bool    trim{true};
    double  trimMaxEdge{100.0};       // only emitted when trim is true
    QString outputName;               // the straightened surface's name; empty
                                      // => <name>_straightened (dialog default)
};

// The tool's stage arguments (without input/output). --overlap-pairs is always
// explicit, so the tool never falls back to its "no flags = full pipeline" default.
inline QStringList straightenArgs(const StraightenRequest& request)
{
    QStringList args;
    if (request.unbend) {
        args << QStringLiteral("--unbend")
             << QStringLiteral("--unbend-smooth-cols") << QString::number(request.unbendSmoothCols, 'f', 0);
    }
    args << QStringLiteral("--overlap-pairs") << QString::number(request.overlapPasses);
    if (request.orthogonalize) args << QStringLiteral("--orthogonalize");
    if (request.trim) {
        args << QStringLiteral("--trim")
             << QStringLiteral("--trim-max-edge") << QString::number(request.trimMaxEdge, 'f', 0);
    }
    return args;
}
