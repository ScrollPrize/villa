#include "SpiralArtifactPins.hpp"

#include <QtTest/QtTest>

class SpiralArtifactPinsTest : public QObject
{
    Q_OBJECT

private slots:
    void keepsEveryDisplayOnlyPclArtifact()
    {
        const QString preview = QStringLiteral("/cache/session/preview/manifest.json");
        const QString diagnostics =
            QStringLiteral("/cache/session/diagnostics/manifest.json");
        const QString sameWinding =
            QStringLiteral("/cache/session/same-winding/manifest.json");
        const QString relativeWinding =
            QStringLiteral("/cache/session/relative-winding/manifest.json");

        const QStringList pins = vc3d::spiralArtifactCachePins(
            preview, diagnostics, {sameWinding, relativeWinding});

        QCOMPARE(pins, QStringList({preview, diagnostics, sameWinding,
                                    relativeWinding}));
        QCOMPARE(vc3d::spiralArtifactCachePins(preview, diagnostics, {}),
                 QStringList({preview, diagnostics}));
    }
};

QTEST_APPLESS_MAIN(SpiralArtifactPinsTest)

#include "test_spiral_artifact_pins.moc"
