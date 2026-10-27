#include "SpiralArtifactPins.hpp"

#include <QtTest/QtTest>

class SpiralArtifactPinsTest : public QObject
{
    Q_OBJECT

private slots:
    void keepsTheInputManifestBesideThePreview()
    {
        const QString preview = QStringLiteral("/cache/session/preview/manifest.json");
        const QString diagnostics =
            QStringLiteral("/cache/session/diagnostics/manifest.json");
        const QString inputs =
            QStringLiteral("/cache/session/inputs/object_store_inputs.json");

        const QStringList pins = vc3d::spiralArtifactCachePins(
            preview, diagnostics, {inputs});

        QCOMPARE(pins, QStringList({preview, diagnostics, inputs}));
        QCOMPARE(vc3d::spiralArtifactCachePins(preview, diagnostics, {}),
                 QStringList({preview, diagnostics}));
    }
};

QTEST_APPLESS_MAIN(SpiralArtifactPinsTest)

#include "test_spiral_artifact_pins.moc"
