#include "SpiralPreviewProvenance.hpp"

#include <QJsonObject>
#include <QtTest/QtTest>

class SpiralPreviewProvenanceTest : public QObject
{
    Q_OBJECT

private slots:
    void readsInstalledArtifactManifest()
    {
        const auto provenance = vc3d::spiralPreviewProvenance({
            {QStringLiteral("source_fit_iteration"), 200},
            {QStringLiteral("initialization_mode"), QStringLiteral("warm")},
        });
        QCOMPARE(provenance.sourceIteration, qint64{200});
        QCOMPARE(provenance.initializationMode, QStringLiteral("warm"));
    }

    void missingProvenanceIsUnknown()
    {
        const auto provenance = vc3d::spiralPreviewProvenance({});
        QCOMPARE(provenance.sourceIteration, qint64{-1});
        QVERIFY(provenance.initializationMode.isEmpty());
    }
};

QTEST_APPLESS_MAIN(SpiralPreviewProvenanceTest)

#include "test_spiral_preview_provenance.moc"
