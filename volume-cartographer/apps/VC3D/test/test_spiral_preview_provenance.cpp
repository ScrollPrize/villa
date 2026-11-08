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
        });
        QCOMPARE(provenance.sourceIteration, qint64{200});
    }

    void missingProvenanceIsUnknown()
    {
        const auto provenance = vc3d::spiralPreviewProvenance({});
        QCOMPARE(provenance.sourceIteration, qint64{-1});
        QVERIFY(provenance.inputDigest.isEmpty());
    }

    void aPreviewOfOtherInputsIsLabelled()
    {
        auto provenance = vc3d::spiralPreviewProvenance({
            {QStringLiteral("source_fit_iteration"), 200},
            {QStringLiteral("input_digest"), QStringLiteral("aaaaaaaaaaaaaaaa")},
        });
        QCOMPARE(provenance.inputDigest, QStringLiteral("aaaaaaaaaaaaaaaa"));
        QVERIFY(vc3d::spiralPreviewInputNote(provenance, QStringLiteral("aaaaaaaaaaaaaaaa")).isEmpty());
        QCOMPARE(vc3d::spiralPreviewInputNote(provenance, QStringLiteral("bbbbbbbbbbbbbbbb")),
                 QStringLiteral("fitted to other inputs (aaaaaaaaaaaa, not bbbbbbbbbbbb)"));
        // A preview or session without a digest is not called different.
        QVERIFY(vc3d::spiralPreviewInputNote({200, {}}, QStringLiteral("bbbb")).isEmpty());
        QVERIFY(vc3d::spiralPreviewInputNote(provenance, {}).isEmpty());
        // One fitted at a store commit names it.
        provenance = vc3d::spiralPreviewProvenance({
            {QStringLiteral("input_digest"), QStringLiteral("aaaaaaaaaaaaaaaa")},
            {QStringLiteral("input_position"), QJsonObject{{QStringLiteral("commit_sequence"), 41}}},
        });
        QCOMPARE(provenance.inputPosition, qint64{41});
        QCOMPARE(vc3d::spiralPreviewInputNote(provenance, QStringLiteral("bbbbbbbbbbbbbbbb")),
                 QStringLiteral("fitted to the inputs at commit 41 (aaaaaaaaaaaa, not bbbbbbbbbbbb)"));
    }
};

QTEST_APPLESS_MAIN(SpiralPreviewProvenanceTest)

#include "test_spiral_preview_provenance.moc"
