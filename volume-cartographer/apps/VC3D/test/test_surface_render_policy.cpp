#include "volume_viewers/SurfaceRenderPolicy.hpp"

#include <QtTest/QtTest>

class SurfaceRenderPolicyTest : public QObject
{
    Q_OBJECT

private slots:
    void fallbackDemand()
    {
        using vc::render::surfaceQueuedFallbackLevels;
        for (int levels = 0; levels <= 5; ++levels) {
            // Disabled/missing/out-of-band caches must retain coarse demand.
            QCOMPARE(surfaceQueuedFallbackLevels(levels, true, false), levels);
            QCOMPARE(surfaceQueuedFallbackLevels(levels, false, false), levels);
            QCOMPARE(surfaceQueuedFallbackLevels(levels, false, true), levels);
            QCOMPARE(surfaceQueuedFallbackLevels(levels, true, true),
                     std::min(levels, 1));
        }
    }

    void independentBaseAndOverlay()
    {
        using vc::render::surfaceQueuedFallbackLevels;
        for (bool baseUsable : {false, true}) {
            const bool overlayUsable = !baseUsable;
            QCOMPARE(surfaceQueuedFallbackLevels(5, true, baseUsable),
                     baseUsable ? 1 : 5);
            QCOMPARE(surfaceQueuedFallbackLevels(4, true, overlayUsable),
                     overlayUsable ? 1 : 4);
        }
    }
};

QTEST_GUILESS_MAIN(SurfaceRenderPolicyTest)
#include "test_surface_render_policy.moc"
