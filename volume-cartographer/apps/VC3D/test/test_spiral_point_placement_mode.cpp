#include "SpiralBrushCursorWidget.hpp"
#include "SpiralPointPlacementMode.hpp"

#include <QImage>
#include <QKeyEvent>
#include <QPainter>
#include <QTest>

#include <cmath>

class SpiralPointPlacementModeTest : public QObject
{
    Q_OBJECT

private slots:
    void qActivatesAndIsIdempotent();
    void eActivatesRelativeAndSwitchesRoles();
    void escapeClearsInteractionInOneStep();
    void activeSelectionGatesCollectionKeys();
    void interruptionsDoNotDeactivate();
    void readyResetAndDiscardDeactivate();
    void surfaceChangesClearInteraction();
    void cursorCueHasPlacementPrecedence();
    void cursorRendersRetainedPclHover();
    void cursorCueBoundsCoverOnlyDrawnCues();
};

namespace {
using Transition = SpiralPointPlacementMode::Transition;

QKeyEvent keyEvent(QEvent::Type type, int key, bool autoRepeat = false)
{
    return QKeyEvent(type, key, Qt::NoModifier, QString(), autoRepeat);
}

QKeyEvent qEvent(QEvent::Type type, bool autoRepeat = false)
{
    return keyEvent(type, Qt::Key_Q, autoRepeat);
}

QKeyEvent eEvent(QEvent::Type type, bool autoRepeat = false)
{
    return keyEvent(type, Qt::Key_E, autoRepeat);
}

bool isCyan(const QColor& color)
{
    return color.alpha() > 100 && color.red() < 100
        && color.green() > 220 && color.blue() > 170;
}

bool isOrange(const QColor& color)
{
    return color.alpha() > 100 && color.red() > 220
        && color.green() > 130 && color.green() < 210 && color.blue() < 100;
}

bool isWhite(const QColor& color)
{
    return color.alpha() > 100 && color.red() > 180
        && color.green() > 180 && color.blue() > 180;
}

int countPixels(const QImage& image, const QPointF& center,
                qreal minimumRadius, qreal maximumRadius,
                bool (*predicate)(const QColor&))
{
    int count = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const qreal radius = std::hypot(x - center.x(), y - center.y());
            if (radius >= minimumRadius && radius <= maximumRadius
                && predicate(image.pixelColor(x, y)))
                ++count;
        }
    }
    return count;
}

QImage renderCursor(SpiralBrushCursorWidget& widget)
{
    QImage image(widget.size(), QImage::Format_ARGB32_Premultiplied);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    widget.render(&painter);
    return image;
}
}

void SpiralPointPlacementModeTest::qActivatesAndIsIdempotent()
{
    SpiralPointPlacementMode mode;

    auto result = mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::Activated));
    QVERIFY(mode.active());

    result = mode.handleEvent(qEvent(QEvent::KeyRelease));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    QVERIFY(mode.active());

    result = mode.handleEvent(qEvent(QEvent::KeyPress, true));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    QVERIFY(mode.active());

    result = mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    QVERIFY(mode.active());
    QVERIFY(mode.activeRole().has_value());
    QCOMPARE(static_cast<int>(*mode.activeRole()),
             static_cast<int>(vc3d::spiral::PclRole::SameWinding));
}

void SpiralPointPlacementModeTest::eActivatesRelativeAndSwitchesRoles()
{
    SpiralPointPlacementMode mode;
    QVERIFY(!mode.activeRole());

    auto result = mode.handleEvent(eEvent(QEvent::KeyPress));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::Activated));
    QVERIFY(mode.active());
    QCOMPARE(static_cast<int>(*mode.activeRole()),
             static_cast<int>(vc3d::spiral::PclRole::Relative));

    // E release, autorepeat, and a repeated press all belong to the mode
    // without changing it.
    result = mode.handleEvent(eEvent(QEvent::KeyRelease));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    result = mode.handleEvent(eEvent(QEvent::KeyPress, true));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    result = mode.handleEvent(eEvent(QEvent::KeyPress));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    QCOMPARE(static_cast<int>(*mode.activeRole()),
             static_cast<int>(vc3d::spiral::PclRole::Relative));

    // The other role's key switches roles in one step, both ways.
    result = mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::SwitchRole));
    QCOMPARE(static_cast<int>(*mode.activeRole()),
             static_cast<int>(vc3d::spiral::PclRole::SameWinding));
    result = mode.handleEvent(eEvent(QEvent::KeyPress));
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::SwitchRole));
    QCOMPARE(static_cast<int>(*mode.activeRole()),
             static_cast<int>(vc3d::spiral::PclRole::Relative));

    // Escape clears the relative mode exactly like the same-winding one.
    result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_Escape));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition),
             static_cast<int>(Transition::ClearInteraction));
    QVERIFY(!mode.active());
    QVERIFY(!mode.activeRole());

    QVERIFY(SpiralPointPlacementMode::roleForKey(Qt::Key_E).has_value());
    QVERIFY(SpiralPointPlacementMode::roleForKey(Qt::Key_Q).has_value());
    QVERIFY(!SpiralPointPlacementMode::roleForKey(Qt::Key_F).has_value());
}

void SpiralPointPlacementModeTest::escapeClearsInteractionInOneStep()
{
    SpiralPointPlacementMode mode;
    mode.handleEvent(qEvent(QEvent::KeyPress));

    auto result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_Escape));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition),
             static_cast<int>(Transition::ClearInteraction));
    QVERIFY(!mode.active());

    result = mode.handleEvent(
        keyEvent(QEvent::KeyPress, Qt::Key_Escape, true));
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    result = mode.handleEvent(keyEvent(QEvent::KeyRelease, Qt::Key_Escape));
    QVERIFY(result.handled);
    result = mode.handleEvent(keyEvent(QEvent::KeyRelease, Qt::Key_Escape));
    QVERIFY(!result.handled);

    result = mode.handleEvent(
        keyEvent(QEvent::KeyPress, Qt::Key_Escape), true);
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition),
             static_cast<int>(Transition::ClearInteraction));
}

void SpiralPointPlacementModeTest::activeSelectionGatesCollectionKeys()
{
    SpiralPointPlacementMode mode;
    auto result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_F));
    QVERIFY(!result.handled);
    result = mode.handleEvent(keyEvent(QEvent::KeyRelease, Qt::Key_F));
    QVERIFY(!result.handled);

    result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_F), true);
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition),
             static_cast<int>(Transition::ReverseActive));
    result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_F, true), true);
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition), static_cast<int>(Transition::None));
    result = mode.handleEvent(keyEvent(QEvent::KeyRelease, Qt::Key_F));
    QVERIFY(result.handled);

    result = mode.handleEvent(keyEvent(QEvent::KeyPress, Qt::Key_Delete), true);
    QVERIFY(result.handled);
    QCOMPARE(static_cast<int>(result.transition),
             static_cast<int>(Transition::DeleteActive));
    result = mode.handleEvent(keyEvent(QEvent::KeyRelease, Qt::Key_Delete));
    QVERIFY(result.handled);
}

void SpiralPointPlacementModeTest::interruptionsDoNotDeactivate()
{
    SpiralPointPlacementMode mode;
    mode.handleEvent(qEvent(QEvent::KeyPress));

    QEvent leave(QEvent::Leave);
    QVERIFY(!mode.handleEvent(leave).handled);
    QVERIFY(mode.active());
    QEvent focusOut(QEvent::FocusOut);
    QVERIFY(!mode.handleEvent(focusOut).handled);
    QVERIFY(mode.active());
    QEvent deactivate(QEvent::WindowDeactivate);
    QVERIFY(!mode.handleEvent(deactivate).handled);
    QVERIFY(mode.active());
}

void SpiralPointPlacementModeTest::readyResetAndDiscardDeactivate()
{
    SpiralPointPlacementMode mode;
    mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(mode.deactivate()); // Shift+E ready workflow
    QVERIFY(!mode.active());

    mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(mode.deactivate()); // session reset
    QVERIFY(!mode.active());

    mode.handleEvent(qEvent(QEvent::KeyPress));
    QVERIFY(mode.deactivate()); // discard unfinalized drafts
    QVERIFY(!mode.active());
    QVERIFY(!mode.deactivate());
}

void SpiralPointPlacementModeTest::surfaceChangesClearInteraction()
{
    SpiralPointPlacementMode mode;
    QCOMPARE(static_cast<int>(mode.surfaceChanged()), static_cast<int>(Transition::None));
    mode.handleEvent(qEvent(QEvent::KeyPress));
    QCOMPARE(static_cast<int>(mode.surfaceChanged()),
             static_cast<int>(Transition::ClearInteractionPreserveDraft));
    QVERIFY(!mode.active());
    QCOMPARE(static_cast<int>(mode.surfaceChanged(true)),
             static_cast<int>(Transition::ClearInteractionPreserveDraft));
}

void SpiralPointPlacementModeTest::cursorCueHasPlacementPrecedence()
{
    SpiralBrushCursorWidget widget;
    widget.resize(101, 101);
    const QPointF center(50.0, 50.0);

    widget.setCursorState(center, 64, true, false);
    const QImage inactive = renderCursor(widget);
    QCOMPARE(countPixels(inactive, center, 0.0, 4.0, isCyan), 0);
    QCOMPARE(countPixels(inactive, center, 17.0, 23.0, isCyan), 0);
    QVERIFY(countPixels(inactive, center, 29.0, 35.0, isWhite) > 20);

    widget.setCursorState(center, 64, true, true);
    const QImage active = renderCursor(widget);
    QVERIFY(countPixels(active, center, 0.0, 4.0, isCyan) > 20);
    for (int y = 49; y <= 51; ++y) {
        for (int x = 49; x <= 51; ++x)
            QVERIFY(isCyan(active.pixelColor(x, y)));
    }
    QCOMPARE(countPixels(active, center, 17.0, 23.0, isCyan), 0);
    QCOMPARE(countPixels(active, center, 29.0, 35.0, isWhite), 0);

    // The relative-winding role draws its own accent so the active mode is
    // readable from the cursor alone.
    widget.setCursorState(
        center, 64, true, true,
        vc3d::spiral::pclRoleAccentColor(vc3d::spiral::PclRole::Relative));
    const QImage relative = renderCursor(widget);
    QVERIFY(countPixels(relative, center, 0.0, 4.0, isOrange) > 20);
    QCOMPARE(countPixels(relative, center, 0.0, 4.0, isCyan), 0);
    QCOMPARE(countPixels(relative, center, 29.0, 35.0, isWhite), 0);
}

void SpiralPointPlacementModeTest::cursorRendersRetainedPclHover()
{
    SpiralBrushCursorWidget widget;
    widget.resize(101, 101);
    const QPointF center(50.0, 50.0);
    widget.setCursorState(center, 32, false, false);
    widget.setEditablePclHover(
        center, QColor(50, 255, 215), true, 8.0, 6.0, 1.5);
    const QImage hovered = renderCursor(widget);
    QVERIFY(isCyan(hovered.pixelColor(50, 50)));
    QVERIFY(countPixels(hovered, center, 0.0, 7.0, isCyan) > 50);

    widget.setEditablePclHover(
        std::nullopt, QColor{}, false, 0.0, 0.0, 0.0);
    const QImage cleared = renderCursor(widget);
    QCOMPARE(countPixels(cleared, center, 0.0, 10.0, isCyan), 0);
}

void SpiralPointPlacementModeTest::cursorCueBoundsCoverOnlyDrawnCues()
{
    // The cursor widget covers a QGraphicsView viewport; a repaint of it
    // redraws the scene beneath. With nothing drawn, plain mouse motion must
    // not dirty anything, and drawn cues must dirty only their own pixels
    // rather than the whole viewport.
    SpiralBrushCursorWidget widget;
    widget.resize(400, 300);
    const QPointF center(200.0, 150.0);

    widget.setCursorState(center, 64, false, false);
    QVERIFY(widget.cueBounds().isEmpty());
    widget.setCursorState(center + QPointF(37.0, -12.0), 64, false, false);
    QVERIFY(widget.cueBounds().isEmpty());

    widget.setCursorState(center, 64, false, true);
    const QRect dot = widget.cueBounds();
    QVERIFY(dot.contains(center.toPoint()));
    QVERIFY(dot.width() < 20 && dot.height() < 20);
    // The cue region tracks the rendered dot.
    const QImage active = renderCursor(widget);
    QVERIFY(countPixels(active, center, 0.0, 4.0, isCyan) > 20);
    for (int y = 0; y < active.height(); ++y) {
        for (int x = 0; x < active.width(); ++x) {
            if (active.pixelColor(x, y).alpha() == 0) continue;
            QVERIFY2(dot.contains(x, y), "painted pixel outside cueBounds()");
        }
    }

    widget.setCursorState(center, 64, true, false);
    const QRect ring = widget.cueBounds();
    QVERIFY(ring.contains(center.toPoint()));
    QVERIFY(ring.width() >= 64 && ring.width() < 80);
    const QImage ringImage = renderCursor(widget);
    for (int y = 0; y < ringImage.height(); ++y) {
        for (int x = 0; x < ringImage.width(); ++x) {
            if (ringImage.pixelColor(x, y).alpha() == 0) continue;
            QVERIFY2(ring.contains(x, y), "painted ring pixel outside cueBounds()");
        }
    }

    const QPointF hoverCenter(60.0, 40.0);
    widget.setEditablePclHover(hoverCenter, QColor(50, 255, 215), true, 8.0, 6.0, 1.5);
    const QRect both = widget.cueBounds();
    QVERIFY(both.contains(hoverCenter.toPoint()));
    QVERIFY(both.contains(center.toPoint()));
    QVERIFY(both.width() < widget.width());
    const QImage hovered = renderCursor(widget);
    for (int y = 0; y < hovered.height(); ++y) {
        for (int x = 0; x < hovered.width(); ++x) {
            if (hovered.pixelColor(x, y).alpha() == 0) continue;
            QVERIFY2(both.contains(x, y), "painted hover pixel outside cueBounds()");
        }
    }
}

QTEST_MAIN(SpiralPointPlacementModeTest)
#include "test_spiral_point_placement_mode.moc"
