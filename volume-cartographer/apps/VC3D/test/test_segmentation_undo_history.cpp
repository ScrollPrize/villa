#include "Keybinds.hpp"
#include "segmentation/SegmentationUndoHistory.hpp"

#include <QKeyEvent>
#include <QTest>

#include <deque>
#include <vector>

using segmentation::UndoHistory;
using segmentation::VertexDelta;

namespace {

using Grid = cv::Mat_<cv::Vec3f>;

Grid makeGrid(int rows, int cols, float value)
{
    return Grid(rows, cols, cv::Vec3f(value, value, value));
}

bool sameGrid(const Grid& a, const Grid& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (int r = 0; r < a.rows; ++r) {
        for (int c = 0; c < a.cols; ++c) {
            if (a(r, c) != b(r, c)) {
                return false;
            }
        }
    }
    return true;
}

// Drives the history the way SegmentationModule::applyHistoryEntry does: an
// edit records the positions it overwrites, and undo or redo records the
// current positions (UndoHistory::inverseFor) before restoring them, so every
// step can be reversed.
class Editor
{
public:
    explicit Editor(Grid grid, segmentation::UndoLimits limits = {})
        : grid(std::move(grid)), history(limits)
    {
        history.setGridSize(this->grid.size());
    }

    void move(int row, int col, float value)
    {
        history.captureDelta({{row, col, grid(row, col)}});
        grid(row, col) = cv::Vec3f(value, value, value);
    }

    void snapshotEdit(float value)
    {
        history.capture(grid);
        grid.setTo(cv::Vec3f(value, value, value));
    }

    // approvalSteps: whether the paired auto-approval could be stepped too.
    bool step(bool redo, bool approvalSteps = true)
    {
        auto entry = redo ? history.takeRedo() : history.takeUndo();
        if (!entry) {
            return false;
        }
        UndoHistory::Entry replaced = UndoHistory::inverseFor(grid, *entry);
        if (entry->isDelta()) {
            UndoHistory::applyDeltas(grid, *entry);
        } else if (entry->points.size() == grid.size()) {
            entry->points.copyTo(grid);
        } else {
            return false;  // Like setPreviewPoints; the entry is dropped
        }
        if (entry->autoApprovalId != 0 && approvalSteps) {
            replaced.autoApprovalId = entry->autoApprovalId;
        }
        history.pushReplaced(redo, std::move(replaced));
        return true;
    }

    bool undo() { return step(false); }
    bool redo() { return step(true); }

    Grid grid;
    UndoHistory history;
};

std::size_t snapshotBytes(const Grid& grid)
{
    return UndoHistory::Entry{grid.clone(), {}, 0}.bytes();
}

QKeyEvent keyPress(Qt::Key key, Qt::KeyboardModifiers modifiers)
{
    return QKeyEvent(QEvent::KeyPress, key, modifiers);
}

} // namespace

class SegmentationUndoHistoryTest : public QObject
{
    Q_OBJECT

private slots:
    void redoRestoresWhatUndoRemoved()
    {
        Editor editor(makeGrid(4, 4, 0.0f));
        const Grid original = editor.grid.clone();
        editor.move(1, 1, 5.0f);
        editor.move(2, 3, 7.0f);
        const Grid edited = editor.grid.clone();

        QVERIFY(editor.undo());
        QVERIFY(editor.undo());
        QVERIFY(sameGrid(editor.grid, original));
        QVERIFY(!editor.history.canUndo());
        QCOMPARE(editor.history.redoSize(), std::size_t{2});

        QVERIFY(editor.redo());
        QVERIFY(editor.redo());
        QVERIFY(sameGrid(editor.grid, edited));
        QVERIFY(!editor.history.canRedo());
        QCOMPARE(editor.history.size(), std::size_t{2});

        // The redone edits can be undone again.
        QVERIFY(editor.undo());
        QCOMPARE(editor.grid(2, 3), cv::Vec3f(0.0f, 0.0f, 0.0f));
        QCOMPARE(editor.grid(1, 1), cv::Vec3f(5.0f, 5.0f, 5.0f));
    }

    void redoAlsoWorksForSnapshots()
    {
        Editor editor(makeGrid(3, 5, 1.0f));
        editor.snapshotEdit(2.0f);
        QVERIFY(editor.undo());
        QCOMPARE(editor.grid(0, 0), cv::Vec3f(1.0f, 1.0f, 1.0f));
        QVERIFY(editor.redo());
        QCOMPARE(editor.grid(2, 4), cv::Vec3f(2.0f, 2.0f, 2.0f));
    }

    void newEditClearsRedo()
    {
        Editor editor(makeGrid(4, 4, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.move(0, 1, 2.0f);
        QVERIFY(editor.undo());
        QVERIFY(editor.history.canRedo());

        editor.move(3, 3, 9.0f);
        QVERIFY(!editor.history.canRedo());
        QVERIFY(!editor.redo());
        QCOMPARE(editor.grid(0, 1), cv::Vec3f(0.0f, 0.0f, 0.0f));
        QCOMPARE(editor.history.size(), std::size_t{2});
    }

    void discardedCaptureKeepsRedo()
    {
        // A mask stroke captures before painting and discards when it
        // changed nothing; that is not an edit, so redo must survive.
        Editor editor(makeGrid(2, 2, 0.0f));
        editor.move(0, 0, 1.0f);
        QVERIFY(editor.undo());
        QVERIFY(editor.history.capture(editor.grid));
        QVERIFY(!editor.history.canRedo());
        editor.history.discardLast();
        QVERIFY(editor.history.canRedo());
        QVERIFY(!editor.history.canUndo());
        QVERIFY(editor.redo());
        QCOMPARE(editor.grid(0, 0), cv::Vec3f(1.0f, 1.0f, 1.0f));
    }

    void defaultLimitsKeepTheOldDepthAndBoundTheRest()
    {
        const segmentation::UndoLimits limits;
        QCOMPARE(limits.minEntries, std::size_t{10});
        QCOMPARE(limits.maxEntries, std::size_t{100});
        QCOMPARE(limits.byteBudget, std::size_t{256} << 20);
        QVERIFY(!limits.exceeded(10, std::size_t{4} << 30));  // Ten steps whatever their size
        QVERIFY(limits.exceeded(11, (std::size_t{256} << 20) + 1));
        QVERIFY(!limits.exceeded(100, 0));
        QVERIFY(limits.exceeded(101, 0));
    }

    void budgetEvictsOldestUndoFirst()
    {
        const Grid grid = makeGrid(16, 16, 0.0f);
        UndoHistory history({snapshotBytes(grid) * 3, 1, 100});
        for (int i = 0; i < 5; ++i) {
            QVERIFY(history.capture(makeGrid(16, 16, static_cast<float>(i))));
        }
        QCOMPARE(history.size(), std::size_t{3});
        QVERIFY(history.bytes() <= history.limits().byteBudget);

        // The survivors are the three newest snapshots.
        for (float expected : {4.0f, 3.0f, 2.0f}) {
            auto entry = history.takeUndo();
            QVERIFY(entry.has_value());
            QCOMPARE(entry->points(0, 0)[0], expected);
        }
        QVERIFY(!history.canUndo());
        QCOMPARE(history.bytes(), std::size_t{0});
    }

    void largeSnapshotsKeepTheOldTenSteps()
    {
        // Before the byte budget there were always ten steps; snapshots too
        // large for the budget still get them.
        const Grid grid = makeGrid(16, 16, 0.0f);
        UndoHistory history({snapshotBytes(grid), 10, 100});
        for (int i = 0; i < 15; ++i) {
            QVERIFY(history.capture(makeGrid(16, 16, static_cast<float>(i))));
        }
        QCOMPARE(history.size(), std::size_t{10});
        QCOMPARE(history.takeUndo()->points(0, 0)[0], 14.0f);
    }

    void smallEditsAreCappedByCount()
    {
        UndoHistory history;
        for (int i = 0; i < 150; ++i) {
            QVERIFY(history.captureDelta({{i, 0, cv::Vec3f(0.0f, 0.0f, 0.0f)}}));
        }
        QCOMPARE(history.size(), std::size_t{100});
        QCOMPARE(history.takeUndo()->deltas.front().row, 149);
    }

    void budgetKeepsTheNewestEntry()
    {
        UndoHistory history({16, 0, 100});  // Smaller than any entry
        QVERIFY(history.capture(makeGrid(8, 8, 1.0f)));
        QVERIFY(history.capture(makeGrid(8, 8, 2.0f)));
        QCOMPARE(history.size(), std::size_t{1});

        // Undoing pushes the replaced state onto redo; that newest entry
        // survives and the undo stack makes room for it.
        auto entry = history.takeUndo();
        QVERIFY(entry.has_value());
        QCOMPARE(entry->points(0, 0)[0], 2.0f);
        history.pushReplaced(false, UndoHistory::Entry{makeGrid(8, 8, 3.0f), {}, 0});
        QVERIFY(history.canRedo());

        history.pushReplaced(true, UndoHistory::Entry{makeGrid(8, 8, 4.0f), {}, 0});
        QVERIFY(history.canUndo());
        QVERIFY(!history.canRedo());
    }

    void discardedCaptureGivesBackWhatItEvicted()
    {
        // A mask click that changes nothing must not cost an older undo step.
        Editor editor(makeGrid(4, 4, 0.0f), {std::size_t{1} << 20, 0, 2});
        editor.move(0, 0, 1.0f);
        editor.move(0, 1, 2.0f);
        QVERIFY(editor.history.capture(editor.grid));
        QCOMPARE(editor.history.size(), std::size_t{2});  // The oldest step was evicted...
        editor.history.discardLast();
        QCOMPARE(editor.history.size(), std::size_t{2});  // ...and is back
        QVERIFY(editor.undo());
        QVERIFY(editor.undo());
        QCOMPARE(editor.grid(0, 0), cv::Vec3f(0.0f, 0.0f, 0.0f));
        QCOMPARE(editor.grid(0, 1), cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    void evictedEntriesStayGoneOnceTheNextChangeComes()
    {
        Editor editor(makeGrid(4, 4, 0.0f), {std::size_t{1} << 20, 0, 2});
        editor.move(0, 0, 1.0f);
        editor.move(0, 1, 2.0f);
        editor.move(0, 2, 3.0f);  // Evicts the first edit
        QVERIFY(editor.undo());   // A change: the eviction is final
        QVERIFY(editor.history.capture(editor.grid));
        editor.history.discardLast();
        QCOMPARE(editor.history.size(), std::size_t{1});
        QVERIFY(editor.history.canRedo());
    }

    void clearRedoKeepsUndo()
    {
        // Growth changes the surface outside the history, so redo must go.
        Editor editor(makeGrid(4, 4, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.move(0, 1, 2.0f);
        QVERIFY(editor.undo());
        editor.history.clearRedo();
        QVERIFY(!editor.history.canRedo());
        QVERIFY(!editor.redo());
        QCOMPARE(editor.history.size(), std::size_t{1});
        QVERIFY(editor.undo());
        QCOMPARE(editor.grid(0, 0), cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    void gridSizeChangeDropsTheHistory()
    {
        Editor editor(makeGrid(4, 4, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.move(1, 1, 1.0f);
        QVERIFY(editor.undo());

        editor.history.setGridSize(cv::Size(4, 4));  // Same size: kept
        QVERIFY(editor.history.canUndo());
        QVERIFY(editor.history.canRedo());

        editor.history.setGridSize(cv::Size(4, 6));  // Grown: nothing fits
        QVERIFY(!editor.history.canUndo());
        QVERIFY(!editor.history.canRedo());
        QCOMPARE(editor.history.bytes(), std::size_t{0});
    }

    void deltasOutsideTheGridAreSkipped()
    {
        Grid grid = makeGrid(2, 2, 0.0f);
        UndoHistory::Entry entry;
        entry.deltas = {{1, 1, cv::Vec3f(5.0f, 5.0f, 5.0f)}, {2, 0, cv::Vec3f(9.0f, 9.0f, 9.0f)}};
        const UndoHistory::Entry inverse = UndoHistory::inverseFor(grid, entry);
        QCOMPARE(inverse.deltas.size(), std::size_t{1});
        QCOMPARE(inverse.deltas.front().previousWorld, cv::Vec3f(0.0f, 0.0f, 0.0f));
        UndoHistory::applyDeltas(grid, entry);
        QCOMPARE(grid(1, 1), cv::Vec3f(5.0f, 5.0f, 5.0f));

        // A step whose deltas all miss the grid leaves nothing to step back to.
        Editor editor(makeGrid(2, 2, 0.0f));
        editor.history.captureDelta({{7, 7, cv::Vec3f(1.0f, 1.0f, 1.0f)}});
        QVERIFY(editor.undo());
        QVERIFY(!editor.history.canRedo());
    }

    void autoApprovalIdFollowsTheEditThroughUndoAndRedo()
    {
        Editor editor(makeGrid(2, 2, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.history.setLastAutoApprovalId(42);
        editor.move(1, 1, 1.0f);

        auto unpaired = editor.history.takeUndo();
        QVERIFY(unpaired.has_value());
        QCOMPARE(unpaired->autoApprovalId, std::uint64_t{0});
        editor.history.pushReplaced(true, std::move(*unpaired));

        QVERIFY(editor.undo());
        QVERIFY(editor.undo());
        auto redoEntry = editor.history.takeRedo();
        QVERIFY(redoEntry.has_value());
        QCOMPARE(redoEntry->autoApprovalId, std::uint64_t{42});
    }

    void autoApprovalIdIsDroppedWhenTheApprovalStepFails()
    {
        // inverseFor leaves the id out; only a successful approval step links it.
        Editor editor(makeGrid(2, 2, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.history.setLastAutoApprovalId(7);
        QVERIFY(editor.step(false, false));
        auto redoEntry = editor.history.takeRedo();
        QVERIFY(redoEntry.has_value());
        QCOMPARE(redoEntry->autoApprovalId, std::uint64_t{0});
    }

    void trimOldestKeepsTheNewest()
    {
        struct Item
        {
            std::size_t bytes;
        };
        auto bytesOf = [](const Item& item) { return item.bytes; };
        std::deque<Item> stack{{10}, {10}, {10}, {50}};
        segmentation::trimOldest(stack, {25, 1, 100}, bytesOf);
        QCOMPARE(stack.size(), std::size_t{1});
        QCOMPARE(stack.front().bytes, std::size_t{50});

        std::deque<Item> many(120, Item{1});
        segmentation::trimOldest(many, {}, bytesOf);
        QCOMPARE(many.size(), std::size_t{100});
    }

    void clearDropsBothStacks()
    {
        Editor editor(makeGrid(2, 2, 0.0f));
        editor.move(0, 0, 1.0f);
        editor.move(0, 1, 1.0f);
        QVERIFY(editor.undo());
        editor.history.clear();
        QVERIFY(!editor.history.canUndo());
        QVERIFY(!editor.history.canRedo());
        QCOMPARE(editor.history.bytes(), std::size_t{0});
    }

    void ctrlShiftZRedoesInsteadOfUndoing()
    {
        using vc3d::keybinds::HistoryRequest;
        using vc3d::keybinds::historyRequestFor;

        // Before the fix the masked Ctrl+Z test matched Ctrl+Shift+Z as undo.
        QCOMPARE(historyRequestFor(keyPress(Qt::Key_Z, Qt::ControlModifier | Qt::ShiftModifier)),
                 HistoryRequest::Redo);
        QCOMPARE(historyRequestFor(keyPress(Qt::Key_Z, Qt::ControlModifier)), HistoryRequest::Undo);
        QCOMPARE(historyRequestFor(keyPress(Qt::Key_Z, Qt::ControlModifier | Qt::KeypadModifier)),
                 HistoryRequest::Undo);
        QCOMPARE(historyRequestFor(keyPress(Qt::Key_Z, Qt::NoModifier)), HistoryRequest::None);
        QCOMPARE(historyRequestFor(keyPress(Qt::Key_Y, Qt::NoModifier)), HistoryRequest::None);

        // Every platform redo binding (Ctrl+Y on Windows) is honored.
        for (const QKeySequence& sequence : QKeySequence::keyBindings(QKeySequence::Redo)) {
            const QKeyCombination combination = sequence[0];
            QCOMPARE(historyRequestFor(keyPress(combination.key(), combination.keyboardModifiers())),
                     HistoryRequest::Redo);
        }
    }

    void keybindsHelpListsRedo()
    {
        const QString help = vc3d::keybinds::buildKeybindsHelpText();
        QVERIFY(help.contains(QString::fromUtf8(vc3d::keybinds::keypress::SegmentationRedo.description)));
    }
};

QTEST_MAIN(SegmentationUndoHistoryTest)
#include "test_segmentation_undo_history.moc"
