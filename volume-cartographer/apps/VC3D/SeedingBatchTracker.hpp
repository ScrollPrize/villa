#pragma once

#include <QSet>
#include <QString>
#include <QStringList>

// Outcome aggregation for a run/expand seeding batch, extracted from
// SeedingWidget so it can be unit-tested without spawning real QProcess
// children (SPEC §1). QObject-free and process-agnostic: it takes a stable
// per-child index key (never a raw QProcess*, whose address a later child can
// reuse after deleteLater — see recordTerminal) and pre-extracted outcome
// fields, never a Qt process type.
//
// SeedingWidget keeps the genuine process lifecycle (draining output, killing,
// deleteLater, the running list, the per-child output tail); this class owns
// only the honest failure/cancel/success bookkeeping and the terminal message.
class SeedingBatchTracker {
public:
    struct Result {
        bool success{false};
        bool canceled{false};
        int completed{0};
        int total{0};
        QString message;
    };

    // Clear all state to the fresh, not-finalized, no-active-batch shape. Call
    // this when a NON-seeding activation (e.g. a neural trace) that shares
    // SeedingWidget's finalizeSeedingBatch() teardown begins, so finalized()
    // reflects only the current activation and does not leak true across from a
    // prior batch (codex #2b).
    void reset();

    // Reset all state for a new batch (mirrors the reset SeedingWidget did
    // before launching the first child). kind is "run" | "expand".
    void begin(const QString& kind, int total);

    // Record one child reaching its terminal state. key is the child's stable
    // batch index, used only for dedup: finished() and errorOccurred() can both
    // fire for the same child, but it must count exactly once. A raw QProcess*
    // must NOT be used here — SeedingWidget deleteLater()s each finished child
    // mid-batch, so a later child can be allocated at a freed address and be
    // mistaken for an already-terminal one, stranding the batch. The index is
    // unique per child and identical across both signals. Returns false (and
    // no-ops) when key is already terminal. Otherwise advances completion and,
    // when failedToStart || crashed || exitCode != 0, counts a failure and
    // appends a bounded diagnostic. label (optional) is the "Segmentation for
    // point N" / "Expansion iteration N" prefix SeedingWidget builds.
    bool recordTerminal(int key, bool failedToStart, bool crashed,
                        int exitCode, const QString& tail,
                        const QString& label = QString());

    // Latch a user cancel (distinct from an execution failure).
    void requestCancel();

    [[nodiscard]] bool isComplete() const { return _completed >= _total; }
    [[nodiscard]] bool finalized() const { return _finalized; }

    // Compute the terminal outcome + message. Idempotent: repeated calls return
    // the same cached Result and mutate nothing further.
    Result finalize();

    // Introspection (mirrors the former SeedingWidget batch members).
    [[nodiscard]] const QString& kind() const { return _kind; }
    [[nodiscard]] int total() const { return _total; }
    [[nodiscard]] int completed() const { return _completed; }
    [[nodiscard]] int failures() const { return _failures; }
    [[nodiscard]] bool cancelRequested() const { return _cancelRequested; }
    [[nodiscard]] const QStringList& failureMessages() const { return _failureMessages; }

private:
    QString _kind;
    int _total{0};
    int _completed{0};
    int _failures{0};
    QStringList _failureMessages;      // bounded diagnostic tail for failures
    bool _cancelRequested{false};
    bool _finalized{false};
    QSet<int> _terminalKeys;           // each child (by batch index) completes once
    Result _result;                    // cached after the first finalize()
};
