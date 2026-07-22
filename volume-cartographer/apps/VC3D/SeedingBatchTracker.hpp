#pragma once

#include <QSet>
#include <QString>
#include <QStringList>

// Outcome aggregation for a run/expand seeding batch, extracted from
// SeedingWidget so it can be unit-tested without spawning real QProcess
// children (SPEC §1). QObject-free and process-agnostic: it takes an opaque
// per-child key (the QProcess* identity, as a const void*) and pre-extracted
// outcome fields, never a Qt process type.
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

    // Reset all state for a new batch (mirrors the reset SeedingWidget did
    // before launching the first child). kind is "run" | "expand".
    void begin(const QString& kind, int total);

    // Record one child reaching its terminal state. key is an opaque process
    // identity used only for dedup: finished() and errorOccurred() can both
    // fire for the same child, but it must count exactly once. Returns false
    // (and no-ops) when key is already terminal. Otherwise advances completion
    // and, when failedToStart || crashed || exitCode != 0, counts a failure and
    // appends a bounded diagnostic. label (optional) is the "Segmentation for
    // point N" / "Expansion iteration N" prefix SeedingWidget builds from the
    // child kind + index; the tracker itself has no notion of an index.
    bool recordTerminal(const void* key, bool failedToStart, bool crashed,
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
    QSet<const void*> _terminalKeys;   // each child contributes to completion once
    Result _result;                    // cached after the first finalize()
};
