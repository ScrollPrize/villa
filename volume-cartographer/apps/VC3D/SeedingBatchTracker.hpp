#pragma once

#include <QSet>
#include <QString>
#include <QStringList>

// Outcome aggregation for a run/expand seeding batch, extracted from SeedingWidget so it
// can be unit-tested without spawning real QProcess children (SPEC §1). QObject-free:
// takes a stable per-child index key (never a raw QProcess*, whose address a later child
// can reuse after deleteLater — see recordTerminal) and pre-extracted outcome fields.
//
// SeedingWidget keeps the process lifecycle (draining output, killing, deleteLater); this
// class owns only the failure/cancel/success bookkeeping and the terminal message.
class SeedingBatchTracker {
public:
    struct Result {
        bool success{false};
        bool canceled{false};
        int completed{0};
        int total{0};
        QString message;
    };

    // Clear finalized state before a non-seeding operation reuses the teardown path.
    void reset();

    // Resets state for a new batch (mirrors SeedingWidget's pre-launch reset). kind is
    // "run" | "expand".
    void begin(const QString& kind, int total);

    // Records one child's terminal state; key is a stable per-child index (never a raw
    // QProcess*, since SeedingWidget deleteLater()s finished children and a later child
    // can reuse the freed address). finished()/errorOccurred() may both fire for the same
    // child, so key dedups to count each exactly once; returns false when key is already
    // terminal. label is the optional "Segmentation for point N" / "Expansion iteration N"
    // prefix.
    bool recordTerminal(int key, bool failedToStart, bool crashed,
                        int exitCode, const QString& tail,
                        const QString& label = QString());

    // Latch a user cancel (distinct from an execution failure).
    void requestCancel();

    [[nodiscard]] bool isComplete() const { return _completed >= _total; }
    [[nodiscard]] bool finalized() const { return _finalized; }

    // Idempotent: repeated calls return the cached Result without further mutation.
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
