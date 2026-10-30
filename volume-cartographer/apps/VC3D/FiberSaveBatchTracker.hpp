#pragma once

#include <functional>
#include <utility>

#include <QString>
#include <QStringList>

class FiberSaveBatchTracker {
public:
    using Completion = std::function<void(bool, const QString&)>;

    explicit FiberSaveBatchTracker(Completion completion)
        : _completion(std::move(completion))
    {
    }

    void addError(const QString& error)
    {
        if (!error.isEmpty()) {
            _errors.push_back(error);
        }
    }

    void finish()
    {
        if (!_completion) return;
        auto completion = std::move(_completion);
        // A moved-from std::function is valid but unspecified, not
        // necessarily empty; libc++ leaves small targets callable. Re-arm the
        // guard above, before the call, so a re-entrant completion is covered.
        _completion = nullptr;
        completion(_errors.isEmpty(), _errors.join('\n'));
    }

private:
    QStringList _errors;
    Completion _completion;
};
