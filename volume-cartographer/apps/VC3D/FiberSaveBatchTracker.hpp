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

    void addJob()
    {
        ++_pendingJobs;
    }

    void addError(const QString& error)
    {
        if (!error.isEmpty()) {
            _errors.push_back(error);
        }
    }

    void finishJob(const QString& error = {})
    {
        addError(error);
        if (_pendingJobs > 0) {
            --_pendingJobs;
        }
        completeIfReady();
    }

    void finishScheduling()
    {
        _scheduling = false;
        completeIfReady();
    }

private:
    void completeIfReady()
    {
        if (_scheduling || _pendingJobs != 0 || !_completion) {
            return;
        }
        auto completion = std::move(_completion);
        completion(_errors.isEmpty(), _errors.join('\n'));
    }

    int _pendingJobs = 0;
    bool _scheduling = true;
    QStringList _errors;
    Completion _completion;
};
