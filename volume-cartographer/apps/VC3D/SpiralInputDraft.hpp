#pragma once

#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace vc3d::spiral {

// A serialized document or immutable working-copy manifest. Renderers and
// editor paths are deliberately not part of revision bookkeeping.
struct InputDraftContent {
    QJsonObject manifest;
    bool deleted = false;

    bool operator==(const InputDraftContent&) const = default;
};

struct InputDraftSnapshot {
    QString id;
    quint64 localRevision = 0;
    quint64 expectedAccepted = 0;
    InputDraftContent content;
};

class InputDraft {
public:
    InputDraft(QString id, InputDraftContent content, quint64 accepted = 0,
               std::optional<InputDraftContent> beforeDelete = std::nullopt)
        : _id(std::move(id)), _content(std::move(content)),
          _acceptedContent(_content), _beforeDelete(std::move(beforeDelete)), _accepted(accepted),
          _applied(accepted), _persisted(accepted),
          _acceptedLocal(accepted ? 1 : 0)
    {
    }

    void edit(InputDraftContent content, QString validationError = {})
    {
        _content = std::move(content);
        _validationError = std::move(validationError);
        ++_localRevision;
    }

    void remove()
    {
        if (_content.deleted) return;
        _beforeDelete = _content;
        _errorBeforeDelete = _validationError;
        edit({_content.manifest, true});
    }

    void restore()
    {
        if (!canRestore()) return;
        edit(*_beforeDelete, _errorBeforeDelete);
    }

    void discardLocalChanges()
    {
        edit(_acceptedContent);
        _acceptedLocal = _localRevision;
        if (!_content.deleted) _beforeDelete.reset();
    }

    InputDraftSnapshot snapshot() const
    {
        return {_id, _localRevision, _accepted, _content};
    }

    void acknowledgeAccepted(const InputDraftSnapshot& submitted, quint64 revision)
    {
        if (submitted.id != _id || submitted.localRevision > _localRevision
            || revision <= submitted.expectedAccepted)
            throw std::invalid_argument("Invalid draft acknowledgement");
        if (revision <= _accepted) return; // Delayed duplicate, never regress.
        _accepted = revision;
        _acceptedContent = submitted.content;
        // An old response must not make local revision N+1 clean.
        _acceptedLocal = submitted.localRevision;
    }

    void acknowledgeApplied(quint64 revision)
    {
        if (revision > _accepted)
            throw std::invalid_argument("Cannot apply an unaccepted revision");
        _applied = std::max(_applied, revision);
    }

    void acknowledgePersisted(quint64 revision)
    {
        if (revision > _applied)
            throw std::invalid_argument("Cannot persist an unapplied revision");
        _persisted = std::max(_persisted, revision);
    }

    void reconcileReviewedContent(InputDraftContent content, quint64 revision,
                                  quint64 applied, quint64 persisted, bool discard)
    {
        if (revision < _accepted) return;
        _acceptedContent = std::move(content);
        _accepted = revision;
        if (discard) { edit(_acceptedContent); _acceptedLocal = _localRevision; }
        reconcileServiceCursors(applied, persisted);
    }

    void reconcileServiceCursors(quint64 applied, quint64 persisted)
    {
        if (applied > _accepted || persisted > _accepted)
            throw std::invalid_argument("Service cursor exceeds accepted revision");
        _applied = applied;
        _persisted = persisted;
    }

    bool dirty() const { return _localRevision != _acceptedLocal; }
    bool valid() const { return _content.deleted || _validationError.isEmpty(); }
    bool deleted() const { return _content.deleted; }
    bool canRestore() const
    {
        return deleted() && _beforeDelete.has_value()
            && (dirty() || _accepted != _persisted);
    }
    bool needsCommit() const { return dirty() || _accepted != _persisted; }
    quint64 accepted() const { return _accepted; }
    quint64 applied() const { return _applied; }
    quint64 persisted() const { return _persisted; }
    const QString& validationError() const { return _validationError; }

private:
    QString _id;
    InputDraftContent _content;
    InputDraftContent _acceptedContent;
    std::optional<InputDraftContent> _beforeDelete;
    QString _validationError;
    QString _errorBeforeDelete;
    quint64 _accepted = 0;
    quint64 _applied = 0;
    quint64 _persisted = 0;
    quint64 _localRevision = 1;
    quint64 _acceptedLocal = 0;
};

struct InputDraftBatch {
    QString commandId;
    QVector<InputDraftSnapshot> entries;
    // An unknown transport outcome retains the same immutable submission.
    bool outcomeUnknown = false;
};

class InputDraftSubmission {
public:
    struct Selection {
        std::optional<InputDraftBatch> batch;
        QMap<QString, QString> errors;
    };

    Selection begin(const QString& commandId, const QVector<InputDraft*>& selected)
    {
        if (_active) return {_active, {}};
        Selection result;
        InputDraftBatch batch{commandId, {}, false};
        QStringList seen;
        if (commandId.isEmpty())
            throw std::invalid_argument("A submission needs a stable command id");
        for (const auto* draft : selected) {
            const auto snapshot = draft->snapshot();
            if (seen.contains(snapshot.id))
                throw std::invalid_argument("Duplicate draft in selected batch");
            seen.push_back(snapshot.id);
            if (!draft->valid()) {
                result.errors.insert(snapshot.id, draft->validationError());
            } else if (draft->dirty()) {
                batch.entries.push_back(snapshot);
            }
        }
        if (!result.errors.isEmpty()) return result;
        if (batch.entries.isEmpty()) return result;
        _active = batch;
        result.batch = batch;
        return result;
    }

    void transportInterrupted()
    {
        if (_active) _active->outcomeUnknown = true;
    }

    void reconciled(const QString& commandId)
    {
        if (_active && _active->commandId == commandId) _active.reset();
    }

    const std::optional<InputDraftBatch>& active() const { return _active; }

private:
    std::optional<InputDraftBatch> _active;
};

} // namespace vc3d::spiral
