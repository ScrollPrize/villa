#pragma once

#include "ObjectStoreRegions.hpp"

#include <functional>
#include <map>

// GUI-thread state for the workspace's asynchronous counts. A refresh during
// a read queues one replacement; failures retain the last successful counts.
class ProjectStoreCounts : public QObject
{
public:
    using Counts = std::map<std::string, int64_t>;
    using Work = std::function<Counts()>;
    using Identity = vc::objects::AttachmentIdentity;
    struct Snapshot {
        Counts values;
        QString error;
    };

    ProjectStoreCounts(QObject* parent, std::function<void()> changed)
        : QObject(parent), changed_(std::move(changed)) {}

    void retain(const std::vector<Identity>& attachments)
    {
        std::erase_if(states_, [&](const auto& entry) {
            return std::none_of(attachments.begin(), attachments.end(), [&](const auto& id) {
                return key(id) == entry.first;
            });
        });
    }

    const Snapshot* snapshot(const Identity& id) const
    {
        const auto found = states_.find(key(id));
        return found == states_.end() ? nullptr : &found->second->snapshot;
    }

    void request(const Identity& id, Work work)
    {
        auto& state = states_[key(id)];
        if (!state) state = std::make_shared<State>();
        if (state->running) {
            state->next = std::move(work);
            return;
        }
        start(id, state, std::move(work));
    }

private:
    struct State {
        Snapshot snapshot;
        bool running = false;
        Work next;
    };
    static std::pair<std::string, uint64_t> key(const Identity& id) { return {id.storeUuid, id.generation}; }

    void start(const Identity& id, std::shared_ptr<State> state, Work work)
    {
        state->running = true;
        vc3d::store::background<Counts>(this, std::move(work),
            [this, id, state](vc3d::store::Outcome<Counts> result) {
                const auto found = states_.find(key(id));
                if (found == states_.end() || found->second != state) return;
                state->running = false;
                if (state->next) {
                    auto next = std::move(state->next);
                    state->next = {};
                    start(id, state, std::move(next));
                    return;
                }
                if (result.error.isEmpty()) state->snapshot.values = std::move(result.value);
                state->snapshot.error = std::move(result.error);
                changed_();
            });
    }

    std::map<std::pair<std::string, uint64_t>, std::shared_ptr<State>> states_;
    std::function<void()> changed_;
};
