#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace vcr::augment::cuda {
// Each lane is an ordered stream of work for one independent sample. Only
// ready commands with the same implementation are combined; sample-local
// dependencies are preserved even when parameters select different stages.
class BatchQueue {
    struct Command {
        const void* type;
        void (*run)(const std::vector<const Command*>&);
        virtual bool compatible(const Command&) const = 0;
        virtual ~Command() = default;
    };
    template<class T> struct Stored final : Command {
        inline static const char token = 0;
        T value;
        bool compatible(const Command& other) const override {
            if (this->type!=other.type) return false;
            if constexpr (requires { value.compatible(value); }) return value.compatible(static_cast<const Stored&>(other).value);
            return true;
        }
        explicit Stored(T v) : value(std::move(v)) {
            this->type = &token;
            this->run = [](const std::vector<const Command*>& commands) {
                std::vector<const T*> values;
                values.reserve(commands.size());
                for (auto* c : commands) values.push_back(&static_cast<const Stored*>(c)->value);
                T::run(values);
            };
        }
    };
    struct Retired { void* ptr; bool pooled; cudaStream_t stream; };
    std::vector<std::vector<std::unique_ptr<Command>>> lanes_;
    std::vector<Retired> retired_;
    std::size_t lane_ = 0;
    bool enabled_;
    BatchQueue* previous_;
    void release() noexcept {
        for (auto p : retired_) {
#if CUDART_VERSION >= 11020
            if (p.pooled) cudaFreeAsync(p.ptr,p.stream); else
#endif
            cudaFree(p.ptr);
        }
        retired_.clear();
    }
public:
    inline static thread_local BatchQueue* current = nullptr;
    explicit BatchQueue(std::size_t count) : lanes_(count), enabled_(count>1), previous_(current) {
        current=enabled_?this:nullptr;
    }
    BatchQueue(const BatchQueue&) = delete;
    ~BatchQueue() { current = previous_; release(); }
    void lane(std::size_t i) { lane_ = i; }
    template<class T> void append(T command) {
        lanes_[lane_].push_back(std::make_unique<Stored<T>>(std::move(command)));
    }
    void retire(void* ptr, bool pooled, cudaStream_t stream) { retired_.push_back({ptr,pooled,stream}); }
    void flush() {
        if(!enabled_)return;
        current = nullptr;
        try {
            std::vector<std::size_t> positions(lanes_.size());
            std::vector<const Command*> ready;
            for (;;) {
                const void* type = nullptr;
                ready.clear();
                for (std::size_t i = 0; i < lanes_.size(); ++i) {
                    if (positions[i] == lanes_[i].size()) continue;
                    const auto* command = lanes_[i][positions[i]].get();
                    if (!type) type = command->type;
                    if (command->type == type && (ready.empty() || ready.front()->compatible(*command)) && ready.size() < 65535) {
                        ready.push_back(command);
                        ++positions[i];
                    }
                }
                if (ready.empty()) break;
                ready.front()->run(ready);
            }
            for (auto& lane : lanes_) lane.clear();
            release();
        } catch (...) {
            for (auto& lane : lanes_) lane.clear();
            release();
            current = this;
            throw;
        }
        current = this;
    }
};
}
