#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace vcr::transforms::cuda {
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
    // Scratch carved from shared pool chunks. The queue keeps every chunk until a flush has
    // launched the work using it; each buffer also keeps its chunk, so the stream-ordered free
    // happens after both, like a retired buffer's.
    std::vector<std::shared_ptr<void>> chunks_;
    std::shared_ptr<void> open_;
    std::byte* open_ptr_ = nullptr;
    std::size_t open_left_ = 0;
    cudaStream_t open_stream_ = nullptr;
    int open_device_ = -1;
    std::size_t lane_ = 0;
    std::size_t allocated_bytes_ = 0;
    bool enabled_;
    BatchQueue* previous_;
    void release() noexcept {
        chunks_.clear(); open_.reset(); open_ptr_ = nullptr; open_left_ = 0;
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
    void allocated(std::size_t bytes) { allocated_bytes_ += bytes; }
    std::size_t allocated_bytes() const { return allocated_bytes_; }
    template<class T> void append(T command) {
        lanes_[lane_].push_back(std::make_unique<Stored<T>>(std::move(command)));
    }
    void retire(void* ptr, bool pooled, cudaStream_t stream) { retired_.push_back({ptr,pooled,stream}); }
#if CUDART_VERSION >= 11020
    // A 256-byte aligned slice of a pool chunk on `stream`/`device` (the current device).
    std::pair<void*,std::shared_ptr<void>> carve(std::size_t bytes, cudaStream_t stream, int device) {
        bytes = (bytes + 255) / 256 * 256;
        if (!open_ || open_left_ < bytes || stream != open_stream_ || device != open_device_) {
            const std::size_t size = std::max<std::size_t>(bytes, std::size_t(8) << 20);
            void* chunk = nullptr;
            if (cudaMallocAsync(&chunk, size, stream) != cudaSuccess) throw std::runtime_error("CUDA transform: out of memory for batch scratch");
            open_ = std::shared_ptr<void>(chunk, [stream, device](void* q) {
                int previous = 0;
                if (cudaGetDevice(&previous) != cudaSuccess) return;
                if (previous != device) cudaSetDevice(device);
                cudaFreeAsync(q, stream);
                if (previous != device) cudaSetDevice(previous);
            });
            chunks_.push_back(open_);
            open_ptr_ = static_cast<std::byte*>(chunk); open_left_ = size; open_stream_ = stream; open_device_ = device;
        }
        void* result = open_ptr_; open_ptr_ += bytes; open_left_ -= bytes;
        return {result, open_};
    }
#endif
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
            allocated_bytes_ = 0;
        } catch (...) {
            for (auto& lane : lanes_) lane.clear();
            release();
            allocated_bytes_ = 0;
            current = this;
            throw;
        }
        current = this;
    }
};
}
