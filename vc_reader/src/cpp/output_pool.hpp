// Recycled output buffers. A fresh allocation costs a page fault per 4 KiB page
// on first write (about as much as the read itself for warm data); buffers handed
// back when their arrays are freed are reused by the next read of the same size.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace vcr {

class OutputPool {
public:
    static OutputPool& instance()
    {
        static OutputPool* pool = new OutputPool();  // never destroyed: arrays may outlive static teardown
        return *pool;
    }

    void* acquire(std::size_t nbytes)
    {
        std::lock_guard lock(mutex_);
        void* p = nullptr;
        auto it = free_.find(nbytes);
        if (it != free_.end() && !it->second.empty()) {
            p = it->second.back();
            it->second.pop_back();
            retained_ -= nbytes;
            ++hits_;
        } else {
            p = std::aligned_alloc(kAlign, round_up(nbytes));
            if (!p)
                throw std::bad_alloc();
            ++misses_;
        }
        live_[p] = nbytes;
        return p;
    }

    // Takes back a buffer from acquire().
    void release(void* p)
    {
        std::size_t nbytes;
        {
            std::lock_guard lock(mutex_);
            auto it = live_.find(p);
            if (it == live_.end())
                return;
            nbytes = it->second;
            live_.erase(it);
            if (retained_ + nbytes <= max_bytes_) {
                free_[nbytes].push_back(p);
                retained_ += nbytes;
                return;
            }
        }
        std::free(p);
    }

    void set_max_bytes(std::size_t n)
    {
        std::vector<void*> drop;
        {
            std::lock_guard lock(mutex_);
            max_bytes_ = n;
            for (auto& [size, list] : free_) {
                while (retained_ > max_bytes_ && !list.empty()) {
                    drop.push_back(list.back());
                    list.pop_back();
                    retained_ -= size;
                }
            }
        }
        for (void* p : drop)
            std::free(p);
    }

    std::size_t max_bytes() { std::lock_guard lock(mutex_); return max_bytes_; }
    std::size_t retained() { std::lock_guard lock(mutex_); return retained_; }
    std::size_t hits() { std::lock_guard lock(mutex_); return hits_; }
    std::size_t misses() { std::lock_guard lock(mutex_); return misses_; }

private:
    static constexpr std::size_t kAlign = 64;
    static std::size_t round_up(std::size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

    std::size_t max_bytes_ = std::size_t{1} << 30;
    std::size_t retained_ = 0, hits_ = 0, misses_ = 0;
    std::unordered_map<std::size_t, std::vector<void*>> free_;
    std::unordered_map<void*, std::size_t> live_;
    std::mutex mutex_;
};

}  // namespace vcr
