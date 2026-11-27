// Zarr v3 objects and indexed shards, over the shared HTTP/local I/O path.
#pragma once
#include "chunk_io.hpp"
#include "http.hpp"
#include "lru.hpp"
#include "thread_pool.hpp"
#include "shard_index.hpp"
#include "channel_decode.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cctype>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vcr {

class ZarrStorage {
    struct IndexData {
        std::vector<std::uint8_t> bytes;
        std::uint64_t total = 0, start = 0;
        std::uint64_t epoch = 0;
        std::uint64_t hash = 0;
        std::string etag;
        std::string modified;
    };
    struct IndexLookup {
        std::shared_ptr<const IndexData> index;
        bool retained = false;
    };
    struct ActiveFreshVisit {
        std::string path;
        std::weak_ptr<const IndexData> index;
        std::uint64_t epoch = 0;
    };
    struct InFlightIndex {
        std::uint64_t epoch = 0;
        bool evicted = false;  // guarded by index_mutex_
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        std::shared_ptr<const IndexData> index;
        std::exception_ptr error;
    };
    struct Position { std::size_t inner, slot; };
    struct ContentRange { std::uint64_t first, last, total; };
    struct RetryShardRange : HttpRangeChanged { using HttpRangeChanged::HttpRangeChanged; };
    enum class RefreshAction { Abort, ForceRefresh, ReacquireUpdated };

    static constexpr int max_shard_refreshes = 1;
public:
    struct ShardRevision {
        bool valid = false;
        bool exists = true;
        std::string path;
        std::string etag;
        std::string modified;
        std::uint64_t total = 0;
        std::uint64_t index_hash = 0;
    };

    struct ShardPin {
        bool initialized = false;
        bool exists = true;
        std::shared_ptr<const IndexData> index;  // nullptr for 404 or foreign-provenance-only pin
        std::shared_ptr<const IndexData> superseded_index;  // retained index replaced by pre-commit refresh
        std::string etag;
        std::string modified;
        std::uint64_t total = 0;
        std::uint64_t index_hash = 0;
        bool retained = false;    // true iff `index` came from retained LRU without fresh suffix GET
        bool committed = false;   // true once ANY chunk or missing entry of this shard is delivered/consumed
        bool refreshing = false;  // true while one worker executes the bounded pre-commit refresh
        int refreshes = 0;        // bounded per-shard across the entire ReadSession (<= 1, only while !committed)
    };

    // Per-Array::run revision-consistency scope spanning all batch windows,
    // intra-plan ranges, foreign SingleFlight joins, and non-batched sharded reads.
    struct ReadSession {
        std::mutex mutex;
        std::condition_variable refresh_cv;
        std::unordered_map<std::string, std::vector<std::size_t>> planned_positions;
        std::unordered_map<std::string, ShardPin> shards;

        void plan_position(std::string path, std::size_t inner)
        {
            std::lock_guard lock(mutex);
            planned_positions[std::move(path)].push_back(inner);
        }

        // Atomically bind or verify a committed revision from a foreign SingleFlight claim
        // (or non-batched remote read). Throws HttpRangeChanged on incompatible revision.
        void verify_or_bind_committed(const std::string& path, bool exists,
                                      const std::string& etag, const std::string& modified,
                                      std::uint64_t total, std::uint64_t index_hash = 0)
        {
            std::lock_guard lock(mutex);
            auto& pin = shards[path];
            if (!pin.initialized) {
                pin.initialized = true;
                pin.exists = exists;
                pin.index = nullptr;
                pin.etag = etag;
                pin.modified = modified;
                pin.total = total;
                pin.index_hash = index_hash;
                pin.retained = false;
                pin.committed = true;
                return;
            }
            if (!foreign_revisions_compatible(pin.exists, pin.etag, pin.modified, pin.total, pin.index_hash,
                                              exists, etag, modified, total, index_hash)) {
                if (pin.committed)
                    throw HttpRangeChanged("shard changed during read: " + path);
                // Pre-commit foreign claim supersedes an uncommitted pin.
                pin.exists = exists;
                pin.index = nullptr;
                pin.etag = etag;
                pin.modified = modified;
                pin.total = total;
                pin.index_hash = index_hash;
                pin.retained = false;
                pin.committed = true;
                return;
            }
            if (pin.modified.empty() && !modified.empty())
                pin.modified = modified;
            if (pin.index_hash == 0 && index_hash != 0)
                pin.index_hash = index_hash;
            pin.committed = true;
        }
    };

    struct BatchChunkResult {
        bool found = false;
        std::vector<std::uint8_t> body;
        std::exception_ptr error;
        ShardRevision revision;
    };
    using ChunkCallback = std::function<void(std::size_t, BatchChunkResult)>;

    ZarrStorage(std::string root, std::string prefix, std::string separator, Index shards,
                bool index_end, bool index_crc, std::size_t chunk_bytes, HttpClient* http,
                std::atomic<std::uint64_t>& requests, std::atomic<std::uint64_t>& bytes)
        : root_(std::move(root)), prefix_(std::move(prefix)), separator_(std::move(separator)),
          layout_(std::move(shards), index_end, index_crc, chunk_bytes), http_(http),
          requests_(requests), bytes_(bytes)
    {}

    bool is_remote_sharded() const
    {
        return http_ && !layout_.ratios.empty();
    }

    std::pair<std::string, std::size_t> locate_chunk(const Index& coordinates) const
    {
        const auto [shard, position] = layout_.locate(coordinates);
        std::string shard_key;
        for (auto coordinate : shard) {
            if (!shard_key.empty()) shard_key += separator_;
            shard_key += std::to_string(coordinate);
        }
        return {joinLocation(root_, prefix_ + shard_key), position};
    }

    bool read(const std::string& key, std::vector<std::uint8_t>& body,
              ShardRevision* out_revision = nullptr, ReadSession* session = nullptr)
    {
        const auto [path, position] = location(key);
        if (layout_.ratios.empty()) {
            if (http_) {
                ++requests_;
                auto result = http_->get_stream(path, [&] { body.clear(); }, [&](const char* p, std::size_t n) {
                    if (n > layout_.maxPayload-body.size()) throw std::runtime_error("encoded chunk exceeds bounded payload size");
                    body.insert(body.end(), p, p+n);
                });
                bytes_ += result.bytes;
                return result.status == 200;
            }
            return local_read(path, body, std::nullopt, 0, layout_.maxPayload);
        }
        if (!http_) return LocalShard(utf8Path(path), layout_).readEntry(*position, body);
        const Position requested{*position, 0};
        bool force_refresh = false;
        RefreshTicket refresh_ticket;
        for (int attempt = 0; attempt <= max_shard_refreshes; ++attempt) {
            const auto lookup = acquire_session_index(path, {&requested, 1}, force_refresh, session);
            refresh_ticket.release();
            const auto rev = make_revision(path, lookup.index);
            if (!lookup.index) {
                commit_or_verify_session(path, lookup.index, session);
                if (out_revision) *out_revision = rev;
                return false;
            }
            const auto entry = layout_.entry(lookup.index->bytes, *position, lookup.index->total);
            if (!entry) {
                if (lookup.retained)
                    throw std::logic_error("unexpected missing entry in retained shard index");
                commit_or_verify_session(path, lookup.index, session);
                if (out_revision) *out_revision = rev;
                return false;
            }
            const auto [offset, length] = *entry;
            const bool can_retry = lookup.retained && attempt < max_shard_refreshes &&
                                   can_retry_session(path, lookup.index, session);
            try {
                read_shard_range(path, lookup, offset, length, body, can_retry);
                commit_or_verify_session(path, lookup.index, session);
                if (out_revision) *out_revision = rev;
                return true;
            } catch (const RetryShardRange&) {
                refresh_ticket = consume_session_refresh(path, lookup.index, session);
                if (refresh_ticket.action == RefreshAction::Abort)
                    throw;
                force_refresh = (refresh_ticket.action == RefreshAction::ForceRefresh);
            }
        }
        throw std::logic_error("shard refresh attempts exhausted");
    }

    static constexpr std::size_t max_merged_bytes = 512 << 10;
    static constexpr std::size_t max_batch_chunks = 16;

    bool can_batch() const
    {
        return http_ && !layout_.ratios.empty() && !sftp::isSftpUrl(root_);
    }

    // The caller owns these remote misses. Plan and read each shard independently
    // on the existing pool, delivering every slot exactly once as soon as its
    // chunk settles so healthy chunks are neither blocked nor poisoned by sibling
    // errors. When a ReadSession is supplied, shard revision pinning and
    // commit-once conflict reporting span all ranges and batch windows.
    void read_batch(const std::vector<std::string>& keys, ThreadPool& pool,
                    const ChunkCallback& deliver, ReadSession* session = nullptr)
    {
        if (keys.size() > max_batch_chunks)
            throw std::invalid_argument("shard batch exceeds chunk limit");
        struct ShardGroup {
            std::string path;
            std::vector<Position> positions;
        };
        std::map<std::string, std::vector<Position>> by_shard;
        for (std::size_t slot = 0; slot < keys.size(); ++slot) {
            try {
                auto [path, position] = location(keys[slot]);
                by_shard[std::move(path)].emplace_back(*position, slot);
            } catch (...) {
                deliver(slot, BatchChunkResult{false, {}, std::current_exception(), {}});
            }
        }
        std::vector<ShardGroup> groups;
        groups.reserve(by_shard.size());
        for (auto& [path, positions] : by_shard)
            groups.push_back({std::move(path), std::move(positions)});

        std::optional<ReadSession> local_session;
        ReadSession* active_session = session;
        if (!active_session) {
            local_session.emplace();
            active_session = &*local_session;
        }
        pool.parallel_for(groups.size(), [&](std::size_t s) {
            process_shard(groups[s].path, groups[s].positions, pool, deliver, active_session);
        });
    }

private:
    static std::uint64_t hash_index_bytes(const std::vector<std::uint8_t>& bytes)
    {
        std::uint64_t h = 14695981039346656037ULL;
        for (std::uint8_t b : bytes) {
            h ^= static_cast<std::uint64_t>(b);
            h *= 1099511628211ULL;
        }
        return h ? h : 1ULL;
    }

    struct RefreshTicket {
        ReadSession* session = nullptr;
        std::string path;
        RefreshAction action = RefreshAction::Abort;

        RefreshTicket() = default;
        RefreshTicket(ReadSession* s, std::string p, RefreshAction a)
            : session(s), path(std::move(p)), action(a) {}
        RefreshTicket(const RefreshTicket&) = delete;
        RefreshTicket& operator=(const RefreshTicket&) = delete;
        RefreshTicket(RefreshTicket&& other) noexcept
            : session(std::exchange(other.session, nullptr)),
              path(std::move(other.path)),
              action(other.action) {}
        RefreshTicket& operator=(RefreshTicket&& other) noexcept
        {
            if (this != &other) {
                release();
                session = std::exchange(other.session, nullptr);
                path = std::move(other.path);
                action = other.action;
            }
            return *this;
        }
        ~RefreshTicket() { release(); }

        void release()
        {
            if (!session || action != RefreshAction::ForceRefresh)
                return;
            ReadSession* target = std::exchange(session, nullptr);
            {
                std::lock_guard lock(target->mutex);
                if (auto it = target->shards.find(path); it != target->shards.end())
                    it->second.refreshing = false;
            }
            target->refresh_cv.notify_all();
        }
    };

    static void bind_pin(ShardPin& pin, const std::shared_ptr<const IndexData>& index,
                         bool retained, bool committed = false)
    {
        pin.initialized = true;
        pin.exists = static_cast<bool>(index);
        pin.index = index;
        pin.etag = index ? index->etag : std::string{};
        pin.modified = index ? index->modified : std::string{};
        pin.total = index ? index->total : 0;
        pin.index_hash = index ? index->hash : 0;
        pin.retained = retained;
        if (committed)
            pin.committed = true;
    }

    static bool foreign_revisions_compatible(bool a_exists, const std::string& a_etag,
                                             const std::string& a_modified, std::uint64_t a_total,
                                             std::uint64_t a_hash,
                                             bool b_exists, const std::string& b_etag,
                                             const std::string& b_modified, std::uint64_t b_total,
                                             std::uint64_t b_hash)
    {
        if (a_exists != b_exists) return false;
        if (!a_exists) return true;
        if (a_total != b_total || a_etag != b_etag) return false;
        if (is_strong_etag(a_etag)) return true;
        if ((!a_modified.empty() || !b_modified.empty()) && a_modified != b_modified)
            return false;
        if (a_hash != 0 && b_hash != 0 && a_hash != b_hash)
            return false;
        return true;
    }

    static bool revisions_compatible(const ShardPin& pin, const std::shared_ptr<const IndexData>& candidate)
    {
        const bool cand_exists = static_cast<bool>(candidate);
        if (pin.exists != cand_exists) return false;
        if (!pin.exists) return true;
        if (candidate->total != pin.total || candidate->etag != pin.etag) return false;
        if (is_strong_etag(pin.etag)) return true;
        // Weak or absent ETag: also check Last-Modified and shard index table bytes/hash.
        // These detect validator/size/index changes but cannot conclusively prove identical
        // payload bytes against same-size in-place overwrites with unchanged headers/index.
        if ((!pin.modified.empty() || !candidate->modified.empty()) && pin.modified != candidate->modified)
            return false;
        if (pin.index && candidate->bytes != pin.index->bytes) return false;
        if (pin.index_hash != 0 && candidate->hash != 0 && pin.index_hash != candidate->hash) return false;
        return true;
    }

    static ShardRevision make_revision(const std::string& path, const std::shared_ptr<const IndexData>& index)
    {
        ShardRevision rev;
        rev.valid = true;
        rev.exists = static_cast<bool>(index);
        rev.path = path;
        if (index) {
            rev.etag = index->etag;
            rev.modified = index->modified;
            rev.total = index->total;
            rev.index_hash = index->hash;
        }
        return rev;
    }

    bool is_active_fresh_locked(const std::string& path, const std::shared_ptr<const IndexData>& index)
    {
        if (!index || active_fresh_.epoch == 0 || active_fresh_.path != path)
            return false;
        const auto current = active_fresh_.index.lock();
        if (!current || current.get() != index.get())
            return false;
        const auto* cached = indexes_.peek(path);
        return cached && cached->get() == index.get();
    }

    bool is_active_fresh(const std::string& path, const std::shared_ptr<const IndexData>& index)
    {
        std::lock_guard lock(index_mutex_);
        return is_active_fresh_locked(path, index);
    }

    bool is_session_committed(const std::string& path, ReadSession* session) const
    {
        if (!session) return false;
        std::lock_guard lock(session->mutex);
        const auto it = session->shards.find(path);
        return it != session->shards.end() && it->second.committed;
    }

    bool can_retry_session(const std::string& path, const std::shared_ptr<const IndexData>& failed_index,
                           ReadSession* session) const
    {
        if (!session) return true;
        std::lock_guard lock(session->mutex);
        const auto it = session->shards.find(path);
        if (it == session->shards.end()) return true;
        const auto& pin = it->second;
        if (!pin.committed && pin.retained && pin.refreshes < max_shard_refreshes)
            return true;
        return failed_index && pin.superseded_index && pin.superseded_index.get() == failed_index.get();
    }

    RefreshTicket consume_session_refresh(const std::string& path,
                                          const std::shared_ptr<const IndexData>& failed_index,
                                          ReadSession* session) const
    {
        if (!session) return RefreshTicket{nullptr, {}, RefreshAction::ForceRefresh};
        std::unique_lock lock(session->mutex);
        auto& pin = session->shards[path];
        if (!pin.committed && pin.retained && pin.refreshes < max_shard_refreshes &&
            (!failed_index || pin.index.get() == failed_index.get())) {
            ++pin.refreshes;
            pin.superseded_index = failed_index ? failed_index : pin.index;
            pin.refreshing = true;
            pin.initialized = false;
            pin.index.reset();
            pin.retained = false;
            return RefreshTicket{session, path, RefreshAction::ForceRefresh};
        }
        if (failed_index && pin.superseded_index && pin.superseded_index.get() == failed_index.get()) {
            session->refresh_cv.wait(lock, [&] {
                const auto it = session->shards.find(path);
                return it == session->shards.end() || !it->second.refreshing;
            });
            const auto it = session->shards.find(path);
            if (it != session->shards.end() && it->second.initialized &&
                (!it->second.index || it->second.index.get() != failed_index.get()))
                return RefreshTicket{nullptr, {}, RefreshAction::ReacquireUpdated};
        }
        return RefreshTicket{nullptr, {}, RefreshAction::Abort};
    }

    // Atomically linearizes the first committed revision of `path` inside `session`,
    // and rejects any conflicting revision before data or missing markers can be delivered.
    void commit_or_verify_session(const std::string& path, const std::shared_ptr<const IndexData>& index,
                                  ReadSession* session) const
    {
        if (!session) return;
        std::lock_guard lock(session->mutex);
        auto& pin = session->shards[path];
        if (!pin.initialized) {
            bind_pin(pin, index, false, true);
            return;
        }
        if (!revisions_compatible(pin, index)) {
            if (pin.committed)
                throw HttpRangeChanged("shard changed during read: " + path);
            bind_pin(pin, index, false, true);
            return;
        }
        if (!pin.index && index)
            pin.index = index;
        if (pin.modified.empty() && index && !index->modified.empty())
            pin.modified = index->modified;
        if (pin.index_hash == 0 && index)
            pin.index_hash = index->hash;
        pin.committed = true;
    }

    IndexLookup acquire_session_index(const std::string& path, std::span<const Position> positions,
                                      bool force_refresh, ReadSession* session)
    {
        if (!session)
            return shard_index(path, positions, force_refresh);

        ShardPin pin_copy;
        std::vector<std::size_t> extra_positions;
        {
            std::unique_lock lock(session->mutex);
            if (!force_refresh) {
                session->refresh_cv.wait(lock, [&] {
                    const auto it = session->shards.find(path);
                    return it == session->shards.end() || !it->second.refreshing;
                });
            }
            if (const auto it = session->shards.find(path); it != session->shards.end())
                pin_copy = it->second;
            if (!pin_copy.initialized) {
                if (const auto pit = session->planned_positions.find(path); pit != session->planned_positions.end())
                    extra_positions = pit->second;
            }
        }

        if (!force_refresh && pin_copy.initialized) {
            // Case 1: Pinned to a confirmed 404 shard.
            if (!pin_copy.exists) {
                const auto reval = shard_index(path, positions, /*force_refresh=*/true);
                if (reval.index)
                    throw HttpRangeChanged("shard changed during read: " + path);
                return IndexLookup{nullptr, false};
            }
            // Case 2: Pinned to a known IndexData from an earlier window/thread in this ReadSession.
            if (pin_copy.index) {
                bool has_missing = false;
                bool has_invalid = false;
                for (const auto& p : positions) {
                    const auto state = layout_.probe_entry(pin_copy.index->bytes, p.inner, pin_copy.index->total).state;
                    if (state == ShardLayout::EntryState::Missing) has_missing = true;
                    else if (state == ShardLayout::EntryState::Invalid) has_invalid = true;
                }
                if (has_invalid && !pin_copy.committed && pin_copy.retained &&
                    pin_copy.refreshes < max_shard_refreshes) {
                    {
                        std::lock_guard lock(index_mutex_);
                        if (const auto* cached = indexes_.peek(path); cached && cached->get() == pin_copy.index.get())
                            indexes_.erase(path);
                    }
                    auto ticket = consume_session_refresh(path, pin_copy.index, session);
                    if (ticket.action == RefreshAction::ForceRefresh) {
                        auto refreshed = acquire_session_index(path, positions, /*force_refresh=*/true, session);
                        ticket.release();
                        return refreshed;
                    }
                    if (ticket.action == RefreshAction::ReacquireUpdated)
                        return acquire_session_index(path, positions, /*force_refresh=*/false, session);
                }
                if (!has_missing && is_strong_etag(pin_copy.etag)) {
                    // Strong ETag + present (or committed invalid) entries: reuse pinned index directly;
                    // server enforces exact revision via If-Match.
                    const bool retained = !pin_copy.committed && pin_copy.retained;
                    return IndexLookup{pin_copy.index, retained};
                }
                if (is_strong_etag(pin_copy.etag) && is_active_fresh(path, pin_copy.index)) {
                    return IndexLookup{pin_copy.index, false};
                }
                // Either missing entry needs confirmation or weak/absent ETag needs fresh check:
                // revalidate via fresh suffix GET and enforce revision compatibility against pin.
                const auto reval = shard_index(path, positions, /*force_refresh=*/true);
                if (!revisions_compatible(pin_copy, reval.index)) {
                    {
                        std::lock_guard lock(session->mutex);
                        auto& pin = session->shards[path];
                        if (!pin.committed && pin.retained && pin.refreshes < max_shard_refreshes) {
                            ++pin.refreshes;
                            pin.superseded_index = pin.index;
                            bind_pin(pin, reval.index, false);
                            return IndexLookup{reval.index, false};
                        }
                    }
                    throw HttpRangeChanged("shard changed during read: " + path);
                }
                return IndexLookup{reval.index, false};
            }
            // Case 3: Pinned from a foreign SingleFlight claim (revision provenance only, index == nullptr).
            auto lookup = shard_index(path, positions, /*force_refresh=*/false);
            if (!revisions_compatible(pin_copy, lookup.index)) {
                // Local retained/active cache may simply predate the foreign claim's revision:
                // check live server once with force_refresh before declaring a conflict.
                lookup = shard_index(path, positions, /*force_refresh=*/true);
                if (!revisions_compatible(pin_copy, lookup.index))
                    throw HttpRangeChanged("shard changed during read: " + path);
            }
            {
                std::lock_guard lock(session->mutex);
                auto& pin = session->shards[path];
                if (!pin.index && lookup.index && revisions_compatible(pin, lookup.index)) {
                    pin.index = lookup.index;
                    if (pin.modified.empty() && !lookup.index->modified.empty())
                        pin.modified = lookup.index->modified;
                    if (pin.index_hash == 0)
                        pin.index_hash = lookup.index->hash;
                }
            }
            return IndexLookup{lookup.index, false};
        }

        // First lookup for `path` in this ReadSession (or explicit pre-commit refresh).
        auto lookup = shard_index(path, positions, force_refresh, extra_positions);
        {
            std::unique_lock lock(session->mutex);
            if (!force_refresh) {
                session->refresh_cv.wait(lock, [&] {
                    const auto it = session->shards.find(path);
                    return it == session->shards.end() || !it->second.refreshing;
                });
            }
            auto& pin = session->shards[path];
            if (!pin.initialized) {
                bind_pin(pin, lookup.index, lookup.retained);
            } else if (!force_refresh && pin.superseded_index && lookup.index &&
                       pin.superseded_index.get() == lookup.index.get()) {
                // Sibling thread already refreshed away from this exact stale retained index.
                return IndexLookup{pin.index, false};
            } else if (!force_refresh && !pin.retained && lookup.retained) {
                // Sibling thread already established a fresh non-retained pin; adopt it.
                return IndexLookup{pin.index, false};
            } else if (!revisions_compatible(pin, lookup.index)) {
                if (pin.committed)
                    throw HttpRangeChanged("shard changed during read: " + path);
                bind_pin(pin, lookup.index, lookup.retained);
            } else {
                if (!pin.index && lookup.index)
                    pin.index = lookup.index;
                if (pin.modified.empty() && lookup.index && !lookup.index->modified.empty())
                    pin.modified = lookup.index->modified;
                if (pin.index_hash == 0 && lookup.index)
                    pin.index_hash = lookup.index->hash;
            }
        }
        return lookup;
    }

    void process_shard(const std::string& path, const std::vector<Position>& all_positions,
                       ThreadPool& pool, const ChunkCallback& deliver, ReadSession* session)
    {
        std::vector<Position> pending = all_positions;
        bool force_refresh = false;
        int local_refreshes = 0;
        RefreshTicket refresh_ticket;

        while (!pending.empty()) {
            IndexLookup lookup;
            try {
                lookup = acquire_session_index(path, pending, force_refresh, session);
                refresh_ticket.release();
            } catch (...) {
                refresh_ticket.release();
                auto err = std::current_exception();
                for (const auto& p : pending)
                    deliver(p.slot, BatchChunkResult{false, {}, err, {}});
                return;
            }
            const auto rev = make_revision(path, lookup.index);
            if (!lookup.index) {
                try {
                    commit_or_verify_session(path, nullptr, session);
                } catch (...) {
                    auto err = std::current_exception();
                    for (const auto& p : pending)
                        deliver(p.slot, BatchChunkResult{false, {}, err, {}});
                    return;
                }
                for (const auto& p : pending)
                    deliver(p.slot, BatchChunkResult{false, {}, nullptr, rev});
                return;
            }

            struct Member {
                std::uint64_t start, stop;
                std::size_t inner, slot;
            };
            struct Range {
                std::uint64_t start, stop;
                std::vector<Member> members;
            };
            struct RetryItem {
                std::size_t inner, slot;
                std::exception_ptr error;
            };

            std::vector<Member> members;
            members.reserve(pending.size());
            for (const auto& p : pending) {
                try {
                    if (auto entry = layout_.entry(lookup.index->bytes, p.inner, lookup.index->total)) {
                        members.push_back({entry->first, entry->first + entry->second, p.inner, p.slot});
                    } else {
                        if (lookup.retained)
                            throw std::logic_error("unexpected missing entry in retained shard index");
                        commit_or_verify_session(path, lookup.index, session);
                        deliver(p.slot, BatchChunkResult{false, {}, nullptr, rev});
                    }
                } catch (...) {
                    deliver(p.slot, BatchChunkResult{false, {}, std::current_exception(), {}});
                }
            }
            if (members.empty()) return;

            std::sort(members.begin(), members.end(), [](const Member& a, const Member& b) {
                return a.start < b.start;
            });

            std::vector<Range> ranges;
            const bool strong_etag = is_strong_etag(lookup.index->etag);
            for (const auto& member : members) {
                if (strong_etag && !ranges.empty()) {
                    auto& last = ranges.back();
                    const auto stop = std::max(last.stop, member.stop);
                    if (member.start <= last.stop && stop - last.start <= max_merged_bytes) {
                        last.stop = stop;
                        last.members.push_back(member);
                        continue;
                    }
                }
                ranges.push_back({member.start, member.stop, {member}});
            }

            std::mutex retry_mutex;
            std::vector<RetryItem> retry_items;
            const bool can_retry = lookup.retained && local_refreshes < max_shard_refreshes &&
                                   can_retry_session(path, lookup.index, session);

            auto deliver_committed = [&](std::size_t slot, std::vector<std::uint8_t> chunk_body) {
                try {
                    commit_or_verify_session(path, lookup.index, session);
                } catch (...) {
                    deliver(slot, BatchChunkResult{false, {}, std::current_exception(), {}});
                    return;
                }
                deliver(slot, BatchChunkResult{true, std::move(chunk_body), nullptr, rev});
            };

            auto handle_retry_error = [&](std::size_t inner, std::size_t slot, std::exception_ptr err) {
                // If this shard already committed a revision in this ReadSession, fail fast
                // immediately without waiting on sibling ranges or attempting Pass 2.
                if (is_session_committed(path, session)) {
                    deliver(slot, BatchChunkResult{false, {}, std::move(err), {}});
                    return;
                }
                std::lock_guard lock(retry_mutex);
                retry_items.push_back({inner, slot, std::move(err)});
            };

            auto read_member = [&](const Member& member) {
                std::vector<std::uint8_t> body;
                try {
                    read_shard_range(path, lookup, member.start, member.stop - member.start, body, can_retry);
                    deliver_committed(member.slot, std::move(body));
                } catch (const RetryShardRange&) {
                    handle_retry_error(member.inner, member.slot, std::current_exception());
                } catch (...) {
                    deliver(member.slot, BatchChunkResult{false, {}, std::current_exception(), {}});
                }
            };

            // Isolate every range and member so no error cancels healthy siblings.
            pool.parallel_for(ranges.size(), [&](std::size_t i) {
                const auto& range = ranges[i];
                if (range.members.size() == 1) {
                    read_member(range.members.front());
                    return;
                }
                std::vector<std::uint8_t> body;
                try {
                    read_shard_range(path, lookup, range.start, range.stop - range.start, body, can_retry);
                } catch (const HttpRangeRejected&) {
                    pool.parallel_for(range.members.size(), [&](std::size_t m) {
                        read_member(range.members[m]);
                    });
                    return;
                } catch (const RetryShardRange&) {
                    auto err = std::current_exception();
                    for (const auto& member : range.members)
                        handle_retry_error(member.inner, member.slot, err);
                    return;
                } catch (...) {
                    auto err = std::current_exception();
                    for (const auto& member : range.members)
                        deliver(member.slot, BatchChunkResult{false, {}, err, {}});
                    return;
                }
                for (const auto& member : range.members) {
                    std::vector<std::uint8_t> slice(
                        body.data() + (member.start - range.start),
                        body.data() + (member.stop - range.start));
                    deliver_committed(member.slot, std::move(slice));
                }
            });

            if (retry_items.empty())
                return;
            // Re-check commitment AFTER all sibling ranges in Pass 1 have joined:
            // if ANY sibling range committed a chunk on the old revision, Pass 2 is
            // strictly forbidden — fail all retry_items with their captured 412 error.
            if (local_refreshes >= max_shard_refreshes) {
                for (auto& item : retry_items)
                    deliver(item.slot, BatchChunkResult{false, {}, std::move(item.error), {}});
                return;
            }
            refresh_ticket = consume_session_refresh(path, lookup.index, session);
            if (refresh_ticket.action == RefreshAction::Abort) {
                for (auto& item : retry_items)
                    deliver(item.slot, BatchChunkResult{false, {}, std::move(item.error), {}});
                return;
            }
            pending.clear();
            pending.reserve(retry_items.size());
            for (const auto& item : retry_items)
                pending.push_back({item.inner, item.slot});
            force_refresh = (refresh_ticket.action == RefreshAction::ForceRefresh);
            ++local_refreshes;
        }
    }

public:
    std::optional<ChannelDecodeResult> read_channels(const std::string& key, const ChunkCodec& codec,
        std::size_t decodedBytes, std::size_t channelBytes, std::span<const std::size_t> requested,
        std::span<const std::size_t> relevant, ReadSession* session = nullptr,
        ShardRevision* out_revision = nullptr)
    {
        const auto [path, position] = location(key);
        if (!http_) {
            if (position) {
                LocalShard shard(utf8Path(path), layout_);
                if (shard.index.empty()) return std::nullopt;
                const auto entry = layout_.entry(shard.index, *position, shard.total);
                if (!entry) return std::nullopt;
                return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, entry->second,
                    [&](auto offset, auto length, auto dst) { shard.readRange(entry->first+offset, length, dst); });
            }
            File file = File::open_read(path);
            if (!file) return std::nullopt;
            platform::FileStat stat;
            if (platform::fstat(file.fd(), stat) != 0)
                throw std::system_error(errno, std::generic_category(), "stat " + path);
            if (stat.size < 0 || std::uint64_t(stat.size) > layout_.maxPayload)
                throw std::runtime_error("oversized channel payload: " + path);
            return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, std::size_t(stat.size),
                [&](auto offset, auto length, auto dst) { pread_exact(file.fd(), dst, length, offset, path); });
        }
        if (position) {
            const Position requested_pos{*position, 0};
            bool force_refresh = false;
            RefreshTicket refresh_ticket;
            for (int attempt = 0; attempt <= max_shard_refreshes; ++attempt) {
                auto lookup = acquire_session_index(path, {&requested_pos, 1}, force_refresh, session);
                refresh_ticket.release();
                if (!lookup.index) {
                    commit_or_verify_session(path, nullptr, session);
                    if (out_revision) *out_revision = make_revision(path, nullptr);
                    return std::nullopt;
                }
                const auto entry = layout_.entry(lookup.index->bytes, *position, lookup.index->total);
                if (!entry) {
                    if (lookup.retained)
                        throw std::logic_error("unexpected missing entry in retained shard index");
                    commit_or_verify_session(path, lookup.index, session);
                    if (out_revision) *out_revision = make_revision(path, lookup.index);
                    return std::nullopt;
                }
                const auto base = entry->first;
                const auto length = entry->second;
                const bool can_retry = lookup.retained && attempt < max_shard_refreshes &&
                                       can_retry_session(path, lookup.index, session);
                try {
                    auto decoded = decode_channels(codec, decodedBytes, channelBytes, requested, relevant, length,
                        [&](std::size_t offset, std::size_t size, std::uint8_t* dst) {
                            if (!size) return;
                            std::vector<std::uint8_t> body;
                            read_shard_range(path, lookup, base + offset, size, body, can_retry);
                            std::memcpy(dst, body.data(), size);
                        });
                    commit_or_verify_session(path, lookup.index, session);
                    if (out_revision) *out_revision = make_revision(path, lookup.index);
                    return decoded;
                } catch (const RetryShardRange&) {
                    refresh_ticket = consume_session_refresh(path, lookup.index, session);
                    if (refresh_ticket.action == RefreshAction::Abort)
                        throw;
                    force_refresh = (refresh_ticket.action == RefreshAction::ForceRefresh);
                }
            }
            throw std::logic_error("shard refresh attempts exhausted");
        }
        std::uint64_t base = 0, length = 0, total = 0;
        std::string etag, modified;
        std::vector<std::uint8_t> prefix;
        {
            // Discover size and revision without a HEAD request. For Blosc this
            // is also the complete codec header, reused by the range planner.
            const std::size_t probe = codec == Codec::Blosc ? BLOSC_MIN_HEADER_LENGTH : 1;
            ++requests_;
            auto response = http_->get_range(path, prefix, "0-"+std::to_string(probe-1), probe);
            bytes_ += response.bytes;
            if (response.status == 404) return std::nullopt;
            unsigned long long first = 0, last = 0, size = 0;
            if (std::sscanf(response.content_range.c_str(), "bytes %llu-%llu/%llu", &first, &last, &size) != 3 ||
                first != 0 || last != probe-1 || size < probe || size > layout_.maxPayload)
                throw std::runtime_error("invalid channel object Content-Range");
            length = total = size; etag = response.etag; modified = response.modified;
        }
        return decode_channels(codec, decodedBytes, channelBytes, requested, relevant, length,
            [&](std::size_t offset, std::size_t size, std::uint8_t* dst) {
                if (offset < prefix.size()) {
                    const auto reused = std::min(size, prefix.size()-offset);
                    std::memcpy(dst, prefix.data()+offset, reused);
                    dst += reused; offset += reused; size -= reused;
                }
                if (!size) return;
                std::vector<std::uint8_t> body;
                ++requests_;
                auto response = http_->get_range(path, body, std::to_string(base+offset)+"-"+
                    std::to_string(base+offset+size-1), size, etag);
                bytes_ += response.bytes;
                if (response.status == 404) throw std::runtime_error("channel object disappeared during ranged read");
                validate_range(parse_range(response.content_range), base+offset, size, total);
                if (!etag.empty() && !response.etag.empty() && etag != response.etag)
                    throw std::runtime_error("channel object changed during ranged read");
                if (!is_strong_etag(etag) && !modified.empty() && !response.modified.empty() && modified != response.modified)
                    throw std::runtime_error("channel object changed during ranged read");
                std::memcpy(dst, body.data(), size);
            });
    }

private:
    std::pair<std::string, std::optional<std::size_t>> location(const std::string& key) const
    {
        if (layout_.ratios.empty()) return {joinLocation(root_, prefix_ + key), std::nullopt};
        Index coordinates;
        std::size_t begin = 0;
        for (std::size_t axis = 0; axis < layout_.ratios.size(); ++axis) {
            const auto stop = key.find(separator_, begin);
            coordinates.push_back(std::stoll(key.substr(begin, stop - begin)));
            begin = stop == std::string::npos ? key.size() : stop + separator_.size();
        }
        const auto [shard, position] = layout_.locate(coordinates);
        std::string shard_key;
        for (auto coordinate : shard) {
            if (!shard_key.empty()) shard_key += separator_;
            shard_key += std::to_string(coordinate);
        }
        return {joinLocation(root_, prefix_ + shard_key), position};
    }
    static bool is_strong_etag(std::string_view tag)
    {
        return tag.size() >= 2 && tag.front() == '"' && tag.back() == '"' &&
            std::all_of(tag.begin()+1, tag.end()-1, [](unsigned char c) {
                return c == '!' || (c >= '#' && c <= '~') || c >= 0x80;
            });
    }

    static ContentRange parse_range(std::string_view header)
    {
        constexpr std::string_view unit = "bytes ";
        if (header.size() < unit.size() || !std::equal(unit.begin(), unit.end(), header.begin(),
                [](char expected, unsigned char received) { return expected == std::tolower(received); }))
            throw std::runtime_error("invalid HTTP Content-Range for shard read");
        const auto* cursor = header.data() + unit.size();
        const auto* end = header.data() + header.size();
        auto number = [&](std::uint64_t& value) {
            auto [next, error] = std::from_chars(cursor, end, value);
            cursor = next;
            return error == std::errc{};
        };
        ContentRange range{};
        if (!number(range.first) || cursor == end || *cursor++ != '-' ||
            !number(range.last) || cursor == end || *cursor++ != '/' ||
            !number(range.total) || cursor != end || range.last < range.first || range.last >= range.total)
            throw std::runtime_error("invalid HTTP Content-Range for shard read");
        return range;
    }

    static void validate_range(const ContentRange& range, std::uint64_t offset,
                               std::uint64_t length, std::uint64_t total)
    {
        if (range.first != offset || range.last-range.first+1 != length || range.total != total)
            throw std::runtime_error("invalid HTTP Content-Range for shard read");
    }

    void evict_shard(const std::string& path, const std::string& etag)
    {
        std::lock_guard lock(index_mutex_);
        ++visit_epoch_;
        if (pending_fetch_path_ == path)
            pending_fetch_path_.clear();
        const auto* cached = indexes_.peek(path);
        if (cached && (*cached)->etag == etag)
            indexes_.erase(path);
        if (active_fresh_.path == path) {
            const auto current = active_fresh_.index.lock();
            if (!current || current->etag == etag)
                active_fresh_ = {};
        }
        if (auto it = active_fetches_.find(path); it != active_fetches_.end()) {
            for (auto& flight : it->second)
                flight->evicted = true;
        }
    }

    [[noreturn]] void changed_shard(const std::string& path, const IndexLookup& lookup,
                                   bool can_retry, const HttpRangeChanged& error)
    {
        evict_shard(path, lookup.index->etag);
        if (can_retry) throw RetryShardRange(error.what());
        throw error;
    }

    void read_shard_range(const std::string& path, const IndexLookup& lookup,
                          std::uint64_t offset, std::size_t length, std::vector<std::uint8_t>& body,
                          bool can_retry)
    {
        const auto& etag = lookup.index->etag;
        const auto& modified = lookup.index->modified;
        const auto total = lookup.index->total;
        ++requests_;
        body.reserve(length);
        HttpResponse response;
        try {
            response = http_->get_range(path, body, std::to_string(offset) + "-" + std::to_string(offset + length - 1),
                                        length, etag);
        } catch (const HttpRangeChanged& error) {
            changed_shard(path, lookup, can_retry, error);
        }
        bytes_ += response.bytes;
        if (response.status == 404)
            changed_shard(path, lookup, can_retry, HttpRangeChanged("shard disappeared after index read"));
        if (!etag.empty() && !response.etag.empty() && response.etag != etag)
            changed_shard(path, lookup, can_retry, HttpRangeChanged("shard changed during ranged read"));
        if (!is_strong_etag(etag) && !modified.empty() && !response.modified.empty() && response.modified != modified)
            changed_shard(path, lookup, can_retry, HttpRangeChanged("shard changed during ranged read"));
        validate_range(parse_range(response.content_range), offset, length, total);
    }

    static bool local_read(const std::string& path, std::vector<std::uint8_t>& body,
                           std::optional<std::uint64_t> offset, std::size_t length, std::size_t maximum)
    {
        int fd = platform::open_file(utf8Path(path), O_RDONLY);
        if (fd < 0) {
            if (errno == ENOENT) return false;
            throw std::system_error(errno, std::generic_category(), "open " + path);
        }
        try {
            platform::FileStat stat;
            if (platform::fstat(fd, stat) != 0) throw std::system_error(errno, std::generic_category(), "fstat " + path);
            if (!offset) length = static_cast<std::size_t>(stat.size);
            if (length > maximum || offset.value_or(0) > std::uint64_t(stat.size) ||
                length > std::uint64_t(stat.size)-offset.value_or(0))
                throw std::runtime_error("truncated or oversized Zarr object: " + path);
            body.resize(length);
            pread_exact(fd, body.data(), length, static_cast<std::int64_t>(offset.value_or(0)), path);
        } catch (...) { ::close(fd); throw; }
        ::close(fd);
        return true;
    }

    void remove_flight_locked(const std::string& path, const InFlightIndex* target)
    {
        auto it = active_fetches_.find(path);
        if (it == active_fetches_.end()) return;
        auto& list = it->second;
        list.erase(std::remove_if(list.begin(), list.end(),
            [&](const std::shared_ptr<InFlightIndex>& item) { return item.get() == target; }),
            list.end());
        if (list.empty())
            active_fetches_.erase(it);
    }

    IndexLookup shard_index(const std::string& path, std::span<const Position> positions = {},
                            bool force_refresh = false,
                            std::span<const std::size_t> extra_positions = {})
    {
        std::shared_ptr<InFlightIndex> my_flight;
        std::shared_ptr<InFlightIndex> join_flight;
        {
            std::lock_guard lock(index_mutex_);
            if (!force_refresh && active_fresh_.epoch != 0 && active_fresh_.path == path) {
                const auto current = active_fresh_.index.lock();
                if (auto* cached = indexes_.get(path); current && cached && cached->get() == current.get())
                    return IndexLookup{current, false};
                active_fresh_ = {};
            }
            if (!force_refresh) {
                if (const auto* cached = indexes_.peek(path)) {
                    const auto& index = **cached;
                    // Reusing a retained index across shard visits requires a strong
                    // ETag and valid, present entries for every requested position
                    // (including planned positions in later windows of the same read).
                    // Never trust a missing entry from a retained index, never promote
                    // a retained index into active_fresh_, and erase/refresh on any
                    // invalid retained entry. Immediately revoke any previous
                    // active_fresh_ visit before returning so a pending visit never
                    // leaves a prior shard trusted.
                    bool has_invalid = false;
                    bool all_present = can_batch() && !positions.empty() && is_strong_etag(index.etag);
                    const auto check_inner = [&](std::size_t inner) {
                        const auto state = layout_.probe_entry(index.bytes, inner, index.total).state;
                        if (state == ShardLayout::EntryState::Invalid) {
                            has_invalid = true;
                            all_present = false;
                            return false;
                        }
                        if (state != ShardLayout::EntryState::Present)
                            all_present = false;
                        return true;
                    };
                    for (const auto& p : positions)
                        if (!check_inner(p.inner)) break;
                    if (!has_invalid) {
                        for (const auto inner : extra_positions)
                            if (!check_inner(inner)) break;
                    }
                    if (has_invalid) {
                        indexes_.erase(path);
                    } else if (all_present) {
                        auto retained = *indexes_.get(path);
                        ++visit_epoch_;
                        active_fresh_ = {};
                        pending_fetch_path_.clear();
                        return IndexLookup{std::move(retained), true};
                    }
                }
                if (active_fresh_.epoch == 0 && pending_fetch_path_ == path) {
                    if (auto it = active_fetches_.find(path); it != active_fetches_.end()) {
                        for (const auto& flight : it->second) {
                            if (!flight->evicted && flight->epoch == visit_epoch_) {
                                join_flight = flight;
                                break;
                            }
                        }
                    }
                }
            }
            if (!join_flight) {
                ++visit_epoch_;
                active_fresh_ = {};
                pending_fetch_path_ = path;
                my_flight = std::make_shared<InFlightIndex>();
                my_flight->epoch = visit_epoch_;
                active_fetches_[path].push_back(my_flight);
            }
        }

        if (join_flight) {
            std::unique_lock wait_lock(join_flight->mutex);
            join_flight->cv.wait(wait_lock, [&] { return join_flight->done; });
            if (join_flight->error)
                std::rethrow_exception(join_flight->error);
            return IndexLookup{join_flight->index, false};
        }

        std::shared_ptr<IndexData> fetched;
        std::exception_ptr error;
        try {
            auto index = std::make_shared<IndexData>();
            ++requests_;
            auto response = http_->get_range(path, index->bytes,
                layout_.indexEnd ? "-"+std::to_string(layout_.indexBytes) : "0-"+std::to_string(layout_.indexBytes-1),
                layout_.indexBytes);
            bytes_ += response.bytes;
            if (response.status != 404) {
                const auto range = parse_range(response.content_range);
                if (range.total < layout_.indexBytes)
                    throw std::runtime_error("invalid HTTP shard index Content-Range");
                index->start = layout_.indexEnd ? range.total - layout_.indexBytes : 0;
                index->total = range.total;
                index->epoch = my_flight->epoch;
                index->etag = response.etag;
                index->modified = response.modified;
                validate_range(range, index->start, layout_.indexBytes, range.total);
                layout_.validate(index->bytes, index->total);
                index->hash = hash_index_bytes(index->bytes);
                fetched = std::move(index);
            }
        } catch (...) {
            error = std::current_exception();
        }

        {
            std::lock_guard lock(index_mutex_);
            const bool evicted = my_flight->evicted;
            remove_flight_locked(path, my_flight.get());
            const bool still_current = !evicted && visit_epoch_ == my_flight->epoch;
            if (still_current && pending_fetch_path_ == path)
                pending_fetch_path_.clear();
            if (!error && !evicted) {
                if (!fetched) {
                    if (const auto* existing = indexes_.peek(path); existing && (*existing)->epoch < my_flight->epoch)
                        indexes_.erase(path);
                } else {
                    const auto cost = fetched->bytes.capacity() + fetched->etag.capacity() +
                                      fetched->modified.capacity() + 2 * path.capacity() + sizeof(IndexData);
                    const auto* existing = indexes_.peek(path);
                    const bool newer = !existing || (*existing)->epoch < my_flight->epoch;
                    if (cost <= indexes_.limit() && newer) {
                        indexes_.erase(path);
                        indexes_.put(path, fetched, cost);
                        if (still_current)
                            active_fresh_ = ActiveFreshVisit{path, fetched, my_flight->epoch};
                    }
                }
            }
        }

        {
            std::lock_guard notify_lock(my_flight->mutex);
            my_flight->index = fetched;
            my_flight->error = error;
            my_flight->done = true;
        }
        my_flight->cv.notify_all();

        if (error)
            std::rethrow_exception(error);
        return IndexLookup{std::move(fetched), false};
    }

    std::string root_, prefix_, separator_;
    ShardLayout layout_;
    HttpClient* http_;
    std::atomic<std::uint64_t>& requests_;
    std::atomic<std::uint64_t>& bytes_;
    std::mutex index_mutex_;
    std::uint64_t visit_epoch_ = 0;
    ActiveFreshVisit active_fresh_;
    std::string pending_fetch_path_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<InFlightIndex>>> active_fetches_;
    Lru<std::shared_ptr<const IndexData>> indexes_{8 << 20};
};
} // namespace vcr
