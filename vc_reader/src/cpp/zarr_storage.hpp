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
#include <vector>

namespace vcr {

class ZarrStorage {
    struct IndexData {
        std::vector<std::uint8_t> bytes;
        std::uint64_t total = 0, start = 0;
        std::uint64_t epoch = 0;
        std::string etag;
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
    struct RetryShardRange : std::runtime_error { using std::runtime_error::runtime_error; };

    static constexpr int max_shard_refreshes = 1;
public:
    struct BatchChunkResult {
        bool found = false;
        std::vector<std::uint8_t> body;
        std::exception_ptr error;
    };
    using ChunkCallback = std::function<void(std::size_t, BatchChunkResult)>;

    ZarrStorage(std::string root, std::string prefix, std::string separator, Index shards,
                bool index_end, bool index_crc, std::size_t chunk_bytes, HttpClient* http,
                std::atomic<std::uint64_t>& requests, std::atomic<std::uint64_t>& bytes)
        : root_(std::move(root)), prefix_(std::move(prefix)), separator_(std::move(separator)),
          layout_(std::move(shards), index_end, index_crc, chunk_bytes), http_(http),
          requests_(requests), bytes_(bytes)
    {}

    bool read(const std::string& key, std::vector<std::uint8_t>& body)
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
        for (int attempt = 0; attempt <= max_shard_refreshes; ++attempt) {
            const auto lookup = shard_index(path, {&requested, 1}, force_refresh);
            if (!lookup.index) return false;
            const auto entry = layout_.entry(lookup.index->bytes, *position, lookup.index->total);
            if (!entry) return false;
            const auto [offset, length] = *entry;
            const bool can_retry = lookup.retained && attempt < max_shard_refreshes;
            try {
                read_shard_range(path, lookup, offset, length, body, can_retry);
                return true;
            } catch (const RetryShardRange&) {
                force_refresh = true;
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
    // errors. Stale retained indexes refresh per shard with a bounded retry limit.
    void read_batch(const std::vector<std::string>& keys, ThreadPool& pool, const ChunkCallback& deliver)
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
                deliver(slot, BatchChunkResult{false, {}, std::current_exception()});
            }
        }
        std::vector<ShardGroup> groups;
        groups.reserve(by_shard.size());
        for (auto& [path, positions] : by_shard)
            groups.push_back({std::move(path), std::move(positions)});

        pool.parallel_for(groups.size(), [&](std::size_t s) {
            process_shard(groups[s].path, groups[s].positions, pool, deliver);
        });
    }

private:
    void process_shard(const std::string& path, const std::vector<Position>& all_positions,
                       ThreadPool& pool, const ChunkCallback& deliver)
    {
        std::vector<Position> pending = all_positions;
        bool force_refresh = false;
        int refreshes = 0;

        while (!pending.empty()) {
            IndexLookup lookup;
            try {
                lookup = shard_index(path, pending, force_refresh);
            } catch (...) {
                auto err = std::current_exception();
                for (const auto& p : pending)
                    deliver(p.slot, BatchChunkResult{false, {}, err});
                return;
            }
            if (!lookup.index) {
                for (const auto& p : pending)
                    deliver(p.slot, BatchChunkResult{false, {}, nullptr});
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

            std::vector<Member> members;
            members.reserve(pending.size());
            for (const auto& p : pending) {
                try {
                    if (auto entry = layout_.entry(lookup.index->bytes, p.inner, lookup.index->total)) {
                        members.push_back({entry->first, entry->first + entry->second, p.inner, p.slot});
                    } else {
                        if (lookup.retained)
                            throw std::logic_error("unexpected missing entry in retained shard index");
                        deliver(p.slot, BatchChunkResult{false, {}, nullptr});
                    }
                } catch (...) {
                    deliver(p.slot, BatchChunkResult{false, {}, std::current_exception()});
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
            std::vector<Position> retry_positions;
            const bool can_retry = lookup.retained && refreshes < max_shard_refreshes;

            auto read_member = [&](const Member& member) {
                std::vector<std::uint8_t> body;
                try {
                    read_shard_range(path, lookup, member.start, member.stop - member.start, body, can_retry);
                    deliver(member.slot, BatchChunkResult{true, std::move(body), nullptr});
                } catch (const RetryShardRange&) {
                    std::lock_guard lock(retry_mutex);
                    retry_positions.push_back({member.inner, member.slot});
                } catch (...) {
                    deliver(member.slot, BatchChunkResult{false, {}, std::current_exception()});
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
                    std::lock_guard lock(retry_mutex);
                    for (const auto& member : range.members)
                        retry_positions.push_back({member.inner, member.slot});
                    return;
                } catch (...) {
                    auto err = std::current_exception();
                    for (const auto& member : range.members)
                        deliver(member.slot, BatchChunkResult{false, {}, err});
                    return;
                }
                for (const auto& member : range.members) {
                    std::vector<std::uint8_t> slice(
                        body.data() + (member.start - range.start),
                        body.data() + (member.stop - range.start));
                    deliver(member.slot, BatchChunkResult{true, std::move(slice), nullptr});
                }
            });

            if (retry_positions.empty())
                return;
            pending = std::move(retry_positions);
            force_refresh = true;
            ++refreshes;
        }
    }

public:
    std::optional<ChannelDecodeResult> read_channels(const std::string& key, const ChunkCodec& codec,
        std::size_t decodedBytes, std::size_t channelBytes, std::span<const std::size_t> requested,
        std::span<const std::size_t> relevant)
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
        std::uint64_t base = 0, length = 0, total = 0;
        std::string etag;
        std::vector<std::uint8_t> prefix;
        if (position) {
            auto index = shard_index(path).index;
            if (!index) return std::nullopt;
            const auto entry = layout_.entry(index->bytes, *position, index->total);
            if (!entry) return std::nullopt;
            base = entry->first; length = entry->second; total = index->total; etag = index->etag;
        } else {
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
            length = total = size; etag = response.etag;
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
                            bool force_refresh = false)
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
                    // ETag and valid, present entries for every requested position.
                    // Never trust a missing entry from a retained index, never promote
                    // a retained index into active_fresh_, and erase/refresh on any
                    // invalid retained entry. Immediately revoke any previous
                    // active_fresh_ visit before returning so a pending visit never
                    // leaves a prior shard trusted.
                    bool has_invalid = false;
                    bool all_present = can_batch() && !positions.empty() && is_strong_etag(index.etag);
                    for (const auto& p : positions) {
                        const auto state = layout_.probe_entry(index.bytes, p.inner, index.total).state;
                        if (state == ShardLayout::EntryState::Invalid) {
                            has_invalid = true;
                            all_present = false;
                            break;
                        }
                        if (state != ShardLayout::EntryState::Present)
                            all_present = false;
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
                validate_range(range, index->start, layout_.indexBytes, range.total);
                layout_.validate(index->bytes, index->total);
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
                                      2 * path.capacity() + sizeof(IndexData);
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
