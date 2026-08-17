#include <utils/thread_pool.hpp>

#if defined(__linux__) && defined(__x86_64__) && __has_include(<valgrind/callgrind.h>)
#include <valgrind/callgrind.h>
#define VC_HAS_CALLGRIND_CLIENT 1
#else
#define VC_HAS_CALLGRIND_CLIENT 0
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <latch>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

enum class Mode {
    Serial,
    Futures,
    FuturesGateOpen,
    FuturesGateClosed,
    Batch,
    Lifecycle,
};

enum class WorkKind {
    Alu,
    FloatingPoint,
    Branch,
    PointerChase,
    StreamRead,
    StreamWrite,
    ReadFour,
    ReadEight,
    WriteEight,
    LineReadOne,
    LineReadEight,
    LineWriteOne,
    LineWriteEight,
    LineReadOneWriteOne,
    LineReadEightWriteOne,
    LineReadOneWriteEight,
    LineReadEightWriteEight,
    CacheRead,
    CacheWrite,
    GridSample,
    MixedGridPhase,
    MixedGridRandom,
    Mixed,
};

struct Args {
    Mode mode = Mode::Futures;
    std::size_t workers = 4;
    std::size_t tasks = 4;
    std::size_t workIterations = 10000;
    std::size_t taskIterationSkew = 0;
    std::size_t rounds = 4096;
    std::size_t warmupRounds = 32;
    std::size_t idleNanoseconds = 0;
    WorkKind workKind = WorkKind::Alu;
    std::size_t workingSetBytes = 256 * 1024;
};

struct WorkData {
    std::vector<std::uint32_t> next;
    std::vector<std::uint64_t> values;
    std::size_t cacheStride = 1;
};

struct Measurement {
    double wallSeconds = 0.0;
    double cpuSeconds = 0.0;
    double rawDispatchSeconds = 0.0;
    double clockOverheadSeconds = 0.0;
    double actualIdleSeconds = 0.0;
    std::uint64_t checksum = 0;
};

const char* modeName(Mode mode)
{
    switch (mode) {
        case Mode::Serial:
            return "serial";
        case Mode::Futures:
            return "futures";
        case Mode::FuturesGateOpen:
            return "futures-gate-open";
        case Mode::FuturesGateClosed:
            return "futures-gate-closed";
        case Mode::Batch:
            return "batch";
        case Mode::Lifecycle:
            return "lifecycle";
    }
    throw std::runtime_error("unknown dispatch mode");
}

Mode parseMode(std::string_view value)
{
    if (value == "serial")
        return Mode::Serial;
    if (value == "futures")
        return Mode::Futures;
    if (value == "futures-gate-open")
        return Mode::FuturesGateOpen;
    if (value == "futures-gate-closed")
        return Mode::FuturesGateClosed;
    if (value == "batch")
        return Mode::Batch;
    if (value == "lifecycle")
        return Mode::Lifecycle;
    throw std::runtime_error("unknown mode: " + std::string(value));
}

const char* workKindName(WorkKind kind)
{
    switch (kind) {
        case WorkKind::Alu:
            return "alu";
        case WorkKind::FloatingPoint:
            return "fp";
        case WorkKind::Branch:
            return "branch";
        case WorkKind::PointerChase:
            return "pointer";
        case WorkKind::StreamRead:
            return "stream-read";
        case WorkKind::StreamWrite:
            return "stream-write";
        case WorkKind::ReadFour:
            return "read-four";
        case WorkKind::ReadEight:
            return "read-eight";
        case WorkKind::WriteEight:
            return "write-eight";
        case WorkKind::LineReadOne:
            return "line-read-one";
        case WorkKind::LineReadEight:
            return "line-read-eight";
        case WorkKind::LineWriteOne:
            return "line-write-one";
        case WorkKind::LineWriteEight:
            return "line-write-eight";
        case WorkKind::LineReadOneWriteOne:
            return "line-r1-w1";
        case WorkKind::LineReadEightWriteOne:
            return "line-r8-w1";
        case WorkKind::LineReadOneWriteEight:
            return "line-r1-w8";
        case WorkKind::LineReadEightWriteEight:
            return "line-r8-w8";
        case WorkKind::CacheRead:
            return "cache-read";
        case WorkKind::CacheWrite:
            return "cache-write";
        case WorkKind::GridSample:
            return "grid-sample";
        case WorkKind::MixedGridPhase:
            return "mixed-grid-phase";
        case WorkKind::MixedGridRandom:
            return "mixed-grid-random";
        case WorkKind::Mixed:
            return "mixed";
    }
    throw std::runtime_error("unknown work kind");
}

WorkKind parseWorkKind(std::string_view value)
{
    if (value == "alu")
        return WorkKind::Alu;
    if (value == "fp")
        return WorkKind::FloatingPoint;
    if (value == "branch")
        return WorkKind::Branch;
    if (value == "pointer")
        return WorkKind::PointerChase;
    if (value == "stream-read")
        return WorkKind::StreamRead;
    if (value == "stream-write")
        return WorkKind::StreamWrite;
    if (value == "read-four")
        return WorkKind::ReadFour;
    if (value == "read-eight")
        return WorkKind::ReadEight;
    if (value == "write-eight")
        return WorkKind::WriteEight;
    if (value == "line-read-one")
        return WorkKind::LineReadOne;
    if (value == "line-read-eight")
        return WorkKind::LineReadEight;
    if (value == "line-write-one")
        return WorkKind::LineWriteOne;
    if (value == "line-write-eight")
        return WorkKind::LineWriteEight;
    if (value == "line-r1-w1")
        return WorkKind::LineReadOneWriteOne;
    if (value == "line-r8-w1")
        return WorkKind::LineReadEightWriteOne;
    if (value == "line-r1-w8")
        return WorkKind::LineReadOneWriteEight;
    if (value == "line-r8-w8")
        return WorkKind::LineReadEightWriteEight;
    if (value == "cache-read")
        return WorkKind::CacheRead;
    if (value == "cache-write")
        return WorkKind::CacheWrite;
    if (value == "grid-sample")
        return WorkKind::GridSample;
    if (value == "mixed-grid-phase")
        return WorkKind::MixedGridPhase;
    if (value == "mixed-grid-random")
        return WorkKind::MixedGridRandom;
    if (value == "mixed")
        return WorkKind::Mixed;
    throw std::runtime_error("unknown work kind: " + std::string(value));
}

std::size_t parseCount(std::string_view option, std::string_view value, bool allowZero = false)
{
    if (value.empty() || value.front() == '-')
        throw std::runtime_error(std::string(option) + " has an invalid value");
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(std::string(value), &consumed);
    if (consumed != value.size() || (!allowZero && parsed == 0) || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(option) + (allowZero ? " must be nonnegative" : " must be positive"));
    }
    return static_cast<std::size_t>(parsed);
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        auto value = [&]() -> std::string_view {
            if (++i >= argc)
                throw std::runtime_error(std::string(option) + " requires a value");
            return argv[i];
        };
        if (option == "--mode")
            args.mode = parseMode(value());
        else if (option == "--workers")
            args.workers = parseCount(option, value());
        else if (option == "--tasks")
            args.tasks = parseCount(option, value());
        else if (option == "--work-iterations")
            args.workIterations = parseCount(option, value(), true);
        else if (option == "--task-iteration-skew")
            args.taskIterationSkew = parseCount(option, value(), true);
        else if (option == "--rounds")
            args.rounds = parseCount(option, value());
        else if (option == "--warmup-rounds")
            args.warmupRounds = parseCount(option, value(), true);
        else if (option == "--idle-nanoseconds")
            args.idleNanoseconds = parseCount(option, value(), true);
        else if (option == "--work-kind")
            args.workKind = parseWorkKind(value());
        else if (option == "--working-set-bytes")
            args.workingSetBytes = parseCount(option, value());
        else
            throw std::runtime_error("unknown option: " + std::string(option));
    }
    return args;
}

#if defined(__GNUC__) || defined(__clang__)
#define VC_NOINLINE __attribute__((noinline))
#else
#define VC_NOINLINE
#endif

VC_NOINLINE std::uint64_t branchEven(std::uint64_t value)
{
    return (value ^ 0xd6e8feb86659fd93ULL) * 0x9e3779b185ebca87ULL;
}

VC_NOINLINE std::uint64_t branchOdd(std::uint64_t value)
{
    return (value + 0xa0761d6478bd642fULL) ^ (value >> 23U);
}

std::uint64_t deterministicWork(std::uint64_t state, std::size_t iterations)
{
    for (std::size_t i = 0; i < iterations; ++i) {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
    }
    return state;
}

inline std::uint64_t readCacheLine(std::uint64_t state, const std::uint64_t* values, bool readEight)
{
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (readEight) {
        asm volatile(
            "xorq 0(%[values]), %[state]\n\t"
            "xorq 8(%[values]), %[state]\n\t"
            "xorq 16(%[values]), %[state]\n\t"
            "xorq 24(%[values]), %[state]\n\t"
            "xorq 32(%[values]), %[state]\n\t"
            "xorq 40(%[values]), %[state]\n\t"
            "xorq 48(%[values]), %[state]\n\t"
            "xorq 56(%[values]), %[state]"
            : [state] "+r"(state)
            : [values] "r"(values)
            : "memory");
    } else {
        asm volatile("xorq 0(%[values]), %[state]" : [state] "+r"(state) : [values] "r"(values) : "memory");
    }
#else
    state ^= values[0];
    if (readEight) {
        for (std::size_t offset = 1; offset < 8; ++offset)
            state ^= values[offset];
    }
#endif
    return state;
}

inline void writeCacheLine(std::uint64_t state, std::uint64_t* values, bool writeEight)
{
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (writeEight) {
        asm volatile(
            "movq %[state], 0(%[values])\n\t"
            "movq %[state], 8(%[values])\n\t"
            "movq %[state], 16(%[values])\n\t"
            "movq %[state], 24(%[values])\n\t"
            "movq %[state], 32(%[values])\n\t"
            "movq %[state], 40(%[values])\n\t"
            "movq %[state], 48(%[values])\n\t"
            "movq %[state], 56(%[values])"
            :
            : [state] "r"(state), [values] "r"(values)
            : "memory");
    } else {
        asm volatile("movq %[state], 0(%[values])" : : [state] "r"(state), [values] "r"(values) : "memory");
    }
#else
    values[0] = state;
    if (writeEight) {
        for (std::size_t offset = 1; offset < 8; ++offset)
            values[offset] = state;
    }
#endif
}

inline void readWriteCacheLine(std::uint64_t state, std::uint64_t stored, std::uint64_t* values, bool readEight, bool writeEight)
{
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (readEight && writeEight) {
        asm volatile(
            "xorq 0(%[values]), %[state]\n\t"
            "xorq 8(%[values]), %[state]\n\t"
            "xorq 16(%[values]), %[state]\n\t"
            "xorq 24(%[values]), %[state]\n\t"
            "xorq 32(%[values]), %[state]\n\t"
            "xorq 40(%[values]), %[state]\n\t"
            "xorq 48(%[values]), %[state]\n\t"
            "xorq 56(%[values]), %[state]\n\t"
            "movq %[stored], 0(%[values])\n\t"
            "movq %[stored], 8(%[values])\n\t"
            "movq %[stored], 16(%[values])\n\t"
            "movq %[stored], 24(%[values])\n\t"
            "movq %[stored], 32(%[values])\n\t"
            "movq %[stored], 40(%[values])\n\t"
            "movq %[stored], 48(%[values])\n\t"
            "movq %[stored], 56(%[values])"
            : [state] "+r"(state)
            : [stored] "r"(stored), [values] "r"(values)
            : "memory");
    } else if (readEight) {
        asm volatile(
            "xorq 0(%[values]), %[state]\n\t"
            "xorq 8(%[values]), %[state]\n\t"
            "xorq 16(%[values]), %[state]\n\t"
            "xorq 24(%[values]), %[state]\n\t"
            "xorq 32(%[values]), %[state]\n\t"
            "xorq 40(%[values]), %[state]\n\t"
            "xorq 48(%[values]), %[state]\n\t"
            "xorq 56(%[values]), %[state]\n\t"
            "movq %[stored], 0(%[values])"
            : [state] "+r"(state)
            : [stored] "r"(stored), [values] "r"(values)
            : "memory");
    } else if (writeEight) {
        asm volatile(
            "xorq 0(%[values]), %[state]\n\t"
            "movq %[stored], 0(%[values])\n\t"
            "movq %[stored], 8(%[values])\n\t"
            "movq %[stored], 16(%[values])\n\t"
            "movq %[stored], 24(%[values])\n\t"
            "movq %[stored], 32(%[values])\n\t"
            "movq %[stored], 40(%[values])\n\t"
            "movq %[stored], 48(%[values])\n\t"
            "movq %[stored], 56(%[values])"
            : [state] "+r"(state)
            : [stored] "r"(stored), [values] "r"(values)
            : "memory");
    } else {
        asm volatile(
            "xorq 0(%[values]), %[state]\n\t"
            "movq %[stored], 0(%[values])"
            : [state] "+r"(state)
            : [stored] "r"(stored), [values] "r"(values)
            : "memory");
    }
#else
    (void)readCacheLine(state, values, readEight);
    writeCacheLine(stored, values, writeEight);
#endif
}

double trilinearSample(const std::vector<std::uint64_t>& values, std::size_t index, std::uint64_t state)
{
    const double fx = double((state >> 8U) & 0xffU) / 255.0;
    const double fy = double((state >> 16U) & 0xffU) / 255.0;
    const double fz = double((state >> 24U) & 0xffU) / 255.0;
    const auto sample = [&](std::size_t offset) { return double(values[index + offset] & 0xffU); };
    const double x00 = sample(0) + fx * (sample(1) - sample(0));
    const double x10 = sample(2) + fx * (sample(3) - sample(2));
    const double x01 = sample(4) + fx * (sample(5) - sample(4));
    const double x11 = sample(6) + fx * (sample(7) - sample(6));
    const double y0 = x00 + fy * (x10 - x00);
    const double y1 = x01 + fy * (x11 - x01);
    return y0 + fz * (y1 - y0);
}

WorkData makeWorkData(std::size_t bytes, std::size_t task)
{
    const std::size_t count = std::max<std::size_t>(64, bytes / sizeof(std::uint64_t));
    WorkData data;
    data.next.resize(count);
    data.values.resize(count);
    std::vector<std::uint32_t> order(count);
    for (std::size_t i = 0; i < count; ++i) {
        order[i] = std::uint32_t(i);
        data.values[i] = 0x9e3779b97f4a7c15ULL * (i + 1 + task * 17);
    }
    std::uint64_t state = 0xa0761d6478bd642fULL ^ task;
    for (std::size_t i = count - 1; i > 0; --i) {
        state = deterministicWork(state, 1);
        std::swap(order[i], order[state % (i + 1)]);
    }
    for (std::size_t i = 0; i < count; ++i)
        data.next[order[i]] = order[(i + 1) % count];
    data.cacheStride = count / 2 | 1U;
    while (std::gcd(data.cacheStride, count) != 1)
        data.cacheStride += 2;
    return data;
}

std::uint64_t runWork(WorkKind kind, std::uint64_t state, std::size_t iterations, WorkData& data)
{
    if (kind == WorkKind::Alu)
        return deterministicWork(state, iterations);
    if (kind == WorkKind::FloatingPoint) {
        double value = double((state & 0xffffU) + 1U) / 65536.0;
        for (std::size_t i = 0; i < iterations; ++i)
            value = value * 1.00000011920928955078125 + 0.000000059604644775390625;
        return std::bit_cast<std::uint64_t>(value) ^ state;
    }
    if (kind == WorkKind::Branch) {
        for (std::size_t i = 0; i < iterations; ++i) {
            state = deterministicWork(state, 1);
            state = (state & 1U) ? branchOdd(state) : branchEven(state);
        }
        return state;
    }

    const std::size_t count = data.values.size();
    std::size_t index = state % count;
    if (kind == WorkKind::PointerChase) {
        for (std::size_t i = 0; i < iterations; ++i) {
            index = data.next[index];
            state ^= data.values[index];
        }
    } else if (kind == WorkKind::StreamRead) {
        for (std::size_t i = 0; i < iterations; ++i) {
            state += data.values[index];
            if (++index == count)
                index = 0;
        }
    } else if (kind == WorkKind::StreamWrite) {
        for (std::size_t i = 0; i < iterations; ++i) {
            data.values[index] = state + i;
            if (++index == count)
                index = 0;
        }
        state ^= data.values[(index + count - 1) % count];
    } else if (kind == WorkKind::ReadFour || kind == WorkKind::ReadEight) {
        std::uint64_t first = state;
        std::uint64_t second = std::rotl(state, 11);
        std::uint64_t third = std::rotl(state, 23);
        std::uint64_t fourth = std::rotl(state, 37);
        for (std::size_t i = 0; i < iterations; ++i) {
            if (index + 8 > count)
                index = 0;
            first += data.values[index];
            second ^= data.values[index + 1];
            third += data.values[index + 2];
            fourth ^= data.values[index + 3];
            if (kind == WorkKind::ReadEight) {
                first ^= data.values[index + 4];
                second += data.values[index + 5];
                third ^= data.values[index + 6];
                fourth += data.values[index + 7];
            }
            index += 8;
        }
        state = first ^ std::rotl(second, 13) ^ std::rotl(third, 29) ^ std::rotl(fourth, 47);
    } else if (kind == WorkKind::WriteEight) {
        for (std::size_t i = 0; i < iterations; ++i) {
            if (index + 8 > count)
                index = 0;
            data.values[index] = state + i;
            data.values[index + 1] = state - i;
            data.values[index + 2] = state ^ i;
            data.values[index + 3] = state | i;
            data.values[index + 4] = state & ~std::uint64_t(i);
            data.values[index + 5] = std::rotl(state, unsigned(i & 63U));
            data.values[index + 6] = state + i * 3U;
            data.values[index + 7] = state ^ (i * 5U);
            index += 8;
        }
        state ^= data.values[(index + count - 1) % count];
    } else if (kind == WorkKind::LineReadOne || kind == WorkKind::LineReadEight) {
        const bool readEight = kind == WorkKind::LineReadEight;
        for (std::size_t i = 0; i < iterations; ++i) {
            if (index + 8 > count)
                index = 0;
            state = readCacheLine(state, data.values.data() + index, readEight);
            index += 8;
        }
    } else if (kind == WorkKind::LineWriteOne || kind == WorkKind::LineWriteEight) {
        const bool writeEight = kind == WorkKind::LineWriteEight;
        for (std::size_t i = 0; i < iterations; ++i) {
            if (index + 8 > count)
                index = 0;
            writeCacheLine(state + i, data.values.data() + index, writeEight);
            index += 8;
        }
        state ^= data.values[(index + count - 1) % count];
    } else if (kind == WorkKind::LineReadOneWriteOne || kind == WorkKind::LineReadEightWriteOne || kind == WorkKind::LineReadOneWriteEight || kind == WorkKind::LineReadEightWriteEight) {
        const bool readEight = kind == WorkKind::LineReadEightWriteOne || kind == WorkKind::LineReadEightWriteEight;
        const bool writeEight = kind == WorkKind::LineReadOneWriteEight || kind == WorkKind::LineReadEightWriteEight;
        for (std::size_t i = 0; i < iterations; ++i) {
            if (index + 8 > count)
                index = 0;
            auto* values = data.values.data() + index;
            readWriteCacheLine(state, state + i, values, readEight, writeEight);
            index += 8;
        }
        state ^= data.values[(index + count - 8) % count];
    } else if (kind == WorkKind::CacheRead) {
        std::uint64_t first = state;
        std::uint64_t second = state ^ 0x9e3779b97f4a7c15ULL;
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < iterations; ++i) {
            index = data.next[cursor];
            if (++cursor == count)
                cursor = 0;
            if (i & 1U)
                first += data.values[index];
            else
                second ^= data.values[index];
        }
        state = first ^ std::rotl(second, 17);
    } else if (kind == WorkKind::CacheWrite) {
        for (std::size_t i = 0; i < iterations; ++i) {
            index += data.cacheStride;
            if (index >= count)
                index -= count;
            data.values[index] = state + i;
        }
        if (iterations != 0)
            state ^= data.values[index];
    } else if (kind == WorkKind::GridSample) {
        double accumulated = double(state & 0xffU);
        const std::size_t limit = count - 8;
        for (std::size_t i = 0; i < iterations; ++i) {
            state = deterministicWork(state, 1);
            if ((state & 15U) == 0)
                continue;
            index = state % limit;
            accumulated += trilinearSample(data.values, index, state);
        }
        state ^= std::bit_cast<std::uint64_t>(accumulated);
    } else if (kind == WorkKind::MixedGridPhase || kind == WorkKind::MixedGridRandom) {
        double accumulated = double(state & 0xffU);
        const std::size_t bucketSize = count / 4;
        const std::size_t bucketLimit = bucketSize - 8;
        const std::size_t phaseSize = std::max<std::size_t>(1, iterations / 4);
        for (std::size_t i = 0; i < iterations; ++i) {
            state = deterministicWork(state, 1);
            const std::size_t bucket = kind == WorkKind::MixedGridRandom ? std::size_t((state >> 32U) & 3U) : std::min<std::size_t>(3, i / phaseSize);
            index = bucket * bucketSize + state % (bucketLimit + 1);
            const double sample = trilinearSample(data.values, index, state);
            switch (bucket) {
                case 0:
                    accumulated += sample;
                    break;
                case 1:
                    accumulated -= sample;
                    break;
                case 2:
                    accumulated += sample * 0.5;
                    break;
                default:
                    accumulated -= sample * 0.5;
                    break;
            }
        }
        state ^= std::bit_cast<std::uint64_t>(accumulated);
    } else {
        for (std::size_t i = 0; i < iterations; ++i) {
            state = deterministicWork(state, 1);
            index = data.next[(index + (state & 7U)) % count];
            state += data.values[index];
            state = (state & 1U) ? branchOdd(state) : branchEven(state);
        }
    }
    return state ^ std::uint64_t(index);
}

std::uint64_t checksum(const std::vector<std::uint64_t>& values)
{
    std::uint64_t result = 1469598103934665603ULL;
    for (const std::uint64_t value : values) {
        result ^= value;
        result *= 1099511628211ULL;
    }
    return result;
}

void runRound(
    Mode mode,
    utils::ThreadPool& pool,
    std::vector<std::uint64_t>& results,
    std::vector<WorkData>& workData,
    WorkKind workKind,
    std::size_t workIterations,
    std::size_t taskIterationSkew,
    std::size_t round)
{
    const auto task = [&](std::size_t index) {
        const std::uint64_t seed = 0x9e3779b97f4a7c15ULL ^ (std::uint64_t(round) << 32U) ^ std::uint64_t(index + 1);
        if (taskIterationSkew != 0 && index > (std::numeric_limits<std::size_t>::max() - workIterations) / taskIterationSkew)
            throw std::runtime_error("per-task work iteration count overflows");
        return runWork(workKind, seed, workIterations + index * taskIterationSkew, workData[index]);
    };

    if (mode == Mode::Serial) {
        for (std::size_t index = 0; index < results.size(); ++index)
            results[index] = task(index);
        return;
    }

    if (mode == Mode::Futures || mode == Mode::FuturesGateOpen || mode == Mode::FuturesGateClosed) {
        const bool gated = mode != Mode::Futures;
        std::latch startGate(mode == Mode::FuturesGateClosed ? 1 : 0);
        std::vector<std::future<std::uint64_t>> futures;
        futures.reserve(results.size());
        for (std::size_t index = 0; index < results.size(); ++index) {
            futures.push_back(pool.submit([&, index] {
                if (gated)
                    startGate.wait();
                return task(index);
            }));
        }
        if (mode == Mode::FuturesGateClosed)
            startGate.count_down();
        for (std::size_t index = 0; index < results.size(); ++index)
            results[index] = futures[index].get();
        return;
    }

    pool.run_indexed_batch(results.size(), [&](std::size_t index) { results[index] = task(index); });
}

Measurement measureLifecycle(const Args& args)
{
    if (args.idleNanoseconds != 0)
        throw std::runtime_error("lifecycle mode does not support inter-wave idle");
    for (std::size_t round = 0; round < 32; ++round) {
        utils::ThreadPool pool(args.workers);
    }

    const std::clock_t cpuStart = std::clock();
    const auto wallStart = std::chrono::steady_clock::now();
    for (std::size_t round = 0; round < args.rounds; ++round) {
        utils::ThreadPool pool(args.workers);
    }
    const auto wallEnd = std::chrono::steady_clock::now();
    const std::clock_t cpuEnd = std::clock();
    const double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
    return Measurement{
        wallSeconds,
        double(cpuEnd - cpuStart) / double(CLOCKS_PER_SEC),
        wallSeconds,
        0.0,
        0.0,
        std::uint64_t(args.workers) ^ (std::uint64_t(args.rounds) << 32U),
    };
}

Measurement measureDispatch(const Args& args)
{
    utils::ThreadPool pool(args.workers);
    std::vector<std::uint64_t> results(args.tasks);
    std::vector<WorkData> workData;
    workData.reserve(args.tasks);
    for (std::size_t task = 0; task < args.tasks; ++task)
        workData.push_back(makeWorkData(args.workingSetBytes, task));
    for (std::size_t round = 0; round < args.warmupRounds; ++round)
        runRound(args.mode, pool, results, workData, args.workKind, args.workIterations, args.taskIterationSkew, round);

    double rawDispatchSeconds = 0.0;
    double clockOverheadSeconds = 0.0;
    double actualIdleSeconds = 0.0;
    const std::clock_t cpuStart = std::clock();
    if (args.idleNanoseconds == 0) {
        const auto wallStart = std::chrono::steady_clock::now();
#if VC_HAS_CALLGRIND_CLIENT
        CALLGRIND_ZERO_STATS;
        CALLGRIND_START_INSTRUMENTATION;
#endif
        for (std::size_t round = 0; round < args.rounds; ++round)
            runRound(args.mode, pool, results, workData, args.workKind, args.workIterations, args.taskIterationSkew, round + args.warmupRounds);
#if VC_HAS_CALLGRIND_CLIENT
        CALLGRIND_STOP_INSTRUMENTATION;
#endif
        const auto wallEnd = std::chrono::steady_clock::now();
        rawDispatchSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
    } else {
#if VC_HAS_CALLGRIND_CLIENT
        CALLGRIND_ZERO_STATS;
        CALLGRIND_START_INSTRUMENTATION;
#endif
        const auto requestedIdle = std::chrono::nanoseconds(args.idleNanoseconds);
        for (std::size_t round = 0; round < args.rounds; ++round) {
            while (pool.pending() != 0 || pool.active() != 0)
                std::this_thread::yield();
            const auto idleStart = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(requestedIdle);
            const auto idleEnd = std::chrono::steady_clock::now();
            actualIdleSeconds += std::chrono::duration<double>(idleEnd - idleStart).count();

            const auto overheadStart = std::chrono::steady_clock::now();
            const auto overheadEnd = std::chrono::steady_clock::now();
            clockOverheadSeconds += std::chrono::duration<double>(overheadEnd - overheadStart).count();

            const auto dispatchStart = std::chrono::steady_clock::now();
            runRound(args.mode, pool, results, workData, args.workKind, args.workIterations, args.taskIterationSkew, round + args.warmupRounds);
            const auto dispatchEnd = std::chrono::steady_clock::now();
            rawDispatchSeconds += std::chrono::duration<double>(dispatchEnd - dispatchStart).count();
        }
#if VC_HAS_CALLGRIND_CLIENT
        CALLGRIND_STOP_INSTRUMENTATION;
#endif
    }
    const std::clock_t cpuEnd = std::clock();
    const double adjustedDispatchSeconds = std::max(0.0, rawDispatchSeconds - clockOverheadSeconds);

    Measurement result{
        adjustedDispatchSeconds,
        double(cpuEnd - cpuStart) / double(CLOCKS_PER_SEC),
        rawDispatchSeconds,
        clockOverheadSeconds,
        actualIdleSeconds,
        checksum(results),
    };
    const std::size_t finalRound = args.warmupRounds + args.rounds - 1;
    for (std::size_t index = 0; index < results.size(); ++index) {
        const std::uint64_t seed = 0x9e3779b97f4a7c15ULL ^ (std::uint64_t(finalRound) << 32U) ^ std::uint64_t(index + 1);
        WorkData data = makeWorkData(args.workingSetBytes, index);
        results[index] = runWork(args.workKind, seed, args.workIterations + index * args.taskIterationSkew, data);
    }
    if (result.checksum != checksum(results))
        throw std::runtime_error("dispatch result checksum mismatch");
    return result;
}

Measurement measure(const Args& args)
{
    if (args.mode == Mode::Lifecycle)
        return measureLifecycle(args);
    return measureDispatch(args);
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parseArgs(argc, argv);
        const Measurement result = measure(args);
        std::cout << std::setprecision(12) << "{\"mode\":\"" << modeName(args.mode) << "\",\"workers\":" << args.workers
                  << ",\"tasks\":" << args.tasks << ",\"work_iterations\":" << args.workIterations
                  << ",\"task_iteration_skew\":" << args.taskIterationSkew << ",\"work_kind\":\"" << workKindName(args.workKind) << "\""
                  << ",\"working_set_bytes\":" << args.workingSetBytes << ",\"rounds\":" << args.rounds
                  << ",\"warmup_rounds\":" << args.warmupRounds << ",\"wall_seconds\":" << result.wallSeconds
                  << ",\"cpu_seconds\":" << result.cpuSeconds << ",\"average_cpu_cores\":" << result.cpuSeconds / result.wallSeconds
                  << ",\"nanoseconds_per_round\":" << result.wallSeconds * 1e9 / double(args.rounds)
                  << ",\"raw_dispatch_nanoseconds_per_round\":" << result.rawDispatchSeconds * 1e9 / double(args.rounds)
                  << ",\"clock_overhead_nanoseconds_per_round\":" << result.clockOverheadSeconds * 1e9 / double(args.rounds)
                  << ",\"requested_idle_nanoseconds_per_round\":" << args.idleNanoseconds
                  << ",\"actual_idle_nanoseconds_per_round\":" << result.actualIdleSeconds * 1e9 / double(args.rounds)
                  << ",\"checksum\":" << result.checksum << ",\"compiler_id\":\"" << VC_DISPATCH_BENCH_COMPILER_ID
                  << "\",\"compiler_version\":\"" << VC_DISPATCH_BENCH_COMPILER_VERSION << "\",\"build_type\":\""
                  << VC_DISPATCH_BENCH_BUILD_TYPE << "\",\"architecture_target\":\"" << VC_DISPATCH_BENCH_ARCHITECTURE << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bench_thread_pool_dispatch: " << error.what() << '\n';
        return 1;
    }
}
