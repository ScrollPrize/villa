#include "vc/fiber_tracer/FiberletStorage.hpp"

#include "vc/lasagna/ChannelSampler.hpp"

#include <utils/hash.hpp>

#include <zstd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace vc::fiber_tracer
{
namespace
{

constexpr std::array<std::byte, 8>
    kMagic{std::byte{'V'}, std::byte{'C'}, std::byte{'F'}, std::byte{'L'}, std::byte{'T'}, std::byte{'V'}, std::byte{'2'}, std::byte{0}};
constexpr std::uint32_t kVersion = 2;
constexpr std::size_t kFixedHeaderBytes = 144;
constexpr std::size_t kDescriptorBytes = 40;
constexpr std::size_t kChecksumOffset = 136;

enum class Scalar : std::uint8_t {
    U8 = 1,
    I8 = 2,
    U16 = 3,
    I16 = 4,
    U32 = 5,
    I32 = 6,
    U64 = 7,
    I64 = 8,
    F32 = 9,
    F64 = 10,
};

enum class BlockCodec : std::uint8_t { Raw = 0, Zstd = 1 };

enum Field : std::uint16_t {
    KeyZ = 1,
    KeyY = 2,
    KeyX = 3,
    Variant = 4,
    PositionX = 5,
    PositionY = 6,
    PositionZ = 7,
    AxisX = 8,
    AxisY = 9,
    AxisZ = 10,
    PredictionAxisX = 11,
    PredictionAxisY = 12,
    PredictionAxisZ = 13,
    PredictionPresence = 14,
    NormalX = 15,
    NormalY = 16,
    NormalZ = 17,
    ScoringFlags = 18,
    FirstZ = 20,
    FirstY = 21,
    FirstX = 22,
    FirstVariant = 23,
    SecondDZ = 24,
    SecondDY = 25,
    SecondDX = 26,
    SecondVariant = 27,
    InteriorCount = 28,
    EntryU = 29,
    EntryV = 30,
    ExitU = 31,
    ExitV = 32,
    PathLength = 33,
    TotalCost = 34,
    RouteOffsets = 40,
    MiddleU = 41,
    MiddleV = 42,
    CostOffsets = 43,
    SegmentCostDensity = 44,
    InvalidPredictionCost = 50,
    AlignmentCost = 51,
    IsotropicSmoothnessCost = 52,
    TangentSmoothnessCost = 53,
    NormalSmoothnessCost = 54,
    FirstStepX = 55,
    FirstStepY = 56,
    FirstStepZ = 57,
    LastStepX = 58,
    LastStepY = 59,
    LastStepZ = 60,
    TraceOrdinal = 70,
    TraceSeedX = 71,
    TraceSeedY = 72,
    TraceSeedZ = 73,
    TraceSeedPresence = 74,
    TraceTotalCost = 75,
    TracePathLength = 76,
    TracePointOffsets = 77,
    TracePointX = 78,
    TracePointY = 79,
    TracePointZ = 80,
};

struct FieldBlock {
    std::uint16_t id = 0;
    Scalar scalar = Scalar::U8;
    std::uint64_t count = 0;
    BlockCodec codec = BlockCodec::Raw;
    std::vector<std::byte> decoded;
    std::vector<std::byte> encoded;
    std::uint64_t offset = 0;
};

template <typename T>
using Unsigned = std::make_unsigned_t<T>;

template <typename T>
void appendLittle(std::vector<std::byte>& output, T value)
{
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (sizeof(T) == 4)
            appendLittle(output, std::bit_cast<std::uint32_t>(value));
        else
            appendLittle(output, std::bit_cast<std::uint64_t>(value));
    } else {
        using U = Unsigned<T>;
        std::uint64_t bits = static_cast<std::uint64_t>(static_cast<U>(value));
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            output.push_back(static_cast<std::byte>(bits & 0xffU));
            bits >>= 8;
        }
    }
}

template <typename T>
T readLittle(std::span<const std::byte> bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw std::invalid_argument("fiberlet payload is truncated");
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (sizeof(T) == 4)
            return std::bit_cast<float>(readLittle<std::uint32_t>(bytes, offset));
        else
            return std::bit_cast<double>(readLittle<std::uint64_t>(bytes, offset));
    } else {
        using U = Unsigned<T>;
        U value = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<U>(std::to_integer<unsigned char>(bytes[offset + index])) << (index * 8);
        }
        return static_cast<T>(value);
    }
}

void overwriteLittle(std::vector<std::byte>& output, std::size_t offset, std::uint64_t value)
{
    if (offset > output.size() || output.size() - offset < sizeof(value))
        throw std::logic_error("fiberlet header overwrite is out of range");
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        output[offset + index] = static_cast<std::byte>(value & 0xffU);
        value >>= 8;
    }
}

std::uint64_t payloadChecksum(std::span<const std::byte> bytes)
{
    std::uint64_t hash = utils::fnv_offset_basis;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const unsigned char value =
            index >= kChecksumOffset && index < kChecksumOffset + sizeof(std::uint64_t) ? 0 : std::to_integer<unsigned char>(bytes[index]);
        hash ^= value;
        hash *= utils::fnv_prime;
    }
    return hash;
}

std::size_t scalarBytes(Scalar scalar)
{
    switch (scalar) {
        case Scalar::U8:
        case Scalar::I8:
            return 1;
        case Scalar::U16:
        case Scalar::I16:
            return 2;
        case Scalar::U32:
        case Scalar::I32:
        case Scalar::F32:
            return 4;
        case Scalar::U64:
        case Scalar::I64:
        case Scalar::F64:
            return 8;
    }
    throw std::invalid_argument("unknown fiberlet scalar type");
}

Scalar unsignedScalar(std::uint8_t bits)
{
    if (bits == 8)
        return Scalar::U8;
    if (bits == 16)
        return Scalar::U16;
    if (bits == 32)
        return Scalar::U32;
    throw std::invalid_argument("fiberlet unsigned scalar width must be 8, 16, or 32");
}

Scalar signedScalar(std::uint8_t bits)
{
    if (bits == 8)
        return Scalar::I8;
    if (bits == 16)
        return Scalar::I16;
    if (bits == 32)
        return Scalar::I32;
    throw std::invalid_argument("fiberlet signed scalar width must be 8, 16, or 32");
}

void appendUnsigned(std::vector<std::byte>& output, Scalar scalar, std::uint64_t value)
{
    const auto fail = [&] { throw std::invalid_argument("fiberlet unsigned field overflows its declared scalar"); };
    switch (scalar) {
        case Scalar::U8:
            if (value > std::numeric_limits<std::uint8_t>::max())
                fail();
            appendLittle(output, static_cast<std::uint8_t>(value));
            return;
        case Scalar::U16:
            if (value > std::numeric_limits<std::uint16_t>::max())
                fail();
            appendLittle(output, static_cast<std::uint16_t>(value));
            return;
        case Scalar::U32:
            if (value > std::numeric_limits<std::uint32_t>::max())
                fail();
            appendLittle(output, static_cast<std::uint32_t>(value));
            return;
        default:
            throw std::logic_error("fiberlet field is not unsigned");
    }
}

void appendSigned(std::vector<std::byte>& output, Scalar scalar, std::int64_t value)
{
    const auto fail = [&] { throw std::invalid_argument("fiberlet signed field overflows its declared scalar"); };
    switch (scalar) {
        case Scalar::I8:
            if (value < std::numeric_limits<std::int8_t>::min() || value > std::numeric_limits<std::int8_t>::max())
                fail();
            appendLittle(output, static_cast<std::int8_t>(value));
            return;
        case Scalar::I16:
            if (value < std::numeric_limits<std::int16_t>::min() || value > std::numeric_limits<std::int16_t>::max())
                fail();
            appendLittle(output, static_cast<std::int16_t>(value));
            return;
        case Scalar::I32:
            if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max())
                fail();
            appendLittle(output, static_cast<std::int32_t>(value));
            return;
        default:
            throw std::logic_error("fiberlet field is not signed");
    }
}

std::uint64_t readUnsigned(std::span<const std::byte> bytes, Scalar scalar, std::size_t index)
{
    const std::size_t offset = index * scalarBytes(scalar);
    switch (scalar) {
        case Scalar::U8:
            return readLittle<std::uint8_t>(bytes, offset);
        case Scalar::U16:
            return readLittle<std::uint16_t>(bytes, offset);
        case Scalar::U32:
            return readLittle<std::uint32_t>(bytes, offset);
        default:
            throw std::invalid_argument("fiberlet field has wrong unsigned scalar type");
    }
}

std::int64_t readSigned(std::span<const std::byte> bytes, Scalar scalar, std::size_t index)
{
    const std::size_t offset = index * scalarBytes(scalar);
    switch (scalar) {
        case Scalar::I8:
            return readLittle<std::int8_t>(bytes, offset);
        case Scalar::I16:
            return readLittle<std::int16_t>(bytes, offset);
        case Scalar::I32:
            return readLittle<std::int32_t>(bytes, offset);
        default:
            throw std::invalid_argument("fiberlet field has wrong signed scalar type");
    }
}

FieldBlock makeField(std::uint16_t id, Scalar scalar, std::uint64_t count)
{
    FieldBlock field;
    field.id = id;
    field.scalar = scalar;
    field.count = count;
    if (count > std::numeric_limits<std::size_t>::max() / scalarBytes(scalar))
        throw std::invalid_argument("fiberlet field byte count overflows size_t");
    field.decoded.reserve(static_cast<std::size_t>(count) * scalarBytes(scalar));
    return field;
}

void finishField(FieldBlock& field, bool compress)
{
    const std::size_t expected = static_cast<std::size_t>(field.count) * scalarBytes(field.scalar);
    if (field.decoded.size() != expected)
        throw std::logic_error("fiberlet field encoder produced the wrong byte count");
    if (!compress) {
        field.codec = BlockCodec::Raw;
        field.encoded = field.decoded;
        return;
    }
    std::vector<std::byte> compressed(ZSTD_compressBound(field.decoded.size()));
    const std::size_t size = ZSTD_compress(compressed.data(), compressed.size(), field.decoded.data(), field.decoded.size(), 3);
    if (ZSTD_isError(size))
        throw std::runtime_error(std::string("fiberlet zstd encode failed: ") + ZSTD_getErrorName(size));
    compressed.resize(size);
    if (size < field.decoded.size()) {
        field.codec = BlockCodec::Zstd;
        field.encoded = std::move(compressed);
    } else {
        field.codec = BlockCodec::Raw;
        field.encoded = field.decoded;
    }
}

void validateConfig(const FiberletStorageCodecConfig& config)
{
    if (config.profile != FiberletStorageProfile::Float32Cache && config.profile != FiberletStorageProfile::CompactQuantized &&
        config.profile != FiberletStorageProfile::CompactDirectionsFixedCost && config.profile != FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("unknown fiberlet storage profile");
    if (config.profile == FiberletStorageProfile::Float64Traces) {
        if (config.coordinateBits != 32 || config.deltaBits != 32 || config.routeCountBits != 32 || config.routeLatticeBits != 32 ||
            config.costBits != 64 || config.positionQuantumBaseVoxels != 0 || !(config.predictionToBaseScale > 0.0) ||
            !std::isfinite(config.predictionToBaseScale)) {
            throw std::invalid_argument("float64 trace profile has invalid canonical settings");
        }
        return;
    }
    (void)unsignedScalar(config.coordinateBits);
    (void)signedScalar(config.deltaBits);
    (void)unsignedScalar(config.routeCountBits);
    (void)signedScalar(config.routeLatticeBits);
    if (config.profile == FiberletStorageProfile::Float32Cache) {
        if (config.costBits != 32 || config.positionQuantumBaseVoxels != 0)
            throw std::invalid_argument("float32 fiberlet profile has invalid physical settings");
    } else if (config.profile == FiberletStorageProfile::CompactQuantized) {
        if (config.costBits != 8 && config.costBits != 16)
            throw std::invalid_argument("compact fiberlet cost width must be 8 or 16");
        if (config.positionQuantumBaseVoxels == 0 || !(config.predictionToBaseScale > 0.0) || !std::isfinite(config.predictionToBaseScale))
            throw std::invalid_argument("compact fiberlet position scale is invalid");
    } else if (config.costBits != 16 || config.positionQuantumBaseVoxels != 0) {
        throw std::invalid_argument("compact-direction fiberlet profile requires float positions and uint16 costs");
    }
}

std::vector<std::byte> encodePayload(
    const FiberletStorageCodecConfig& config,
    FiberletStorageChunkKind kind,
    std::uint64_t recordCount,
    std::uint64_t auxiliaryCount,
    float costOffset,
    float costScale,
    std::vector<FieldBlock> fields,
    bool compress = true)
{
    validateConfig(config);
    std::sort(fields.begin(), fields.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
    for (std::size_t index = 1; index < fields.size(); ++index) {
        if (fields[index - 1].id == fields[index].id)
            throw std::logic_error("duplicate fiberlet field id");
    }
    for (auto& field : fields)
        finishField(field, compress);

    if (fields.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("too many fiberlet fields");
    const std::size_t headerBytes = kFixedHeaderBytes + fields.size() * kDescriptorBytes;
    std::size_t totalBytes = headerBytes;
    for (auto& field : fields) {
        field.offset = totalBytes;
        if (field.encoded.size() > std::numeric_limits<std::size_t>::max() - totalBytes)
            throw std::invalid_argument("fiberlet payload size overflows size_t");
        totalBytes += field.encoded.size();
    }

    std::vector<std::byte> output;
    output.reserve(totalBytes);
    output.insert(output.end(), kMagic.begin(), kMagic.end());
    appendLittle(output, kVersion);
    appendLittle(output, static_cast<std::uint8_t>(kind));
    appendLittle(output, static_cast<std::uint8_t>(config.profile));
    appendLittle(output, config.coordinateBits);
    appendLittle(output, config.deltaBits);
    appendLittle(output, config.routeCountBits);
    appendLittle(output, config.routeLatticeBits);
    appendLittle(output, config.costBits);
    appendLittle(output, std::uint8_t{0});
    for (const auto value : config.chunkZYX)
        appendLittle(output, value);
    for (const auto value : config.coordinateOriginZYX)
        appendLittle(output, value);
    for (const auto value : config.datasetFingerprint)
        appendLittle(output, value);
    appendLittle(output, recordCount);
    appendLittle(output, auxiliaryCount);
    appendLittle(output, config.positionQuantumBaseVoxels);
    appendLittle(output, std::uint32_t{0});
    appendLittle(output, config.predictionToBaseScale);
    appendLittle(output, costOffset);
    appendLittle(output, costScale);
    appendLittle(output, static_cast<std::uint32_t>(fields.size()));
    appendLittle(output, static_cast<std::uint32_t>(headerBytes));
    appendLittle(output, std::uint64_t{0});
    if (output.size() != kFixedHeaderBytes)
        throw std::logic_error("fiberlet fixed header size is inconsistent");

    for (const auto& field : fields) {
        appendLittle(output, field.id);
        appendLittle(output, static_cast<std::uint8_t>(field.scalar));
        appendLittle(output, static_cast<std::uint8_t>(field.codec));
        appendLittle(output, field.count);
        appendLittle(output, field.offset);
        appendLittle(output, static_cast<std::uint64_t>(field.encoded.size()));
        appendLittle(output, static_cast<std::uint64_t>(field.decoded.size()));
        appendLittle(output, std::uint32_t{0});
    }
    for (const auto& field : fields)
        output.insert(output.end(), field.encoded.begin(), field.encoded.end());
    if (output.size() != totalBytes)
        throw std::logic_error("fiberlet payload size is inconsistent");
    overwriteLittle(output, kChecksumOffset, payloadChecksum(output));
    return output;
}

struct DecodedPayload {
    FiberletStorageCodecConfig config;
    FiberletStorageChunkKind kind = FiberletStorageChunkKind::Anchors;
    std::uint64_t recordCount = 0;
    std::uint64_t auxiliaryCount = 0;
    float costOffset = 0.0F;
    float costScale = 0.0F;
    std::map<std::uint16_t, std::pair<Scalar, std::vector<std::byte>>> fields;
};

DecodedPayload decodePayload(std::span<const std::byte> bytes, FiberletStorageChunkKind expectedKind)
{
    if (bytes.size() < kFixedHeaderBytes || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        throw std::invalid_argument("invalid fiberlet payload magic");
    if (readLittle<std::uint32_t>(bytes, 8) != kVersion)
        throw std::invalid_argument("unsupported fiberlet payload version");
    const auto kind = static_cast<FiberletStorageChunkKind>(readLittle<std::uint8_t>(bytes, 12));
    if (kind != expectedKind)
        throw std::invalid_argument("fiberlet payload kind does not match decoder");
    if (readLittle<std::uint8_t>(bytes, 19) != 0 || readLittle<std::uint32_t>(bytes, 108) != 0)
        throw std::invalid_argument("fiberlet payload reserved fields are nonzero");
    const auto storedChecksum = readLittle<std::uint64_t>(bytes, kChecksumOffset);
    if (storedChecksum != payloadChecksum(bytes))
        throw std::invalid_argument("fiberlet payload checksum mismatch");

    DecodedPayload result;
    result.kind = kind;
    result.config.profile = static_cast<FiberletStorageProfile>(readLittle<std::uint8_t>(bytes, 13));
    result.config.coordinateBits = readLittle<std::uint8_t>(bytes, 14);
    result.config.deltaBits = readLittle<std::uint8_t>(bytes, 15);
    result.config.routeCountBits = readLittle<std::uint8_t>(bytes, 16);
    result.config.routeLatticeBits = readLittle<std::uint8_t>(bytes, 17);
    result.config.costBits = readLittle<std::uint8_t>(bytes, 18);
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.config.chunkZYX[axis] = readLittle<std::int32_t>(bytes, 20 + axis * 4);
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.config.coordinateOriginZYX[axis] = readLittle<std::int64_t>(bytes, 32 + axis * 8);
    for (std::size_t index = 0; index < result.config.datasetFingerprint.size(); ++index)
        result.config.datasetFingerprint[index] = readLittle<std::uint8_t>(bytes, 56 + index);
    result.recordCount = readLittle<std::uint64_t>(bytes, 88);
    result.auxiliaryCount = readLittle<std::uint64_t>(bytes, 96);
    result.config.positionQuantumBaseVoxels = readLittle<std::uint32_t>(bytes, 104);
    result.config.predictionToBaseScale = readLittle<double>(bytes, 112);
    result.costOffset = readLittle<float>(bytes, 120);
    result.costScale = readLittle<float>(bytes, 124);
    const auto descriptorCount = readLittle<std::uint32_t>(bytes, 128);
    const auto headerBytes = readLittle<std::uint32_t>(bytes, 132);
    validateConfig(result.config);
    if (headerBytes != kFixedHeaderBytes + static_cast<std::uint64_t>(descriptorCount) * kDescriptorBytes || headerBytes > bytes.size())
        throw std::invalid_argument("fiberlet descriptor table size is invalid");

    std::uint64_t previousEnd = headerBytes;
    for (std::uint32_t index = 0; index < descriptorCount; ++index) {
        const std::size_t offset = kFixedHeaderBytes + static_cast<std::size_t>(index) * kDescriptorBytes;
        const auto id = readLittle<std::uint16_t>(bytes, offset);
        const auto scalar = static_cast<Scalar>(readLittle<std::uint8_t>(bytes, offset + 2));
        const auto codec = static_cast<BlockCodec>(readLittle<std::uint8_t>(bytes, offset + 3));
        const auto count = readLittle<std::uint64_t>(bytes, offset + 4);
        const auto blockOffset = readLittle<std::uint64_t>(bytes, offset + 12);
        const auto encodedBytes = readLittle<std::uint64_t>(bytes, offset + 20);
        const auto decodedBytes = readLittle<std::uint64_t>(bytes, offset + 28);
        if (readLittle<std::uint32_t>(bytes, offset + 36) != 0)
            throw std::invalid_argument("fiberlet descriptor reserved field is nonzero");
        const auto elementBytes = scalarBytes(scalar);
        if (count > std::numeric_limits<std::uint64_t>::max() / elementBytes || decodedBytes != count * elementBytes)
            throw std::invalid_argument("fiberlet descriptor decoded length is invalid");
        if (blockOffset < previousEnd || encodedBytes > bytes.size() || blockOffset > bytes.size() - encodedBytes)
            throw std::invalid_argument("fiberlet descriptor range is invalid");
        previousEnd = blockOffset + encodedBytes;
        if (result.fields.contains(id))
            throw std::invalid_argument("fiberlet payload contains a duplicate field");
        std::vector<std::byte> decoded(static_cast<std::size_t>(decodedBytes));
        const auto source = bytes.subspan(static_cast<std::size_t>(blockOffset), static_cast<std::size_t>(encodedBytes));
        if (codec == BlockCodec::Raw) {
            if (source.size() != decoded.size())
                throw std::invalid_argument("fiberlet raw field length is invalid");
            std::copy(source.begin(), source.end(), decoded.begin());
        } else if (codec == BlockCodec::Zstd) {
            const auto size = ZSTD_decompress(decoded.data(), decoded.size(), source.data(), source.size());
            if (ZSTD_isError(size) || size != decoded.size())
                throw std::invalid_argument("fiberlet zstd field is invalid");
        } else {
            throw std::invalid_argument("fiberlet field codec is unknown");
        }
        result.fields.emplace(id, std::make_pair(scalar, std::move(decoded)));
    }
    if (previousEnd != bytes.size())
        throw std::invalid_argument("fiberlet payload has trailing or unreferenced bytes");
    return result;
}

const std::pair<Scalar, std::vector<std::byte>>& requireField(const DecodedPayload& payload, std::uint16_t id, Scalar scalar, std::uint64_t count)
{
    const auto found = payload.fields.find(id);
    if (found == payload.fields.end() || found->second.first != scalar || found->second.second.size() != count * scalarBytes(scalar))
        throw std::invalid_argument("fiberlet payload is missing a required field or has the wrong field type");
    return found->second;
}

std::array<std::int64_t, 3> keyLocal(const FiberletStorageCodecConfig& config, const FiberletStorageKey& key)
{
    std::array<std::int64_t, 3> local{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        local[axis] = key.coordinateZYX[axis] - config.coordinateOriginZYX[axis];
        if (local[axis] < 0)
            throw std::invalid_argument("fiberlet key lies before its chunk coordinate origin");
    }
    if (key.variant > 1)
        throw std::invalid_argument("fiberlet anchor variant must be zero or one");
    return local;
}

FiberletStorageKey decodeKey(const FiberletStorageCodecConfig& config, const std::array<std::uint64_t, 3>& local, std::uint8_t variant)
{
    if (variant > 1)
        throw std::invalid_argument("fiberlet anchor variant must be zero or one");
    FiberletStorageKey key;
    key.variant = variant;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (local[axis] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
            config.coordinateOriginZYX[axis] > std::numeric_limits<std::int64_t>::max() - static_cast<std::int64_t>(local[axis]))
            throw std::invalid_argument("fiberlet key coordinate overflows int64");
        key.coordinateZYX[axis] = config.coordinateOriginZYX[axis] + static_cast<std::int64_t>(local[axis]);
    }
    return key;
}

void validateFinite(const cv::Vec3f& value, const char* name)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]))
        throw std::invalid_argument(std::string("fiberlet ") + name + " is not finite");
}

void validateFinite(const cv::Vec3d& value, const char* name)
{
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2])) {
        throw std::invalid_argument(std::string("fiberlet ") + name + " is not finite");
    }
}

}  // namespace

std::uint16_t encodeFiberletStoredCostDensity(float density)
{
    if (!(density >= 0.0F) || !std::isfinite(density))
        throw std::invalid_argument("fiberlet route cost density must be finite and nonnegative");
    const float transformed = std::sqrt(std::min(density / kFiberletStoredCostDensityMaximum, 1.0F));
    return static_cast<std::uint16_t>(std::lround(transformed * 65535.0F));
}

float decodeFiberletStoredCostDensity(std::uint16_t code)
{
    const float transformed = static_cast<float>(code) / 65535.0F;
    return transformed * transformed * kFiberletStoredCostDensityMaximum;
}

std::vector<std::byte> serializeFiberletAnchors(const FiberletStorageCodecConfig& config, std::span<const FiberletStoredAnchor> anchors)
{
    validateConfig(config);
    if (config.profile == FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("trace profile cannot encode Fiberlet anchors");
    const Scalar coordinateScalar = unsignedScalar(config.coordinateBits);
    std::vector<FieldBlock> fields;
    for (const auto id : {KeyZ, KeyY, KeyX})
        fields.push_back(makeField(id, coordinateScalar, anchors.size()));
    fields.push_back(makeField(Variant, Scalar::U8, anchors.size()));
    const bool compactDirections = config.profile != FiberletStorageProfile::Float32Cache;
    const bool floatPositions = config.profile != FiberletStorageProfile::CompactQuantized;
    if (!compactDirections) {
        for (const auto id :
             {PositionX, PositionY, PositionZ, AxisX, AxisY, AxisZ, PredictionAxisX, PredictionAxisY, PredictionAxisZ, PredictionPresence, NormalX, NormalY, NormalZ}) {
            fields.push_back(makeField(id, Scalar::F32, anchors.size()));
        }
    } else {
        if (floatPositions) {
            for (const auto id : {PositionX, PositionY, PositionZ})
                fields.push_back(makeField(id, Scalar::F32, anchors.size()));
        }
        for (const auto id : {AxisX, AxisY, PredictionAxisX, PredictionAxisY, PredictionPresence, NormalX, NormalY}) {
            fields.push_back(makeField(id, Scalar::U8, anchors.size()));
        }
    }
    fields.push_back(makeField(ScoringFlags, Scalar::U8, anchors.size()));
    auto field = [&](Field id) -> FieldBlock& {
        return *std::find_if(fields.begin(), fields.end(), [&](const auto& item) { return item.id == id; });
    };
    FiberletStorageKey previous;
    bool havePrevious = false;
    for (const auto& anchor : anchors) {
        if (havePrevious && !(previous < anchor.key))
            throw std::invalid_argument("fiberlet anchors must be strictly sorted by stable key");
        havePrevious = true;
        previous = anchor.key;
        const auto local = keyLocal(config, anchor.key);
        appendUnsigned(field(KeyZ).decoded, coordinateScalar, static_cast<std::uint64_t>(local[0]));
        appendUnsigned(field(KeyY).decoded, coordinateScalar, static_cast<std::uint64_t>(local[1]));
        appendUnsigned(field(KeyX).decoded, coordinateScalar, static_cast<std::uint64_t>(local[2]));
        appendLittle(field(Variant).decoded, anchor.key.variant);
        validateFinite(anchor.fittedAxisXYZ, "anchor axis");
        if (!std::isfinite(anchor.predictionPresence))
            throw std::invalid_argument("fiberlet anchor presence is not finite");
        if (anchor.predictionValid)
            validateFinite(anchor.predictionAxisXYZ, "anchor prediction axis");
        if (anchor.normalValid)
            validateFinite(anchor.normalXYZ, "anchor normal");
        const std::uint8_t scoringFlags = (anchor.predictionValid ? std::uint8_t{1} : std::uint8_t{0}) |
            (anchor.predictionPresenceValid ? std::uint8_t{2} : std::uint8_t{0}) |
            (anchor.normalValid ? std::uint8_t{4} : std::uint8_t{0});
        appendLittle(field(ScoringFlags).decoded, scoringFlags);
        if (!compactDirections) {
            validateFinite(anchor.positionPredictionXYZ, "anchor position");
            for (int axis = 0; axis < 3; ++axis) {
                appendLittle(field(static_cast<Field>(PositionX + axis)).decoded, anchor.positionPredictionXYZ[axis]);
                appendLittle(field(static_cast<Field>(AxisX + axis)).decoded, anchor.fittedAxisXYZ[axis]);
                appendLittle(field(static_cast<Field>(PredictionAxisX + axis)).decoded, anchor.predictionAxisXYZ[axis]);
                appendLittle(field(static_cast<Field>(NormalX + axis)).decoded, anchor.normalXYZ[axis]);
            }
            appendLittle(field(PredictionPresence).decoded, anchor.predictionPresence);
        } else {
            if (floatPositions) {
                validateFinite(anchor.positionPredictionXYZ, "anchor position");
                for (int axis = 0; axis < 3; ++axis) {
                    appendLittle(field(static_cast<Field>(PositionX + axis)).decoded, anchor.positionPredictionXYZ[axis]);
                }
            }
            const auto encoded =
                vc::lasagna::encodeCompactNormalToRaw(cv::Vec3d(anchor.fittedAxisXYZ[0], anchor.fittedAxisXYZ[1], anchor.fittedAxisXYZ[2]));
            if (!encoded)
                throw std::invalid_argument("fiberlet anchor axis cannot be compactly encoded");
            appendLittle(field(AxisX).decoded, (*encoded)[0]);
            appendLittle(field(AxisY).decoded, (*encoded)[1]);
            const auto prediction = anchor.predictionValid
                                        ? vc::lasagna::encodeCompactNormalToRaw(
                                              cv::Vec3d(anchor.predictionAxisXYZ[0], anchor.predictionAxisXYZ[1], anchor.predictionAxisXYZ[2]))
                                        : std::optional<std::array<std::uint8_t, 2>>{std::array<std::uint8_t, 2>{128, 128}};
            const auto normal =
                anchor.normalValid
                    ? vc::lasagna::encodeCompactNormalToRaw(cv::Vec3d(anchor.normalXYZ[0], anchor.normalXYZ[1], anchor.normalXYZ[2]))
                    : std::optional<std::array<std::uint8_t, 2>>{std::array<std::uint8_t, 2>{128, 128}};
            if (!prediction || !normal)
                throw std::invalid_argument("fiberlet anchor scoring axis cannot be compactly encoded");
            appendLittle(field(PredictionAxisX).decoded, (*prediction)[0]);
            appendLittle(field(PredictionAxisY).decoded, (*prediction)[1]);
            appendLittle(field(PredictionPresence).decoded, static_cast<std::uint8_t>(std::lround(std::clamp(anchor.predictionPresence, 0.0F, 1.0F) * 255.0F)));
            appendLittle(field(NormalX).decoded, (*normal)[0]);
            appendLittle(field(NormalY).decoded, (*normal)[1]);
        }
    }
    return encodePayload(config, FiberletStorageChunkKind::Anchors, anchors.size(), 0, 0.0F, 0.0F, std::move(fields));
}

std::vector<std::byte> serializeFiberletPrefixes(const FiberletStorageCodecConfig& config, std::span<const FiberletStoredPrefix> prefixes)
{
    validateConfig(config);
    if (config.profile == FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("trace profile cannot encode Fiberlet prefixes");
    const Scalar coordinateScalar = unsignedScalar(config.coordinateBits);
    const Scalar deltaScalar = signedScalar(config.deltaBits);
    const Scalar countScalar = unsignedScalar(config.routeCountBits);
    const Scalar latticeScalar = signedScalar(config.routeLatticeBits);
    const Scalar costScalar = config.costBits == 8 ? Scalar::U8 : config.costBits == 16 ? Scalar::U16 : Scalar::F32;
    std::vector<FieldBlock> fields;
    for (const auto id : {FirstZ, FirstY, FirstX})
        fields.push_back(makeField(id, coordinateScalar, prefixes.size()));
    fields.push_back(makeField(FirstVariant, Scalar::U8, prefixes.size()));
    for (const auto id : {SecondDZ, SecondDY, SecondDX})
        fields.push_back(makeField(id, deltaScalar, prefixes.size()));
    fields.push_back(makeField(SecondVariant, Scalar::U8, prefixes.size()));
    fields.push_back(makeField(InteriorCount, countScalar, prefixes.size()));
    for (const auto id : {EntryU, EntryV, ExitU, ExitV})
        fields.push_back(makeField(id, latticeScalar, prefixes.size()));
    fields.push_back(makeField(PathLength, Scalar::F32, prefixes.size()));
    if (config.profile == FiberletStorageProfile::Float32Cache) {
        for (const auto id : {InvalidPredictionCost, AlignmentCost, IsotropicSmoothnessCost, TangentSmoothnessCost, NormalSmoothnessCost})
            fields.push_back(makeField(id, Scalar::F32, prefixes.size()));
    } else {
        fields.push_back(makeField(TotalCost, costScalar, prefixes.size()));
    }
    for (const auto id : {FirstStepX, FirstStepY, FirstStepZ, LastStepX, LastStepY, LastStepZ})
        fields.push_back(makeField(id, Scalar::F32, prefixes.size()));
    auto field = [&](Field id) -> FieldBlock& {
        return *std::find_if(fields.begin(), fields.end(), [&](const auto& item) { return item.id == id; });
    };

    float costOffset = 0.0F;
    float costScale = 0.0F;
    if (config.profile == FiberletStorageProfile::CompactQuantized && !prefixes.empty()) {
        auto [minimum, maximum] = std::minmax_element(prefixes.begin(), prefixes.end(), [](const auto& a, const auto& b) {
            return a.cost.total() < b.cost.total();
        });
        costOffset = minimum->cost.total();
        const float range = maximum->cost.total() - costOffset;
        const float maximumCode = config.costBits == 8 ? 255.0F : 65535.0F;
        costScale = range == 0.0F ? 0.0F : range / maximumCode;
    }
    FiberletStorageId previous;
    bool havePrevious = false;
    for (const auto& prefix : prefixes) {
        if (!(prefix.id.first < prefix.id.second))
            throw std::invalid_argument("fiberlet endpoints are not canonical");
        if (havePrevious && !(previous < prefix.id))
            throw std::invalid_argument("fiberlet prefixes must be strictly sorted by stable id");
        previous = prefix.id;
        havePrevious = true;
        const auto validCost = [](float value) { return std::isfinite(value) && value >= 0.0F; };
        if (!std::isfinite(prefix.pathLengthPredictionVoxels) || !(prefix.pathLengthPredictionVoxels > 0.0F) ||
            !validCost(prefix.cost.invalidPrediction) || !validCost(prefix.cost.alignment) || !validCost(prefix.cost.isotropicSmoothness) ||
            !validCost(prefix.cost.tangentSmoothness) || !validCost(prefix.cost.normalSmoothness))
            throw std::invalid_argument("fiberlet prefix length or cost is invalid");
        validateFinite(prefix.firstStepBaseXYZ, "prefix first step");
        validateFinite(prefix.lastStepBaseXYZ, "prefix last step");
        if (!(prefix.firstStepBaseXYZ.dot(prefix.firstStepBaseXYZ) > 0.0F) || !(prefix.lastStepBaseXYZ.dot(prefix.lastStepBaseXYZ) > 0.0F))
            throw std::invalid_argument("fiberlet prefix endpoint step is zero");
        const auto local = keyLocal(config, prefix.id.first);
        for (std::size_t axis = 0; axis < 3; ++axis)
            appendUnsigned(field(static_cast<Field>(FirstZ + axis)).decoded, coordinateScalar, static_cast<std::uint64_t>(local[axis]));
        appendLittle(field(FirstVariant).decoded, prefix.id.first.variant);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto delta = prefix.id.second.coordinateZYX[axis] - prefix.id.first.coordinateZYX[axis];
            appendSigned(field(static_cast<Field>(SecondDZ + axis)).decoded, deltaScalar, delta);
        }
        if (prefix.id.second.variant > 1)
            throw std::invalid_argument("fiberlet anchor variant must be zero or one");
        appendLittle(field(SecondVariant).decoded, prefix.id.second.variant);
        appendUnsigned(field(InteriorCount).decoded, countScalar, prefix.interiorPointCount);
        appendSigned(field(EntryU).decoded, latticeScalar, prefix.entryUV[0]);
        appendSigned(field(EntryV).decoded, latticeScalar, prefix.entryUV[1]);
        appendSigned(field(ExitU).decoded, latticeScalar, prefix.exitUV[0]);
        appendSigned(field(ExitV).decoded, latticeScalar, prefix.exitUV[1]);
        appendLittle(field(PathLength).decoded, prefix.pathLengthPredictionVoxels);
        if (config.profile == FiberletStorageProfile::Float32Cache) {
            appendLittle(field(InvalidPredictionCost).decoded, prefix.cost.invalidPrediction);
            appendLittle(field(AlignmentCost).decoded, prefix.cost.alignment);
            appendLittle(field(IsotropicSmoothnessCost).decoded, prefix.cost.isotropicSmoothness);
            appendLittle(field(TangentSmoothnessCost).decoded, prefix.cost.tangentSmoothness);
            appendLittle(field(NormalSmoothnessCost).decoded, prefix.cost.normalSmoothness);
        } else if (config.profile == FiberletStorageProfile::CompactDirectionsFixedCost) {
            constexpr float densityMaximum = 256.0F;
            const float density = prefix.cost.total() / prefix.pathLengthPredictionVoxels;
            const float normalized = std::sqrt(std::clamp(density / densityMaximum, 0.0F, 1.0F));
            appendUnsigned(field(TotalCost).decoded, costScalar, static_cast<std::uint64_t>(std::lround(normalized * 65535.0F)));
        } else {
            const std::uint64_t maximumCode = config.costBits == 8 ? 255 : 65535;
            const auto code = costScale == 0.0F ? std::uint64_t{0}
                                                : static_cast<std::uint64_t>(std::floor((prefix.cost.total() - costOffset) / costScale + 0.5F));
            if (code > maximumCode)
                throw std::invalid_argument("fiberlet compact cost exceeds its chunk-local range");
            appendUnsigned(field(TotalCost).decoded, costScalar, code);
        }
        for (int axis = 0; axis < 3; ++axis) {
            appendLittle(field(static_cast<Field>(FirstStepX + axis)).decoded, prefix.firstStepBaseXYZ[axis]);
            appendLittle(field(static_cast<Field>(LastStepX + axis)).decoded, prefix.lastStepBaseXYZ[axis]);
        }
    }
    return encodePayload(config, FiberletStorageChunkKind::FiberletPrefix, prefixes.size(), 0, costOffset, costScale, std::move(fields));
}

std::vector<std::byte> serializeFiberletRoutes(const FiberletStorageCodecConfig& config, std::span<const FiberletStoredRoute> routes)
{
    validateConfig(config);
    if (config.profile == FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("trace profile cannot encode Fiberlet routes");
    const Scalar latticeScalar = signedScalar(config.routeLatticeBits);
    std::uint64_t total = 0;
    std::uint64_t totalCosts = 0;
    for (const auto& route : routes) {
        if (route.middleUV.size() > std::numeric_limits<std::uint32_t>::max() - total)
            throw std::invalid_argument("fiberlet route offsets overflow uint32");
        if (route.segmentCostDensities.empty())
            throw std::invalid_argument("fiberlet route has no segment cost densities");
        if (route.segmentCostDensities.size() > std::numeric_limits<std::uint32_t>::max() - totalCosts)
            throw std::invalid_argument("fiberlet route cost offsets overflow uint32");
        total += route.middleUV.size();
        totalCosts += route.segmentCostDensities.size();
    }
    std::vector<FieldBlock> fields;
    fields.push_back(makeField(RouteOffsets, Scalar::U32, routes.size() + 1));
    fields.push_back(makeField(MiddleU, latticeScalar, total));
    fields.push_back(makeField(MiddleV, latticeScalar, total));
    fields.push_back(makeField(CostOffsets, Scalar::U32, routes.size() + 1));
    fields.push_back(makeField(SegmentCostDensity, Scalar::U16, totalCosts));
    appendLittle(fields[0].decoded, std::uint32_t{0});
    appendLittle(fields[3].decoded, std::uint32_t{0});
    std::uint32_t offset = 0;
    std::uint32_t costOffset = 0;
    for (const auto& route : routes) {
        for (const auto& uv : route.middleUV) {
            appendSigned(fields[1].decoded, latticeScalar, uv[0]);
            appendSigned(fields[2].decoded, latticeScalar, uv[1]);
        }
        offset += static_cast<std::uint32_t>(route.middleUV.size());
        appendLittle(fields[0].decoded, offset);
        for (const float density : route.segmentCostDensities)
            appendLittle(fields[4].decoded, encodeFiberletStoredCostDensity(density));
        costOffset += static_cast<std::uint32_t>(route.segmentCostDensities.size());
        appendLittle(fields[3].decoded, costOffset);
    }
    return encodePayload(config, FiberletStorageChunkKind::FiberletRoutes, routes.size(), total, 0.0F, 0.0F, std::move(fields));
}

std::vector<std::byte> serializeFiberletTraces(const FiberletStorageCodecConfig& config, std::span<const FiberletStoredTrace> traces)
{
    validateConfig(config);
    if (config.profile != FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("Fiber traces require the float64 trace profile");

    std::uint64_t pointCount = 0;
    for (const auto& trace : traces) {
        validateFinite(trace.seedBaseXYZ, "trace seed");
        if (!std::isfinite(trace.seedPresence))
            throw std::invalid_argument("fiberlet trace seed presence is not finite");
        if (!(trace.totalMetricCost >= 0.0) || !std::isfinite(trace.totalMetricCost)) {
            throw std::invalid_argument("fiberlet trace total cost must be finite and nonnegative");
        }
        if (!(trace.pathLengthPredictionVoxels > 0.0) || !std::isfinite(trace.pathLengthPredictionVoxels)) {
            throw std::invalid_argument("fiberlet trace path length must be finite and positive");
        }
        if (trace.pointsBaseXYZ.size() < 2)
            throw std::invalid_argument("fiberlet trace requires at least two points");
        bool containsSeed = false;
        for (const auto& point : trace.pointsBaseXYZ) {
            validateFinite(point, "trace point");
            containsSeed = containsSeed || point == trace.seedBaseXYZ;
        }
        if (!containsSeed)
            throw std::invalid_argument("fiberlet trace path does not contain its seed");
        if (trace.pointsBaseXYZ.size() > std::numeric_limits<std::uint64_t>::max() - pointCount) {
            throw std::invalid_argument("fiberlet trace point count overflows uint64");
        }
        pointCount += trace.pointsBaseXYZ.size();
    }

    std::vector<FieldBlock> fields;
    fields.push_back(makeField(TraceOrdinal, Scalar::U64, traces.size()));
    fields.push_back(makeField(TraceSeedX, Scalar::F64, traces.size()));
    fields.push_back(makeField(TraceSeedY, Scalar::F64, traces.size()));
    fields.push_back(makeField(TraceSeedZ, Scalar::F64, traces.size()));
    fields.push_back(makeField(TraceSeedPresence, Scalar::F32, traces.size()));
    fields.push_back(makeField(TraceTotalCost, Scalar::F64, traces.size()));
    fields.push_back(makeField(TracePathLength, Scalar::F64, traces.size()));
    fields.push_back(makeField(TracePointOffsets, Scalar::U64, traces.size() + 1));
    fields.push_back(makeField(TracePointX, Scalar::F64, pointCount));
    fields.push_back(makeField(TracePointY, Scalar::F64, pointCount));
    fields.push_back(makeField(TracePointZ, Scalar::F64, pointCount));
    const auto field = [&](Field id) -> FieldBlock& {
        return *std::find_if(fields.begin(), fields.end(), [id](const auto& candidate) { return candidate.id == id; });
    };
    appendLittle(field(TracePointOffsets).decoded, std::uint64_t{0});
    std::uint64_t offset = 0;
    for (const auto& trace : traces) {
        appendLittle(field(TraceOrdinal).decoded, trace.ordinal);
        appendLittle(field(TraceSeedX).decoded, trace.seedBaseXYZ[0]);
        appendLittle(field(TraceSeedY).decoded, trace.seedBaseXYZ[1]);
        appendLittle(field(TraceSeedZ).decoded, trace.seedBaseXYZ[2]);
        appendLittle(field(TraceSeedPresence).decoded, trace.seedPresence);
        appendLittle(field(TraceTotalCost).decoded, trace.totalMetricCost);
        appendLittle(field(TracePathLength).decoded, trace.pathLengthPredictionVoxels);
        for (const auto& point : trace.pointsBaseXYZ) {
            appendLittle(field(TracePointX).decoded, point[0]);
            appendLittle(field(TracePointY).decoded, point[1]);
            appendLittle(field(TracePointZ).decoded, point[2]);
        }
        offset += trace.pointsBaseXYZ.size();
        appendLittle(field(TracePointOffsets).decoded, offset);
    }
    return encodePayload(config, FiberletStorageChunkKind::FiberTraces, traces.size(), pointCount, 0.0F, 0.0F, std::move(fields));
}

FiberletDecodedAnchors deserializeFiberletAnchors(std::span<const std::byte> bytes)
{
    auto payload = decodePayload(bytes, FiberletStorageChunkKind::Anchors);
    const Scalar coordinateScalar = unsignedScalar(payload.config.coordinateBits);
    const auto& z = requireField(payload, KeyZ, coordinateScalar, payload.recordCount).second;
    const auto& y = requireField(payload, KeyY, coordinateScalar, payload.recordCount).second;
    const auto& x = requireField(payload, KeyX, coordinateScalar, payload.recordCount).second;
    const auto& variant = requireField(payload, Variant, Scalar::U8, payload.recordCount).second;
    const bool compactDirections = payload.config.profile != FiberletStorageProfile::Float32Cache;
    const bool quantizedPositions = payload.config.profile == FiberletStorageProfile::CompactQuantized;
    const auto& axisX = requireField(payload, AxisX, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const auto& axisY = requireField(payload, AxisY, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const std::vector<std::byte>* axisZ = compactDirections ? nullptr : &requireField(payload, AxisZ, Scalar::F32, payload.recordCount).second;
    const auto& predictionAxisX = requireField(payload, PredictionAxisX, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const auto& predictionAxisY = requireField(payload, PredictionAxisY, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const std::vector<std::byte>* predictionAxisZ =
        compactDirections ? nullptr : &requireField(payload, PredictionAxisZ, Scalar::F32, payload.recordCount).second;
    const auto& predictionPresence =
        requireField(payload, PredictionPresence, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const auto& normalX = requireField(payload, NormalX, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const auto& normalY = requireField(payload, NormalY, compactDirections ? Scalar::U8 : Scalar::F32, payload.recordCount).second;
    const std::vector<std::byte>* normalZ = compactDirections ? nullptr : &requireField(payload, NormalZ, Scalar::F32, payload.recordCount).second;
    const auto& scoringFlags = requireField(payload, ScoringFlags, Scalar::U8, payload.recordCount).second;
    const std::vector<std::byte>* positionX =
        quantizedPositions ? nullptr : &requireField(payload, PositionX, Scalar::F32, payload.recordCount).second;
    const std::vector<std::byte>* positionY =
        quantizedPositions ? nullptr : &requireField(payload, PositionY, Scalar::F32, payload.recordCount).second;
    const std::vector<std::byte>* positionZ =
        quantizedPositions ? nullptr : &requireField(payload, PositionZ, Scalar::F32, payload.recordCount).second;
    const std::size_t expectedFields = quantizedPositions ? 12 : compactDirections ? 15 : 18;
    if (payload.fields.size() != expectedFields)
        throw std::invalid_argument("fiberlet anchor payload contains unknown fields");

    FiberletDecodedAnchors result;
    result.config = payload.config;
    result.anchors.reserve(static_cast<std::size_t>(payload.recordCount));
    for (std::size_t index = 0; index < payload.recordCount; ++index) {
        FiberletStoredAnchor anchor;
        anchor.key =
            decodeKey(payload.config, {readUnsigned(z, coordinateScalar, index), readUnsigned(y, coordinateScalar, index), readUnsigned(x, coordinateScalar, index)}, readLittle<std::uint8_t>(variant, index));
        const auto flags = readLittle<std::uint8_t>(scoringFlags, index);
        if ((flags & ~std::uint8_t{7}) != 0)
            throw std::invalid_argument("fiberlet anchor scoring flags are invalid");
        anchor.predictionValid = (flags & 1U) != 0;
        anchor.predictionPresenceValid = (flags & 2U) != 0;
        anchor.normalValid = (flags & 4U) != 0;
        if (compactDirections) {
            const auto decoded =
                vc::lasagna::decodeCompactNormalFromRaw(readLittle<std::uint8_t>(axisX, index), readLittle<std::uint8_t>(axisY, index));
            anchor.fittedAxisXYZ = cv::Vec3f(decoded[0], decoded[1], decoded[2]);
            const auto predictionDecoded =
                vc::lasagna::decodeCompactNormalFromRaw(readLittle<std::uint8_t>(predictionAxisX, index), readLittle<std::uint8_t>(predictionAxisY, index));
            anchor.predictionAxisXYZ = cv::Vec3f(predictionDecoded[0], predictionDecoded[1], predictionDecoded[2]);
            anchor.predictionPresence = static_cast<float>(readLittle<std::uint8_t>(predictionPresence, index)) / 255.0F;
            const auto normalDecoded =
                vc::lasagna::decodeCompactNormalFromRaw(readLittle<std::uint8_t>(normalX, index), readLittle<std::uint8_t>(normalY, index));
            anchor.normalXYZ = cv::Vec3f(normalDecoded[0], normalDecoded[1], normalDecoded[2]);
            if (quantizedPositions) {
                const double scale = static_cast<double>(payload.config.positionQuantumBaseVoxels) / payload.config.predictionToBaseScale;
                anchor.positionPredictionXYZ = cv::Vec3f(
                    static_cast<float>(anchor.key.coordinateZYX[2] * scale),
                    static_cast<float>(anchor.key.coordinateZYX[1] * scale),
                    static_cast<float>(anchor.key.coordinateZYX[0] * scale));
            } else {
                anchor.positionPredictionXYZ =
                    cv::Vec3f(readLittle<float>(*positionX, index * 4), readLittle<float>(*positionY, index * 4), readLittle<float>(*positionZ, index * 4));
            }
        } else {
            anchor.positionPredictionXYZ =
                cv::Vec3f(readLittle<float>(*positionX, index * 4), readLittle<float>(*positionY, index * 4), readLittle<float>(*positionZ, index * 4));
            anchor.fittedAxisXYZ =
                cv::Vec3f(readLittle<float>(axisX, index * 4), readLittle<float>(axisY, index * 4), readLittle<float>(*axisZ, index * 4));
            anchor.predictionAxisXYZ =
                cv::Vec3f(readLittle<float>(predictionAxisX, index * 4), readLittle<float>(predictionAxisY, index * 4), readLittle<float>(*predictionAxisZ, index * 4));
            anchor.predictionPresence = readLittle<float>(predictionPresence, index * 4);
            anchor.normalXYZ =
                cv::Vec3f(readLittle<float>(normalX, index * 4), readLittle<float>(normalY, index * 4), readLittle<float>(*normalZ, index * 4));
        }
        validateFinite(anchor.positionPredictionXYZ, "decoded anchor position");
        validateFinite(anchor.fittedAxisXYZ, "decoded anchor axis");
        if (!std::isfinite(anchor.predictionPresence))
            throw std::invalid_argument("decoded fiberlet anchor presence is not finite");
        if (anchor.predictionValid)
            validateFinite(anchor.predictionAxisXYZ, "decoded anchor prediction axis");
        if (anchor.normalValid)
            validateFinite(anchor.normalXYZ, "decoded anchor normal");
        if (!result.anchors.empty() && !(result.anchors.back().key < anchor.key))
            throw std::invalid_argument("decoded fiberlet anchors are not strictly sorted");
        result.anchors.push_back(anchor);
    }
    return result;
}

FiberletDecodedPrefixes deserializeFiberletPrefixes(std::span<const std::byte> bytes)
{
    auto payload = decodePayload(bytes, FiberletStorageChunkKind::FiberletPrefix);
    const Scalar coordinateScalar = unsignedScalar(payload.config.coordinateBits);
    const Scalar deltaScalar = signedScalar(payload.config.deltaBits);
    const Scalar countScalar = unsignedScalar(payload.config.routeCountBits);
    const Scalar latticeScalar = signedScalar(payload.config.routeLatticeBits);
    const Scalar costScalar = payload.config.costBits == 8 ? Scalar::U8 : payload.config.costBits == 16 ? Scalar::U16 : Scalar::F32;
    const auto get = [&](Field id, Scalar scalar) -> const std::vector<std::byte>& {
        return requireField(payload, id, scalar, payload.recordCount).second;
    };
    const auto& firstZ = get(FirstZ, coordinateScalar);
    const auto& firstY = get(FirstY, coordinateScalar);
    const auto& firstX = get(FirstX, coordinateScalar);
    const auto& firstVariant = get(FirstVariant, Scalar::U8);
    const auto& secondDZ = get(SecondDZ, deltaScalar);
    const auto& secondDY = get(SecondDY, deltaScalar);
    const auto& secondDX = get(SecondDX, deltaScalar);
    const auto& secondVariant = get(SecondVariant, Scalar::U8);
    const auto& interior = get(InteriorCount, countScalar);
    const auto& entryU = get(EntryU, latticeScalar);
    const auto& entryV = get(EntryV, latticeScalar);
    const auto& exitU = get(ExitU, latticeScalar);
    const auto& exitV = get(ExitV, latticeScalar);
    const auto& length = get(PathLength, Scalar::F32);
    const bool floatCache = payload.config.profile == FiberletStorageProfile::Float32Cache;
    const std::vector<std::byte>* totalCost = floatCache ? nullptr : &get(TotalCost, costScalar);
    const std::vector<std::byte>* invalidPredictionCost = floatCache ? &get(InvalidPredictionCost, Scalar::F32) : nullptr;
    const std::vector<std::byte>* alignmentCost = floatCache ? &get(AlignmentCost, Scalar::F32) : nullptr;
    const std::vector<std::byte>* isotropicSmoothnessCost = floatCache ? &get(IsotropicSmoothnessCost, Scalar::F32) : nullptr;
    const std::vector<std::byte>* tangentSmoothnessCost = floatCache ? &get(TangentSmoothnessCost, Scalar::F32) : nullptr;
    const std::vector<std::byte>* normalSmoothnessCost = floatCache ? &get(NormalSmoothnessCost, Scalar::F32) : nullptr;
    std::array<const std::vector<std::byte>*, 3> firstStep{};
    std::array<const std::vector<std::byte>*, 3> lastStep{};
    for (int axis = 0; axis < 3; ++axis) {
        firstStep[axis] = &get(static_cast<Field>(FirstStepX + axis), Scalar::F32);
        lastStep[axis] = &get(static_cast<Field>(LastStepX + axis), Scalar::F32);
    }
    const std::size_t expectedFields = floatCache ? 25 : 21;
    if (payload.fields.size() != expectedFields)
        throw std::invalid_argument("fiberlet prefix payload contains unknown fields");
    if (!std::isfinite(payload.costOffset) || !std::isfinite(payload.costScale) || payload.costScale < 0.0F)
        throw std::invalid_argument("fiberlet cost affine range is invalid");

    FiberletDecodedPrefixes result;
    result.config = payload.config;
    result.prefixes.reserve(static_cast<std::size_t>(payload.recordCount));
    for (std::size_t index = 0; index < payload.recordCount; ++index) {
        FiberletStoredPrefix prefix;
        prefix.id.first =
            decodeKey(payload.config, {readUnsigned(firstZ, coordinateScalar, index), readUnsigned(firstY, coordinateScalar, index), readUnsigned(firstX, coordinateScalar, index)}, readLittle<std::uint8_t>(firstVariant, index));
        prefix.id.second = prefix.id.first;
        const std::array<std::int64_t, 3> delta{readSigned(secondDZ, deltaScalar, index), readSigned(secondDY, deltaScalar, index), readSigned(secondDX, deltaScalar, index)};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if ((delta[axis] > 0 && prefix.id.second.coordinateZYX[axis] > std::numeric_limits<std::int64_t>::max() - delta[axis]) ||
                (delta[axis] < 0 && prefix.id.second.coordinateZYX[axis] < std::numeric_limits<std::int64_t>::min() - delta[axis]))
                throw std::invalid_argument("fiberlet second endpoint coordinate overflows int64");
            prefix.id.second.coordinateZYX[axis] += delta[axis];
        }
        prefix.id.second.variant = readLittle<std::uint8_t>(secondVariant, index);
        if (prefix.id.second.variant > 1 || !(prefix.id.first < prefix.id.second))
            throw std::invalid_argument("decoded fiberlet endpoints are not canonical");
        const auto count = readUnsigned(interior, countScalar, index);
        if (count > std::numeric_limits<std::uint16_t>::max())
            throw std::invalid_argument("fiberlet interior count exceeds logical range");
        prefix.interiorPointCount = static_cast<std::uint16_t>(count);
        const auto lattice = [&](const auto& field) {
            const auto value = readSigned(field, latticeScalar, index);
            if (value < std::numeric_limits<std::int16_t>::min() || value > std::numeric_limits<std::int16_t>::max())
                throw std::invalid_argument("fiberlet lattice coordinate exceeds logical range");
            return static_cast<std::int16_t>(value);
        };
        prefix.entryUV = {lattice(entryU), lattice(entryV)};
        prefix.exitUV = {lattice(exitU), lattice(exitV)};
        prefix.pathLengthPredictionVoxels = readLittle<float>(length, index * 4);
        if (floatCache) {
            prefix.cost = {
                readLittle<float>(*invalidPredictionCost, index * 4),
                readLittle<float>(*alignmentCost, index * 4),
                readLittle<float>(*isotropicSmoothnessCost, index * 4),
                readLittle<float>(*tangentSmoothnessCost, index * 4),
                readLittle<float>(*normalSmoothnessCost, index * 4),
            };
        } else if (payload.config.profile == FiberletStorageProfile::CompactDirectionsFixedCost) {
            constexpr float densityMaximum = 256.0F;
            const float normalized = static_cast<float>(readUnsigned(*totalCost, costScalar, index)) / 65535.0F;
            prefix.cost.alignment = densityMaximum * normalized * normalized * prefix.pathLengthPredictionVoxels;
        } else {
            prefix.cost.alignment = payload.costOffset + payload.costScale * static_cast<float>(readUnsigned(*totalCost, costScalar, index));
        }
        for (int axis = 0; axis < 3; ++axis) {
            prefix.firstStepBaseXYZ[axis] = readLittle<float>(*firstStep[axis], index * 4);
            prefix.lastStepBaseXYZ[axis] = readLittle<float>(*lastStep[axis], index * 4);
        }
        const auto validCost = [](float value) { return std::isfinite(value) && value >= 0.0F; };
        if (!(prefix.pathLengthPredictionVoxels > 0.0F) || !std::isfinite(prefix.pathLengthPredictionVoxels) ||
            !validCost(prefix.cost.invalidPrediction) || !validCost(prefix.cost.alignment) || !validCost(prefix.cost.isotropicSmoothness) ||
            !validCost(prefix.cost.tangentSmoothness) || !validCost(prefix.cost.normalSmoothness))
            throw std::invalid_argument("decoded fiberlet length or cost is invalid");
        validateFinite(prefix.firstStepBaseXYZ, "decoded prefix first step");
        validateFinite(prefix.lastStepBaseXYZ, "decoded prefix last step");
        if (!(prefix.firstStepBaseXYZ.dot(prefix.firstStepBaseXYZ) > 0.0F) || !(prefix.lastStepBaseXYZ.dot(prefix.lastStepBaseXYZ) > 0.0F))
            throw std::invalid_argument("decoded fiberlet prefix endpoint step is zero");
        if (!result.prefixes.empty() && !(result.prefixes.back().id < prefix.id))
            throw std::invalid_argument("decoded fiberlet prefixes are not strictly sorted");
        result.prefixes.push_back(prefix);
    }
    return result;
}

FiberletDecodedRoutes deserializeFiberletRoutes(std::span<const std::byte> bytes)
{
    auto payload = decodePayload(bytes, FiberletStorageChunkKind::FiberletRoutes);
    const Scalar latticeScalar = signedScalar(payload.config.routeLatticeBits);
    const auto& offsets = requireField(payload, RouteOffsets, Scalar::U32, payload.recordCount + 1).second;
    const auto& u = requireField(payload, MiddleU, latticeScalar, payload.auxiliaryCount).second;
    const auto& v = requireField(payload, MiddleV, latticeScalar, payload.auxiliaryCount).second;
    const auto& costOffsets = requireField(payload, CostOffsets, Scalar::U32, payload.recordCount + 1).second;
    const auto costCount = readLittle<std::uint32_t>(costOffsets, payload.recordCount * 4);
    const auto& costDensities = requireField(payload, SegmentCostDensity, Scalar::U16, costCount).second;
    if (payload.fields.size() != 5)
        throw std::invalid_argument("fiberlet route payload contains unknown fields");
    FiberletDecodedRoutes result;
    result.config = payload.config;
    result.routes.resize(static_cast<std::size_t>(payload.recordCount));
    std::uint32_t previous = 0;
    std::uint32_t previousCost = 0;
    for (std::size_t index = 0; index <= payload.recordCount; ++index) {
        const auto current = readLittle<std::uint32_t>(offsets, index * 4);
        const auto currentCost = readLittle<std::uint32_t>(costOffsets, index * 4);
        if (current < previous || current > payload.auxiliaryCount)
            throw std::invalid_argument("fiberlet route offsets are invalid");
        if (currentCost < previousCost || currentCost > costCount)
            throw std::invalid_argument("fiberlet route cost offsets are invalid");
        if (index > 0) {
            auto& storedRoute = result.routes[index - 1];
            auto& route = storedRoute.middleUV;
            route.reserve(current - previous);
            for (std::uint32_t point = previous; point < current; ++point) {
                const auto ru = readSigned(u, latticeScalar, point);
                const auto rv = readSigned(v, latticeScalar, point);
                if (ru < std::numeric_limits<std::int16_t>::min() || ru > std::numeric_limits<std::int16_t>::max() ||
                    rv < std::numeric_limits<std::int16_t>::min() || rv > std::numeric_limits<std::int16_t>::max())
                    throw std::invalid_argument("fiberlet route coordinate exceeds logical range");
                route.push_back({static_cast<std::int16_t>(ru), static_cast<std::int16_t>(rv)});
            }
            if (currentCost == previousCost)
                throw std::invalid_argument("fiberlet route has no segment cost densities");
            storedRoute.segmentCostDensities.reserve(currentCost - previousCost);
            for (std::uint32_t sample = previousCost; sample < currentCost; ++sample) {
                storedRoute.segmentCostDensities.push_back(decodeFiberletStoredCostDensity(readLittle<std::uint16_t>(costDensities, sample * 2)));
            }
        }
        previous = current;
        previousCost = currentCost;
    }
    if (previous != payload.auxiliaryCount || previousCost != costCount)
        throw std::invalid_argument("fiberlet route offsets do not consume all points");
    return result;
}

FiberletDecodedTraces deserializeFiberletTraces(std::span<const std::byte> bytes)
{
    auto payload = decodePayload(bytes, FiberletStorageChunkKind::FiberTraces);
    if (payload.config.profile != FiberletStorageProfile::Float64Traces)
        throw std::invalid_argument("Fiber traces require the float64 trace profile");
    const auto& ordinal = requireField(payload, TraceOrdinal, Scalar::U64, payload.recordCount).second;
    const auto& seedX = requireField(payload, TraceSeedX, Scalar::F64, payload.recordCount).second;
    const auto& seedY = requireField(payload, TraceSeedY, Scalar::F64, payload.recordCount).second;
    const auto& seedZ = requireField(payload, TraceSeedZ, Scalar::F64, payload.recordCount).second;
    const auto& seedPresence = requireField(payload, TraceSeedPresence, Scalar::F32, payload.recordCount).second;
    const auto& totalCost = requireField(payload, TraceTotalCost, Scalar::F64, payload.recordCount).second;
    const auto& pathLength = requireField(payload, TracePathLength, Scalar::F64, payload.recordCount).second;
    const auto& offsets = requireField(payload, TracePointOffsets, Scalar::U64, payload.recordCount + 1).second;
    const auto& pointX = requireField(payload, TracePointX, Scalar::F64, payload.auxiliaryCount).second;
    const auto& pointY = requireField(payload, TracePointY, Scalar::F64, payload.auxiliaryCount).second;
    const auto& pointZ = requireField(payload, TracePointZ, Scalar::F64, payload.auxiliaryCount).second;
    if (payload.fields.size() != 11)
        throw std::invalid_argument("fiberlet trace payload contains unknown fields");

    FiberletDecodedTraces result;
    result.config = payload.config;
    result.traces.reserve(static_cast<std::size_t>(payload.recordCount));
    std::uint64_t previous = 0;
    if (readLittle<std::uint64_t>(offsets, 0) != 0)
        throw std::invalid_argument("fiberlet trace offsets must start at zero");
    for (std::size_t index = 0; index < payload.recordCount; ++index) {
        const auto current = readLittle<std::uint64_t>(offsets, (index + 1) * sizeof(std::uint64_t));
        if (current < previous || current > payload.auxiliaryCount || current - previous < 2) {
            throw std::invalid_argument("fiberlet trace point offsets are invalid");
        }
        FiberletStoredTrace trace;
        trace.ordinal = readLittle<std::uint64_t>(ordinal, index * sizeof(std::uint64_t));
        trace.seedBaseXYZ = {
            readLittle<double>(seedX, index * sizeof(double)),
            readLittle<double>(seedY, index * sizeof(double)),
            readLittle<double>(seedZ, index * sizeof(double)),
        };
        trace.seedPresence = readLittle<float>(seedPresence, index * sizeof(float));
        trace.totalMetricCost = readLittle<double>(totalCost, index * sizeof(double));
        trace.pathLengthPredictionVoxels = readLittle<double>(pathLength, index * sizeof(double));
        trace.pointsBaseXYZ.reserve(static_cast<std::size_t>(current - previous));
        bool containsSeed = false;
        for (std::uint64_t point = previous; point < current; ++point) {
            const cv::Vec3d value{
                readLittle<double>(pointX, point * sizeof(double)),
                readLittle<double>(pointY, point * sizeof(double)),
                readLittle<double>(pointZ, point * sizeof(double)),
            };
            validateFinite(value, "decoded trace point");
            containsSeed = containsSeed || value == trace.seedBaseXYZ;
            trace.pointsBaseXYZ.push_back(value);
        }
        validateFinite(trace.seedBaseXYZ, "decoded trace seed");
        if (!std::isfinite(trace.seedPresence) || !(trace.totalMetricCost >= 0.0) || !std::isfinite(trace.totalMetricCost) ||
            !(trace.pathLengthPredictionVoxels > 0.0) || !std::isfinite(trace.pathLengthPredictionVoxels) || !containsSeed) {
            throw std::invalid_argument("decoded fiberlet trace metadata is invalid");
        }
        result.traces.push_back(std::move(trace));
        previous = current;
    }
    if (previous != payload.auxiliaryCount)
        throw std::invalid_argument("fiberlet trace offsets do not consume all points");
    return result;
}

std::vector<std::byte> materializeFiberletPayload(std::span<const std::byte> bytes)
{
    if (bytes.size() < 13)
        throw std::invalid_argument("fiberlet payload is truncated");
    const auto kind = static_cast<FiberletStorageChunkKind>(readLittle<std::uint8_t>(bytes, 12));
    auto payload = decodePayload(bytes, kind);
    std::vector<FieldBlock> fields;
    fields.reserve(payload.fields.size());
    for (auto& [id, value] : payload.fields) {
        FieldBlock field;
        field.id = id;
        field.scalar = value.first;
        field.count = value.second.size() / scalarBytes(value.first);
        field.decoded = std::move(value.second);
        fields.push_back(std::move(field));
    }
    return encodePayload(payload.config, payload.kind, payload.recordCount, payload.auxiliaryCount, payload.costOffset, payload.costScale, std::move(fields), false);
}

}  // namespace vc::fiber_tracer
