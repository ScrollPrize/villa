#include "io/j2k_chunk.hpp"

#include <algorithm>
#include <cstring>
#include <array>
#include <cmath>

#include "helirec/common.hpp"

#if HELIREC_WITH_OPENJPEG
#include <openjpeg.h>
#endif

#if HELIREC_WITH_OPENJPH
#include <ojph_codestream.h>
#include <ojph_file.h>
#include <ojph_mem.h>
#include <ojph_params.h>
#endif

namespace helirec::j2k {

namespace {

std::uint32_t be32(const std::uint8_t* p) { return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3]; }
std::uint32_t be16(const std::uint8_t* p) { return (std::uint32_t(p[0]) << 8) | p[1]; }
std::uint32_t le32(const std::uint8_t* p) { return (std::uint32_t(p[3]) << 24) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[1]) << 8) | p[0]; }
void put_be32(std::uint8_t* p, std::uint32_t v) { p[0] = v >> 24, p[1] = v >> 16, p[2] = v >> 8, p[3] = v; }
void put_be64(std::uint8_t* p, std::uint64_t v) { put_be32(p, static_cast<std::uint32_t>(v >> 32)), put_be32(p + 4, static_cast<std::uint32_t>(v)); }
void put_le32(std::uint8_t* p, std::uint32_t v) { p[0] = v, p[1] = v >> 8, p[2] = v >> 16, p[3] = v >> 24; }

// Layout of a blosc2 contiguous frame's header, as far as it is read here (all multi-byte fields big-endian,
// each preceded by its msgpack type byte): "b2frame" at 2, header length at 11, frame length at 16, the
// uncompressed size at 30. The chunk follows the header: a 32-byte blosc2 header (little-endian nbytes at 4,
// blocksize at 8, cbytes at 12), the block's offset at 32, and at that offset the stream's length and bytes.
constexpr std::size_t kFrameHeaderMin = 62, kChunkHeader = 32;
constexpr std::uint8_t kCodecUser = 6;       // "user-defined codec" in a blosc2 chunk's flags (bits 5..7)
constexpr std::uint8_t kCodecJ2k = 40;       // the JPEG 2000 codec's id (blosc2_htj2k)

[[noreturn]] void bad(const std::string& what) { throw Error("compressed frame chunk: " + what); }

}  // namespace

ChunkHeader chunk_header(const ReadAt& read, std::size_t size, std::size_t nbytes, int sample_bytes) {
    auto take = [&](void* dst, std::size_t n, std::size_t off) {
        if (off > size || n > size - off) bad("truncated container");
        read(dst, n, off);
    };
    std::array<std::uint8_t, 64> f{};
    take(f.data(), kFrameHeaderMin, 0);
    if (std::memcmp(f.data() + 2, "b2frame", 7) != 0) bad("not a blosc2 frame");
    const std::size_t header = be32(f.data() + 11);
    if (header < kFrameHeaderMin) bad("unexpected blosc2 frame header length");
    take(f.data(), kChunkHeader + 4, header);
    const std::size_t c_nbytes = le32(f.data() + 4), c_block = le32(f.data() + 8), c_bytes = le32(f.data() + 12);
    if (c_nbytes != nbytes) bad("uncompressed size does not match the dataset chunk");
    if (c_block != c_nbytes) bad("more than one block per chunk is not supported");
    if (f[3] != sample_bytes) bad("sample size does not match the dataset dtype");
    if ((f[2] >> 5) != kCodecUser || f[22] != kCodecJ2k) bad("not the JPEG 2000 codec (blosc2 codec 40)");
    if (c_bytes < kChunkHeader + 8 || c_bytes > size - header) bad("truncated blosc2 chunk");
    const std::size_t start = le32(f.data() + kChunkHeader);
    if (start < kChunkHeader + 4 || start > c_bytes - 4) bad("unexpected block offset");
    take(f.data(), 4, header + start);
    const std::size_t length = le32(f.data());
    if (length > c_bytes - start - 4 || length < 4) bad("unexpected stream length");
    ChunkHeader out;
    out.offset = header + start + 4;
    out.size = length;
    if (sample_bytes == 4) {
        if (length < 64) bad("truncated float32 header");
        take(f.data(), 64, out.offset);
        auto le16 = [](const std::uint8_t* p) { return unsigned(p[0]) | (unsigned(p[1]) << 8); };
        auto le64 = [](const std::uint8_t* p) { return std::uint64_t(le32(p)) | (std::uint64_t(le32(p + 4)) << 32); };
        auto f64 = [&](const std::uint8_t* p) { const auto bits = le64(p); double v; std::memcpy(&v, &bits, 8); return v; };
        // blosc2_htj2k B2JPF32 v1: uint quantization plus an affine float32 mapping.
        // Signed codestream samples are centred about 2^(quant_bits-1).
        if (std::memcmp(f.data(), "B2JPF32\0", 8) || le16(f.data() + 8) != 1 || le16(f.data() + 10) != 64)
            bad("unsupported B2JPF32 header");
        out.floating = true;
        out.quant_bits = f[12];
        const unsigned flags = le16(f.data() + 14);
        if ((out.quant_bits != 8 && out.quant_bits != 16 && out.quant_bits != 32) || f[13] != 0 || (flags & ~15u))
            bad("unsupported float quantization or NaN policy");
        out.constant = (flags & 4) != 0;
        out.scale_min = f64(f.data() + 24);
        out.scale_max = f64(f.data() + 32);
        if (!std::isfinite(out.scale_min) || !std::isfinite(out.scale_max) || out.scale_max < out.scale_min)
            bad("invalid float quantization range");
        const auto payload = le64(f.data() + 16);
        if (payload != length - 64 || (!out.constant && payload < 4)) bad("invalid float payload length");
        out.offset += 64;
        out.size = static_cast<std::size_t>(payload);
    } else if (sample_bytes != 2) {
        bad("unsupported sample size");
    }
    if (!out.constant) {
        take(f.data(), 4, out.offset);
        if (be32(f.data()) != 0xff4fff51u) bad("the stream is not a JPEG 2000 codestream");
    }
    return out;
}

Codestream codestream_of_chunk(const std::uint8_t* chunk, std::size_t size, std::size_t nbytes) {
    const auto h = chunk_header([&](void* dst, std::size_t n, std::size_t off) { std::memcpy(dst, chunk + off, n); }, size, nbytes, 2);
    return {chunk + h.offset, h.size};
}

Info info(const Codestream& cs) {
    const std::uint8_t* p = cs.data;
    const std::uint8_t* const end = cs.data + cs.size;
    if (cs.size < 2 + 2 + 38 + 3 || be32(p) != 0xff4fff51u) bad("the stream is not a JPEG 2000 codestream");
    Info out;
    p += 2;  // at SIZ
    {
        const std::size_t length = be16(p + 2);
        if (length < 41 || length > static_cast<std::size_t>(end - p) - 2) bad("truncated SIZ marker");
        const std::uint32_t rsiz = be16(p + 4), x1 = be32(p + 6), y1 = be32(p + 10), x0 = be32(p + 14), y0 = be32(p + 18);
        out.high_throughput = (rsiz & 0x4000u) != 0;
        out.width = static_cast<int>(x1 - x0);
        out.height = static_cast<int>(y1 - y0);
        out.components = static_cast<int>(be16(p + 38));
        out.bit_depth = (p[40] & 0x7f) + 1;
        out.is_signed = (p[40] & 0x80) != 0;
        p += 2 + length;
    }
    while (p + 4 <= end && p[0] == 0xff && p[1] != 0x90) {  // up to the first tile part
        const std::size_t length = be16(p + 2);
        if (length < 2 || length > static_cast<std::size_t>(end - p) - 2) bad("truncated main header");
        if (p[1] == 0x52 && length >= 12) {            // COD
            out.layers = static_cast<int>(be16(p + 6));
            out.levels = p[9];
            out.reversible = p[13] == 1;
            return out;
        }
        p += 2 + length;
    }
    return out;
}

Info info(const ReadAt& read, const ChunkHeader& chunk) {
    auto take = [&](void* dst, std::size_t n, std::size_t off) {
        if (off > chunk.size || n > chunk.size - off) bad("truncated main header");
        read(dst, n, chunk.offset + off);
    };
    std::array<std::uint8_t, 6> marker;
    take(marker.data(), marker.size(), 0);
    if (be32(marker.data()) != 0xff4fff51u) bad("missing SIZ marker");
    const auto siz_length = be16(marker.data() + 4);
    if (siz_length < 41) bad("invalid SIZ marker length");
    std::vector<std::uint8_t> header(4 + siz_length);
    take(header.data(), header.size(), 0);
    // Inspect only SIZ and COD. Comment and other marker bodies may be large;
    // seeking past them keeps storage() cheap even on a remote filesystem.
    std::size_t at = header.size();
    for (;;) {
        take(marker.data(), 4, at);
        if (marker[0] != 0xff || marker[1] == 0x90 || marker[1] == 0x93) bad("missing COD marker");
        const std::size_t length = be16(marker.data() + 2);
        if (length < 2 || length + 2 > chunk.size - at) bad("truncated main header");
        if (marker[1] == 0x52) {
            const auto tail = header.size();
            header.resize(tail + 2 + length);
            take(header.data() + tail, 2 + length, at);
            return info({header.data(), header.size()});
        }
        at += 2 + length;
    }
}

bool available() {
#if HELIREC_WITH_OPENJPEG
    return true;
#else
    return false;
#endif
}

bool openjph_available() {
#if HELIREC_WITH_OPENJPH
    return true;
#else
    return false;
#endif
}

#if HELIREC_WITH_OPENJPH
namespace {
// A bounded, buffered codestream view. A seek changes the cursor without reading
// the skipped bytes. In RPCL/RLCP streams, this omits the high-resolution packets.
class SeekableInput final : public ojph::infile_base {
    const ReadAt& read_;
    std::size_t base_, size_, at_ = 0, begin_ = 0, valid_ = 0;
    std::vector<std::uint8_t> buffer_;
    std::vector<std::uint8_t> whole_;
  public:
    SeekableInput(const ReadAt& read, const ChunkHeader& h, bool selective)
        : read_(read), base_(h.offset), size_(h.size), buffer_(h.floating ? 65536 : 8192) {
        if (!selective) {
            whole_.resize(size_);
            read_(whole_.data(), size_, base_);
        }
    }
    std::size_t read(void* dst, std::size_t n) override {
        n = std::min(n, size_ - at_);
        if (!whole_.empty()) {
            std::memcpy(dst, whole_.data() + at_, n);
            at_ += n;
            return n;
        }
        std::size_t done = 0;
        while (done < n) {
            if (at_ < begin_ || at_ - begin_ >= valid_) {
                begin_ = at_;
                valid_ = std::min(buffer_.size(), size_ - at_);
                read_(buffer_.data(), valid_, base_ + at_);
            }
            const auto take = std::min(n - done, valid_ - (at_ - begin_));
            std::memcpy(static_cast<std::uint8_t*>(dst) + done, buffer_.data() + (at_ - begin_), take);
            at_ += take;
            done += take;
        }
        return done;
    }
    int seek(ojph::si64 off, enum ojph::infile_base::seek origin) override {
        const auto base = static_cast<ojph::si64>(origin == OJPH_SEEK_CUR ? at_ : origin == OJPH_SEEK_END ? size_ : 0);
        if (off < -base || off > static_cast<ojph::si64>(size_) - base) return -1;
        at_ = static_cast<std::size_t>(base + off);
        return 0;
    }
    ojph::si64 tell() override { return static_cast<ojph::si64>(at_); }
    bool eof() override { return at_ == size_; }
};
}  // namespace
#endif

void decode_htj2k(const ReadAt& read, std::size_t size, std::size_t nbytes, int sample_bytes,
                  int stored_width, int stored_height, int reduce, void* out, bool selective) try {
#if HELIREC_WITH_OPENJPH
    if (reduce < 0 || reduce > 3 || stored_width < 1 || stored_height < 1) bad("invalid shape or resolution");
    const auto h = chunk_header(read, size, nbytes, sample_bytes);
    const int width = 1 + ((stored_width - 1) >> reduce), height = 1 + ((stored_height - 1) >> reduce);
    if (h.constant) {
        std::fill_n(static_cast<float*>(out), static_cast<std::size_t>(width) * height, static_cast<float>(h.scale_min));
        return;
    }
    SeekableInput input(read, h, selective);
    ojph::codestream cs;
    cs.read_headers(&input);
    const auto siz = cs.access_siz();
    const auto extent = siz.get_image_extent(), origin = siz.get_image_offset();
    if (siz.get_num_components() != 1 || origin.x || origin.y || extent.x != static_cast<unsigned>(stored_width) ||
        extent.y != static_cast<unsigned>(stored_height) || siz.get_downsampling(0).x != 1 || siz.get_downsampling(0).y != 1)
        bad("unexpected HTJ2K component geometry");
    const int bits = static_cast<int>(siz.get_bit_depth(0));
    const bool signed_samples = siz.is_signed(0);
    if (h.floating ? bits != h.quant_bits : (bits > 16 || signed_samples)) bad("unexpected HTJ2K sample precision");
    if (static_cast<unsigned>(reduce) > cs.access_cod().get_num_decompositions())
        bad("the codestream has no resolution " + std::to_string(reduce) + " levels below the stored one");
    cs.restrict_input_resolution(reduce, reduce);
    cs.set_planar(true);
    if (siz.get_recon_width(0) != static_cast<unsigned>(width) || siz.get_recon_height(0) != static_cast<unsigned>(height))
        bad("unexpected HTJ2K output shape");
    cs.create();
    const double scale = h.floating ? (h.scale_max - h.scale_min) / static_cast<double>((std::uint64_t{1} << bits) - 1) : 0;
    for (int y = 0; y < height; ++y) {
        ojph::ui32 comp = 0;
        auto* line = cs.pull(comp);
        if (!line || comp != 0 || !line->i32) bad("missing HTJ2K output row");
        const auto offset = static_cast<std::size_t>(y) * width;
        if (h.floating) {
            float* dst = static_cast<float*>(out) + offset;
            for (int x = 0; x < width; ++x) {
                const std::uint64_t q = signed_samples ? static_cast<std::int64_t>(line->i32[x]) + (std::int64_t{1} << (bits - 1))
                                                       : static_cast<std::uint32_t>(line->i32[x]);
                dst[x] = static_cast<float>(h.scale_min + static_cast<double>(q) * scale);
            }
        } else {
            auto* dst = static_cast<std::uint16_t*>(out) + offset;
            for (int x = 0; x < width; ++x) dst[x] = static_cast<std::uint16_t>(std::clamp(line->i32[x], 0, 65535));
        }
    }
#else
    (void)read; (void)size; (void)nbytes; (void)sample_bytes; (void)stored_width; (void)stored_height; (void)reduce; (void)out; (void)selective;
    bad("float32 HTJ2K needs a build with HELIREC_OPENJPH=ON");
#endif
} catch (const char* error) {
    bad(std::string("OpenJPH: ") + error);
}

#if HELIREC_WITH_OPENJPEG

namespace {

struct Reader {
    const std::uint8_t* data;
    std::size_t size, at;
};
OPJ_SIZE_T read_fn(void* buffer, OPJ_SIZE_T n, void* user) {
    Reader* r = static_cast<Reader*>(user);
    if (r->at >= r->size) return static_cast<OPJ_SIZE_T>(-1);
    const std::size_t take = std::min<std::size_t>(n, r->size - r->at);
    std::memcpy(buffer, r->data + r->at, take);
    r->at += take;
    return take;
}
OPJ_OFF_T skip_fn(OPJ_OFF_T n, void* user) {
    Reader* r = static_cast<Reader*>(user);
    const std::size_t to = static_cast<std::size_t>(std::clamp<OPJ_OFF_T>(static_cast<OPJ_OFF_T>(r->at) + n, 0, static_cast<OPJ_OFF_T>(r->size)));
    const OPJ_OFF_T moved = static_cast<OPJ_OFF_T>(to) - static_cast<OPJ_OFF_T>(r->at);
    r->at = to;
    return moved;
}
OPJ_BOOL seek_fn(OPJ_OFF_T to, void* user) {
    Reader* r = static_cast<Reader*>(user);
    if (to < 0 || static_cast<std::size_t>(to) > r->size) return OPJ_FALSE;
    r->at = static_cast<std::size_t>(to);
    return OPJ_TRUE;
}

struct Writer {
    std::vector<std::uint8_t> bytes;
    std::size_t at = 0;
};
OPJ_SIZE_T write_fn(void* buffer, OPJ_SIZE_T n, void* user) {
    Writer* w = static_cast<Writer*>(user);
    if (w->at + n > w->bytes.size()) w->bytes.resize(w->at + n);
    std::memcpy(w->bytes.data() + w->at, buffer, n);
    w->at += n;
    return n;
}
OPJ_OFF_T write_skip_fn(OPJ_OFF_T n, void* user) {
    Writer* w = static_cast<Writer*>(user);
    if (n < 0 && static_cast<std::size_t>(-n) > w->at) return -1;
    w->at = static_cast<std::size_t>(static_cast<OPJ_OFF_T>(w->at) + n);
    if (w->at > w->bytes.size()) w->bytes.resize(w->at);
    return n;
}
OPJ_BOOL write_seek_fn(OPJ_OFF_T to, void* user) {
    Writer* w = static_cast<Writer*>(user);
    if (to < 0) return OPJ_FALSE;
    w->at = static_cast<std::size_t>(to);
    if (w->at > w->bytes.size()) w->bytes.resize(w->at);
    return OPJ_TRUE;
}

void quiet(const char*, void*) {}

}  // namespace

void decode(const Codestream& cs, int width, int height, std::uint16_t* out, int reduce) {
    if (reduce < 0 || (reduce > 0 && reduce > info(cs).levels))
        bad("the codestream has no resolution " + std::to_string(reduce) + " levels below the stored one");
    Reader reader{cs.data, cs.size, 0};
    opj_stream_t* stream = opj_stream_create(std::min<std::size_t>(cs.size, 1u << 20), OPJ_TRUE);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_image_t* image = nullptr;
    std::string error;
    if (!stream || !codec) {
        error = "OpenJPEG could not be set up";
    } else {
        opj_stream_set_user_data(stream, &reader, nullptr);
        opj_stream_set_user_data_length(stream, cs.size);
        opj_stream_set_read_function(stream, read_fn);
        opj_stream_set_skip_function(stream, skip_fn);
        opj_stream_set_seek_function(stream, seek_fn);
        opj_set_error_handler(codec, quiet, nullptr);
        opj_set_warning_handler(codec, quiet, nullptr);
        opj_dparameters_t par;
        opj_set_default_decoder_parameters(&par);
        par.cp_reduce = static_cast<OPJ_UINT32>(reduce);
        if (!opj_setup_decoder(codec, &par) || !opj_read_header(stream, codec, &image) || !opj_decode(codec, stream, image) ||
            !opj_end_decompress(codec, stream)) {
            error = "OpenJPEG could not decode the codestream";
        } else if (image->numcomps != 1 || static_cast<int>(image->comps[0].w) != width || static_cast<int>(image->comps[0].h) != height ||
                   image->comps[0].sgnd || image->comps[0].prec > 16 || !image->comps[0].data) {
            error = "the codestream is not one unsigned component of " + std::to_string(width) + " x " + std::to_string(height) + " values of at most 16 bits";
        } else {
            const OPJ_INT32* v = image->comps[0].data;
            const std::size_t n = static_cast<std::size_t>(width) * height;
            for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::uint16_t>(std::clamp<OPJ_INT32>(v[i], 0, 65535));
        }
    }
    if (image) opj_image_destroy(image);
    if (codec) opj_destroy_codec(codec);
    if (stream) opj_stream_destroy(stream);
    if (!error.empty()) bad(error);
}

std::vector<std::uint8_t> encode_chunk(const std::uint16_t* values, int width, int height) {
    opj_cparameters_t par;
    opj_set_default_encoder_parameters(&par);
    par.irreversible = 0;
    par.tcp_numlayers = 1;
    par.tcp_rates[0] = 0;  // everything: exact
    par.cp_disto_alloc = 1;
    par.numresolution = 1;
    for (int n = std::min(width, height); n >= 64 && par.numresolution < 6; n /= 2) ++par.numresolution;
    opj_image_cmptparm_t cp;
    std::memset(&cp, 0, sizeof cp);
    cp.dx = cp.dy = 1;
    cp.w = static_cast<OPJ_UINT32>(width);
    cp.h = static_cast<OPJ_UINT32>(height);
    cp.prec = 16;
    cp.sgnd = 0;
    opj_image_t* image = opj_image_create(1, &cp, OPJ_CLRSPC_GRAY);
    if (!image) bad("OpenJPEG could not make an image");
    image->x0 = image->y0 = 0;
    image->x1 = static_cast<OPJ_UINT32>(width);
    image->y1 = static_cast<OPJ_UINT32>(height);
    for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) image->comps[0].data[i] = values[i];
    Writer writer;
    opj_stream_t* stream = opj_stream_create(1u << 20, OPJ_FALSE);
    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    bool ok = stream && codec;
    if (ok) {
        opj_stream_set_user_data(stream, &writer, nullptr);
        opj_stream_set_write_function(stream, write_fn);
        opj_stream_set_skip_function(stream, write_skip_fn);
        opj_stream_set_seek_function(stream, write_seek_fn);
        opj_set_error_handler(codec, quiet, nullptr);
        opj_set_warning_handler(codec, quiet, nullptr);
        ok = opj_setup_encoder(codec, &par, image) && opj_start_compress(codec, image, stream) && opj_encode(codec, stream) && opj_end_compress(codec, stream);
    }
    if (stream) opj_stream_destroy(stream);  // flushes
    if (codec) opj_destroy_codec(codec);
    opj_image_destroy(image);
    if (!ok) bad("OpenJPEG could not encode");
    const std::vector<std::uint8_t>& cs = writer.bytes;

    // the container: frame header, one chunk of one block, no trailer
    const std::size_t nbytes = static_cast<std::size_t>(width) * height * 2;
    const std::size_t header = kFrameHeaderMin, start = kChunkHeader + 4, c_bytes = start + 4 + cs.size();
    std::vector<std::uint8_t> out(header + c_bytes, 0);
    std::uint8_t* f = out.data();
    f[0] = 0x9e, f[1] = 0xa8;
    std::memcpy(f + 2, "b2frame", 8);
    f[10] = 0xd2, put_be32(f + 11, static_cast<std::uint32_t>(header));
    f[15] = 0xcf, put_be64(f + 16, out.size());
    f[24] = 0xa4, f[25] = 0x12, f[26] = 0x00, f[27] = 0x06, f[28] = 0x01;
    f[29] = 0xd3, put_be64(f + 30, nbytes);
    f[38] = 0xd3, put_be64(f + 39, c_bytes);
    f[47] = 0xd2, put_be32(f + 48, 2);
    f[52] = 0xd2, put_be32(f + 53, static_cast<std::uint32_t>(nbytes));
    f[57] = 0xd2, put_be32(f + 58, static_cast<std::uint32_t>(nbytes));
    std::uint8_t* c = f + header;
    c[0] = 5, c[1] = 1, c[2] = static_cast<std::uint8_t>(kCodecUser << 5), c[3] = 2;
    put_le32(c + 4, static_cast<std::uint32_t>(nbytes));
    put_le32(c + 8, static_cast<std::uint32_t>(nbytes));
    put_le32(c + 12, static_cast<std::uint32_t>(c_bytes));
    c[22] = kCodecJ2k;
    put_le32(c + kChunkHeader, static_cast<std::uint32_t>(start));
    put_le32(c + start, static_cast<std::uint32_t>(cs.size()));
    std::memcpy(c + start + 4, cs.data(), cs.size());
    return out;
}

#else

void decode(const Codestream&, int, int, std::uint16_t*, int) {
    throw Error("frames stored as JPEG 2000 need helirec built with OpenJPEG (libopenjp2 >= 2.5: apt install libopenjp2-7-dev, brew install openjpeg)");
}
std::vector<std::uint8_t> encode_chunk(const std::uint16_t*, int, int) {
    throw Error("making JPEG 2000 chunks needs helirec built with OpenJPEG (libopenjp2 >= 2.5)");
}

#endif

}  // namespace helirec::j2k
