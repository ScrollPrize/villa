#include "io/j2k_chunk.hpp"

#include <algorithm>
#include <cstring>

#include "helirec/common.hpp"

#if HELIREC_WITH_OPENJPEG
#include <openjpeg.h>
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

Codestream codestream_of_chunk(const std::uint8_t* chunk, std::size_t size, std::size_t nbytes) {
    if (size < kFrameHeaderMin || std::memcmp(chunk + 2, "b2frame", 7) != 0) bad("not a blosc2 frame");
    const std::size_t header = be32(chunk + 11);
    if (header < kFrameHeaderMin || header + kChunkHeader + 8 > size) bad("blosc2 frame header of an unexpected length");
    const std::uint8_t* c = chunk + header;
    const std::size_t c_nbytes = le32(c + 4), c_block = le32(c + 8), c_bytes = le32(c + 12);
    if (c_nbytes != nbytes) bad("it holds " + std::to_string(c_nbytes) + " bytes of values where the dataset's chunk has " + std::to_string(nbytes));
    if (c_block != c_nbytes) bad("more than one block per chunk is not supported");
    if (c[3] != 2) bad("values are not two bytes wide");
    if ((c[2] >> 5) != kCodecUser || c[22] != kCodecJ2k) bad("its codec is not the JPEG 2000 codec (blosc2 codec " + std::to_string(c[2] >> 5) + ", user codec " + std::to_string(c[22]) + ")");
    if (header + c_bytes > size) bad("truncated");
    const std::size_t start = le32(c + kChunkHeader);
    if (start < kChunkHeader + 4 || start + 4 > c_bytes) bad("unexpected block offset");
    const std::size_t length = le32(c + start);
    if (start + 4 + length > c_bytes || length < 4) bad("unexpected stream length");
    Codestream cs{c + start + 4, length};
    if (be32(cs.data) != 0xff4fff51u) bad("the stream is not a JPEG 2000 codestream");
    return cs;
}

Info info(const Codestream& cs) {
    const std::uint8_t* p = cs.data;
    const std::uint8_t* const end = cs.data + cs.size;
    if (cs.size < 2 + 2 + 38 + 3 || be32(p) != 0xff4fff51u) bad("the stream is not a JPEG 2000 codestream");
    Info out;
    p += 2;  // at SIZ
    {
        const std::size_t length = be16(p + 2);
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
        if (p[1] == 0x52 && p + 4 + 10 <= end) {            // COD
            out.layers = static_cast<int>(be16(p + 6));
            out.levels = p[9];
            out.reversible = p[13] == 1;
        }
        p += 2 + length;
    }
    return out;
}

bool available() {
#if HELIREC_WITH_OPENJPEG
    return true;
#else
    return false;
#endif
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

void decode(const Codestream& cs, int width, int height, std::uint16_t* out) {
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

void decode(const Codestream&, int, int, std::uint16_t*) {
    throw Error("frames stored as JPEG 2000 need helirec built with OpenJPEG (libopenjp2 >= 2.5: apt install libopenjp2-7-dev, brew install openjpeg)");
}
std::vector<std::uint8_t> encode_chunk(const std::uint16_t*, int, int) {
    throw Error("making JPEG 2000 chunks needs helirec built with OpenJPEG (libopenjp2 >= 2.5)");
}

#endif

}  // namespace helirec::j2k
