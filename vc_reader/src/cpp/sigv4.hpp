// AWS Signature Version 4 for S3 GETs, and the SHA-256 / HMAC-SHA256 it needs.
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

namespace vcr {

class Sha256 {
public:
    using Digest = std::array<std::uint8_t, 32>;

    void update(const void* data, std::size_t n)
    {
        const auto* p = static_cast<const std::uint8_t*>(data);
        total_ += n;
        if (used_) {
            const std::size_t take = std::min(n, sizeof buf_ - used_);
            std::memcpy(buf_ + used_, p, take);
            used_ += take;
            p += take;
            n -= take;
            if (used_ < sizeof buf_)
                return;
            block(buf_);
            used_ = 0;
        }
        for (; n >= sizeof buf_; p += sizeof buf_, n -= sizeof buf_)
            block(p);
        std::memcpy(buf_, p, n);
        used_ = n;
    }

    void update(const std::string& s) { update(s.data(), s.size()); }

    Digest finish()
    {
        const std::uint64_t bits = total_ * 8;
        const std::uint8_t pad = 0x80;
        update(&pad, 1);
        const std::uint8_t zero = 0;
        while (used_ != 56)
            update(&zero, 1);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i)
            len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        update(len, 8);
        Digest d;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                d[4 * i + j] = static_cast<std::uint8_t>(h_[i] >> (24 - 8 * j));
        return d;
    }

    static Digest of(const std::string& s)
    {
        Sha256 h;
        h.update(s);
        return h.finish();
    }

private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const std::uint8_t* p)
    {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = std::uint32_t(p[4 * i]) << 24 | std::uint32_t(p[4 * i + 1]) << 16 |
                   std::uint32_t(p[4 * i + 2]) << 8 | std::uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    std::uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                           0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::uint8_t buf_[64];
    std::size_t used_ = 0;
    std::uint64_t total_ = 0;
};

inline Sha256::Digest hmac_sha256(const std::string& key, const std::string& msg)
{
    std::uint8_t k[64] = {};
    if (key.size() > sizeof k) {
        const auto d = Sha256::of(key);
        std::memcpy(k, d.data(), d.size());
    } else {
        std::memcpy(k, key.data(), key.size());
    }
    std::uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }
    Sha256 inner;
    inner.update(ipad, sizeof ipad);
    inner.update(msg);
    const auto ih = inner.finish();
    Sha256 outer;
    outer.update(opad, sizeof opad);
    outer.update(ih.data(), ih.size());
    return outer.finish();
}

inline std::string to_hex(const Sha256::Digest& d)
{
    static const char* digits = "0123456789abcdef";
    std::string s(2 * d.size(), '0');
    for (std::size_t i = 0; i < d.size(); ++i) {
        s[2 * i] = digits[d[i] >> 4];
        s[2 * i + 1] = digits[d[i] & 15];
    }
    return s;
}

inline std::string digest_string(const Sha256::Digest& d) { return std::string(d.begin(), d.end()); }

// AWS URI encoding: every byte but the unreserved characters (and '/' if keep_slash) as %XX.
inline std::string aws_uri_encode(const std::string& s, bool keep_slash)
{
    static const char* digits = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (keep_slash && c == '/')) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += digits[c >> 4];
            out += digits[c & 15];
        }
    }
    return out;
}

inline std::string percent_decode(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

struct S3Key {
    std::string access_key, secret_key, session_token, region;
};

// "YYYYMMDDTHHMMSSZ" for a UTC time.
inline std::string amz_date(std::time_t t)
{
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[17];
    std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
    return buf;
}

// Headers that sign a GET of `url` (http(s)://host[:port]/path[?query]) for S3.
// `extra` headers (lowercase names) are sent and signed as well.
inline std::vector<std::pair<std::string, std::string>> sigv4_get_headers(
    const std::string& url, const S3Key& key, const std::string& date,
    const std::vector<std::pair<std::string, std::string>>& extra = {})
{
    // Split the URL.
    const std::size_t scheme_end = url.find("://");
    const std::string scheme = scheme_end == std::string::npos ? "https" : url.substr(0, scheme_end);
    const std::size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
    const std::size_t path_start = std::min(url.find('/', host_start), url.find('?', host_start));
    std::string host = url.substr(host_start, path_start == std::string::npos ? std::string::npos
                                                                              : path_start - host_start);
    if ((scheme == "https" && host.ends_with(":443")) || (scheme == "http" && host.ends_with(":80")))
        host.erase(host.rfind(':'));  // curl leaves default ports out of the Host header
    std::string path, query;
    if (path_start != std::string::npos) {
        const std::string rest = url.substr(path_start);
        const std::size_t q = rest.find('?');
        path = rest.substr(0, q);
        if (q != std::string::npos)
            query = rest.substr(q + 1);
    }
    const std::string canonical_uri = path.empty() ? "/" : aws_uri_encode(percent_decode(path), true);
    std::vector<std::pair<std::string, std::string>> params;
    for (std::size_t pos = 0; pos < query.size();) {
        std::size_t amp = query.find('&', pos);
        if (amp == std::string::npos)
            amp = query.size();
        const std::string item = query.substr(pos, amp - pos);
        if (!item.empty()) {
            const std::size_t eq = item.find('=');
            params.emplace_back(aws_uri_encode(percent_decode(item.substr(0, eq)), false),
                                eq == std::string::npos ? "" : aws_uri_encode(percent_decode(item.substr(eq + 1)), false));
        }
        pos = amp + 1;
    }
    std::sort(params.begin(), params.end());
    std::string canonical_query;
    for (const auto& [k, v] : params)
        canonical_query += (canonical_query.empty() ? "" : "&") + k + "=" + v;

    static const std::string empty_payload = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    std::vector<std::pair<std::string, std::string>> headers = extra;
    headers.emplace_back("x-amz-content-sha256", empty_payload);
    headers.emplace_back("x-amz-date", date);
    if (!key.session_token.empty())
        headers.emplace_back("x-amz-security-token", key.session_token);
    std::vector<std::pair<std::string, std::string>> signed_headers = headers;
    signed_headers.emplace_back("host", host);
    std::sort(signed_headers.begin(), signed_headers.end());
    std::string canonical_headers, names;
    for (const auto& [k, v] : signed_headers) {
        canonical_headers += k + ":" + v + "\n";
        names += (names.empty() ? "" : ";") + k;
    }
    const std::string canonical_request =
        "GET\n" + canonical_uri + "\n" + canonical_query + "\n" + canonical_headers + "\n" + names + "\n" + empty_payload;

    const std::string day = date.substr(0, 8);
    const std::string scope = day + "/" + key.region + "/s3/aws4_request";
    const std::string to_sign =
        "AWS4-HMAC-SHA256\n" + date + "\n" + scope + "\n" + to_hex(Sha256::of(canonical_request));
    auto k = hmac_sha256("AWS4" + key.secret_key, day);
    k = hmac_sha256(digest_string(k), key.region);
    k = hmac_sha256(digest_string(k), "s3");
    k = hmac_sha256(digest_string(k), "aws4_request");
    const std::string signature = to_hex(hmac_sha256(digest_string(k), to_sign));
    headers.emplace_back("authorization", "AWS4-HMAC-SHA256 Credential=" + key.access_key + "/" + scope +
                                              ", SignedHeaders=" + names + ", Signature=" + signature);
    return headers;
}

}  // namespace vcr
