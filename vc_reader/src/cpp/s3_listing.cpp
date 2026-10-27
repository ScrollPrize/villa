#include "vc_reader/s3_listing.hpp"

#include <cctype>
#include <charconv>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vcr::s3 {
namespace {

constexpr char kHex[] = "0123456789ABCDEF";

std::string percentEncode(std::string_view value, bool keepSlash = false)
{
    std::string out;
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (keepSlash && c == '/')) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 15]);
        }
    }
    return out;
}

int hexValue(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// encoding-type=url values: percent-encoded, spaces written as '+'.
std::string urlDecode(std::string_view value)
{
    std::string out;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+') {
            out.push_back(' ');
        } else if (value[i] == '%' && i + 2 < value.size() && hexValue(value[i + 1]) >= 0 &&
                   hexValue(value[i + 2]) >= 0) {
            out.push_back(static_cast<char>(hexValue(value[i + 1]) * 16 + hexValue(value[i + 2])));
            i += 2;
        } else {
            out.push_back(value[i]);
        }
    }
    return out;
}

void appendUtf8(std::string& out, unsigned long codePoint)
{
    if (codePoint < 0x80) {
        out.push_back(static_cast<char>(codePoint));
    } else if (codePoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint < 0x110000) {
        out.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
        throw std::runtime_error("S3 listing: invalid XML character reference");
    }
}

// Character data with its entity and character references replaced.
void appendXmlText(std::string& out, std::string_view raw)
{
    for (std::size_t i = 0; i < raw.size();) {
        if (raw[i] != '&') {
            out.push_back(raw[i++]);
            continue;
        }
        const auto end = raw.find(';', i);
        if (end == std::string_view::npos) throw std::runtime_error("S3 listing: unterminated XML reference");
        const auto name = raw.substr(i + 1, end - i - 1);
        if (name == "amp") out.push_back('&');
        else if (name == "lt") out.push_back('<');
        else if (name == "gt") out.push_back('>');
        else if (name == "quot") out.push_back('"');
        else if (name == "apos") out.push_back('\'');
        else if (name.size() > 1 && name[0] == '#') {
            const bool hex = name[1] == 'x' || name[1] == 'X';
            const auto digits = name.substr(hex ? 2 : 1);
            unsigned long codePoint = 0;
            const auto [ptr, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), codePoint, hex ? 16 : 10);
            if (digits.empty() || ec != std::errc{} || ptr != digits.data() + digits.size())
                throw std::runtime_error("S3 listing: invalid XML character reference");
            appendUtf8(out, codePoint);
        } else {
            throw std::runtime_error("S3 listing: unknown XML entity &" + std::string(name) + ";");
        }
        i = end + 1;
    }
}

std::string_view elementName(std::string_view tag)
{
    const auto end = tag.find_first_of(" \t\r\n");
    return end == std::string_view::npos ? tag : tag.substr(0, end);
}

} // namespace

std::optional<Prefix> splitPrefixUrl(std::string_view url)
{
    const auto scheme = url.find("://");
    if (scheme == std::string_view::npos) return std::nullopt;
    const auto hostEnd = url.find('/', scheme + 3);
    if (hostEnd == std::string_view::npos) return std::nullopt;
    auto key = url.substr(hostEnd + 1);
    while (!key.empty() && key.back() == '/') key.remove_suffix(1);
    if (key.empty()) return std::nullopt;
    return Prefix{std::string(url.substr(0, hostEnd)), std::string(key) + "/"};
}

std::string listUrl(const Prefix& prefix, std::string_view delimiter, std::string_view continuationToken, int maxKeys)
{
    if (maxKeys < 1 || maxKeys > 1000) throw std::invalid_argument("S3 listing: max-keys must be in [1, 1000]");
    // Parameters in canonical (sorted) order, as SigV4 signs them.
    std::string url = prefix.origin + "/?";
    if (!continuationToken.empty()) url += "continuation-token=" + percentEncode(continuationToken) + "&";
    if (!delimiter.empty()) url += "delimiter=" + percentEncode(delimiter) + "&";
    url += "encoding-type=url&list-type=2&max-keys=" + std::to_string(maxKeys);
    if (!prefix.key.empty()) url += "&prefix=" + percentEncode(prefix.key);
    return url;
}

ListPage parseListPage(std::string_view xml)
{
    ListPage page;
    std::vector<std::string_view> open;  // the element path
    std::string text;                    // character data since the last tag
    Object object;
    std::string errorCode, errorMessage, token;
    bool truncated = false;
    bool urlEncoded = false;

    // An element closed with its character data.
    const auto close = [&](std::string_view name) {
        if (open.empty() || open.back() != name)
            throw std::runtime_error("S3 listing: mismatched XML element </" + std::string(name) + ">");
        const auto depth = open.size();
        const auto parent = depth >= 2 ? open[depth - 2] : std::string_view{};
        if (open.front() == "Error") {
            if (depth == 2 && name == "Code") errorCode = text;
            if (depth == 2 && name == "Message") errorMessage = text;
        } else if (depth == 2) {
            if (name == "NextContinuationToken") token = text;
            else if (name == "IsTruncated") truncated = text == "true";
            else if (name == "EncodingType") urlEncoded = text == "url";
            else if (name == "Contents") page.objects.push_back(std::exchange(object, {}));
        } else if (depth == 3 && parent == "Contents") {
            if (name == "Key") object.key = text;
            else if (name == "Size") {
                const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), object.size);
                if (text.empty() || ec != std::errc{} || ptr != text.data() + text.size())
                    throw std::runtime_error("S3 listing: invalid object size '" + text + "'");
            }
        } else if (depth == 3 && parent == "CommonPrefixes" && name == "Prefix") {
            page.commonPrefixes.push_back(text);
        }
        open.pop_back();
        text.clear();
    };

    for (std::size_t i = 0; i < xml.size();) {
        const auto lt = xml.find('<', i);
        appendXmlText(text, xml.substr(i, lt == std::string_view::npos ? std::string_view::npos : lt - i));
        if (lt == std::string_view::npos) break;
        if (xml.substr(lt, 9) == "<![CDATA[") {
            const auto end = xml.find("]]>", lt);
            if (end == std::string_view::npos) throw std::runtime_error("S3 listing: unterminated CDATA");
            text.append(xml.substr(lt + 9, end - lt - 9));
            i = end + 3;
            continue;
        }
        if (xml.substr(lt, 4) == "<!--") {
            const auto end = xml.find("-->", lt);
            if (end == std::string_view::npos) throw std::runtime_error("S3 listing: unterminated XML comment");
            i = end + 3;
            continue;
        }
        const auto gt = xml.find('>', lt);
        if (gt == std::string_view::npos) throw std::runtime_error("S3 listing: unterminated XML tag");
        auto tag = xml.substr(lt + 1, gt - lt - 1);
        i = gt + 1;
        if (!tag.empty() && (tag.front() == '?' || tag.front() == '!')) continue;  // declaration, doctype
        if (!tag.empty() && tag.front() == '/') {
            close(elementName(tag.substr(1)));
            continue;
        }
        const bool empty = !tag.empty() && tag.back() == '/';
        if (empty) tag.remove_suffix(1);
        const auto name = elementName(tag);
        if (name.empty()) throw std::runtime_error("S3 listing: empty XML element name");
        if (open.empty() && name != "ListBucketResult" && name != "Error")
            throw std::runtime_error("S3 listing: not a ListObjectsV2 response (<" + std::string(name) + ">)");
        open.push_back(name);
        text.clear();
        if (empty) close(name);
    }
    if (!open.empty()) throw std::runtime_error("S3 listing: unterminated XML element <" + std::string(open.back()) + ">");
    if (!errorCode.empty() || !errorMessage.empty())
        throw std::runtime_error("S3 listing failed: " + errorCode + (errorMessage.empty() ? "" : ": " + errorMessage));

    if (urlEncoded) {
        for (auto& listed : page.objects) listed.key = urlDecode(listed.key);
        for (auto& prefix : page.commonPrefixes) prefix = urlDecode(prefix);
    }
    if (truncated && token.empty()) throw std::runtime_error("S3 listing: truncated page without a continuation token");
    if (truncated) page.nextContinuationToken = std::move(token);
    return page;
}

std::string objectUrl(std::string_view origin, std::string_view key)
{
    return std::string(origin) + "/" + percentEncode(key, true);
}

} // namespace vcr::s3
