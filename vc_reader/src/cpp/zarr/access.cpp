// How remote arrays are requested: their request URLs and credentials, and
// reading one metadata object of a node.
#include "vc_reader/catalog.hpp"
#include "vc_reader/s3_listing.hpp"
#include "vc_reader/sftp.hpp"
#include "vc_reader/zarr.hpp"

#include <curl/curl.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

namespace vcr::zarr {
namespace {
std::string env(const char* name)
{
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::pair<std::string, std::string> splitS3(const std::string& url)
{
    const auto rest = url.substr(5);
    const auto slash = rest.find('/');
    auto bucket = rest.substr(0, slash);
    if (bucket.empty())
        throw std::invalid_argument("no bucket in '" + url + "'");
    return {bucket, slash == std::string::npos ? std::string() : rest.substr(slash + 1)};
}

std::string percentEncode(const std::string& key)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out;
    for (const char c : key) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || std::string_view("/-_.~").find(c) != std::string_view::npos) {
            out += c;
        } else {
            out += '%';
            out += digits[u >> 4];
            out += digits[u & 15];
        }
    }
    return out;
}

std::string trim(std::string s)
{
    const auto b = s.find_first_not_of(" \t\r\n\"");
    const auto e = s.find_last_not_of(" \t\r\n\"");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

// section -> key -> value of an INI file (AWS config/credentials, VC3D.ini).
using Ini = std::map<std::string, std::map<std::string, std::string>>;
Ini readIni(const std::filesystem::path& path)
{
    Ini out;
    std::ifstream in(path);
    std::string line, section;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';')
            continue;
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        const auto eq = line.find('=');
        if (eq != std::string::npos)
            out[section][trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return out;
}

std::filesystem::path awsFile(const char* variable, const char* name)
{
    const auto configured = env(variable);
    if (!configured.empty())
        return configured;
    return std::filesystem::path(env("HOME")) / ".aws" / name;
}

std::string lookup(const Ini& ini, const std::string& section, const std::string& key)
{
    const auto s = ini.find(section);
    if (s == ini.end())
        return {};
    const auto k = s->second.find(key);
    return k == s->second.end() ? std::string() : k->second;
}

std::size_t headerRegion(char* data, std::size_t size, std::size_t count, void* user)
{
    const std::string line(data, size * count);
    const auto colon = line.find(':');
    if (colon != std::string::npos) {
        std::string name = line.substr(0, colon);
        for (auto& c : name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (name == "x-amz-bucket-region")
            *static_cast<std::string*>(user) = trim(line.substr(colon + 1));
    }
    return size * count;
}

std::string bucketRoot(const Access& access, const std::string& bucket)
{
    if (!access.endpoint.empty())
        return access.endpoint + "/" + bucket + "/";
    if (bucket.find('.') != std::string::npos)  // virtual-host names with dots fail TLS verification
        return "https://s3." + access.auth.region + ".amazonaws.com/" + bucket + "/";
    return "https://" + bucket + ".s3." + access.auth.region + ".amazonaws.com/";
}
}  // namespace

bool isPublicLocation(const std::string& location)
{
    return catalog::openDataS3Url(catalog::resolveOpenDataUrl(location)).has_value();
}

bool isRemote(const std::string& location)
{
    return location.starts_with("http://") || location.starts_with("https://") || location.starts_with("s3://") ||
           location.starts_with("s3+") || sftp::isSftpUrl(location);
}

std::string httpUrl(const std::string& location)
{
    if (location.starts_with("s3://") || location.starts_with("s3+"))
        return catalog::resolveOpenDataUrl(location);
    return location;
}

std::string Access::url(const std::string& location) const
{
    if (kind != Kind::S3 || !location.starts_with("s3://"))
        return httpUrl(location);
    const auto [bucket, key] = splitS3(location);
    return bucketRoot(*this, bucket) + percentEncode(key);
}

std::string Access::cacheUrl(const std::string& location) const
{
    return location.starts_with("s3://") || kind == Kind::S3 ? location : httpUrl(location);
}

std::optional<Auth> awsCredentials(const std::string& profileName)
{
    const auto key = env("AWS_ACCESS_KEY_ID"), secret = env("AWS_SECRET_ACCESS_KEY");
    std::string profile = profileName;
    if (profile.empty()) {
        if (!key.empty() && !secret.empty())
            return Auth{"s3", key, secret, env("AWS_SESSION_TOKEN"), env("AWS_REGION")};
        profile = env("AWS_PROFILE");
        if (profile.empty())
            profile = env("AWS_DEFAULT_PROFILE");
        if (profile.empty())
            profile = "default";
    }
    const auto credentials = readIni(awsFile("AWS_SHARED_CREDENTIALS_FILE", "credentials"));
    const auto config = readIni(awsFile("AWS_CONFIG_FILE", "config"));
    const auto configSection = profile == "default" ? std::string("default") : "profile " + profile;
    const auto region = lookup(config, configSection, "region");
    auto id = lookup(credentials, profile, "aws_access_key_id");
    auto secretKey = lookup(credentials, profile, "aws_secret_access_key");
    if (!id.empty() && !secretKey.empty())
        return Auth{"s3", id, secretKey, lookup(credentials, profile, "aws_session_token"), region};
    id = lookup(config, configSection, "aws_access_key_id");
    secretKey = lookup(config, configSection, "aws_secret_access_key");
    if (!id.empty() && !secretKey.empty())
        return Auth{"s3", id, secretKey, lookup(config, configSection, "aws_session_token"), region};
    return std::nullopt;
}

std::optional<std::string> bucketRegion(const std::string& bucket)
{
    static std::mutex mutex;
    static std::map<std::string, std::optional<std::string>> regions;
    {
        std::lock_guard lock(mutex);
        if (const auto it = regions.find(bucket); it != regions.end())
            return it->second;
    }
    static std::once_flag init;
    std::call_once(init, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    std::string region;
    if (CURL* h = curl_easy_init()) {
        const auto url = "https://" + bucket + ".s3.amazonaws.com/";
        curl_easy_setopt(h, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
        curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(h, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &headerRegion);
        curl_easy_setopt(h, CURLOPT_HEADERDATA, &region);
        curl_easy_perform(h);  // 403 (private) and 301 (other region) carry the header too
        curl_easy_cleanup(h);
    }
    std::optional<std::string> result;
    if (!region.empty())
        result = region;
    std::lock_guard lock(mutex);
    regions[bucket] = result;
    return result;
}

namespace {
Access signedAccess(const std::string& location, Auth keys, std::string endpoint, const std::string& fallbackRegion)
{
    const auto bucket = splitS3(location).first;
    if (endpoint.empty())
        endpoint = env("AWS_ENDPOINT_URL_S3");
    if (endpoint.empty())
        endpoint = env("AWS_ENDPOINT_URL");
    while (!endpoint.empty() && endpoint.back() == '/')
        endpoint.pop_back();
    if (keys.region.empty() && endpoint.empty())
        keys.region = bucketRegion(bucket).value_or("");
    if (keys.region.empty())
        keys.region = env("AWS_REGION");
    if (keys.region.empty())
        keys.region = env("AWS_DEFAULT_REGION");
    if (keys.region.empty())
        keys.region = fallbackRegion;
    if (keys.region.empty())
        keys.region = "us-east-1";
    keys.kind = "s3";
    Access access;
    access.kind = Access::Kind::S3;
    access.auth = std::move(keys);
    access.endpoint = std::move(endpoint);
    return access;
}
}  // namespace

Access s3Access(const std::string& location, Auth keys, std::string endpoint)
{
    if (!location.starts_with("s3://"))
        throw std::invalid_argument("S3 credentials need an s3:// URL, not '" + location + "'");
    return signedAccess(location, std::move(keys), std::move(endpoint), {});
}

Access basicAccess(std::string user, std::string password)
{
    Access access;
    access.kind = Access::Kind::Basic;
    access.auth = Auth{"basic", std::move(user), std::move(password), "", ""};
    return access;
}

void reloadEnvironmentKeys(Access& access)
{
    if (!access.fromEnvironment)
        return;
    const auto keys = awsCredentials();
    if (!keys)
        throw HttpDenied("no AWS credentials found (environment, AWS_PROFILE or ~/.aws)");
    access.auth.user = keys->user;
    access.auth.secret = keys->secret;
    access.auth.token = keys->token;
}

std::pair<std::string, std::string> loadBasicAuth(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::filesystem::filesystem_error("cannot read credentials", path, std::error_code(errno, std::generic_category()));
    const auto data = nlohmann::json::parse(in);
    if (!data.is_object())
        throw std::invalid_argument(path.string() + ": expected a JSON object");
    const auto& root = data.contains("auth") && data.at("auth").is_object() ? data.at("auth") : data;
    const auto field = [&](const char* name) {
        const auto it = root.find(name);
        return it == root.end() || it->is_null() ? std::optional<std::string>{}
                                                 : std::optional<std::string>(it->is_string() ? it->get<std::string>() : it->dump());
    };
    auto user = field("username");
    if (!user)
        user = field("user");
    const auto password = field("password");
    if (!user || user->empty() || !password)
        throw std::invalid_argument(path.string() + ": needs 'username' (or 'user') and 'password'");
    return {*user, *password};
}

std::vector<std::string> listChildren(const Access& access, const std::string& location)
{
    const auto s3 = location.starts_with("s3://") ? std::optional<std::string>(location) : catalog::openDataS3Url(location);
    if (!s3)
        throw std::invalid_argument("cannot list " + location + ": plain HTTP has no listing and the group has no "
                                    "consolidated .zmetadata; open its children by name");
    const auto [bucket, key] = splitS3(*s3);
    auto origin = access.kind == Access::Kind::S3 ? bucketRoot(access, bucket) : "https://" + bucket + ".s3.amazonaws.com/";
    origin.pop_back();
    auto prefixKey = key;
    while (!prefixKey.empty() && prefixKey.back() == '/')
        prefixKey.pop_back();
    if (!prefixKey.empty())
        prefixKey += '/';
    const s3::Prefix prefix{origin, prefixKey};
    HttpOptions options;
    options.connections = 1;
    options.auth = access.auth;
    HttpClient http(options);
    std::vector<std::string> names;
    std::string token;
    do {
        std::vector<std::uint8_t> body;
        if (!http.get(s3::listUrl(prefix, "/", token), body))
            throw NotFound("cannot list " + *s3 + ": bucket not found");
        const auto page = s3::parseListPage(std::string_view(reinterpret_cast<const char*>(body.data()), body.size()));
        for (const auto& child : page.commonPrefixes) {
            auto name = child.substr(prefixKey.size());
            while (!name.empty() && name.back() == '/')
                name.pop_back();
            if (!name.empty())
                names.push_back(std::move(name));
        }
        token = page.nextContinuationToken;
    } while (!token.empty());
    return names;
}

Access resolveAccess(const std::string& location, AuthMode mode, bool offline)
{
    const bool s3 = location.starts_with("s3://");
    if (!isRemote(location) || mode == AuthMode::Anonymous)
        return {};
    // Cached reads need neither signing credentials nor a bucket region.
    if (offline && s3)
        return {};
    if (mode == AuthMode::Auto && (!s3 || isPublicLocation(location)))
        return {};
    if (!s3)
        throw std::invalid_argument("AWS credentials need an s3:// URL, not '" + location + "'");
    auto keys = awsCredentials();
    if (!keys) {
        if (mode == AuthMode::Auto)  // no credentials: anonymous, as for a public bucket
            return {};
        throw HttpDenied("no AWS credentials found (environment, AWS_PROFILE or ~/.aws)");
    }
    // The bucket's own region, then AWS_REGION, then the profile's.
    const auto profileRegion = std::move(keys->region);
    keys->region.clear();
    auto access = signedAccess(location, std::move(*keys), {}, profileRegion);
    access.fromEnvironment = true;
    return access;
}

MetadataReader metadataReader(const Access& access, int connections)
{
    HttpOptions options;
    options.connections = connections;
    options.auth = access.auth;
    auto http = std::make_shared<HttpClient>(options);
    return [access, http](const std::string& location, const std::string& name) -> std::optional<std::string> {
        if (!isRemote(location)) {
            const auto path = std::filesystem::path(location) / name;
            std::error_code ec;
            const auto status = std::filesystem::status(path, ec);
            if (status.type() == std::filesystem::file_type::not_found)
                return std::nullopt;
            std::ifstream in(path, std::ios::binary);
            if (!in)
                throw std::filesystem::filesystem_error("cannot read", path, std::error_code(errno, std::generic_category()));
            std::ostringstream text;
            text << in.rdbuf();
            return text.str();
        }
        std::vector<std::uint8_t> body;
        try {
            if (!http->get(access.url(location + "/" + name), body))
                return std::nullopt;
        } catch (const HttpDenied& e) {
            if (access.kind == Access::Kind::Anonymous)
                throw HttpDenied(std::string(e.what()) + " (requested anonymously: pass credentials)");
            throw;
        }
        return std::string(body.begin(), body.end());
    };
}

}  // namespace vcr::zarr
