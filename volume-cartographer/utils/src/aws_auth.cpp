#include "utils/http_fetch.hpp"

#if UTILS_HAS_CURL

#include "utils/Json.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace utils {

namespace {

// ---------------------------------------------------------------------------
// IMDSv2 (EC2 Instance Metadata Service v2)
//
// The previous implementation resolved instance-role credentials by forking
// the `aws` CLI (`aws configure export-credentials`) on every AwsAuth::load().
// Under high client concurrency (e.g. the recompress tool's 96 inner workers)
// this spawns dozens of `aws` subprocesses simultaneously, each hammering the
// link-local IMDS endpoint. IMDS rate-limits aggressively, so some calls get
// an empty/throttled response and load() returned empty credentials, failing
// the S3 op. Querying IMDSv2 directly (no subprocess) plus caching the
// resolved credentials in-process until shortly before expiry eliminates the
// fork-storm and survives the multi-hour STS rotation on long-running jobs.
// ---------------------------------------------------------------------------

constexpr const char* kImdsBase = "http://169.254.169.254";

size_t imds_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

// Minimal libcurl GET/PUT against the link-local metadata endpoint. Kept
// self-contained (not via HttpClient) because HttpClient::Config embeds an
// AwsAuth and the metadata endpoint must never be SigV4-signed.
std::optional<std::string> imds_request(const std::string& url,
                                        bool put,
                                        const std::string& token)
{
    CURL* curl = curl_easy_init();
    if (!curl) return std::nullopt;

    std::string body;
    struct curl_slist* headers = nullptr;
    if (put) {
        headers = curl_slist_append(
            headers, "X-aws-ec2-metadata-token-ttl-seconds: 21600");
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
    } else if (!token.empty()) {
        std::string h = "X-aws-ec2-metadata-token: " + token;
        headers = curl_slist_append(headers, h.c_str());
    }
    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, imds_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    // Link-local: must be fast. A non-EC2 host has no 169.254.169.254 route,
    // so keep the timeout tight to fail over to other methods quickly.
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);
    curl_easy_setopt(curl, CURLOPT_NOPROXY, "169.254.169.254");

    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK || status < 200 || status >= 300) return std::nullopt;
    return body;
}

struct CachedCreds {
    AwsAuth auth;
    std::chrono::system_clock::time_point expiry{};
    bool has_expiry = false;
};

std::mutex g_cred_mutex;
std::optional<CachedCreds> g_cached;

std::chrono::system_clock::time_point parse_iso8601(const std::string& s)
{
    // Expiration looks like "2026-05-16T18:42:00Z".
    std::tm tm{};
    if (sscanf(s.c_str(), "%d-%d-%dT%d:%d:%dZ",
               &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
               &tm.tm_hour, &tm.tm_min, &tm.tm_sec) != 6) {
        return {};
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
#if defined(_WIN32)
    std::time_t t = _mkgmtime(&tm);
#else
    std::time_t t = timegm(&tm);
#endif
    return std::chrono::system_clock::from_time_t(t);
}

// Returns instance-role creds via IMDSv2, or nullopt if not on EC2 / no role.
std::optional<CachedCreds> fetch_imds_creds()
{
    auto token = imds_request(std::string(kImdsBase) + "/latest/api/token",
                              /*put=*/true, "");
    if (!token || token->empty()) {
        fprintf(stderr, "[AWS] IMDSv2: no token (not on EC2?)\n");
        return std::nullopt;
    }

    auto role = imds_request(
        std::string(kImdsBase) + "/latest/meta-data/iam/security-credentials/",
        false, *token);
    if (!role || role->empty()) {
        fprintf(stderr, "[AWS] IMDSv2: no instance role attached\n");
        return std::nullopt;
    }
    // The listing may contain a trailing newline; take the first line.
    std::string role_name = *role;
    if (auto nl = role_name.find('\n'); nl != std::string::npos)
        role_name.resize(nl);

    auto creds_json = imds_request(
        std::string(kImdsBase) + "/latest/meta-data/iam/security-credentials/"
            + role_name,
        false, *token);
    if (!creds_json || creds_json->empty()) {
        fprintf(stderr, "[AWS] IMDSv2: role '%s' creds fetch failed\n",
                role_name.c_str());
        return std::nullopt;
    }

    try {
        auto j = Json::parse(*creds_json);
        if (!j.contains("AccessKeyId") || !j.contains("SecretAccessKey"))
            return std::nullopt;
        CachedCreds cc;
        cc.auth.access_key = j["AccessKeyId"].get_string();
        cc.auth.secret_key = j["SecretAccessKey"].get_string();
        if (j.contains("Token"))
            cc.auth.session_token = j["Token"].get_string();
        if (j.contains("Expiration")) {
            cc.expiry = parse_iso8601(j["Expiration"].get_string());
            cc.has_expiry =
                cc.expiry != std::chrono::system_clock::time_point{};
        }
        fprintf(stderr,
                "[AWS] IMDSv2: got role '%s' creds (key=%s..., expires %s)\n",
                role_name.c_str(), cc.auth.access_key.substr(0, 8).c_str(),
                j.contains("Expiration")
                    ? j["Expiration"].get_string().c_str() : "(none)");
        return cc;
    } catch (...) {
        fprintf(stderr, "[AWS] IMDSv2: creds JSON parse failed\n");
        return std::nullopt;
    }
}

// Serve instance-role creds from an in-process cache, refreshing ~5 min
// before expiry. Thread-safe; the network fetch happens at most once per
// rotation window regardless of how many workers call load() concurrently.
std::optional<AwsAuth> cached_imds_creds()
{
    std::lock_guard<std::mutex> lk(g_cred_mutex);
    auto now = std::chrono::system_clock::now();

    if (g_cached) {
        if (!g_cached->has_expiry ||
            now + std::chrono::minutes(5) < g_cached->expiry) {
            return g_cached->auth;
        }
    }

    auto fresh = fetch_imds_creds();
    if (fresh) {
        g_cached = std::move(*fresh);
        return g_cached->auth;
    }
    // Refresh failed but a still-valid cached copy exists — keep using it.
    if (g_cached && (!g_cached->has_expiry || now < g_cached->expiry))
        return g_cached->auth;
    return std::nullopt;
}

} // namespace

AwsAuth AwsAuth::load(const std::string& profile)
{
    AwsAuth auth;

    auto getEnv = [](const char* name) -> std::string {
        const char* v = std::getenv(name);
        return v ? v : "";
    };

    // Method 1: `aws configure export-credentials` — resolves SSO, assume-role,
    // credential_process, etc.
    auto tryExportCreds = [&](const std::string& profileArg) -> bool {
        std::string cmd = "aws configure export-credentials";
        if (!profileArg.empty())
            cmd += " --profile " + profileArg;
        cmd += " 2>/dev/null";
        fprintf(stderr, "[AWS] trying: %s\n", cmd.c_str());
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) { fprintf(stderr, "[AWS]   popen failed\n"); return false; }
        std::string output;
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe.get()))
            output += buf;
        if (output.empty()) { fprintf(stderr, "[AWS]   empty output\n"); return false; }
        try {
            auto j = Json::parse(output);
            if (j.contains("AccessKeyId") && j.contains("SecretAccessKey")) {
                auth.access_key = j["AccessKeyId"].get_string();
                auth.secret_key = j["SecretAccessKey"].get_string();
                if (j.contains("SessionToken"))
                    auth.session_token = j["SessionToken"].get_string();
                fprintf(stderr, "[AWS]   got creds: key=%s...  token=%s\n",
                    auth.access_key.substr(0, 8).c_str(),
                    auth.session_token.empty() ? "(none)" : (auth.session_token.substr(0, 20) + "...").c_str());
                return true;
            }
        } catch (...) { fprintf(stderr, "[AWS]   JSON parse failed\n"); }
        return false;
    };

    // Scan ~/.aws/config for SSO-enabled profiles
    auto findSsoProfiles = [&]() -> std::vector<std::string> {
        std::vector<std::string> ssoProfiles;
        std::filesystem::path home = getEnv("HOME");
        if (home.empty()) home = "/tmp";
        std::ifstream cfg(home / ".aws" / "config");
        if (!cfg) return ssoProfiles;
        std::string curSection;
        bool hasSso = false;
        std::string line;
        while (std::getline(cfg, line)) {
            auto start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;
            if (line[0] == '[') {
                if (!curSection.empty() && hasSso)
                    ssoProfiles.push_back(curSection);
                auto end = line.find(']');
                curSection = (end != std::string::npos) ? line.substr(1, end - 1) : "";
                if (curSection.starts_with("profile "))
                    curSection = curSection.substr(8);
                hasSso = false;
                continue;
            }
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            key.erase(key.find_last_not_of(" \t") + 1);
            if (key == "sso_account_id" || key == "sso_session")
                hasSso = true;
        }
        if (!curSection.empty() && hasSso)
            ssoProfiles.push_back(curSection);
        return ssoProfiles;
    };

    {
        auto envProfile = getEnv("AWS_PROFILE");
        bool got = false;

        fprintf(stderr, "[AWS] AwsAuth::load(profile=\"%s\") AWS_PROFILE=\"%s\"\n",
            profile.c_str(), envProfile.c_str());

        // 1. Explicit profile (env or argument) wins — caller asked for a
        //    specific named profile, honour it before instance role.
        if (!envProfile.empty())
            got = tryExportCreds(envProfile);
        else if (profile != "default")
            got = tryExportCreds(profile);

        // 2. EC2 instance role via IMDSv2 (cached, no subprocess). This is
        //    the hot path for the recompress tool running on EC2; serving it
        //    from the in-process cache avoids the `aws` CLI fork-storm that
        //    throttled IMDS and intermittently yielded empty credentials.
        if (!got) {
            if (auto imds = cached_imds_creds()) {
                auth.access_key    = imds->access_key;
                auth.secret_key    = imds->secret_key;
                auth.session_token = imds->session_token;
                got = true;
            }
        }

        // 3. SSO profiles discovered in ~/.aws/config
        if (!got) {
            auto ssoProfiles = findSsoProfiles();
            fprintf(stderr, "[AWS] found %zu SSO profiles in config\n", ssoProfiles.size());
            for (auto& p : ssoProfiles) {
                fprintf(stderr, "[AWS]   SSO profile: %s\n", p.c_str());
                if (tryExportCreds(p)) {
                    got = true;
                    break;
                }
            }
        }

        // 4. Default profile (no --profile flag)
        if (!got)
            got = tryExportCreds("");

        fprintf(stderr, "[AWS] credential resolution result: %s (key=%s...)\n",
            got ? "success" : "FAILED",
            auth.access_key.empty() ? "(empty)" : auth.access_key.substr(0, 8).c_str());
    }

    // Method 2: INI files (~/.aws/credentials, ~/.aws/config)
    {
        auto effectiveProfile = getEnv("AWS_PROFILE");
        if (effectiveProfile.empty()) effectiveProfile = profile;

        auto parseIni = [&](const std::filesystem::path& path,
                            const std::vector<std::pair<std::string, std::string*>>& keys) {
            std::ifstream f(path);
            if (!f) return;
            bool inProfile = false;
            std::string line;
            while (std::getline(f, line)) {
                auto start = line.find_first_not_of(" \t");
                if (start == std::string::npos) continue;
                line = line.substr(start);
                if (line.empty() || line[0] == '#' || line[0] == ';') continue;
                if (line[0] == '[') {
                    auto end = line.find(']');
                    std::string section = (end != std::string::npos) ? line.substr(1, end - 1) : "";
                    inProfile = (section == effectiveProfile || section == "profile " + effectiveProfile);
                    continue;
                }
                if (!inProfile) continue;
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq);
                std::string val = line.substr(eq + 1);
                key.erase(key.find_last_not_of(" \t") + 1);
                val.erase(0, val.find_first_not_of(" \t"));
                for (auto& [k, dst] : keys) {
                    if (key == k && dst->empty()) *dst = val;
                }
            }
        };

        std::filesystem::path home = getEnv("HOME");
        if (home.empty()) home = "/tmp";

        if (auth.access_key.empty() || auth.secret_key.empty()) {
            parseIni(home / ".aws" / "credentials", {
                {"aws_access_key_id",     &auth.access_key},
                {"aws_secret_access_key", &auth.secret_key},
                {"aws_session_token",     &auth.session_token},
            });
        }
        parseIni(home / ".aws" / "config", {
            {"region", &auth.region},
        });
    }

    // Method 3: environment variables
    if (auth.access_key.empty())    auth.access_key = getEnv("AWS_ACCESS_KEY_ID");
    if (auth.secret_key.empty())    auth.secret_key = getEnv("AWS_SECRET_ACCESS_KEY");
    if (auth.session_token.empty()) auth.session_token = getEnv("AWS_SESSION_TOKEN");
    if (auth.region.empty())        auth.region = getEnv("AWS_DEFAULT_REGION");

    fprintf(stderr, "[AWS] FINAL: key=%s... secret=%s token=%s region=%s\n",
        auth.access_key.empty() ? "(empty)" : auth.access_key.substr(0, 8).c_str(),
        auth.secret_key.empty() ? "(empty)" : "***",
        auth.session_token.empty() ? "(empty)" : (auth.session_token.substr(0, 20) + "...").c_str(),
        auth.region.empty() ? "(empty)" : auth.region.c_str());

    return auth;
}

} // namespace utils

#endif // UTILS_HAS_CURL
