#include "utils/http_fetch.hpp"

#if UTILS_HAS_CURL

#include "utils/Json.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace utils {

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
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) return false;
        std::string output;
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe.get()))
            output += buf;
        if (output.empty()) return false;
        try {
            auto j = Json::parse(output);
            if (j.contains("AccessKeyId") && j.contains("SecretAccessKey")) {
                auth.access_key = j["AccessKeyId"].get_string();
                auth.secret_key = j["SecretAccessKey"].get_string();
                if (j.contains("SessionToken"))
                    auth.session_token = j["SessionToken"].get_string();
                return true;
            }
        } catch (...) {
        }
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

        // 1. Try explicit profile (env or argument)
        if (!envProfile.empty())
            got = tryExportCreds(envProfile);
        else if (profile != "default")
            got = tryExportCreds(profile);

        // 2. Try SSO profiles from ~/.aws/config
        if (!got) {
            auto ssoProfiles = findSsoProfiles();
            for (auto& p : ssoProfiles) {
                if (tryExportCreds(p)) {
                    got = true;
                    break;
                }
            }
        }

        // 3. Try default (no --profile flag)
        if (!got)
            got = tryExportCreds("");
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

    return auth;
}

} // namespace utils

#endif // UTILS_HAS_CURL
