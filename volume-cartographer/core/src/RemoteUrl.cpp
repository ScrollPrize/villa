#include "vc/core/util/RemoteUrl.hpp"

#include <cstdlib>

namespace vc {

ResolvedUrl resolveRemoteUrl(const std::string& input)
{
    // Check for s3:// or s3+REGION:// prefix
    if (input.rfind("s3://", 0) == 0) {
        // s3://bucket/key — default region us-east-1
        std::string rest = input.substr(5);  // after "s3://"
        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string region = "us-east-1";
        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        return ResolvedUrl{httpsUrl, region, true};
    }

    if (input.rfind("s3+", 0) == 0) {
        // s3+REGION://bucket/key
        auto schemeEnd = input.find("://");
        if (schemeEnd == std::string::npos) {
            // Malformed — treat as plain URL
            return ResolvedUrl{input, {}, false};
        }

        std::string region = input.substr(3, schemeEnd - 3);  // between "s3+" and "://"
        std::string rest = input.substr(schemeEnd + 3);

        auto slash = rest.find('/');
        std::string bucket = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
        std::string key = (slash != std::string::npos) ? rest.substr(slash + 1) : "";

        std::string httpsUrl = "https://" + bucket + ".s3." + region + ".amazonaws.com";
        if (!key.empty()) {
            httpsUrl += "/" + key;
        }

        return ResolvedUrl{httpsUrl, region, true};
    }

    // Check if this is an S3 HTTPS URL by hostname pattern
    // e.g. https://bucket.s3.region.amazonaws.com/key
    //   or https://bucket.s3.amazonaws.com/key (us-east-1 default)
    if (input.rfind("https://", 0) == 0 || input.rfind("http://", 0) == 0) {
        auto schemeEnd = input.find("://");
        std::string rest = input.substr(schemeEnd + 3);
        auto slash = rest.find('/');
        std::string host = (slash != std::string::npos) ? rest.substr(0, slash) : rest;

        // Match *.s3.REGION.amazonaws.com or *.s3.amazonaws.com
        auto s3Pos = host.find(".s3.");
        if (s3Pos != std::string::npos && host.find(".amazonaws.com") != std::string::npos) {
            // Extract region from between .s3. and .amazonaws.com
            std::string afterS3 = host.substr(s3Pos + 4);  // after ".s3."
            auto amzPos = afterS3.find(".amazonaws.com");
            std::string region;
            if (amzPos != std::string::npos && amzPos > 0) {
                region = afterS3.substr(0, amzPos);
            } else {
                region = "us-east-1";
            }
            return ResolvedUrl{input, region, true};
        }
    }

    // HTTP/HTTPS — pass through
    return ResolvedUrl{input, {}, false};
}

}  // namespace vc
