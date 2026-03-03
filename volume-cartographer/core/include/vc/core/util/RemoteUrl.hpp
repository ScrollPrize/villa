#pragma once

#include <string>

namespace vc {

struct ResolvedUrl {
    std::string httpsUrl;      // resolved HTTPS URL
    std::string awsRegion;     // empty if not S3
    bool useAwsSigv4 = false;  // true if s3:// and credentials detected
};

// Resolve s3://bucket/key -> https://... with optional SigV4 flag.
// Supports s3://bucket/key (defaults to us-east-1) and
// s3+REGION://bucket/key (explicit region).
// Passes through http:// and https:// URLs unchanged.
ResolvedUrl resolveRemoteUrl(const std::string& input);

}  // namespace vc
