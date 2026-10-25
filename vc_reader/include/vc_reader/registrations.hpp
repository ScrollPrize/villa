#pragma once
#include "vc_reader/affine.hpp"
#include <string>
#include <vector>
#include <optional>
namespace vcr::catalog {
struct OpenDataManifest;
struct OpenDataSample;
struct Registration {
    std::string id, revision, source, target;
    Affine matrix;
    std::string provenance;
};
struct TransformCandidate {
    Affine forward, inverse;
    std::vector<std::string> path;
};
enum class TransformStatus { Unavailable, Resolved, Ambiguous };
struct TransformResolution {
    std::string source, target;
    TransformStatus status=TransformStatus::Unavailable;
    std::vector<TransformCandidate> candidates;
    const TransformCandidate& resolved() const;
};
// Immutable once shared with a geometry snapshot. No network or database I/O.
class RegistrationGraph {
public:
    void add(Registration registration);
    void addCatalog(const OpenDataManifest&);
    void addSample(const OpenDataSample&);
    TransformResolution resolve(const std::string& source,const std::string& target,
                                const std::optional<std::vector<std::string>>& choice=std::nullopt) const;
    std::vector<TransformResolution> available(const std::string& source) const;
    const std::vector<Registration>& registrations() const{return registrations_;}
private:
    std::vector<Registration> registrations_;
};
}
