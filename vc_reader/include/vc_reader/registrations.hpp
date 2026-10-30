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
    // Every volume with a route to `target`: its resolution from that volume.
    std::vector<TransformResolution> reaching(const std::string& target) const;
    // A route's steps by registration provenance (else id), with the volumes
    // each relates and inverse steps marked, e.g. "inverse of scan (a → b)".
    std::string describe(const std::vector<std::string>& path) const;
    const std::vector<Registration>& registrations() const{return registrations_;}
private:
    std::vector<Registration> registrations_;
};
// The route from one of a sample's volumes to another, and the derivation path the
// sample publishes for that direct pair (empty when it has none). nullopt: no route.
struct VolumeTransform {
    Affine forward;
    std::string derivationPath;
};
std::optional<VolumeTransform> resolveVolumeTransform(const OpenDataSample&, const std::string& source, const std::string& target);
}
