#pragma once
#include "vc/constraints/ConstraintGraph.hpp"

namespace vc::constraints {
// Source nodes are shared; UV placements belong to an individual association.
enum class ReviewStatus { Accepted, Rejected, Skipped };
struct ReviewDecision {
    ReviewStatus status = ReviewStatus::Rejected;
    uint64_t order = 0;
    Graph alignment;
    // Recorded by the geometric auto-linker rather than a person; cleared when
    // the pair is decided by hand. Auto-rejected pairs are what a reviewer
    // walks through afterwards.
    bool automatic = false;
    std::string note;
    // Manual keypoint pairs to apply when this pair is (re)fitted; the compact
    // links workspace stores these instead of fitted placements.
    std::vector<Link> keypoints;
    // Linker measurements carried through from a vc_fiber_patch_links file.
    nlohmann::json measurements;
};
// An accepted decision whose placement has been fitted in this session. The
// compact workspace stores decisions only; placements are refitted on demand.
inline bool fitted(const ReviewDecision& d) { return d.status == ReviewStatus::Accepted && !d.alignment.nodes.empty(); }
struct ReviewFiber {
    std::filesystem::path path;
    char direction = '?';
    Graph base;
    std::map<std::string, ReviewDecision> decisions;
    std::vector<Candidate> proposals;
    bool discovered = false, reverse = false;
    int contactPolicyVersion = 1;
    int current = -1;
    std::optional<Graph> pending;
    // Keypoints saved for undecided proposals (the compact workspace keeps the
    // pending proposal's manual pairs this way).
    std::map<std::string, std::vector<Link>> savedKeypoints;
};
struct FiberReview {
    Settings settings;
    std::map<std::string, ReviewFiber> fibers;
    std::string selected;
    uint64_t nextOrder = 1;
    int stride = 1, proposalOpacity = 50, committedOpacity = 20;
    int committedTint = 4;
    int fiberLineOpacity = 35;
    double fiberLineWidth = 1.0;
    bool boundaries = false;
    struct QuarantinedPatch {std::filesystem::path source,destination;};
    std::map<std::string,QuarantinedPatch> quarantined;
    // Revision of the last saved document, excluding unsaved review edits.
    std::string sessionRevision;
};
void orderReviewProposals(ReviewFiber&);
int nextReviewProposal(const ReviewFiber&, int from = -1, int delta = 1);
size_t acceptedPatchCount(const ReviewFiber&);
bool regenerateReviewFiber(ReviewFiber&, const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr);
// Drop stale automatic proposals, retaining decisions and embedded alignments.
// Call only with a successfully loaded catalog, never an unavailable source.
size_t pruneUnavailableReviewProposals(ReviewFiber&, const std::set<std::string>& available);
// Replace overlap targets using only accepted patches of this fiber; the
// proposal and accepted source XYZ/UV remain immutable during matching.
void addReviewOverlapTargets(GraphEdit&, const ReviewFiber&, const std::atomic_bool* cancel = nullptr);
nlohmann::json reviewToJson(const FiberReview&);
FiberReview reviewFromJson(const nlohmann::json&);
// Compact workspace: the vc_fiber_patch_links document vc_fiber_auto_link
// writes (version 1) or VC3D saves (version 2, with statuses, notes and
// keypoints). Only decisions are stored; strips, proposals and placements are
// rebuilt when a fiber is selected. Accepted decisions load unfitted.
FiberReview reviewFromLinks(const nlohmann::json&);
nlohmann::json linksFromReview(const FiberReview&);
bool isLinksDocument(const nlohmann::json&);
// Reads either format by its type field.
FiberReview reviewFromDocument(const nlohmann::json&);
// saveReview writes the legacy full workspace; saveLinks the compact one.
std::string saveReview(const FiberReview&, const std::filesystem::path&);
std::string saveLinks(const FiberReview&, const std::filesystem::path&);
FiberReview loadReview(const std::filesystem::path&);
FiberReview quarantineReviewPatch(const FiberReview&, const std::filesystem::path& workspace,
                                 const PatchEntry&, const Progress& = {});
} // namespace vc::constraints
