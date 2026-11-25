// Approval of a measured winding constraint: its identity, its rows, the
// superseding that revokes it and the span fingerprints that invalidate it.
#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberConstraintApprovalExtension.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"

#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace vc::objects;
namespace {
using Json = nlohmann::json;
using Kind = FiberConstraintKind;
using State = FiberConstraintApprovalState;

const std::string fiberLow = "40000000-0000-4000-8000-000000000001";
const std::string fiberHigh = "40000000-0000-4000-8000-000000000002";

// Eight controls (ids 10..80) on the x axis, one line point per unit.
FiberPieceContent fixture(double offsetY = 0.0)
{
    vc::geometry::Fiber fiber;
    Json annotations = Json::array();
    for (size_t i = 0; i < 8; ++i) {
        fiber.controlPoints.push_back({10.0 * double(i), offsetY, 0.0});
        fiber.controlPointIds.push_back(int64_t(10 * (i + 1)));
        annotations.push_back(Json{{"direction", {1.0, 0.0, 0.0}}, {"tags", Json::array()}});
    }
    for (size_t x = 0; x <= 70; ++x) fiber.linePoints.push_back({double(x), offsetY, 0.0});
    return fiberPieceContent(fiber, annotations);
}

void identities()
{
    const auto forward = fiberConstraintIdentity(fiberLow, 100., fiberHigh, 30., Kind::Perpendicular);
    const auto reverse = fiberConstraintIdentity(fiberHigh, 30., fiberLow, 100., Kind::Perpendicular);
    require(forward == reverse, "the identity does not depend on the order the caller holds the fibers in");
    require(forward.fiberA == fiberLow && forward.fiberB == fiberHigh, "the fibers are stored in UUID order");
    // 64 base voxels per quantum: 100 vx lands in quantum one, 30 vx in zero.
    require(forward.arcA == 1 && forward.arcB == 0, "anchors are quantized to the extraction's event spacing");
    require(fiberConstraintIdentity(fiberLow, 100., fiberHigh, 30., Kind::Perpendicular) ==
                fiberConstraintIdentity(fiberLow, 120., fiberHigh, 60., Kind::Perpendicular),
            "a re-extracted crossing within one quantum keeps its identity");
    require(!(fiberConstraintIdentity(fiberLow, 100., fiberHigh, 30., Kind::Perpendicular) ==
              fiberConstraintIdentity(fiberLow, 400., fiberHigh, 30., Kind::Perpendicular)),
            "a different crossing of the same pair is a different constraint");
    require(!(fiberConstraintIdentity(fiberLow, 100., fiberHigh, 30., Kind::Perpendicular) ==
              fiberConstraintIdentity(fiberLow, 100., fiberHigh, 30., Kind::Parallel)),
            "the kind is part of the identity");
    mustFail([&] { fiberConstraintIdentity(fiberLow, 10., fiberLow, 20., Kind::Perpendicular); });
    mustFail([&] { fiberConstraintIdentity(fiberLow, -1., fiberHigh, 20., Kind::Perpendicular); });
    mustFail([&] { fiberConstraintIdentity(fiberLow, 10., fiberHigh, 20., Kind::Perpendicular, 0.); });
}

void rowsAndValidation()
{
    const auto a = fixture(), b = fixture(5.);
    const auto identity = fiberConstraintIdentity(fiberLow, 35., fiberHigh, 35., Kind::Perpendicular);
    const auto row = approveFiberConstraint(identity, a, 3, 7, b, 3, 2, FiberConstraintDistance::Direct, "alice", "2026-11-17T10:00:00Z", "host-a");
    require(row.state == State::Approved && !row.supersedes, "a fresh approval supersedes nothing");
    require(row.approvalUuid == row.rowUuid, "the first row of an approval names the event");
    require(row.spanA && row.spanB, "an approval fingerprints both covering spans");
    require(row.spanA->idLow == 40 && row.spanA->idHigh == 50, "the span is named by its two control ids");

    const auto encoded = fiberConstraintApprovalRow(row);
    require(encoded.size() == fiberConstraintApprovalTable().columns.size(), "a row fills every column");
    require(decodeFiberConstraintApprovalRow(encoded) == row, "a row round-trips through the table");

    const auto revoked = revokeFiberConstraintApproval(row, "bob", "2026-11-17T11:00:00Z");
    require(revoked.state == State::Revoked && revoked.supersedes == row.rowUuid, "a revocation supersedes its row");
    require(revoked.approvalUuid == row.approvalUuid, "a revocation keeps the approval event");
    require(!revoked.spanA && !revoked.spanB, "a revocation carries no fingerprints");
    require(decodeFiberConstraintApprovalRow(fiberConstraintApprovalRow(revoked)) == revoked,
            "a revocation round-trips through the table");

    // The live set: the approval until it is revoked, then nothing.
    const std::array<FiberConstraintApproval, 1> one{row};
    require(liveFiberConstraintApprovals(one).size() == 1, "an unrevoked approval is live");
    const std::array<FiberConstraintApproval, 2> both{row, revoked};
    require(liveFiberConstraintApprovals(both).empty(), "a revoked approval is not live");

    auto broken = row;
    broken.identity.fiberA = fiberHigh;
    broken.identity.fiberB = fiberLow;
    mustFail([&] { fiberConstraintApprovalRow(broken); });
    broken = row;
    broken.author.clear();
    mustFail([&] { fiberConstraintApprovalRow(broken); });
    broken = row;
    broken.identity.kind = Kind(0);
    mustFail([&] { fiberConstraintApprovalRow(broken); });
    broken = row;
    broken.spanB.reset();
    mustFail([&] { fiberConstraintApprovalRow(broken); });
    broken = revoked;
    broken.supersedes.reset();
    mustFail([&] { fiberConstraintApprovalRow(broken); });
    mustFail([&] { approveFiberConstraint(identity, a, 99, 7, b, 3, 2, FiberConstraintDistance::Direct, "alice", "2026-11-17T10:00:00Z", "host-a"); });
}

void currencyFollowsTheCoveringSpans()
{
    const auto a = fixture(), b = fixture(5.);
    const auto identity = fiberConstraintIdentity(fiberLow, 35., fiberHigh, 35., Kind::Perpendicular);
    const auto row = approveFiberConstraint(identity, a, 3, 7, b, 3, 2, FiberConstraintDistance::Direct, "alice", "2026-11-17T10:00:00Z", "host-a");
    require(fiberConstraintApprovalCurrent(row, a, 3, b, 3), "an untouched pair stays approved");
    auto edited = a;
    edited.linePoints[35] = {35.0, 0.5, 0.0};
    require(!fiberConstraintApprovalCurrent(row, edited, 3, b, 3), "editing the covering span invalidates it");
    auto elsewhere = a;
    elsewhere.linePoints[5] = {5.0, 0.5, 0.0};
    require(fiberConstraintApprovalCurrent(row, elsewhere, 3, b, 3), "an edit in another span does not");
    require(!fiberConstraintApprovalCurrent(revokeFiberConstraintApproval(row, "bob", "2026-11-17T11:00:00Z"), a, 3, b, 3),
            "a revocation is never current");

    // The covering span of an anchor, by the line point nearest it.
    require(fiberSpanCoveringLineIndex(a, 35) == size_t{3}, "a line index inside the fourth span resolves to it");
    require(fiberSpanCoveringLineIndex(a, 0) == size_t{0}, "the first line point belongs to the first span");
    require(fiberSpanCoveringLineIndex(a, 70) == size_t{6}, "the last line point belongs to the last span");
    require(fiberSpanCoveringLineIndex(a, 71) == size_t{6}, "a line point on a tail belongs to the nearest span");
    require(!fiberSpanCoveringLineIndex(a, -1).has_value(), "a negative line index covers no span");
}

void storeRoundTrip()
{
    TemporaryDirectory directory;
    const auto path = (directory.path / "constraints.vcobj").string();
    const auto a = fixture(), b = fixture(5.);
    const auto identity = fiberConstraintIdentity(fiberLow, 35., fiberHigh, 35., Kind::Perpendicular);
    const auto row = approveFiberConstraint(identity, a, 3, 7, b, 3, 2, FiberConstraintDistance::Direct, "alice", "2026-11-17T10:00:00Z", "host-a");
    const auto revoked = revokeFiberConstraintApproval(row, "bob", "2026-11-17T11:00:00Z");
    {
        OpenOptions options;
        options.author = "test";
        auto store = ObjectStore::create(path, options);
        auto txn = store.write();
        store.createExtensionTable(txn, fiberConstraintApprovalTable());
        const std::array<ExtensionRow, 2> rows{fiberConstraintApprovalRow(row), fiberConstraintApprovalRow(revoked)};
        store.appendExtensionRows(txn, kFiberConstraintApprovalTable, rows);
        txn.commit();
    }
    OpenOptions options;
    options.readOnly = true;
    auto store = ObjectStore::open(path, options);
    const auto txn = store.read();
    const LocalStoreReader reader(store, txn);
    const auto read = readFiberConstraintApprovals(reader);
    require(read.size() == 2, "both rows come back");
    require(std::any_of(read.begin(), read.end(), [&](const auto& entry) { return entry == row; }),
            "the approval row round-trips through the store");
    require(liveFiberConstraintApprovals(read).empty(), "the revocation wins after a reread");
    require(readFiberConstraintApprovals(reader, fiberLow).size() == 2, "a fiber finds the rows of either endpoint");
    require(readFiberConstraintApprovals(reader, fiberHigh).size() == 2, "a fiber finds the rows of either endpoint");
    require(readFiberConstraintApprovals(reader, "40000000-0000-4000-8000-000000000003").empty(),
            "an unrelated fiber finds none");
}
} // namespace

int main() try
{
    identities();
    rowsAndValidation();
    currencyFollowsTheCoveringSpans();
    storeRoundTrip();
    std::cout << "fiber constraint approval tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
