#include "TestSupport.hpp"
#include "StoreTestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/FiberPieceApprovalExtension.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace vc::objects;
namespace {
using Json = nlohmann::json;
using Aspect = FiberPieceAspect;

// Eight controls (ids 10..80) on the x axis, one line point per unit.
FiberPieceContent fixture(size_t controls = 8)
{
    vc::geometry::Fiber fiber;
    Json annotations = Json::array();
    for (size_t i = 0; i < controls; ++i) {
        fiber.controlPoints.push_back({10.0 * double(i), 0.0, 0.0});
        fiber.controlPointIds.push_back(int64_t(10 * (i + 1)));
        Json a{{"direction", {1.0, 0.0, 0.0}}, {"tags", Json::array()}};
        if (i + 1 < controls) a["segment_to_next"] = Json{{"goal", "trace"}, {"mode", "auto"}};
        annotations.push_back(a);
    }
    for (size_t x = 0; x <= 10 * (controls - 1); ++x) fiber.linePoints.push_back({double(x), 0.0, 0.0});
    return fiberPieceContent(fiber, annotations);
}

void tag(FiberPieceContent& content, size_t index, const char* value)
{
    content.annotations[index]["tags"].push_back(value);
}

FiberPieceRange range(const FiberPieceContent& content, int64_t a, std::optional<int64_t> b, int64_t inside)
{
    const auto r = fiberPieceRange(content, a, b, inside);
    require(r.has_value(), "piece range resolves");
    return *r;
}

FiberPieceApproval approve(const FiberPieceContent& content, int64_t a, std::optional<int64_t> b, int64_t inside,
                           Aspect aspect = Aspect::Geometry, const char* author = "alice")
{
    return approveFiberPiece(content, "40000000-0000-4000-8000-000000000001", range(content, a, b, inside), aspect,
                             3, author, "2026-11-17T10:00:00Z", "host-a");
}

FiberPieceContent reversed(const FiberPieceContent& source)
{
    vc::geometry::Fiber fiber;
    const size_t n = source.controlPoints.size();
    Json annotations = Json::array();
    for (size_t i = 0; i < n; ++i) {
        const size_t j = n - 1 - i;
        fiber.controlPoints.push_back(source.controlPoints[j]);
        fiber.controlPointIds.push_back(source.controlIds[j]);
        Json a = source.annotations[j];
        if (a.contains("direction")) for (auto& c : a["direction"]) c = -c.get<double>();
        a.erase("segment_to_next");
        // Span i of the reversed fiber is span n-2-i of the source; its
        // descriptor moves to the reversed owner.
        if (i + 1 < n && source.annotations[n - 2 - i].contains("segment_to_next"))
            a["segment_to_next"] = source.annotations[n - 2 - i]["segment_to_next"];
        annotations.push_back(a);
    }
    fiber.linePoints.assign(source.linePoints.rbegin(), source.linePoints.rend());
    return fiberPieceContent(fiber, annotations);
}

std::vector<size_t> approvedSpans(const FiberPieceContent& content, std::span<const FiberPieceApproval> rows, Aspect aspect = Aspect::Geometry)
{
    return fiberPieceApprovalStates(content, rows).approvedSpans[size_t(aspect)];
}
} // namespace

int main() try
{
    // Codec round trip through the typed row.
    {
        auto content = fixture();
        tag(content, 3, kFiberLockControlPointTag);
        tag(content, 5, kFiberLockControlPointTag);
        const auto row = approve(content, 40, 60, 50);
        require(row.controls.size() == 3 && row.spans.size() == 2, "a two-span piece has three controls and two spans");
        require(row.controls[0].id == 40 && row.controls[1].id == 50 && row.controls[2].id == 60, "controls walk from lock A");
        require(row.spans[0].idLow == 40 && row.spans[0].idHigh == 50, "span ids are ordered");
        const auto decoded = decodeFiberPieceApprovalRow(fiberPieceApprovalRow(row));
        require(decoded == row, "row codec round trips");
        require(fiberPieceApprovalTable().primaryKey == std::vector<std::string>{"fiber_uuid", "row_uuid"}, "primary key");
        require(fiberPieceApprovalRow(row).size() == fiberPieceApprovalTable().columns.size(), "row matches the column count");
        auto broken = row; broken.insideCp = 60;
        mustFail([&] { fiberPieceApprovalRow(broken); });
        // The backward walk from the other lock yields the same hashes with reversed order.
        const auto backward = approve(content, 60, 40, 50);
        require(backward.spans[0].hash == row.spans[1].hash && backward.spans[1].hash == row.spans[0].hash,
                "span hashes do not depend on the walking direction");
        require(backward.controls[0].hash == row.controls[2].hash, "control hashes do not depend on the walking direction");
    }
    // Fingerprint invariants.
    {
        auto content = fixture();
        const auto base = approve(content, 40, 60, 50);
        auto moved = content; moved.linePoints[45][1] = 0.5;
        const auto movedRow = approve(moved, 40, 60, 50);
        require(movedRow.spans[1].hash != base.spans[1].hash && movedRow.spans[0].hash == base.spans[0].hash,
                "a moved line point changes exactly its span");
        require(movedRow.controls == base.controls, "a moved line point changes no control");
        auto turned = content; turned.annotations[4]["direction"] = {0.0, 1.0, 0.0};
        const auto turnedRow = approve(turned, 40, 60, 50);
        require(turnedRow.controls[1].hash != base.controls[1].hash && turnedRow.controls[0] == base.controls[0] &&
                turnedRow.controls[2] == base.controls[2], "a changed direction changes exactly that control");
        require(turnedRow.spans == base.spans, "a changed direction changes no span");
        auto flipped = content; flipped.annotations[4]["direction"] = {-1.0, 0.0, 0.0};
        require(approve(flipped, 40, 60, 50).controls == base.controls, "direction is hashed up to sign");
        auto locked = content; tag(locked, 4, kFiberLockControlPointTag);
        require(approve(locked, 40, 60, 50).controls == base.controls, "the lock tag is not part of the hash");
        auto tagged = content; tag(tagged, 4, "break");
        require(approve(tagged, 40, 60, 50).controls[1].hash != base.controls[1].hash, "other tags are hashed");
        auto described = content; described.annotations[3]["segment_to_next"]["goal"] = "cspline";
        const auto describedRow = approve(described, 40, 60, 50);
        require(describedRow.spans[0].hash != base.spans[0].hash && describedRow.spans[1] == base.spans[1],
                "a span descriptor change changes exactly that span");
        auto normal = content; normal.annotations[4]["display_normal"] = {0.0, 0.0, 1.0}; normal.annotations[4]["display_normal_source"] = "manual";
        require(approve(normal, 40, 60, 50).controls == base.controls, "display normals are not in the geometry hash");
        const auto normalsBase = approve(content, 40, 60, 50, Aspect::Normals), normalsRow = approve(normal, 40, 60, 50, Aspect::Normals);
        require(normalsBase.spans.empty() && normalsRow.controls[1].hash != normalsBase.controls[1].hash &&
                normalsRow.controls[0] == normalsBase.controls[0], "normals aspect hashes display normals only");
        require(approve(moved, 40, 60, 50, Aspect::Normals).controls == normalsBase.controls, "geometry edits leave normals hashes alone");
        // Reversal of the whole fiber leaves every fingerprint unchanged.
        const auto mirror = reversed(content);
        const auto mirrored = approve(mirror, 40, 60, 50);
        require(mirrored.spans == base.spans && mirrored.controls == base.controls && mirrored.lockCpB == base.lockCpB,
                "fingerprints are invariant under fiber reversal");
        require(approve(mirror, 40, std::nullopt, 30).controls.size() == 4, "an end piece walks to the fiber end");
        // Inserting a lock control on an existing line point with the tangent
        // direction and the split descriptor leaves the untouched spans' entries unchanged.
        auto split = content;
        split.controlPoints.insert(split.controlPoints.begin() + 5, {45.0, 0.0, 0.0});
        split.controlIds.insert(split.controlIds.begin() + 5, 55);
        Json inserted{{"direction", {1.0, 0.0, 0.0}}, {"tags", Json::array({kFiberLockControlPointTag})},
                      {"segment_to_next", split.annotations[4]["segment_to_next"]}};
        split.annotations.insert(split.annotations.begin() + 5, inserted);
        split.resolveLineIndices();
        const auto splitRow = approve(split, 40, 60, 50);
        require(splitRow.spans.size() == 3 && splitRow.spans[0] == base.spans[0] && splitRow.controls[0] == base.controls[0] &&
                splitRow.controls[1] == base.controls[1] && splitRow.controls[3] == base.controls[2],
                "a lock insertion leaves the other spans' and controls' fingerprints unchanged");
    }
    // Live rows, states, currency.
    {
        auto content = fixture();
        tag(content, 3, kFiberLockControlPointTag);
        tag(content, 5, kFiberLockControlPointTag);
        const auto a = approve(content, 40, 60, 50);
        std::vector<FiberPieceApproval> rows{a};
        require(approvedSpans(content, rows) == std::vector<size_t>{3, 4}, "the piece approves its two spans");
        auto states = fiberPieceApprovalStates(content, rows);
        require(states.rowCurrent == std::vector<bool>{true} && states.lockCpIds == std::set<int64_t>{40, 60} &&
                states.releasableLockCpIds.empty(), "a current row references its locks");
        auto moved = content; moved.linePoints[45][2] = 1.0;
        states = fiberPieceApprovalStates(moved, rows);
        require(states.approvedSpans[0] == std::vector<size_t>{3} && !states.rowCurrent[0] && states.spans[0][4].row == 0 &&
                !states.spans[0][4].current, "an edited span is covered but stale, its neighbour stays approved");
        rows.push_back(revokeFiberPieceApproval(a, "bob", "2026-11-17T11:00:00Z"));
        require(liveFiberPieceApprovals(rows).empty(), "a revocation row retires the approval");
        require(decodeFiberPieceApprovalRow(fiberPieceApprovalRow(rows.back())) == rows.back(), "revocation codec round trips");
        states = fiberPieceApprovalStates(content, rows);
        require(approvedSpans(content, rows).empty() && states.releasableLockCpIds == std::set<int64_t>{40, 60},
                "locks of superseded rows become releasable");
        rows.push_back(approve(content, 60, 80, 70));
        states = fiberPieceApprovalStates(content, rows);
        require(states.releasableLockCpIds == std::set<int64_t>{40} && states.lockCpIds == std::set<int64_t>{60, 80},
                "a lock still referenced by a live row is not releasable");
        // Missing ids make a row stale; the spans between its remaining controls stay covered.
        auto shrunk = content;
        shrunk.controlPoints.erase(shrunk.controlPoints.begin() + 4);
        shrunk.controlIds.erase(shrunk.controlIds.begin() + 4);
        shrunk.annotations.erase(shrunk.annotations.begin() + 4);
        shrunk.resolveLineIndices();
        states = fiberPieceApprovalStates(shrunk, std::vector<FiberPieceApproval>{a});
        require(!states.rowCurrent[0] && states.spans[0][3].row == 0 && !states.spans[0][3].current && states.approvedSpans[0].empty(),
                "a row whose control vanished is stale and covers the span between its remaining controls");
        require(!fiberPieceRange(shrunk, a).has_value(), "a range with a missing inside control does not resolve");
        // End piece growth: new spans past the covered range are uncovered.
        auto end = approve(content, 60, std::nullopt, 70);
        require(!end.lockCpB && end.controls.size() == 3, "an end piece reaches the last control");
        auto grown = fixture(9);
        tag(grown, 5, kFiberLockControlPointTag);
        states = fiberPieceApprovalStates(grown, std::vector<FiberPieceApproval>{end});
        require(states.rowCurrent[0] && states.approvedSpans[0] == std::vector<size_t>{5, 6} && !states.spans[0][7].row,
                "growth past an end piece is uncovered while the piece stays current");
    }
    // Cropping at the first edit of a neighbouring piece.
    {
        auto content = fixture();
        for (size_t i : {0, 4, 5, 7}) tag(content, i, kFiberLockControlPointTag);
        const auto a = approve(content, 10, 60, 20);                 // [0..5]
        const auto c = approve(content, 60, std::nullopt, 70, Aspect::Geometry, "carol"); // [5..7], open end
        const auto n = approve(content, 10, 60, 20, Aspect::Normals);
        std::vector<FiberPieceApproval> live{a, c, n};
        // B opens [4..7]: A is cropped to [0..4], C is entirely inside and revoked, the normals row cropped too.
        const auto crops = cropFiberPieceApprovals(content, live, 4, 7, "bob", "2026-11-17T12:00:00Z");
        require(crops.size() == 3, "every intersecting live row gets one superseding row");
        const auto find = [&](const std::string& approvalUuid) {
            const auto it = std::find_if(crops.begin(), crops.end(), [&](const auto& r) { return r.approvalUuid == approvalUuid; });
            require(it != crops.end(), "crop row for the approval exists");
            return *it;
        };
        const auto croppedA = find(a.approvalUuid);
        require(croppedA.state == FiberPieceState::Approved && croppedA.supersedes == a.rowUuid && croppedA.lockCpA == 10 &&
                croppedA.lockCpB == 50 && croppedA.insideCp == 20 && croppedA.controls.size() == 5 && croppedA.spans.size() == 4,
                "A keeps its lock and is cut at the opened range's first control");
        require(croppedA.author == "alice" && croppedA.approvedAt == a.approvedAt && croppedA.changedBy == "bob" &&
                croppedA.changedAt == "2026-11-17T12:00:00Z", "a crop keeps the approval stamps and records who cropped");
        require(std::equal(croppedA.spans.begin(), croppedA.spans.end(), a.spans.begin()), "cropped span fingerprints are the original ones");
        const auto revokedC = find(c.approvalUuid);
        require(revokedC.state == FiberPieceState::Revoked && revokedC.supersedes == c.rowUuid && revokedC.author == "carol",
                "a row entirely inside the opened range is revoked");
        require(find(n.approvalUuid).aspect == Aspect::Normals && find(n.approvalUuid).spans.empty(), "normals rows are cropped alike");
        std::vector<FiberPieceApproval> all{a, c, n};
        all.insert(all.end(), crops.begin(), crops.end());
        auto states = fiberPieceApprovalStates(content, all);
        require(states.approvedSpans[0] == std::vector<size_t>{0, 1, 2, 3}, "after the crop only A's remaining spans are approved");
        require(states.releasableLockCpIds == std::set<int64_t>{60} && states.lockCpIds == std::set<int64_t>{10, 50},
                "locks only superseded rows reference are releasable");
        // B approves its piece: the overlap is approved again, by B.
        all.push_back(approve(content, 50, 80, 60, Aspect::Geometry, "bob"));
        require(approvedSpans(content, all) == std::vector<size_t>{0, 1, 2, 3, 4, 5, 6}, "B's approval covers the overlap");
        // Opening strictly inside a row splits it in two.
        const auto wide = approve(content, 10, 80, 20);
        const auto parts = cropFiberPieceApprovals(content, std::vector<FiberPieceApproval>{wide}, 4, 5, "bob", "t");
        require(parts.size() == 2, "a range inside a row leaves two parts");
        require(parts[0].lockCpA == 10 && parts[0].lockCpB == 50 && parts[1].lockCpA == 60 && parts[1].lockCpB == 80 &&
                parts[1].insideCp == 70, "both parts are delimited by the opened range's locks");
        // A backward end piece cropped from the right keeps its open side.
        const auto left = approve(content, 50, std::nullopt, 40); // [0..4] walking backward
        const auto cut = cropFiberPieceApprovals(content, std::vector<FiberPieceApproval>{left}, 0, 2, "bob", "t");
        require(cut.size() == 1 && cut[0].lockCpA == 50 && cut[0].lockCpB == 30 && cut[0].insideCp == 40,
                "a backward end piece cut at its open side ends at the opened range");
        const auto cutLeft = cropFiberPieceApprovals(content, std::vector<FiberPieceApproval>{left}, 3, 7, "bob", "t");
        require(cutLeft.size() == 1 && cutLeft[0].lockCpA == 40 && !cutLeft[0].lockCpB && cutLeft[0].insideCp == 30,
                "cutting the lock side moves the lock and keeps the open end");
        // The piece's own row (the same range) is not cropped by its own edits.
        require(cropFiberPieceApprovals(content, std::vector<FiberPieceApproval>{a}, 0, 5, "bob", "t").empty(),
                "a row of exactly the opened range is left alone");
        // Sharing only a lock control is not an intersection.
        require(cropFiberPieceApprovals(content, std::vector<FiberPieceApproval>{a}, 5, 7, "bob", "t").empty(),
                "a neighbouring piece sharing the lock control crops nothing");
        mustFail([&] { cropFiberPieceApprovals(content, live, 4, 4, "bob", "t"); });
    }
    // Store round trip: table registration, append, read back, content from the reader.
    {
        TemporaryDirectory directory;
        OpenOptions options; options.author = "piece-test";
        const auto path = directory.path / "pieces.vcobj";
        { auto store = ObjectStore::create(path, options); }
        auto db = ObjectStore::open(path, options);
        const auto uuid = util::newUuid();
        auto content = fixture();
        tag(content, 3, kFiberLockControlPointTag);
        tag(content, 5, kFiberLockControlPointTag);
        vc::geometry::Fiber geometry;
        geometry.controlPoints = content.controlPoints;
        geometry.controlPointIds = content.controlIds;
        geometry.linePoints = content.linePoints;
        storeAct(db, {{uuid, {}}}, [&](auto& store, auto& txn) {
            ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
            record.fiber.emplace(); record.fiber->controlPointCount = int64_t(content.controlPoints.size());
            record.fiber->length = geometry.length();
            record.meta = Json{{"source", {{"control_points_attributes", Json(content.annotations)}}}}.dump();
            store.upsertObject(txn, record);
            store.putGeometry(txn, uuid, vc::geometry::encode(geometry));
        });
        FiberPieceApproval row;
        storeReader(db, [&](const StoreReader& reader, const auto&) {
            const auto revision = reader.currentObject(uuid);
            require(revision.has_value(), "fiber is current");
            const auto read = fiberPieceContent(reader, *revision);
            require(read.controlIds == content.controlIds && read.linePoints == content.linePoints &&
                    read.controlLineIndex == content.controlLineIndex, "content reads back from the store");
            require(read.isLockControl(3) && !read.isLockControl(4), "lock tags read back");
            row = approveFiberPiece(read, uuid, range(read, 40, 60, 50), Aspect::Geometry, revision->version,
                                    "alice", "2026-11-17T10:00:00Z", "host-a");
        });
        storeAct(db, {}, [&](auto& store, auto& txn) {
            store.createExtensionTable(txn, fiberPieceApprovalTable());
            const auto values = fiberPieceApprovalRow(row);
            store.appendExtensionRows(txn, kFiberPieceApprovalTable, std::span<const ExtensionRow>(&values, 1));
        });
        storeRead(db, [&](const auto& store, const auto& txn, const auto&) {
            const auto rows = readFiberPieceApprovals(store, txn, uuid);
            require(rows.size() == 1 && rows[0] == row, "the row reads back by fiber");
            require(readFiberPieceApprovals(store, txn, util::newUuid()).empty(), "other fibers have no rows");
            const LocalStoreReader reader(store, txn);
            const auto states = fiberPieceApprovalStates(fiberPieceContent(reader, *reader.currentObject(uuid)), rows);
            require(states.approvedSpans[0] == std::vector<size_t>{3, 4}, "the stored row is current against the stored fiber");
        });
        storeAct(db, {}, [&](auto& store, auto& txn) {
            const auto values = fiberPieceApprovalRow(row);
            mustFail([&] { store.appendExtensionRows(txn, kFiberPieceApprovalTable, std::span<const ExtensionRow>(&values, 1)); });
        });
    }
    // Session submit: table creation and row append as transaction operations
    // expecting the fiber's version; JSON codec of the operations; reader scan.
    {
        TemporaryDirectory directory;
        OpenOptions options; options.author = "piece-session-test";
        const auto path = directory.path / "pieces.vcobj";
        { auto store = ObjectStore::create(path, options); }
        const auto uuid = util::newUuid();
        auto content = fixture();
        vc::geometry::Fiber geometry;
        geometry.controlPoints = content.controlPoints;
        geometry.controlPointIds = content.controlIds;
        geometry.linePoints = content.linePoints;
        {
            auto db = ObjectStore::open(path, options);
            storeAct(db, {{uuid, {}}}, [&](auto& store, auto& txn) {
                ObjectRecord record; record.uuid = uuid; record.kind = "fiber";
                record.fiber.emplace(); record.fiber->controlPointCount = int64_t(content.controlPoints.size());
                record.meta = Json{{"source", {{"control_points_attributes", Json(content.annotations)}}}}.dump();
                store.upsertObject(txn, record);
                store.putGeometry(txn, uuid, vc::geometry::encode(geometry));
            });
        }
        ObjectStoreSession session;
        const auto attachment = session.attach(path, options);
        const auto revision = session.currentObject(attachment, uuid);
        require(revision.has_value(), "fiber is current through the session");
        FiberPieceApproval row;
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            require(readFiberPieceApprovals(reader, uuid).empty(), "no rows before the table exists");
            const auto read = fiberPieceContent(reader, *revision);
            row = approveFiberPiece(read, uuid, range(read, 40, 60, 50), Aspect::Geometry, revision->version,
                                    "alice", "2026-11-17T10:00:00Z", "host-a");
        });
        auto request = session.newRequest(attachment);
        expectObject(request, *revision);
        request.operations.push_back(CreateExtensionTableOp{fiberPieceApprovalTable()});
        request.operations.push_back(AppendExtensionRowsOp{kFiberPieceApprovalTable, {fiberPieceApprovalRow(row)}});
        // The operations survive the wire format.
        TransactionPayloads payloads;
        const auto text = transactionRequestToJson(request, &payloads);
        const auto decoded = transactionRequestFromJson(text, payloads);
        require(decoded.operations.size() == 2 && transactionRequestToJson(decoded, nullptr) == text, "extension operations round trip through JSON");
        const auto& decodedAppend = std::get<AppendExtensionRowsOp>(decoded.operations[1]);
        require(decodedAppend.rows.size() == 1 && decodeFiberPieceApprovalRow(decodedAppend.rows[0]) == row, "typed values survive the wire format");
        require(std::get<CreateExtensionTableOp>(decoded.operations[0]).table.columns == fiberPieceApprovalTable().columns,
                "the table definition survives the wire format");
        const auto receipt = session.submit(attachment, request);
        require(receipt.objects.at(uuid) == revision->version, "an approval row leaves the fiber's version unchanged");
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            const auto rows = readFiberPieceApprovals(reader, uuid);
            require(rows.size() == 1 && rows[0] == row, "the row reads back through the session reader");
            require(reader.currentObject(uuid)->version == revision->version, "no owner version advance");
        });
        // A stale expectation conflicts and writes nothing.
        auto stale = session.newRequest(attachment);
        auto older = *revision; older.version = revision->version + 1;
        expectObject(stale, older);
        auto second = revokeFiberPieceApproval(row, "bob", "2026-11-17T11:00:00Z");
        stale.operations.push_back(AppendExtensionRowsOp{kFiberPieceApprovalTable, {fiberPieceApprovalRow(second)}});
        mustFail([&] { session.submit(attachment, stale); });
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            require(readFiberPieceApprovals(reader, uuid).size() == 1, "a conflicting request appends nothing");
        });
        // Replaying the first request yields the same receipt without a second row.
        const auto replay = session.submit(attachment, request);
        require(replay.replayed, "an identical retry replays its receipt");
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
            require(readFiberPieceApprovals(reader, uuid).size() == 1, "a replayed request appends nothing");
        });
    }
    std::cout << "fiber piece approval tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
