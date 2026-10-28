#include "TestSupport.hpp"
#include "HubTestSupport.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include <nlohmann/json.hpp>
#include <bit>
#include <sqlite3.h>
#include <iostream>
#include <thread>

using namespace vc::objects;

namespace {
GeometryArrayEdit moveX(int64_t point, double x)
{
    const auto bits = std::bit_cast<uint64_t>(x);
    GeometryArrayEdit edit{"control_points", point * 3, std::vector<uint8_t>(8)};
    for (int i = 0; i < 8; ++i) edit.bytes[i] = uint8_t(bits >> (i * 8));
    return edit;
}

void sql(const std::filesystem::path& path, const std::string& statement)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "raw open");
    const int rc = sqlite3_exec(db, statement.c_str(), nullptr, nullptr, nullptr);
    sqlite3_close(db);
    require(rc == SQLITE_OK, "raw statement");
}

// What a position holds, by identities every replica shares: per object its
// metadata, tags, retirement and geometry revision; per relationship its
// record (approval included) and applicability.
nlohmann::json state(ObjectStore& replica, int64_t position)
{
    auto result = nlohmann::json::object();
    readHubPosition(replica, position, [&](ObjectStore& hub, const ReadTxn& txn, const RevisionSelection& selection) {
        RevisionSelection active;
        for (const auto& [object, revision] : selection) {
            const auto value = hub.objectRevision(txn, *revision);
            result["objects"][object] = {{"meta", nlohmann::json::parse(value->record.meta)}, {"tags", value->record.tags},
                                         {"retired", value->record.retired}, {"geometry", value->geometryRevision.value_or("")}};
            if (!value->record.retired) active.emplace(object, revision);
        }
        for (const auto& selected : hub.relationshipsAt(txn, active))
            result["relationships"][selected.relationship.record.uuid] = {
                {"approval", selected.relationship.record.approvalState},
                {"meta", nlohmann::json::parse(selected.relationship.record.meta)}, {"applicable", selected.applicable}};
    });
    return result;
}

std::string addFiber(ObjectStore& store, const std::vector<std::string>& tags = {}, bool geometry = true)
{
    vc::geometry::Fiber fiber;
    fiber.controlPoints = {{0,0,0}, {10,0,0}, {20,0,0}};
    fiber.controlPointIds = {1,2,3};
    auto txn = store.write();
    ObjectRecord record; record.uuid = util::newUuid(); record.kind = "fiber"; record.tags = tags;
    store.upsertObject(txn, record);
    if (geometry) store.putGeometry(txn, record.uuid, vc::geometry::encode(fiber));
    txn.commit();
    return record.uuid;
}

std::string link(ObjectStore& store, const std::string& a, const std::string& b)
{
    auto txn = store.write();
    EdgeRecord edge; edge.kind = "fiber_branch"; edge.aUuid = a; edge.bUuid = b;
    edge.aAnchor.points = {{"control_points", {0}}}; edge.bAnchor.points = edge.aAnchor.points;
    auto uuid = store.addEdge(txn, edge);
    txn.commit();
    return uuid;
}
}

int main() try
{
    TemporaryDirectory directory;
    const auto hubPath = directory.path / "hub.vcobj";
    std::string a, b;
    {
        auto origin = ObjectStore::create(hubPath, as("curator"));
        a = addFiber(origin, {"first"});
        b = addFiber(origin);
        origin.checkpoint();
    }
    initHub(hubPath, "curator");
    const auto alicePath = directory.path / "alice.vcobj", bobPath = directory.path / "bob.vcobj";
    cloneHub(hubPath, alicePath, as("alice"));
    cloneHub(hubPath, bobPath, as("bob"));
    auto alice = ObjectStore::open(alicePath, as("alice"));
    auto bob = ObjectStore::open(bobPath, as("bob"));

    // A clone starts at the hub's position; pulls and pushes report positions.
    const auto start = hubStatus(alice)->position;
    require(start > 0 && hubStatus(bob)->position == start, "clones start at the hub head");
    require(hubStatus(alice)->hubReplica == hubStatus(bob)->hubReplica && !hubStatus(alice)->hubReplica.empty(),
            "replicas name the same hub");
    require(hubPull(alice).pulledPosition == start, "a pull with nothing new stays at the head");
    const auto initial = state(alice, start);
    require(initial["objects"].size() == 2, "the initial position holds the ingested objects");

    // Position P1: a metadata-only edit and a new relationship.
    acquireLeases(alice, {a});
    setNote(alice, a, "one");
    const auto ab = link(alice, a, b);
    auto report = hubPush(alice);
    const auto p1 = report.publishedPosition;
    require(report.pushed > 0 && p1 > start, "a push reports its publication position");

    // Position P2: approval, tags and a deletion, then P3: a retirement,
    // pushed by bob, who pulls only now (alice keeps pulling every step).
    require(hubPull(bob).pulledPosition == p1, "pull reaches the push");
    const auto c = addFiber(bob, {}, false), r = addFiber(bob);  // only objects without geometry can be deleted
    const auto pc = hubPush(bob).publishedPosition;
    require(pc > p1, "creation published");
    hubPull(alice);
    {
        acquireLeases(bob, {b});
        auto txn = bob.write();
        bob.setApproval(txn, ab, "approved");
        auto record = *bob.objectByUuid(txn, b);
        record.tags = {"second"};
        bob.upsertObject(txn, record);
        bob.deleteObject(txn, c);
        txn.commit();
    }
    const auto p2 = hubPush(bob).publishedPosition;
    {
        RevisionSelection expected;
        { auto read = bob.read(); expected[r] = bob.currentObjectRevision(read, r)->uuid; }
        auto txn = bob.write(expected);
        bob.retireObject(txn, r);
        txn.commit();
    }
    const auto p3 = hubPush(bob).publishedPosition;
    require(p1 < pc && pc < p2 && p2 < p3, "positions increase");

    // Every replica, whatever it pulled, reads one state per position: alice
    // pulled each step, bob skipped P1's intermediate states, carol is a fresh clone.
    hubPull(alice);
    const auto carolPath = directory.path / "carol.vcobj";
    cloneHub(hubPath, carolPath, as("carol"));
    auto carol = ObjectStore::open(carolPath, as("carol"));
    for (const auto position : {start, p1, pc, p2, p3}) {
        const auto expected = state(alice, position);
        require(state(bob, position) == expected && state(carol, position) == expected, "replicas agree on a position");
    }
    require(state(alice, start) == initial, "an old position is immutable");
    const auto atP1 = state(alice, p1), atP2 = state(alice, p2), atP3 = state(alice, p3);
    require(atP1["objects"][a]["meta"]["note"] == "one" && initial["objects"][a]["meta"].value("note", "") == "",
            "metadata-only edits are positions");
    require(atP1["relationships"][ab]["approval"] == "pending" && atP2["relationships"][ab]["approval"] == "approved",
            "approval changes are positions");
    require(atP1["objects"][b]["tags"].empty() && atP2["objects"][b]["tags"] == nlohmann::json{"second"},
            "tag changes are positions");
    require(state(alice, pc)["objects"].contains(c) && !atP1["objects"].contains(c) && !atP2["objects"].contains(c),
            "deletion is a position");
    require(!atP2["objects"][r]["retired"].get<bool>() && atP3["objects"][r]["retired"].get<bool>(), "retirement");
    mustFail([&] { state(alice, p3 + 1000); });  // beyond the head

    // Geometry of a position must be local: a replica behind it refuses.
    {
        acquireLeases(alice, {a});
        auto txn = alice.write();
        alice.editGeometry(txn, a, {moveX(2, 25)}, alice.currentGeometry(txn, a)->uuid);
        txn.commit();
    }
    const auto p4 = hubPush(alice).publishedPosition;
    mustFail([&] { state(bob, p4); });
    hubPull(bob);
    require(state(bob, p4) == state(alice, p4), "after pulling, the position reads");

    // Conditional publication. A foreign commit after H refuses the push and
    // publishes nothing, even though it touches an unrelated record.
    const auto h = hubPull(alice).pulledPosition;
    link(alice, addFiber(alice), a);
    hubPull(bob);
    addFiber(bob);
    const auto foreign = hubPush(bob).publishedPosition;
    report = hubPush(alice, h);
    require(report.headMismatch && report.hubHead == foreign && report.pushed == 0 && report.publishedPosition == 0,
            "head mismatch publishes nothing");
    require(!hubStatus(alice)->pending.empty(), "local changes stay pending after a mismatch");
    // Relink at the new head: pull, then push against it.
    const auto h2 = hubPull(alice).pulledPosition;
    require(h2 == foreign, "pulled the foreign commit");
    report = hubPush(alice, h2);
    require(!report.headMismatch && report.publishedPosition > h2, "conditional push at the head publishes");
    const auto n = report.publishedPosition;
    // No-op linking validates the head too.
    report = hubPush(alice, n);
    require(report.publishedPosition == n && !report.headMismatch, "no-op push at the head returns N=H");
    // Against the head its own publication followed, it recovers that
    // publication (a lost response); against an older head, a mismatch.
    report = hubPush(alice, h2);
    require(!report.headMismatch && report.publishedPosition == n, "no-op retry recovers its publication");
    report = hubPush(alice, h);
    require(report.headMismatch && report.hubHead == n, "no-op push against a foreign successor is a mismatch");

    // A lost response: the publication happened but this replica lost its
    // bookkeeping. The retry against the same head recovers N, even after
    // another replica committed on top of it.
    const auto h3 = hubPull(alice).pulledPosition;
    setNote(alice, ab, "lost response");
    const auto lost = hubPush(alice, h3).publishedPosition;
    require(lost > h3, "publication before the simulated loss");
    sql(alicePath, "INSERT INTO hub_pending VALUES('" + ab + "',7,0);DELETE FROM hub_versions WHERE record_uuid='" + ab + "';");
    hubPull(bob);
    addFiber(bob);  // a later foreign commit
    require(hubPush(bob).publishedPosition > lost, "foreign commit after the lost publication");
    report = hubPush(alice, h3);
    require(!report.headMismatch && report.publishedPosition == lost && hubStatus(alice)->pending.empty(),
            "retry recovers the original position");
    require(state(alice, lost)["relationships"][ab]["meta"]["note"] == "lost response", "the recovered position reads");

    // Reverting discards unpushed local changes: a new link disappears, an
    // edited one gets the hub's record back, and nothing is pending.
    {
        hubPull(alice);
        const auto before = meta(alice, ab);
        setNote(alice, ab, "unpushed");
        const auto fresh = addFiber(alice);
        const auto extra = link(alice, fresh, b);
        require(hubRevert(alice) == 2, "both links reverted");
        require(meta(alice, ab) == before, "the hub's record is back");
        require(hubStatus(alice)->pending == std::vector<std::string>{fresh}, "object changes stay pending");
        auto txn = alice.read();
        require(!alice.edgeByUuid(txn, extra), "a link the hub never had is gone");
    }
    require(hubAddress("ssh://host/a b.vcobj") == "sftp://host/a%20b.vcobj" || hubAddress("ssh://host/a b.vcobj").rfind("sftp://", 0) == 0,
            "a remote hub address is canonical sftp://");

    // A hub made before the publication index reads its head, and starts the
    // index with its next write.
    {
        sql(hubPath, "DROP TABLE hub_publications;");
        const auto head = hubPull(alice).pulledPosition;
        require(state(alice, head)["objects"].contains(a), "head of a hub without an index");
        mustFail([&] { state(alice, p1); });
        setNote(alice, ab, "indexed again");
        const auto next = hubPush(alice).publishedPosition;
        require(state(alice, head)["objects"].contains(a) && state(alice, next)["relationships"][ab]["meta"]["note"] == "indexed again",
                "the index restarts at the next write");
    }

    // A session push hold keeps background pushes back; a waited sync pushes
    // and reports the position that includes the edit.
    {
        ObjectStoreSession session;
        HubSessionOptions fast; fast.syncInterval = std::chrono::milliseconds(100); fast.pushDelay = std::chrono::milliseconds(20);
        hubSync(alice);
        { auto closing = std::move(alice); }
        const auto attachment = session.attach(alicePath, as("alice"), fast);
        auto endpoints = [&] {
            RevisionSelection expected;
            session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
                for (const auto& uuid : {a, b}) expected[uuid] = store.currentObjectRevision(txn, uuid)->uuid;
            });
            return expected;
        };
        auto hold = session.holdPushes(attachment);
        require(session.hubStatus(attachment)->pushesHeld, "status shows the hold");
        require(session.hubStatus(attachment)->hubReplica == hubStatus(bob)->hubReplica, "status names the hub");
        session.act(attachment, endpoints(), [&](auto& store, auto& txn) { store.setMeta(txn, ab, "note", "\"held\""); });
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        require(session.hubStatus(attachment)->pending.size() == 1, "background sync does not push under a hold");
        const auto synced = session.syncHubAndWait(attachment, std::chrono::seconds(20));
        require(synced && synced->pending.empty() && synced->error.empty(), "a waited sync pushes under the hold");
        hubPull(bob);
        require(synced->position == hubStatus(bob)->position, "the waited sync reports the hub position it reached");
        require(state(bob, synced->position)["relationships"][ab]["meta"]["note"] == "held", "that position holds the edit");
        const auto journal = session.hubJournal(attachment, synced->position - 1, synced->position);
        require(!journal.empty() && journal.back().recordUuid == ab && journal.back().author == "alice",
                "the hub journal names the records a position changed");
        session.act(attachment, endpoints(), [&](auto& store, auto& txn) { store.setMeta(txn, ab, "note", "\"released\""); });
        hold.reset();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!session.hubStatus(attachment)->pending.empty()) {
            require(std::chrono::steady_clock::now() < deadline, "released hold pushes");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(!session.hubStatus(attachment)->pushesHeld, "hold released");
        session.detach(attachment);
    }
    std::cout << "object_store_hub_positions: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
