#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <future>
#include <thread>
#include <iostream>
#include <algorithm>
#include <cctype>

using namespace vc::objects;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "session-test";
    const auto path = directory.path / "mixed.vcobj";
    auto external = ObjectStore::create(path, options);
    std::string a, b;
    {
        auto txn = external.write();
        ObjectRecord object; object.kind = "fiber"; a = external.upsertObject(txn, object);
        object.kind = "point_collection"; b = external.upsertObject(txn, object);
        txn.commit();
    }
    ObjectStoreSession session;
    auto identity = session.attach(path, options);
    auto events = session.takeEvents();
    require(events.size() == 1 && events[0].kind == StoreSessionEvent::Kind::Attachment, "attachment event");
    mustFail([&] { session.attach(directory.path / "absent.vcobj", options); });
    require(!std::filesystem::exists(directory.path / "absent.vcobj"), "attach must not create");
    require(session.current(identity), "failed replacement preserved old attachment");
    auto readonly = options; readonly.readOnly = true;
    identity = session.attach(path, readonly);
    require(!session.writable(), "read-only attachment reports access mode");
    session.read(identity, [&](const auto& store, const auto& txn, const auto&) {
        require(store.objectByUuid(txn, a).has_value(), "read-only attachment reads objects");
    });
    bool callbackRan = false;
    mustFail([&] { session.act(identity, {}, [&](auto&, auto&) { callbackRan = true; }); });
    require(!callbackRan, "read-only actions rejected before callback");
    identity = session.attach(path, options);
    session.takeEvents();
    require(session.writable(), "writable reattachment");

    auto heads = [&] {
        RevisionSelection result;
        session.read(identity, [&](const auto& store, const auto& txn, const auto&) {
            result[a] = store.currentObjectRevision(txn, a)->uuid;
            result[b] = store.currentObjectRevision(txn, b)->uuid;
        });
        return result;
    };
    auto expected = heads();
    std::string edge;
    session.act(identity, expected, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "note", "1");
        store.setMeta(txn, b, "note", "2");
        EdgeRecord link; link.kind = "mixed"; link.aUuid = a; link.bUuid = b;
        edge = store.addEdge(txn, link);
    });
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 3, "mixed action notification");
    require(events[0].attachment == identity && events[0].journalPosition > 0, "event identity and journal");
    for (const auto& change : events[0].changes)
        require(change.kind == (change.recordUuid == edge ? "mixed" : change.recordUuid == a ? "fiber" : "point_collection"), "changes carry record kinds");
    const auto committed = heads();
    mustFail([&] { session.act(identity, expected, [&](auto& store, auto& txn) { store.setMeta(txn, a, "stale", "true"); }); });
    require(heads() == committed && session.takeEvents().empty(), "conflict leaves no changes");
    mustFail([&] { session.act(identity, committed, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "rollback", "true");
        throw std::runtime_error("abort");
    }); });
    require(heads() == committed && session.takeEvents().empty(), "callback rollback");
    mustFail([&] { session.act(identity, {{a, committed.at(a)}}, [&](auto& store, auto& txn) { store.removeEdge(txn, edge); }); });
    require(heads() == committed, "all relationship endpoints must be checked");

    auto upper = a;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    auto receipt = session.act(identity, {{upper, committed.at(a)}}, [&](auto& store, auto& txn) {
        store.setMeta(txn, a, "receipt", "true");
    });
    require(receipt.at(upper) == heads().at(a), "receipt supports accepted UUID spellings");
    session.takeEvents();

    // External changes to a non-fiber object must be observed.
    { auto txn = external.write(); external.setMeta(txn, b, "external", "true"); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes.size() == 1 && events[0].changes[0].recordUuid == b, "external notification");
    { auto txn = external.write(); external.removeEdge(txn, edge); txn.commit(); }
    events = session.takeEvents();
    require(events.size() == 1 && events[0].changes[0].kind == "mixed", "deletions keep their record kind");

    // Same expected head submitted concurrently: exactly one action wins.
    expected = heads();
    std::atomic_int successes{0}, conflicts{0};
    auto writer = [&] {
        try {
            session.act(identity, expected, [&](auto& store, auto& txn) { store.setMeta(txn, a, "race", "true"); });
            ++successes;
        } catch (const std::exception&) { ++conflicts; }
    };
    auto one = std::async(std::launch::async, writer), two = std::async(std::launch::async, writer);
    one.get(); two.get(); session.finalize();
    require(successes == 1 && conflicts == 1, "serialized checked actions");
    session.takeEvents();

    // Detachment cancels and waits for a worker-local snapshot. The snapshot
    // remains consistent while another handle commits.
    std::promise<void> entered, changed;
    auto changeReady = changed.get_future();
    auto reader = std::async(std::launch::async, [&] {
        session.read(identity, [&](const auto& store, const auto& txn, const auto& cancelled) {
            const auto before = store.currentObjectRevision(txn, b)->uuid;
            entered.set_value();
            changeReady.wait();
            require(store.currentObjectRevision(txn, b)->uuid == before, "worker snapshot consistency");
            while (!cancelled.load()) std::this_thread::yield();
        });
    });
    entered.get_future().wait();
    { auto txn = external.write(); external.setMeta(txn, b, "external", "2"); txn.commit(); }
    changed.set_value();
    session.detach(); reader.get();
    require(!session.attachment(), "detached");
    events = session.takeEvents();
    require(events.size() == 2 && events[0].kind == StoreSessionEvent::Kind::Committed &&
            events[1].kind == StoreSessionEvent::Kind::Attachment, "old commits precede detachment");
    mustFail([&] { session.act(identity, {}, [](auto&, auto&) {}); });
    const auto second = session.attach(path, options);
    require(second.storeUuid == identity.storeUuid && second.generation > identity.generation, "same store invalidates old requests");
    mustFail([&] { session.read(identity, [](const auto&, const auto&, const auto&) {}); });
    // Accepted writes finish before detachment. A writer submitted after the
    // transition begins must not slip onto the old attachment.
    std::promise<void> writing, finishWriting;
    auto finishReady = finishWriting.get_future();
    RevisionSelection latest;
    session.read(second, [&](const auto& store, const auto& txn, const auto&) { latest[a] = store.currentObjectRevision(txn, a)->uuid; });
    auto pending = std::async(std::launch::async, [&] {
        session.act(second, latest, [&](auto& store, auto& txn) {
            writing.set_value(); finishReady.wait(); store.setMeta(txn, a, "drained", "true");
        });
    });
    writing.get_future().wait();
    auto detach = std::async(std::launch::async, [&] { session.detach(); });
    while (session.current(second)) std::this_thread::yield();
    mustFail([&] { session.act(second, latest, [](auto&, auto&) {}); });
    finishWriting.set_value(); pending.get(); detach.get();
    { auto txn = external.read(); require(external.objectByUuid(txn, a)->meta.find("drained") != std::string::npos, "accepted write drained"); }
    auto replacementPath = directory.path / "replacement.vcobj";
    { auto replacement = ObjectStore::create(replacementPath, options); }
    auto replacement = session.attach(replacementPath, options);
    require(replacement.storeUuid != second.storeUuid && replacement.generation > second.generation, "different-store replacement");
    session.detach();
    std::cout << "session tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
