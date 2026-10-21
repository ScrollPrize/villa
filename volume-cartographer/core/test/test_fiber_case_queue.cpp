#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberCaseQueue.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <set>
#include <mutex>

using namespace vc::fiber_tracer;
using namespace std::chrono_literals;
namespace {
template<class Predicate> bool eventually(Predicate ready) {
    const auto end=std::chrono::steady_clock::now()+3s;
    while(!ready() && std::chrono::steady_clock::now()<end) std::this_thread::sleep_for(1ms);
    return ready();
}
struct Scope {
    std::atomic<int>& count;
    explicit Scope(std::atomic<int>& value):count(value) { ++count; }
    ~Scope() { --count; }
};
}

TEST_CASE("bounded suspended cases retain stacks and original OS threads") {
    if(!FiberCaseQueue::supported()) {
        CHECK_THROWS(FiberCaseQueue(1,1,[]{}));
        return;
    }
    std::atomic<size_t> entered=0,completed=0;
    std::atomic<bool> ready=false,correct=true;
    std::mutex mutex;
    std::set<std::thread::id> ids;
    FiberCaseQueue queue(2,600,[&] {
        const auto thread=std::this_thread::get_id();
        { std::lock_guard lock(mutex);ids.insert(thread); }
        const auto value=entered.fetch_add(1);
        volatile size_t preserved=value;
        FiberCaseQueue::await([&] {return ready.load();});
        if(thread!=std::this_thread::get_id() || preserved!=value) correct=false;
        ++completed;
    },4);
    REQUIRE(eventually([&] {return entered==8;}));
    CHECK(completed==0);
    ready=true;FiberCaseQueue::notify();
    queue.join();
    CHECK(completed==600);
    CHECK(ids.size()==2);
    CHECK(correct);
}

TEST_CASE("nested children progress with all contexts occupied by parents") {
    if(!FiberCaseQueue::supported()) return;
    std::atomic<int> calls=0;
    FiberCaseQueue queue(2,12,[&] {
        FiberCaseQueue::parallelFor(7,[&](size_t) {
            FiberCaseQueue::parallelFor(3,[&](size_t) { ++calls; });
        });
    },1);
    queue.join();
    CHECK(calls==12*7*3);
}

TEST_CASE("sibling failure unwinds suspended cases and reports first error") {
    if(!FiberCaseQueue::supported()) return;
    std::atomic<int> live=0,entered=0;
    FiberCaseQueue queue(1,20,[&] {
        Scope scope(live);
        if(entered++==2) throw std::runtime_error("sibling failed");
        FiberCaseQueue::await([] {return false;});
    },4);
    CHECK_THROWS_AS(queue.join(),std::runtime_error);
    CHECK(live==0);
    CHECK(entered==3);
}

TEST_CASE("idle workers remain available for children submitted later") {
    if(!FiberCaseQueue::supported()) return;
    std::atomic<int> entered=0;
    std::atomic<bool> release=false;
    FiberCaseQueue queue(2,1,[&] {
        std::this_thread::sleep_for(20ms);
        FiberCaseQueue::parallelFor(2,[&](size_t) {
            ++entered;
            // Deliberately OS-block: verifies two different workers remain.
            while(!release.load()) std::this_thread::yield();
        });
    },1);
    const bool both=eventually([&] {return entered==2;});
    release=true;
    queue.join();
    CHECK(both);
}

TEST_CASE("cancel drains child references before unwinding their parent") {
    if(!FiberCaseQueue::supported()) return;
    std::atomic<int> live=0,children=0;
    FiberCaseQueue queue(2,4,[&] {
        Scope parent(live);
        int parentLocal=42;
        FiberCaseQueue::parallelFor(12,[&](size_t) {
            Scope child(live);
            ++children;
            FiberCaseQueue::await([] {return false;});
            if(parentLocal!=42) throw std::runtime_error("parent freed early");
        });
    },1);
    REQUIRE(eventually([&] {return children>=2;}));
    queue.cancel("test stop");
    CHECK_THROWS_AS(queue.join(),std::runtime_error);
    CHECK(live==0);
}

TEST_CASE("permit scopes are case local and released during publication waits") {
    if(!FiberCaseQueue::supported()) return;
    std::counting_semaphore<> permits(1);
    std::atomic<int> entered=0;
    std::atomic<bool> ready=false,correct=true;
    FiberCaseQueue queue(1,2,[&] {
        FiberConstructionCpuPermit permit(permits);
        if(FiberConstructionCpuPermit::current()!=&permit) correct=false;
        const int index=entered++;
        if(index==0) permit.suspend([&] {FiberCaseQueue::await([&] {return ready.load();});});
        else ready=true;
        if(FiberConstructionCpuPermit::current()!=&permit) correct=false;
    },2);
    queue.join();
    CHECK(correct);
    CHECK(entered==2);
    CHECK(FiberConstructionCpuPermit::current()==nullptr);
    CHECK(permits.try_acquire());
    CHECK_FALSE(permits.try_acquire());
}

TEST_CASE("predicate errors and destructor cancellation unwind local resources") {
    if(!FiberCaseQueue::supported()) return;
    std::atomic<int> live=0;
    {
        int calls=0;
        FiberCaseQueue queue(1,1,[&] {
            Scope scope(live);
            FiberCaseQueue::await([&] {
                if(++calls>1) throw std::runtime_error("bad predicate");
                return false;
            });
        });
        CHECK_THROWS_AS(queue.join(),std::runtime_error);
        CHECK(live==0);
    }
    {
        FiberCaseQueue queue(1,1,[&] {
            Scope scope(live);
            FiberCaseQueue::await([] {return false;});
        });
        REQUIRE(eventually([&] {return live==1;}));
    }
    CHECK(live==0);
    CHECK_THROWS(FiberCaseQueue::await([] {return true;}));
    CHECK_THROWS(FiberCaseQueue(0,1,[]{}));
    FiberCaseQueue empty(1,0,[]{});
    empty.join();
}
