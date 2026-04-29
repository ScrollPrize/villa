#include "test.hpp"

#include <atomic>
#include <chrono>
#include <thread>

#include "utils/http_fetch.hpp"

// Sanity tests for utils::CancelScope / HttpClient::shouldAbort(). The
// rest of the codebase depends on these to isolate one remote-load's
// cancellation from any concurrent remote load running on the same
// Qt thread pool.

TEST(HttpCancel, NoScopeMeansNoCancel)
{
    utils::HttpClient::resetAbort();
    EXPECT_FALSE(utils::HttpClient::shouldAbort());
    EXPECT_FALSE(utils::HttpClient::isAborted());
}

TEST(HttpCancel, ScopeObservesItsOwnToken)
{
    utils::HttpClient::resetAbort();
    auto token = std::make_shared<std::atomic<bool>>(false);
    utils::CancelScope scope(token.get());

    EXPECT_FALSE(utils::HttpClient::shouldAbort());
    token->store(true);
    EXPECT_TRUE(utils::HttpClient::shouldAbort());
    token->store(false);
    EXPECT_FALSE(utils::HttpClient::shouldAbort());
}

TEST(HttpCancel, ScopeRestoresPreviousToken)
{
    utils::HttpClient::resetAbort();
    auto outer = std::make_shared<std::atomic<bool>>(false);
    auto inner = std::make_shared<std::atomic<bool>>(false);

    utils::CancelScope s1(outer.get());
    {
        utils::CancelScope s2(inner.get());
        // While nested, the inner token is observed.
        inner->store(true);
        EXPECT_TRUE(utils::HttpClient::shouldAbort());
        inner->store(false);
    }
    // After inner pops, outer token is observed again.
    outer->store(true);
    EXPECT_TRUE(utils::HttpClient::shouldAbort());
    outer->store(false);
    EXPECT_FALSE(utils::HttpClient::shouldAbort());
}

TEST(HttpCancel, PerThreadIsolation)
{
    // Two threads, each installs its own CancelScope. Firing one token
    // must not cause the other thread's shouldAbort() to report true.
    utils::HttpClient::resetAbort();

    auto tokenA = std::make_shared<std::atomic<bool>>(false);
    auto tokenB = std::make_shared<std::atomic<bool>>(false);

    std::atomic<int> step{0};
    std::atomic<bool> bSawCancel{false};

    std::thread tB([&]() {
        utils::CancelScope scopeB(tokenB.get());
        step.store(1);  // tell A that B is in its scope
        while (step.load() < 2) std::this_thread::yield();

        // A has now set tokenA, but we hold only tokenB — no cancel.
        if (utils::HttpClient::shouldAbort()) {
            bSawCancel.store(true);
        }

        step.store(3);
        while (step.load() < 4) std::this_thread::yield();
    });

    while (step.load() < 1) std::this_thread::yield();
    {
        utils::CancelScope scopeA(tokenA.get());
        tokenA->store(true);
        EXPECT_TRUE(utils::HttpClient::shouldAbort());  // A sees its own
        step.store(2);                                   // unblock B
        while (step.load() < 3) std::this_thread::yield();
    }
    step.store(4);
    tB.join();

    EXPECT_FALSE(bSawCancel.load());
}

TEST(HttpCancel, GlobalAbortStillSeen)
{
    // Process-global abortAll() is an escape hatch for shutdown — it
    // intentionally wins over per-thread tokens. Verify it still works.
    utils::HttpClient::resetAbort();
    auto token = std::make_shared<std::atomic<bool>>(false);
    utils::CancelScope scope(token.get());

    EXPECT_FALSE(utils::HttpClient::shouldAbort());
    utils::HttpClient::abortAll();
    EXPECT_TRUE(utils::HttpClient::shouldAbort());
    utils::HttpClient::resetAbort();
    EXPECT_FALSE(utils::HttpClient::shouldAbort());
}
