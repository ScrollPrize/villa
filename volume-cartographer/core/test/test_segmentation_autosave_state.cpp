#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "segmentation/SegmentationAutosaveState.hpp"

TEST_CASE("current autosave completion applies")
{
    segmentation::AutosaveState state;
    state.beginSession();
    state.markPending();

    const auto ticket = state.startSave();
    const auto completion = state.completeSuccess(ticket);

    CHECK(completion == segmentation::AutosaveState::Completion::Current);
    CHECK_FALSE(state.pending());
    CHECK_FALSE(state.saveInProgress());
    CHECK_FALSE(state.dirtyAfterSave());
}
TEST_CASE("stale autosave completion is ignored without clearing current pending work")
{
    segmentation::AutosaveState state;
    state.beginSession();
    state.markPending();
    const auto oldTicket = state.startSave();

    state.beginSession();
    state.markPending();
    const auto completion = state.completeSuccess(oldTicket);

    CHECK(completion == segmentation::AutosaveState::Completion::Stale);
    CHECK(state.pending());
}

TEST_CASE("stale autosave failure does not requeue after session end")
{
    segmentation::AutosaveState state;
    state.beginSession();
    state.markPending();
    const auto oldTicket = state.startSave();

    state.endSession();
    const auto completion = state.completeFailure(oldTicket, true);

    CHECK(completion == segmentation::AutosaveState::Completion::Stale);
    CHECK_FALSE(state.pending());
    CHECK_FALSE(state.saveInProgress());
    CHECK_FALSE(state.dirtyAfterSave());
}
