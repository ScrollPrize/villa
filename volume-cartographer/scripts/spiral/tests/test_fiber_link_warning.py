"""The warning that an uploaded fiber's cross-fiber links are inert.

Links are resolved once, when the fit is built, so branches drawn on a fiber
added to a live session do nothing until it is committed and the fit rebuilt.
"""

from fit_spiral import unresolved_fiber_link_warning


def _fiber(*branches):
    return {'branches': [{'pending': pending} for pending in branches]}


def test_no_warning_without_links():
    assert unresolved_fiber_link_warning(
        [('fiber-a', _fiber()), ('fiber-b', {})],
        use_links=True, use_pending_links=False) is None


def test_counts_links_and_names_the_fibers():
    warning = unresolved_fiber_link_warning(
        [('fiber-a', _fiber(False, False)), ('fiber-b', _fiber(False))],
        use_links=True, use_pending_links=False)

    assert warning.startswith('3 cross-fiber link(s) on 2 added fiber(s)')
    assert 'fiber-a (2)' in warning and 'fiber-b (1)' in warning
    assert 'rebuild' in warning


def test_pending_links_count_only_when_they_are_configured_in():
    fibers = [('fiber-a', _fiber(False, True))]

    assert '1 cross-fiber link(s)' in unresolved_fiber_link_warning(
        fibers, use_links=True, use_pending_links=False)
    assert '2 cross-fiber link(s)' in unresolved_fiber_link_warning(
        fibers, use_links=True, use_pending_links=True)


def test_pending_only_fiber_is_silent_when_pending_links_are_off():
    assert unresolved_fiber_link_warning(
        [('fiber-a', _fiber(True))],
        use_links=True, use_pending_links=False) is None


def test_links_configured_off_warn_about_nothing():
    # Nothing is being lost: a rebuild would not use these links either.
    assert unresolved_fiber_link_warning(
        [('fiber-a', _fiber(False))],
        use_links=False, use_pending_links=False) is None


def test_long_fiber_lists_are_truncated():
    fibers = [(f'fiber-{i}', _fiber(False)) for i in range(9)]

    warning = unresolved_fiber_link_warning(
        fibers, use_links=True, use_pending_links=False)

    assert warning.startswith('9 cross-fiber link(s) on 9 added fiber(s)')
    assert 'fiber-5 (1)' in warning
    assert 'fiber-6' not in warning
    assert 'and 3 more' in warning
