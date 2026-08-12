# Task log

## 2026-08-12

- Confirmed the main window creates separate permanent cache-stat and
  Z-sensitivity labels. Their independent sizing allows the expanded download
  diagnostics to be obscured.
- Plan review: composing both fields from retained state in one label directly
  addresses overlap while preserving both existing update signals. The change
  is display-only and does not alter cache or rendering behavior.
- Added a tested storage formatter with one shared trailing GiB unit for RAM
  and persistent-disk values.
- Removed the separate Z-sensitivity status widget. `CWindow` now retains the
  latest cache fields and composes them with the current sensitivity into the
  existing cache status label on either update signal.
- Updated the specification, remote-cache documentation, and changelog.
- Validation:
  - `git diff --check`
  - `cmake --build volume-cartographer/build --target test_download_queue_stats VC3D -j4`
  - `ctest --test-dir volume-cartographer/build --output-on-failure -R '^download_queue_stats$'`
- Result: the focused formatter test passed and VC3D linked successfully.
  Existing Qt SFINAE completeness warnings remained during the application
  build; this task introduced no build errors.
- Follow-up visual correction: the first merged-label implementation retained
  a fixed 320-pixel minimum, allowing Qt to compress longer active-download
  text. The label now derives its minimum width from the current rendered text
  after every cache or Z-sensitivity update.
- Changed active network formatting from `N@S.SMiB/s` to
  `Nx S.SMiB/s` for clearer separation of concurrency and speed.
- No task simplifications, deferred requirements, or implementation deviations.
