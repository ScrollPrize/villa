# QuadSurface cache invalidation race

Fix the reproduced render-worker crash when switching projects or active
surfaces clears derived caches during QuadSurface::gen(). Branch from current
main. Preserve rendering results and in-flight work. Submit the PR title/body
for user approval before publishing the PR.
