# Task

Fix the synthetic rendering CI gate so it runs in the dependency container,
which has no NumPy. Move deterministic event-feature and thread-cost scoring
into the C++ replay engine; retain Python only for Valgrind orchestration and
artifact parsing. Add a clean-Python/container dependency smoke check.
