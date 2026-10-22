"""Durable parent-side state for cooperative training maintenance."""

from __future__ import annotations


def maintenance_key(request: dict) -> tuple[str, int, int]:
    if not isinstance(request, dict):
        raise TypeError("maintenance request must be an object")
    kind = request.get("kind")
    step = request.get("step")
    samples_seen = request.get("samples_seen")
    if not isinstance(kind, str) or not kind:
        raise ValueError("maintenance kind must be a nonempty string")
    if type(step) is not int or step < 0:
        raise ValueError("maintenance step must be a nonnegative integer")
    if type(samples_seen) is not int or samples_seen < 0:
        raise ValueError("maintenance samples_seen must be a nonnegative integer")
    return kind, step, samples_seen


class MaintenanceMailbox:
    """Keep one exact maintenance request across replaceable GPU cohorts."""

    def __init__(self):
        self.request: dict | None = None
        self._completed: set[tuple[str, int, int]] = set()
        self.attempt_token = None

    @property
    def pending(self) -> bool:
        return self.request is not None

    def publish(self, request: dict) -> bool:
        key = maintenance_key(request)
        if key in self._completed:
            return False
        if self.request is None:
            self.request = dict(request)
            return True
        if maintenance_key(self.request) != key or self.request != request:
            raise RuntimeError(
                f"conflicting maintenance requests: current={self.request!r} new={request!r}"
            )
        return False

    def complete(self, request: dict) -> None:
        key = maintenance_key(request)
        if self.request is None or maintenance_key(self.request) != key:
            raise RuntimeError("completed maintenance does not match the pending request")
        if self.attempt_token is not None:
            raise RuntimeError("cannot complete maintenance while an attempt is active")
        self._completed.add(key)
        self.request = None

    @property
    def attempt_active(self) -> bool:
        return self.attempt_token is not None

    def begin_attempt(self, request: dict, token) -> None:
        if self.request is None or maintenance_key(self.request) != maintenance_key(request):
            raise RuntimeError("maintenance attempt does not match the pending request")
        if self.attempt_token is not None:
            raise RuntimeError("maintenance attempt is already active")
        self.attempt_token = token

    def end_attempt(self, token) -> None:
        if self.attempt_token != token:
            raise RuntimeError("maintenance attempt ownership token changed")
        self.attempt_token = None
