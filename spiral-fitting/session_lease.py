"""Ownership of a Spiral service's resident session by one client.

A client claims the lease with a token and presents it on every mutating
request; lifecycle commands (initialize, rebuild, run, ...) run one at a
time through the lease's MutationCoordinator, which replays a repeated
command ID instead of acting twice. Releasing the lease tears the resident
session down (ServiceState.release_lease).

The lease holds no inputs. Patches, PCLs and fibers come from the object
store the service is bound to and are frozen for a resident session, so
claiming snapshots nothing and there is nothing to apply, commit or replay.
"""
from __future__ import annotations

import hashlib
from uuid import uuid4

from service_mutations import ClientLease, MutationCoordinator
from service_http import ApiError


class SessionLease:
    """One client token owns a service's mutation surface."""

    def __init__(self, output_root):
        self.id = str(uuid4())
        self.coordinator = MutationCoordinator()
        # The file lock lives under the service's output root: two services
        # writing one output root cannot both have an owner.
        self._lease = ClientLease(output_root)
        # Responses of initialize/rebuild commands already started, by
        # command ID, so a retried request does not build twice.
        self.lifecycle_started = {}

    def claim(self, token, command_id):
        def claim(_):
            try:
                self._lease.claim(token)
            except ApiError as exc:
                raise self._reworded(exc) from exc
            return {"lease_id": self.id, "owned": True}
        return self.coordinator.execute(
            command_id, "claim",
            {"token_digest": hashlib.sha256(str(token).encode()).hexdigest()},
            claim, recoverable=lambda exc: isinstance(exc, TimeoutError))

    def require(self, token):
        try:
            self._lease.require(token)
        except ApiError as exc:
            raise self._reworded(exc) from exc

    @staticmethod
    def _reworded(error):
        """ClientLease's refusals, in terms of the session lease."""
        messages = {400: "An ASCII session token is required",
                    403: "This client does not own the session lease",
                    409: "Another service owns this output root's session lease"}
        return ApiError(error.status, messages.get(error.status, error.message),
                        error.details, error.payload)

    def close(self, timeout=15):
        """Drain the running lifecycle command, then drop ownership."""
        self.coordinator.shutdown(self._lease.close, timeout=timeout)
