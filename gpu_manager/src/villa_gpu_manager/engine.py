"""Single-owner reservation state machine, independent of transport/CUDA."""

import copy
import hmac
import time
import uuid

from .policy import empty, fits, targets, validate_allocation, validate_inventory, validate_request
from .protocol import CAPABILITIES, VERSION, ProtocolError, encode, fields, identity, integer, require
from .processes import descendants, process_state


def validate_process(value):
    fields(value, ("pid", "start_us", "boot_us"))
    for item in value.values():
        integer(item, 1)


class Engine:
    """All mutations are transactional; caller persists before sending response."""

    def __init__(self, inventory, *, controller_uids, state=None, inspect=process_state,
                 discover=descendants, max_tasks=512, protected_processes=lambda: ()):
        validate_inventory(inventory)
        self.inventory = copy.deepcopy(inventory)
        self.controllers = set(controller_uids)
        self.inspect = inspect
        self.discover = discover
        self.protected_processes = protected_processes
        self._protected = list(protected_processes())
        self.max_tasks = max_tasks
        self.draining = False
        self.blocked_gpus = set()
        self.epoch = str(uuid.uuid4())
        self.state = copy.deepcopy(state) if state is not None else {
            "inventory": inventory, "tasks": {}, "override": None,
            "policy_revision": 0, "next_order": 0, "dedup": {}, "retired_ids": []}
        require(self.state["inventory"] == inventory, "journal inventory differs; reconcile before changing pool")
        self.state["dedup"] = {}
        self._exclude_service_members()
        if state is not None:
            for task in self.tasks.values():
                task.setdefault("pausable", True)
                task.setdefault("adaptable", True)
                task.setdefault("fixed_allocation", None)
                if task["terminal"]:
                    continue
                if not task["pausable"]:
                    # Restart cannot revoke a non-preemptible ownership promise.
                    continue
                task["generation"] += 1
                task["phase"] = "release"
                task["reason"] = "server_restart"
                # Preserve reservation even if recipient had not yet acknowledged startup.
                task["held"] = copy.deepcopy(task["granted"])

    @property
    def tasks(self):
        return self.state["tasks"]

    def _exclude_service_members(self):
        """Service children are infrastructure, never workload ownership evidence.

        Match full identities (not reusable PIDs) and preserve task roots. This
        also removes mistakenly discovered service members in older journals.
        """
        for task in self.tasks.values():
            task["processes"] = [p for p in task["processes"]
                                 if p == task["root"] or p not in self._protected]

    def _task_children(self, root, uid):
        return [p for p in self.discover(root, uid) if p not in self._protected]

    def _public(self, task):
        return {k: copy.deepcopy(v) for k, v in task.items() if k not in ("token", "last_report", "processes", "fixed_allocation")}

    def _authorize(self, body, uid, *, controller=False):
        key = identity(body["task_id"])
        require(key in self.tasks, "unknown task", "not_found")
        task = self.tasks[key]
        if not (controller and uid in self.controllers):
            require(uid == task["uid"] and type(body.get("token")) is str
                    and hmac.compare_digest(body["token"], task["token"]), "task authentication failed", "forbidden")
        return task

    def _reconcile(self):
        reasons = {}
        desired = ({key: (task["granted"] if not task["pausable"] and not any(task["paused"].values()) else empty())
                    for key, task in self.tasks.items()} if self.draining
                   else targets(self.tasks, self.inventory, self.state["override"], reasons))
        for key, task in self.tasks.items():
            task["target"] = copy.deepcopy(desired[key])
            if task["terminal"]:
                continue
            if task["phase"] == "waiting":
                task["reason"] = reasons.get(key, "paused" if any(task["paused"].values())
                                             else "waiting_for_resources")
            if task["phase"] in ("applied", "waiting") and task["granted"] != task["target"]:
                if task["granted"] != empty():
                    task["generation"] += 1
                    task["phase"] = "release"
                    task["reason"] = "allocation_changed"
        exclusive = any(self.tasks[k]["request"]["exclusive"] and a != empty()
                        for k, a in desired.items()) and self.state["override"] is None
        for key, task in sorted(self.tasks.items(), key=lambda item: item[1]["order"]):
            if task["terminal"] or task["phase"] != "waiting" or task["target"] == empty():
                continue
            if self.blocked_gpus.intersection(task["target"]["gpus"]):
                task["reason"] = "inventory_unavailable"
                continue
            others = [t["granted"] for k, t in self.tasks.items() if k != key]
            # Both directions need a release barrier: ordinary work may not
            # enter while an outgoing exclusive owner is still draining.
            old_exclusive = self.state["override"] is None and any(
                t["request"]["exclusive"] and t["granted"] != empty()
                for k, t in self.tasks.items() if k != key)
            if (exclusive and any(a != empty() for a in others)) or old_exclusive:
                task["reason"] = "waiting_for_release"
                continue
            if fits(others + [task["target"]], self.inventory):
                task["generation"] += 1
                task["granted"] = copy.deepcopy(task["target"])
                task["phase"] = "initialize"
                task["reason"] = "grant_available"
                if not task["adaptable"] and task["fixed_allocation"] is None:
                    task["fixed_allocation"] = copy.deepcopy(task["granted"])
            else:
                task["reason"] = "waiting_for_release"
        assert fits([t["granted"] for t in self.tasks.values()], self.inventory)

    def _invalidate_override(self):
        if self.state["override"] is not None:
            try:
                targets(self.tasks, self.inventory, self.state["override"])
            except ProtocolError:
                self.state["override"] = None
                self.state["policy_revision"] += 1
                for task in self.tasks.values():
                    task["reason"] = "override_invalidated"

    def dispatch(self, request, uid):
        fields(request, ("version", "id", "op", "body"), ("server_epoch",))
        require(type(request["version"]) is int and request["version"] == VERSION, "unsupported protocol version", "version_mismatch")
        identity(request["id"])
        require(type(request["op"]) is str and type(request["body"]) is dict, "invalid operation/body")
        op, body = request["op"], request["body"]
        if op in ("register", "report"):
            self._protected = list(self.protected_processes())
            self._exclude_service_members()
        if op == "hello":
            fields(body, ("versions", "required_capabilities"))
            require(type(body["versions"]) is list and all(type(v) is int for v in body["versions"])
                    and VERSION in body["versions"], "no common protocol version", "version_mismatch")
            require(type(body["required_capabilities"]) is list
                    and all(type(v) is str for v in body["required_capabilities"]), "invalid capabilities")
            require(set(body["required_capabilities"]) <= set(CAPABILITIES), "unsupported required capability")
            return {"capabilities": CAPABILITIES, "inventory": self.inventory, "peer_uid": uid}
        require(request.get("server_epoch") == self.epoch, "server restarted; reconnect and reconcile", "stale_epoch")
        cache_key = f"{uid}:{request['id']}"
        cached = self.state["dedup"].get(cache_key)
        if cached:
            require(encode(cached["request"]) == encode(request), "request ID reused with different contents", "conflict")
            return copy.deepcopy(cached["response"])
        previous = copy.deepcopy(self.state)
        try:
            response = self._operate(op, body, uid)
            # Read-only polls/status are not retained; bound control retry history.
            if op not in ("poll", "status"):
                self.state["dedup"][cache_key] = {"request": request, "response": response}
                while len(self.state["dedup"]) > 256:
                    del self.state["dedup"][next(iter(self.state["dedup"]))]
            return copy.deepcopy(response)
        except Exception:
            self.state = previous
            raise

    def _operate(self, op, body, uid):
        if op == "register":
            fields(body, ("task_id", "label", "root", "request", "reconfigure_mode", "token"),
                   ("pausable", "adaptable"))
            pausable, adaptable = body.get("pausable", True), body.get("adaptable", True)
            require(type(pausable) is bool and type(adaptable) is bool, "capabilities must be boolean")
            require(pausable or not adaptable, "full_release adaptation requires pausable=true")
            key = identity(body["task_id"])
            require(type(body["token"]) is str and len(body["token"]) == 64
                    and all(c in "0123456789abcdef" for c in body["token"]), "registration requires client-generated 256-bit hex token")
            require(type(body["label"]) is str and 0 < len(body["label"]) <= 256, "invalid label")
            require(body["reconfigure_mode"] == "full_release", "only full_release capability supported")
            validate_request(body["request"], self.inventory)
            validate_process(body["root"])
            require(body["root"] not in self._protected, "manager service cannot register as a task")
            require(key not in self.state["retired_ids"], "task ID is retired", "terminal")
            if key in self.tasks:
                task = self._authorize(body, uid)
                require(not task["terminal"], "task is terminal", "terminal")
                require((pausable, adaptable) == (task["pausable"], task["adaptable"]),
                        "registration capabilities are immutable", "conflict")
                return {"task": self._public(task), "token": task["token"]}
            require(not self.draining, "manager is draining", "unavailable")
            require(sum(not t["terminal"] for t in self.tasks.values()) < self.max_tasks, "active task capacity reached", "capacity")
            require(self.inspect(body["root"], uid) == "alive", "root identity/ownership not verifiable")
            members = [body["root"]] + self._task_children(body["root"], uid)
            for t in self.tasks.values():
                if t["terminal"]:
                    continue
                # A new child can register between periodic scans. Refresh the
                # existing tree *at registration*, not only the proposed tree.
                for p in list(t["processes"]):
                    for child in self._task_children(p, t["uid"]):
                        if child not in t["processes"]:
                            t["processes"].append(child)
                require(not any(p in t["processes"] for p in members), "overlapping task process trees")
            task = {"task_id": key, "label": body["label"], "uid": uid,
                    "root": body["root"], "processes": members, "request": body["request"],
                    "token": body["token"], "request_revision": 0,
                    "pausable": pausable, "adaptable": adaptable, "fixed_allocation": None,
                    "order": self.state["next_order"], "generation": 0,
                    "phase": "waiting", "reason": "registered", "paused": {"self": False, "controller": False},
                    "held": empty(), "granted": empty(), "target": empty(),
                    "terminal": False, "last_contact_ms": int(time.time() * 1000), "last_report": None}
            self.tasks[key] = task
            self.state["next_order"] += 1
            self._reconcile()
            return {"task": self._public(task), "token": task["token"]}
        if op == "status":
            fields(body, (), ("after_order", "limit"))
            after = integer(body.get("after_order", -1), -1)
            limit = integer(body.get("limit", 16), 1)
            require(limit <= 16, "status page limit is 16")
            remaining = sorted((t for t in self.tasks.values() if t["order"] > after), key=lambda t: t["order"])
            page = remaining[:limit]
            return {"inventory": self.inventory, "policy_revision": self.state["policy_revision"],
                    "override": self.state["override"], "draining": self.draining,
                    "blocked_gpu_uuids": sorted(self.blocked_gpus),
                    "tasks": [self._public(t) for t in page],
                    "next_after_order": page[-1]["order"] if len(remaining) > limit else None}
        if op in ("set_override", "clear_override"):
            require(uid in self.controllers, "controller permission required", "forbidden")
            fields(body, ("expected_policy_revision",) + (("allocations", "bound_task_id") if op == "set_override" else ()))
            integer(body["expected_policy_revision"])
            require(body["expected_policy_revision"] == self.state["policy_revision"], "stale policy revision", "conflict")
            if op == "clear_override":
                self.state["override"] = None
            else:
                require(type(body["allocations"]) is dict, "allocations must be an object")
                require(body["allocations"].keys() <= self.tasks.keys(), "unknown override task")
                bound = body["bound_task_id"]
                require(bound is None or type(bound) is str and bound in self.tasks and not self.tasks[bound]["terminal"], "invalid bound task")
                self.state["override"] = {"allocations": body["allocations"], "bound_task_id": bound}
                targets(self.tasks, self.inventory, self.state["override"])
            self.state["policy_revision"] += 1
            self._reconcile()
            return {"policy_revision": self.state["policy_revision"]}
        allowed = {
            "poll": ("last_generation",), "set_request": ("request", "expected_request_revision"),
            "set_paused": ("paused", "scope"), "report": ("generation", "phase", "held", "processes", "detail"),
            "finish": ("outcome",)}
        require(op in allowed, f"unknown operation: {op}", "unknown_operation")
        fields(body, ("task_id",) + allowed[op], ("token",))
        task = self._authorize(body, uid, controller=op in ("set_request", "set_paused"))
        require(not task["terminal"], "task is terminal", "terminal")
        task["last_contact_ms"] = int(time.time() * 1000)
        if op == "poll":
            integer(body["last_generation"])
        elif op == "set_request":
            integer(body["expected_request_revision"])
            require(body["expected_request_revision"] == task["request_revision"], "stale request revision", "conflict")
            validate_request(body["request"], self.inventory)
            fixed = task["fixed_allocation"]
            if fixed:
                validate_allocation(fixed, body["request"], self.inventory)
            require(task["granted"] == empty() or body["request"]["exclusive"] == task["request"]["exclusive"],
                    "release before changing exclusivity", "conflict")
            task["request"] = body["request"]
            task["request_revision"] += 1
            self._invalidate_override()
        elif op == "set_paused":
            require(type(body["paused"]) is bool and body["scope"] in ("self", "controller"), "invalid pause")
            require(body["scope"] != "controller" or uid in self.controllers, "controller permission required", "forbidden")
            require(body["scope"] != "self" or uid == task["uid"], "only task owner may self-pause", "forbidden")
            require(not body["paused"] or body["scope"] == "self" or task["pausable"]
                    or task["granted"] == empty(), "task is non-pausable", "not_pausable")
            task["paused"][body["scope"]] = body["paused"]
            self._invalidate_override()
        elif op == "report":
            integer(body["generation"])
            require(body["generation"] == task["generation"], "stale generation", "conflict")
            # Validate before retry comparison: Python considers True/1/1.0 equal,
            # but the wire protocol requires actual integers for all resources.
            held = body["held"]
            fields(held, ("cpu_cores", "gpus"))
            integer(held["cpu_cores"])
            require(type(held["gpus"]) is dict, "invalid held allocation")
            for gpu, size in held["gpus"].items():
                require(type(gpu) is str, "invalid held GPU identity")
                integer(size)
            if task["last_report"] is not None and encode(task["last_report"]) == encode(body):
                return {"task": self._public(task)}
            require(type(body["detail"]) is str and len(body["detail"]) <= 4096, "invalid report detail")
            require(type(body["processes"]) is list and len(body["processes"]) <= 4096, "invalid process list")
            for p in body["processes"]:
                require(p not in self._protected, "manager service cannot be reported as a worker")
                validate_process(p)
                require(self.inspect(p, uid) == "alive", "reported worker identity not verifiable")
                for other in self.tasks.values():
                    require(other is task or other["terminal"] or p not in other["processes"], "worker belongs to another task")
                if p not in task["processes"]:
                    task["processes"].append(p)
            phase = body["phase"]
            if phase == "applied":
                require(task["phase"] == "initialize" and body["held"] == task["granted"], "applied report differs from grant")
                task["held"] = copy.deepcopy(body["held"])
                task["phase"] = "applied"
            elif phase in ("released", "failed"):
                require(task["phase"] in ("release", "initialize", "applied"), "no active grant to release")
                # Nonzero reports preserve *all* reservations until both GPU and CPU drain.
                held = body["held"]
                fields(held, ("cpu_cores", "gpus"))
                integer(held["cpu_cores"])
                require(type(held["gpus"]) is dict, "invalid held allocation")
                require(held["cpu_cores"] <= task["granted"]["cpu_cores"]
                        and all(g in task["granted"]["gpus"] and type(v) is int and 0 <= v <= task["granted"]["gpus"][g]
                                for g, v in held["gpus"].items()), "held exceeds grant")
                task["held"] = copy.deepcopy(held)
                if held == empty():
                    task["granted"] = empty()
                    task["phase"] = "waiting"
                    task["processes"] = [task["root"]] + [p for p in task["processes"]
                        if p != task["root"] and self.inspect(p, uid) != "dead"]
                else:
                    task["phase"] = "release"
                if phase == "failed":
                    task["paused"]["self"] = True
                    task["reason"] = body["detail"] or "client_failed"
                    self._invalidate_override()
            else:
                raise ProtocolError("invalid_request", "unknown report phase")
            task["last_report"] = copy.deepcopy(body)
        elif op == "finish":
            require(type(body["outcome"]) is str and len(body["outcome"]) <= 4096, "invalid outcome")
            require(task["granted"] == empty() and task["held"] == empty(), "release all resources before finish")
            task["terminal"] = True
            task["phase"] = "finished"
            task["reason"] = body["outcome"]
            self._clear_bound(task["task_id"])
            self._invalidate_override()
        self._reconcile()
        return {"task": self._public(task)}

    def _clear_bound(self, key):
        override = self.state["override"]
        if override and override["bound_task_id"] == key:
            self.state["override"] = None
            self.state["policy_revision"] += 1

    def drain(self):
        self.draining = True
        self._reconcile()

    def update_blocked_gpus(self, gpu_uuids):
        self.blocked_gpus = set(gpu_uuids)
        self._reconcile()

    def scan_processes(self):
        self._protected = list(self.protected_processes())
        self._exclude_service_members()
        for key, task in self.tasks.items():
            if task["terminal"]:
                continue
            for root in list(task["processes"]):
                for child in self._task_children(root, task["uid"]):
                    if child not in task["processes"]:
                        task["processes"].append(child)
            states = [self.inspect(p, task["uid"]) for p in task["processes"]]
            if all(s == "dead" for s in states):
                task.update(terminal=True, phase="finished", reason="process_tree_dead", held=empty(), granted=empty())
                self._clear_bound(key)
            elif states[0] != "alive":
                task["paused"]["self"] = True
                task["reason"] = "orphaned_or_unverifiable_process_tree"
        self._invalidate_override()
        self._reconcile()
        # Compact terminal records; retain UUID tombstones so late registration
        # cannot resurrect a completed task. Tombstones contain no secrets/state.
        terminal = sorted((t for t in self.tasks.values() if t["terminal"]), key=lambda t: t["order"])
        for task in terminal[:-32]:
            self.state["retired_ids"].append(task["task_id"])
            del self.tasks[task["task_id"]]
