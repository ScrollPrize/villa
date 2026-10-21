"""Reference synchronous metadata client; call from the CPU supervisor."""

import socket
import secrets
import uuid

from .processes import process_identity
from .protocol import MAX_FRAME, VERSION, ProtocolError, decode, encode, require


class Client:
    def __init__(self, path, *, timeout=5, registration_root=None):
        self.path, self.timeout = str(path), timeout
        self.socket = None
        self.epoch = None
        self.task_id = None
        self.token = None
        self.task = None
        self.capabilities = None
        self.registration_root = registration_root

    def connect(self):
        self.close()
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect(self.path)
        reply = self.call("hello", {"versions": [VERSION], "required_capabilities": ["full_release", "poll", "task_scheduling"]})
        return reply

    def call(self, op, body, *, request_id=None):
        require(self.socket is not None, "client not connected")
        request_id = request_id or str(uuid.uuid4())
        request = {"version": VERSION, "id": request_id, "op": op, "body": body}
        if op != "hello":
            request["server_epoch"] = self.epoch
        try:
            self.socket.sendall(encode(request))
            data = bytearray()
            while b"\n" not in data:
                part = self.socket.recv(65536)
                if not part:
                    raise ConnectionError("manager disconnected; release at safe boundary")
                data.extend(part)
                require(len(data) <= MAX_FRAME + 1, "oversized server response")
            require(data.endswith(b"\n") and data.count(b"\n") == 1, "unexpected response framing")
            response = decode(data[:-1])
            require(response.get("id") == request_id and response.get("version") == VERSION, "response identity mismatch")
            if not response["ok"]:
                error = response["error"]
                raise ProtocolError(error["code"], error["message"], error["details"])
            self.epoch = response["server_epoch"]
            return response["body"]
        except (OSError, ProtocolError):
            # A timeout can leave a delayed reply on this socket; never reuse it.
            self.close()
            raise

    def register(self, label, request, *, task_id=None, root=None, token=None,
                 pausable=None, adaptable=None):
        if self.capabilities is None:
            self.capabilities = {"pausable": True if pausable is None else pausable,
                                 "adaptable": True if adaptable is None else adaptable}
        else:
            require(pausable is None or pausable == self.capabilities["pausable"], "capabilities are immutable")
            require(adaptable is None or adaptable == self.capabilities["adaptable"], "capabilities are immutable")
        self.task_id = task_id or self.task_id or str(uuid.uuid4())
        # Retain the secret *before* sending, so losing the registration response
        # or restarting the manager cannot strand an allocated task.
        self.token = token or self.token or secrets.token_hex(32)
        body = {"task_id": self.task_id, "label": label, "root": root or self.registration_root or process_identity(),
                "request": request, "reconfigure_mode": "full_release", "token": self.token,
                **self.capabilities}
        result = self.call("register", body)
        self.token, self.task = result["token"], result["task"]
        return self.task

    def task_call(self, op, **body):
        result = self.call(op, {"task_id": self.task_id, "token": self.token, **body})
        if "task" in result:
            self.task = result["task"]
        return result

    def poll(self):
        return self.task_call("poll", last_generation=self.task["generation"] if self.task else 0)["task"]

    def status(self):
        result = self.call("status", {})
        cursor = result["next_after_order"]
        while cursor is not None:
            page = self.call("status", {"after_order": cursor})
            result["tasks"].extend(page["tasks"])
            cursor = page["next_after_order"]
        result["next_after_order"] = None
        return result

    def report(self, phase, held, *, processes=(), detail=""):
        return self.task_call("report", generation=self.task["generation"], phase=phase,
                              held=held, processes=list(processes), detail=detail)["task"]

    def set_request(self, request):
        return self.task_call("set_request", request=request, expected_request_revision=self.task["request_revision"])["task"]

    def pause(self, paused=True):
        return self.task_call("set_paused", paused=paused, scope="self")["task"]

    def finish(self, outcome="completed"):
        return self.task_call("finish", outcome=outcome)["task"]

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        # Socket close does NOT claim that a caller's workers have released GPUs.
        self.close()
