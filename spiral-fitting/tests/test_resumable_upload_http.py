"""Exercise recovery endpoints through the authenticated service handler."""

import json
import urllib.error
import urllib.request

from test_resumable_uploads import DATA, request as upload_request
from test_spiral_service_v2 import HttpServiceFixture, _attach_fake_session


class ResumableUploadHttpTests(HttpServiceFixture):
    def setUp(self):
        super().setUp()
        dataset = self.root / "dataset"
        (dataset / "verified_patches").mkdir(parents=True)
        (dataset / "fibers").mkdir()
        output = self.root / "output"
        output.mkdir()
        _attach_fake_session(self.state, output, dataset)

    def put(self, upload_id, data, offset):
        request = urllib.request.Request(
            self.base + f"/session/inputs/{upload_id}/files/fiber.json?offset={offset}",
            data=data, method="PUT", headers={
                "Authorization": "Bearer secret-key",
                "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_reconnect_reconciles_each_transfer_response(self):
        manifest = upload_request()
        # Ignore the create response, as when a transport drops after success.
        self.request("POST", "/session/inputs", body=manifest)
        code, payload, _ = self.request("POST", "/session/inputs", body=manifest)
        self.assertEqual(code, 200)
        upload_id = json.loads(payload)["upload_id"]
        self.put(upload_id, DATA[:8], 0)
        code, result = self.put(upload_id, DATA[:8], 0)
        self.assertEqual(code, 409)
        self.assertEqual(result["offset"], 8)
        code, payload, _ = self.request("GET", f"/session/inputs/{upload_id}")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(payload)["files"][0]["offset"], 8)
        self.assertEqual(self.put(upload_id, DATA[8:], 8)[0], 200)
        path = f"/session/inputs/{upload_id}/finalize"
        self.assertEqual(self.request("POST", path, body={})[0], 200)
        self.assertEqual(self.request("POST", path, body={})[0], 200)
        code, payload, _ = self.request("GET", f"/session/inputs/{upload_id}")
        self.assertEqual(json.loads(payload)["state"], "finalized")
        self.assertEqual(len(self.state.status()["ephemeral_inputs"]), 1)

    def test_status_and_cancellation_require_authentication(self):
        upload_id = upload_request()["upload_id"]
        self.request("POST", "/session/inputs", body=upload_request())
        path = f"/session/inputs/{upload_id}"
        for method in ("GET", "DELETE"):
            self.assertEqual(self.request(method, path, token=None)[0], 401)
        self.assertEqual(self.request("DELETE", path)[0], 200)
        self.assertEqual(self.request("DELETE", path)[0], 200)
        _, payload, _ = self.request("GET", path)
        self.assertEqual(json.loads(payload)["state"], "cancelled")
        self.assertEqual(self.put(upload_id, DATA, 0)[0], 410)
