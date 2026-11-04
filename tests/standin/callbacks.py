"""A stand-in for NEMAR's compute routes and for object storage, as the launcher uses them.

Routes (the shapes of ``shared/contract/compute-launch.schema.json``, ``$defs`` of the bundle
response, the outputs request and response):

``POST /compute/jobs/<id>/bundle``
    header ``X-Compute-Job-Token`` required; answers ``{url, expires_at}``, a presigned-style GET
    of the job's archive, with an extra key a reader must ignore.
``GET  /s3/bundle/<id>.tar.gz``
    the archive bytes the test supplied, with ``Range`` support.
``POST /compute/jobs/<id>/outputs``
    header required; the body is the outputs request; answers ``{uploads, expires_at}`` with one
    presigned-style PUT address per requested file, each carrying a signature in its query string.
``PUT  /s3/jobs/<id>/<outputs|launcher>/<path>``
    stores the body (kept in ``objects``), checks the declared length against the bytes received.
``POST /compute/jobs/<id>/outputs/complete``
    header required; records the call.

No response here was recorded from NEMAR's Worker (it does not exist yet); the shapes come from the
contract file. Failure modes are configured on the attributes below. ``events`` is the ordered log
of the meaningful steps, so a test can assert that ``complete`` came last.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote

from server import Request, Response, StandinServer, parse_range

TOKEN_HEADER = "x-compute-job-token"
# Distinctive strings so a test can grep every log and provenance file for a leak.
PRESIGN_SIGNATURE = "SIGNATURESECRET0123456789abcdef"
PRESIGN_QUERY = "X-Amz-Signature=" + PRESIGN_SIGNATURE + "&X-Amz-Expires=3600"


def _expires() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Callbacks:
    def __init__(self, job_id: str, token: str) -> None:
        self.job_id = job_id
        self.token = token
        self.server: StandinServer | None = None
        self.bundle_bytes: bytes = b""
        # failure modes -------------------------------------------------------------------------
        self.bundle_route_statuses: list[int] = []  # answered in order before success
        self.bundle_refusal: tuple[int, str] | None = None
        self.bundle_redirect: str | None = None  # answer 302 to this path on the same server
        self.bundle_response_override: dict | None = None
        self.echo_secrets_in_errors = False  # a hostile route that puts the token in its errors
        self.put_redirect_to: str | None = None  # answer every PUT with a 307 to this path
        self.bundle_get_statuses: list[int] = []
        self.bundle_get_truncate_once = False
        self.outputs_refusal: tuple[int, str] | None = None  # refuses a non-empty outputs list
        self.outputs_route_statuses: list[int] = []
        self.put_statuses: dict[str, list[int]] = {}  # key -> statuses answered in order
        self.put_always: dict[str, int] = {}  # key prefix -> status, every time
        self.complete_status: int = 200
        self.complete_refusal_code: str | None = None
        self.omit_from_plan: set[str] = set()
        self.expire_urls_once = False  # first PUT of the run answers 403
        # records -------------------------------------------------------------------------------
        self.objects: dict[str, bytes] = {}
        self.events: list[str] = []
        self.outputs_requests: list[dict] = []
        self._lock = threading.Lock()
        self._expired = False

    def attach(self, server: StandinServer) -> None:
        self.server = server

    def _token_ok(self, request: Request) -> bool:
        return request.headers.get(TOKEN_HEADER) == self.token

    def _event(self, text: str) -> None:
        with self._lock:
            self.events.append(text)

    def handle(self, request: Request) -> Response | None:
        assert self.server is not None
        base = "/compute/jobs/{}/".format(self.job_id)
        if request.method == "POST" and request.path == base + "bundle":
            return self._bundle(request)
        if request.method == "POST" and request.path == base + "outputs":
            return self._outputs(request)
        if request.method == "POST" and request.path == base + "outputs/complete":
            return self._complete(request)
        if request.method == "GET" and request.path == "/s3/bundle/{}.tar.gz".format(self.job_id):
            return self._bundle_get(request)
        if request.method == "PUT" and request.path.startswith("/s3/jobs/{}/".format(self.job_id)):
            return self._put(request)
        return None

    # ---- routes -----------------------------------------------------------------------------

    def _refuse(self, status: int, code: str) -> Response:
        payload = {"error": "refused: " + code, "code": code}
        if self.echo_secrets_in_errors:
            payload["echo"] = "token={} url=https://example.test/x?{}".format(
                self.token, PRESIGN_QUERY
            )
        return Response.json_body(status, payload)

    def _bundle(self, request: Request) -> Response:
        self._event("bundle")
        if not self._token_ok(request):
            return self._refuse(404, "job_not_found")
        with self._lock:
            if self.bundle_route_statuses:
                return Response.json_body(
                    self.bundle_route_statuses.pop(0), {"error": "try again later"}
                )
        if self.bundle_refusal is not None:
            return self._refuse(*self.bundle_refusal)
        if self.bundle_redirect is not None:
            return Response(status=302, headers={"Location": self.bundle_redirect})
        if self.bundle_response_override is not None:
            return Response.json_body(200, self.bundle_response_override)
        assert self.server is not None
        url = "{}/s3/bundle/{}.tar.gz?{}".format(self.server.base_url, self.job_id, PRESIGN_QUERY)
        return Response.json_body(
            200, {"url": url, "expires_at": _expires(), "a_key_added_later": {"x": 1}}
        )

    def _bundle_get(self, request: Request) -> Response:
        with self._lock:
            if self.bundle_get_statuses:
                return Response.json_body(self.bundle_get_statuses.pop(0), {"error": "injected"})
            truncate = self.bundle_get_truncate_once
            self.bundle_get_truncate_once = False
        data = self.bundle_bytes
        if truncate:
            return Response(status=200, body=data, truncate_after=len(data) // 2)
        mode, start, end = parse_range(request.headers.get("range"), len(data))
        if mode == "unsatisfiable":
            return Response(status=416, headers={"Content-Range": f"bytes */{len(data)}"})
        if mode == "partial":
            return Response(
                status=206,
                body=data[start:end],
                headers={"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"},
            )
        return Response(status=200, body=data)

    def _outputs(self, request: Request) -> Response:
        if not self._token_ok(request):
            return self._refuse(404, "job_not_found")
        body = request.json()
        with self._lock:
            self.outputs_requests.append(body)
            statuses = self.outputs_route_statuses
            if statuses:
                self.events.append("outputs")
                return Response.json_body(statuses.pop(0), {"error": "try again later"})
        self._event("outputs")
        if self.outputs_refusal is not None and body.get("outputs"):
            return self._refuse(*self.outputs_refusal)
        assert self.server is not None
        wanted = ["outputs/" + o["path"] for o in body["outputs"]] + [
            "launcher/" + f["name"] for f in body["launcher"]
        ]
        uploads = []
        for key in wanted:
            if key in self.omit_from_plan:
                continue
            uploads.append(
                {
                    "path": key,
                    "url": "{}/s3/jobs/{}/{}?{}".format(
                        self.server.base_url, self.job_id, quote(key), PRESIGN_QUERY
                    ),
                    "an_upload_key_added_later": True,
                }
            )
        return Response.json_body(
            200, {"uploads": uploads, "expires_at": _expires(), "a_key_added_later": [1]}
        )

    def _put(self, request: Request) -> Response:
        key = unquote(request.path[len("/s3/jobs/{}/".format(self.job_id)) :])
        declared = request.headers.get("content-length")
        if declared is None or int(declared) != len(request.body):
            return Response.json_body(400, {"error": "length mismatch"})
        if self.put_redirect_to is not None:
            self._event("put-307:" + key)
            return Response(status=307, headers={"Location": self.put_redirect_to})
        with self._lock:
            if self.expire_urls_once and not self._expired:
                self._expired = True
                self.events.append("put-403:" + key)
                return Response(status=403, body=b"<Error>expired</Error>")
            queue = self.put_statuses.get(key)
            if queue:
                status = queue.pop(0)
                self.events.append(f"put-{status}:{key}")
                return Response(status=status, body=b"<Error>injected</Error>")
            for prefix, status in self.put_always.items():
                if key.startswith(prefix):
                    self.events.append(f"put-{status}:{key}")
                    return Response(status=status, body=b"<Error>injected</Error>")
            self.objects[key] = request.body
            self.events.append("put:" + key)
        return Response(status=200, headers={"ETag": '"stand-in"'})

    def _complete(self, request: Request) -> Response:
        if not self._token_ok(request):
            return self._refuse(404, "job_not_found")
        self._event("complete")
        if self.complete_status != 200:
            code = self.complete_refusal_code or "bad_state"
            return self._refuse(self.complete_status, code)
        return Response.json_body(200, {"ok": True})

    def serve(self) -> StandinServer:
        return StandinServer(self).start()
