"""A stand-in for the registration routes of a Tapis tenant's Applications and Systems services.

It replaces a system that lives outside this repository. NO RESPONSE HERE WAS RECORDED FROM A REAL
TENANT: the routes, the status codes (200 and 404 for a read, 201 and 409 for a create, 401 for a
refused token) and the ``{status, message, result}`` envelope follow Tapis's published OpenAPI
specifications (``deploy/tapis/openapi/``), and the token travels in ``X-Tapis-Token`` as the
specifications' ``TapisJWT`` security scheme says.

It never replaces ``scripts/register.py``'s own logic: the script decides what to send and what to
do with each answer. The knobs below make the stand-in misbehave the ways a tenant can, so a test
can see what the script does then.
"""

from __future__ import annotations

import re
from typing import Any

from server import Request, Response

APP_ROUTE = re.compile(r"^/v3/apps/([^/]+)/([^/]+)$")
SYSTEM_ROUTE = re.compile(r"^/v3/systems/([^/]+)$")


class TapisStandin:
    def __init__(
        self,
        token: str = "stand-in-token",
        *,
        get_status: int | None = None,
        post_status: int | None = None,
        redirect_reads_to: str | None = None,
        echo_token_in_errors: bool = False,
        echo_slice: tuple[int, int] | None = None,
    ) -> None:
        self.token = token
        self.apps: dict[tuple[str, str], dict[str, Any]] = {}
        self.systems: dict[str, dict[str, Any]] = {}
        # Force the answer to every read or every create, whatever is stored.
        self.get_status = get_status
        self.post_status = post_status
        # Answer a read with a redirect to this path.
        self.redirect_reads_to = redirect_reads_to
        # Put the request's token into an error body, as a careless service could.
        self.echo_token_in_errors = echo_token_in_errors
        # Put only this slice of the request's token into an error body.
        self.echo_slice = echo_slice
        # Answer a read of anything stored with exactly these bytes (an answer that cannot be compared).
        self.raw_found: bytes | None = None

    # ---- the routes --------------------------------------------------------------------------

    def handle(self, request: Request) -> Response | None:
        app = APP_ROUTE.match(request.path)
        system = SYSTEM_ROUTE.match(request.path)
        creating = request.method == "POST" and request.path in ("/v3/apps", "/v3/systems")
        if not (app or system or creating):
            return None
        presented = request.headers.get("x-tapis-token")
        if presented != self.token:
            return self._error(401, "token refused", presented)
        if request.method == "GET":
            return self._read(request, app, system, presented)
        if request.method == "POST":
            return self._create(request, presented)
        return Response.json_body(405, {"status": "error", "message": "method not allowed"})

    def _read(self, request, app, system, presented) -> Response:
        if self.redirect_reads_to is not None:
            return Response(status=302, headers={"Location": self.redirect_reads_to})
        if self.get_status is not None:
            return self._error(self.get_status, "forced", presented)
        stored = None
        if app is not None:
            stored = self.apps.get((app.group(1), app.group(2)))
        elif system is not None:
            stored = self.systems.get(system.group(1))
        if stored is None:
            return self._error(404, "not found", presented)
        if self.raw_found is not None:
            return Response(status=200, body=self.raw_found)
        # A real tenant adds fields to what was posted (its tenant, identifiers, dates, defaults):
        # the script must ignore them and notice a change to anything it sent.
        added = {
            "tenant": "stand-in-tenant",
            "uuid": "00000000-0000-0000-0000-000000000001",
            "created": "2026-11-04T00:00:00Z",
            "updated": "2026-11-04T00:00:00Z",
            "deleted": False,
        }
        return Response.json_body(
            200, {"status": "success", "message": "found", "result": {**added, **stored}}
        )

    def _create(self, request: Request, presented: str | None) -> Response:
        if self.post_status is not None:
            return self._error(self.post_status, "forced", presented)
        try:
            body = request.json()
        except ValueError:
            return self._error(400, "invalid JSON", presented)
        if request.path == "/v3/apps":
            app_key = (str(body.get("id")), str(body.get("version")))
            if app_key in self.apps:
                return self._error(409, "already exists", presented)
            self.apps[app_key] = body
            url = "/v3/apps/{}/{}".format(*app_key)
        else:
            system_key = str(body.get("id"))
            if system_key in self.systems:
                return self._error(409, "already exists", presented)
            self.systems[system_key] = body
            url = "/v3/systems/{}".format(system_key)
        return Response.json_body(201, {"status": "success", "result": {"url": url}})

    def _error(self, status: int, message: str, presented: str | None) -> Response:
        text = message
        if self.echo_token_in_errors and presented:
            text = "{} (token {})".format(message, presented)
        if self.echo_slice is not None and presented:
            start, end = self.echo_slice
            text = "{} {} {}".format("x" * 120, presented[start:end], "y" * 120)
        return Response.json_body(status, {"status": "error", "message": text})


class ZipStandin:
    """A stand-in for where an application's zip is published (a release asset URL).

    ``/zips/<name>`` serves the bytes it was given, ``/via/<name>`` redirects there (a release asset
    is served from object storage), ``/loop`` redirects to itself and ``/elsewhere`` to a plain http
    address that is not loopback. It never sees a Tapis token, and a test checks that.
    """

    def __init__(self, files: dict[str, bytes], *, status: int | None = None) -> None:
        self.files = files
        self.status = status

    def handle(self, request: Request) -> Response | None:
        if request.method != "GET":
            return None
        path = request.path
        if path.startswith("/zips/"):
            if self.status is not None:
                return Response.json_body(self.status, {"error": "forced"})
            body = self.files.get(path[len("/zips/") :])
            if body is None:
                return Response.json_body(404, {"error": "no such file"})
            return Response(status=200, body=body, headers={"Content-Type": "application/zip"})
        if path.startswith("/via/"):
            return Response(status=302, headers={"Location": "/zips/" + path[len("/via/") :]})
        if path == "/loop":
            return Response(status=302, headers={"Location": "/loop"})
        if path == "/elsewhere":
            return Response(status=302, headers={"Location": "http://zips.example.org/x.zip"})
        return None
