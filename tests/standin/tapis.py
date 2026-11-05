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
        return Response.json_body(200, {"status": "success", "message": "found", "result": stored})

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
        return Response.json_body(status, {"status": "error", "message": text})
