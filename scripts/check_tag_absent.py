"""Exit 0 only when a registry tag is known to be ABSENT; stop the run in every other case.

The image workflow publishes a version once: ``<version>-cpu`` and the SIF's ``sha256-<hex>`` tag
are never moved to something new. Before it pushes, it asks the registry whether the tag exists
with one request, a manifest ``HEAD``, and reads only the HTTP status:

    404      the tag is absent: exit 0, the run goes on
    200      the tag exists: exit 1, "already exists"
    anything else (another 2xx, a redirect, 401, 403, 429, 5xx), a token that cannot be had, a
    network or TLS error, an empty reply, a missing credential: exit 1, "could not tell"

Nothing in a reply's body or text is read, so no reply that merely contains "404" or "unknown" or
"not found" is taken for an absence. Redirects are not followed. Authentication is the registry's
own token flow: HTTP basic authentication with the actor and the workflow's token at ``/token``
returns a bearer token for the ``pull`` scope of the repository.

The environment gives ``REGISTRY_USER`` and ``REGISTRY_TOKEN``; ``REGISTRY_URL`` (default
``https://ghcr.io``) exists for the tests, which point it at a stand-in on the loopback address
(plain http is refused for any other host). The standard library only.

    python scripts/check_tag_absent.py nemarorg/nemar-python 1.2.3-cpu
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REGISTRY = "https://ghcr.io"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
TIMEOUT_SECONDS = 30
MANIFEST_TYPES = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


class Undecided(Exception):
    """The registry's answer does not say whether the tag exists."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is an answer, not an instruction: following it could end at a 404 elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def registry_base() -> str:
    base = os.environ.get("REGISTRY_URL", DEFAULT_REGISTRY).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in LOOPBACK):
        return base
    raise Undecided("the registry address must be https (plain http only on the loopback address)")


def status_of(method: str, url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method=method, headers=headers)
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def bearer_token(base: str, repository: str, user: str, token: str) -> str:
    host = urllib.parse.urlparse(base).hostname or ""
    query = urllib.parse.urlencode(
        {"service": host, "scope": "repository:{}:pull".format(repository)}
    )
    credentials = base64.b64encode("{}:{}".format(user, token).encode()).decode()
    status, body = status_of(
        "GET", "{}/token?{}".format(base, query), {"Authorization": "Basic " + credentials}
    )
    if status != 200:
        raise Undecided("the token endpoint answered HTTP {}".format(status))
    try:
        value = json.loads(body)["token"]
    except (ValueError, KeyError, TypeError):
        raise Undecided("the token endpoint's reply holds no token") from None
    if not isinstance(value, str) or not value:
        raise Undecided("the token endpoint's reply holds no token")
    return value


def manifest_status(repository: str, tag: str) -> int:
    user = os.environ.get("REGISTRY_USER", "")
    token = os.environ.get("REGISTRY_TOKEN", "")
    if not user or not token:
        raise Undecided("REGISTRY_USER and REGISTRY_TOKEN must both be set")
    base = registry_base()
    bearer = bearer_token(base, repository, user, token)
    url = "{}/v2/{}/manifests/{}".format(base, repository, urllib.parse.quote(tag, safe=""))
    status, _body = status_of(
        "HEAD", url, {"Authorization": "Bearer " + bearer, "Accept": MANIFEST_TYPES}
    )
    return status


def stop(message: str) -> int:
    print("::error::" + message)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repository", help="for example nemarorg/nemar-python")
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    ref = "{}:{}".format(args.repository, args.tag)
    undecided = "could not tell whether {} exists, so nothing is pushed: ".format(ref)
    try:
        status = manifest_status(args.repository, args.tag)
    except (Undecided, OSError, ValueError) as error:
        # URLError, TLS failures, timeouts and a hung-up connection are all OSError
        return stop(undecided + str(error))
    if status == 404:
        print("{} is absent (HTTP 404)".format(ref))
        return 0
    if status == 200:
        return stop("{} already exists: it is never overwritten, publish a new version".format(ref))
    return stop(undecided + "HTTP {}".format(status))


if __name__ == "__main__":
    sys.exit(main())
