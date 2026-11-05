"""Render NEMAR's Tapis definitions and register them. A dry run unless told otherwise.

    python scripts/register.py [--vars FILE] [--apply] app    deploy/tapis/apps/nemar-launch.json ...
    python scripts/register.py [--vars FILE] [--apply] system deploy/tapis/systems/exec.json.tmpl

THE RENDER. A definition is JSON whose string values may hold ``${NAME}`` placeholders
(``string.Template`` syntax, ``NAME`` in capitals). Operator values come from the environment or from
``--vars FILE`` (``NAME=value`` lines, raw, no quoting; the file wins over the environment). An
unresolved placeholder is an error, and so is a value with a newline or any other control character.
Tapis's own macros share the syntax, so a definition writes them ``$${JobUUID}`` and they come out as
``${JobUUID}``. The render works on the parsed document, so a value can fill a placeholder and can
never change the document's structure. A placeholder named like a credential is refused: a
definition never carries one.

THE CHECK. The rendered document must validate against the schema derived from Tapis's own OpenAPI
file (``deploy/tapis/schemas/``) before anything is sent, so a typo is an error here and not a 400,
and it must keep NEMAR's rules (``scripts/tapis_policy.py``, the same function the tests run on the
committed files: launch files not archived, no archive system, the fixed paths, ``locked``, one static
account, the directories as the launcher reads them). A definition that does not is exit status 2.

THE DRY RUN (the default). Prints the method, path and body hash of each request it would make and
sends nothing, not even a read. It needs no credentials.

``--apply``. First, for each application, the file at ``containerImage`` is fetched and its sha256
must equal the digest in the application's ``notes``: Tapis fetches and unzips that file with no digest
check of its own, so this is where the record is compared with the file (exit status 1, nothing
registered, if they differ). The repository's example values are refused (exit status 2). Then, for
each definition: ``GET`` it; if it exists, compare it with the rendered one (Tapis's added fields are
ignored, but what it adds that the rules forbid is not) and leave it exactly as it is, but exit status
1 naming the PATHS that differ, never their values (a registered version is never changed: publish a
new version); if it is absent, ``POST`` it.
A 409 on the ``POST`` means it appeared after the read: that is an error, because this run did not
look at what is there.
No other verb is ever sent, and nothing about permissions, sharing or credentials: those are the
operator's, by other means (docs/operator.md).

CREDENTIALS. The Tapis token is ``TAPIS_TOKEN`` and the tenant's base address is ``TAPIS_BASE_URL``,
both from the environment, never from an argument. The token travels in ``X-Tapis-Token`` only, is in
no output (a service that echoes it in an error body is redacted), and no redirect is followed.
The base address is not printed either: it names a tenant.

Exit status: 0 done (or dry run); 1 the tenant refused or failed, a registered definition differs, or
the zip is not the recorded one; 2 the definition, the values or the environment are wrong, with
nothing sent.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import string
import sys
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, TextIO
from urllib.parse import quote, urljoin, urlsplit

import tapis_policy
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "deploy" / "tapis" / "schemas"

NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
# A placeholder with one of these words in its name would put a secret into a definition.
CREDENTIAL_WORDS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE", "KEY")
MAX_VALUE_CHARS = 1024
LOOPBACK = ("127.0.0.1", "localhost", "::1")
TIMEOUT_SECONDS = 30
MAX_BODY_BYTES = 1024 * 1024
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 5
REDACT_SLICE = 16
MAX_DRIFT_PATHS = 25
LEDGER = ROOT / "deploy" / "tapis" / "released-versions.json"
COMMITTED_SYSTEM = ROOT / "deploy" / "tapis" / "systems" / "exec.json.tmpl"
COMMITTED_APPS = ROOT / "deploy" / "tapis" / "apps"
# The stand-in values of deploy/tapis/example.env: fine for a dry run and CI, never for a tenant.
RESERVED_VALUE = re.compile(r"(?i)example[.-]|/path/to/")


class RenderError(ValueError):
    """A definition cannot be rendered with the values it was given."""


class RequestFailed(RuntimeError):
    """The tenant refused, failed or answered something the script will not act on."""


# ---- shapes of the operator values ---------------------------------------------------------------


class Shape(NamedTuple):
    check: Callable[[str], bool]
    hint: str


def _host(value: str) -> bool:
    return (
        len(value) <= 253
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value) is not None
    )


def _login(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}", value) is not None


def _absolute_dir(value: str) -> bool:
    if not value.startswith("/") or len(value) > MAX_VALUE_CHARS:
        return False
    parts = value[1:].split("/")
    return all(
        re.fullmatch(r"[A-Za-z0-9._+-]+", part) is not None and part not in (".", "..")
        for part in parts
    )


def _slurm_account(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", value) is not None


_PATH_HINT = "an absolute path of plain segments (letters, digits and . _ + -), not the root, no trailing slash"

# What each placeholder the committed definitions use must look like. A placeholder not listed here
# takes any value without control characters.
SHAPES: dict[str, Shape] = {
    "EXEC_HOST": Shape(_host, "a host name or address, with no scheme and no path"),
    "COMMUNITY_ACCOUNT": Shape(_login, "a login name of up to 32 characters"),
    "EXEC_ROOT_DIR": Shape(_absolute_dir, _PATH_HINT),
    "SIF_ROOT": Shape(_absolute_dir, _PATH_HINT),
    "LOCAL_DATASETS_ROOT": Shape(_absolute_dir, _PATH_HINT),
    "NODE_LOCAL_TMPDIR": Shape(_absolute_dir, _PATH_HINT),
    "SLURM_ACCOUNT_CPU": Shape(_slurm_account, "one scheduler account name of up to 32 characters"),
    "SLURM_ACCOUNT_GPU": Shape(_slurm_account, "one scheduler account name of up to 32 characters"),
    "AUTHN_METHOD": Shape(
        lambda value: value in ("PKI_KEYS", "TMS_KEYS", "CERT", "PASSWORD"),
        "PKI_KEYS, TMS_KEYS, CERT or PASSWORD",
    ),
}


# ---- the values ------------------------------------------------------------------------------------


def load_vars(text: str) -> dict[str, str]:
    """``NAME=value`` lines: blank lines and ``#`` lines skipped, the value raw (no quoting)."""
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, equals, value = line.partition("=")
        name = name.strip()
        if not equals or not NAME_RE.fullmatch(name):
            raise RenderError("line {}: expected NAME=value with NAME in capitals".format(number))
        if name in values:
            raise RenderError("line {}: {} is set twice".format(number, name))
        values[name] = value.strip()
    return values


def _check_name(name: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise RenderError(
            "{} is not a placeholder name (capitals, digits and _); a Tapis macro is written "
            "$${{{}}} in a definition".format(name, name)
        )
    if any(word in name for word in CREDENTIAL_WORDS):
        raise RenderError(
            "{} is named like a credential: a definition never carries one".format(name)
        )


def check_value(name: str, value: object) -> None:
    """Refuse a value a definition must not hold. The message names the placeholder, never the value."""
    if not isinstance(value, str):
        raise RenderError("the value of {} is not text".format(name))
    if len(value) > MAX_VALUE_CHARS:
        raise RenderError(
            "the value of {} is longer than {} characters".format(name, MAX_VALUE_CHARS)
        )
    if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in value):
        raise RenderError("the value of {} has a newline or another control character".format(name))
    shape = SHAPES.get(name)
    if shape is not None and not shape.check(value):
        raise RenderError("the value of {} must be {}".format(name, shape.hint))


def placeholders(document: object) -> set[str]:
    """The placeholder names in the string values of a document (``$$`` is an escape, not a name)."""
    names: set[str] = set()
    for text in _strings(document):
        template = string.Template(text)
        if not template.is_valid():
            raise RenderError("a value has a $ that is not a placeholder or an escaped $$")
        names |= set(template.get_identifiers())
    return names


def _strings(node: object):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, item in node.items():
            if "$" in str(key):
                raise RenderError("placeholders are for values, and a key has a $ in it")
            yield from _strings(item)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def resolve(
    names: set[str], env: Mapping[str, str], file_values: Mapping[str, str]
) -> dict[str, str]:
    """The value of each name: the vars file first, then the environment. Nothing else is read."""
    for name in sorted(names):
        _check_name(name)
    missing = sorted(n for n in names if n not in file_values and n not in env)
    if missing:
        raise RenderError("unresolved placeholders: {}".format(", ".join(missing)))
    return {name: file_values[name] if name in file_values else env[name] for name in names}


def _substitute(node: Any, values: Mapping[str, str]) -> Any:
    if isinstance(node, str):
        return string.Template(node).substitute(values)
    if isinstance(node, dict):
        return {key: _substitute(item, values) for key, item in node.items()}
    if isinstance(node, list):
        return [_substitute(item, values) for item in node]
    return node


def render_document(document: Any, values: Mapping[str, str]) -> Any:
    names = placeholders(document)
    for name in sorted(names):
        _check_name(name)
    unresolved = sorted(n for n in names if n not in values)
    if unresolved:
        raise RenderError("unresolved placeholders: {}".format(", ".join(unresolved)))
    for name in sorted(names):
        check_value(name, values[name])
    return _substitute(document, values)


def render_text(text: str, values: Mapping[str, str]) -> dict[str, Any]:
    """Parse a definition and fill its placeholders from ``values``."""
    try:
        document = json.loads(text)
    except ValueError:
        raise RenderError("the definition is not JSON") from None
    if not isinstance(document, dict):
        raise RenderError("the definition is not a JSON object")
    return render_document(document, values)


def canonical(document: object) -> bytes:
    """The exact bytes that are sent, and hashed in the dry run."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def redact(text: str, token: str) -> str:
    """``text`` with the token, and any 16-character slice of it, replaced by ``[token]``.

    A real token is a long JWT: a service that echoes part of one must not get it past a filter that
    only knows the whole. The whole text is redacted, before anything is cut or filtered.
    """
    if not token:
        return text
    text = text.replace(token, "[token]")
    if len(token) < REDACT_SLICE:
        return text
    slices = {token[i : i + REDACT_SLICE] for i in range(len(token) - REDACT_SLICE + 1)}
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i : i + REDACT_SLICE] in slices:
            j = i + REDACT_SLICE
            while j < n and text[j - REDACT_SLICE + 1 : j + 1] in slices:
                j += 1
            out.append("[token]")
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _same_scalar(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """The paths at which ``actual`` is not what ``expected`` says; never a value.

    ONE-DIRECTIONAL: ``actual`` may hold more than ``expected``, because Tapis adds fields (its tenant,
    identifiers, dates and the defaults of what was left out). What it adds that the rules forbid is
    caught separately by ``tapis_policy.stored_violations``. A list must be as long as expected, element
    by element, so a list Tapis reorders is reported as a difference (fail-closed noise).
    """
    here = path or "/"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [here]
        found: list[str] = []
        for key, value in expected.items():
            sub = "{}/{}".format(path, key)
            if key not in actual:
                found.append(sub)
            else:
                found += differences(value, actual[key], sub)
        return found
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [here]
        found = []
        for index, (want, have) in enumerate(zip(expected, actual, strict=True)):
            found += differences(want, have, "{}/{}".format(path, index))
        return found
    return [] if _same_scalar(expected, actual) else [here]


def reserved_values(values: Mapping[str, str]) -> list[str]:
    """The placeholders whose values are the repository's stand-ins (``example.org``, ``/path/to/``, ``example-*``)."""
    return sorted(name for name, value in values.items() if RESERVED_VALUE.search(value))


def load_ledger(path: Path) -> dict[str, Any]:
    try:
        released = json.loads(path.read_text(encoding="utf-8")).get("released")
    except (OSError, ValueError, AttributeError):
        raise RenderError("the released-versions ledger cannot be read") from None
    if not isinstance(released, dict):
        raise RenderError('the released-versions ledger has no "released" map')
    return released


# ---- what each kind of definition is registered with ---------------------------------------------


class Kind(NamedTuple):
    schema: str
    create_path: str
    read_path: Callable[[dict[str, Any]], str]
    label: Callable[[dict[str, Any]], str]


def _segment(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise RenderError("the definition has no {}".format(key))
    return value


KINDS: dict[str, Kind] = {
    "app": Kind(
        schema="app",
        create_path="/v3/apps",
        read_path=lambda d: "/v3/apps/{}/{}".format(
            quote(_segment(d, "id"), safe=""), quote(_segment(d, "version"), safe="")
        ),
        label=lambda d: "app {} {}".format(_segment(d, "id"), _segment(d, "version")),
    ),
    "system": Kind(
        schema="system",
        create_path="/v3/systems",
        read_path=lambda d: "/v3/systems/{}".format(quote(_segment(d, "id"), safe="")),
        label=lambda d: "system {}".format(_segment(d, "id")),
    ),
}


def schema_problems(kind: str, document: Any) -> list[str]:
    schema = json.loads((SCHEMAS / "{}.schema.json".format(kind)).read_text(encoding="utf-8"))
    found = sorted(
        Draft202012Validator(schema).iter_errors(document), key=lambda e: list(map(str, e.path))
    )
    return ["{}: {}".format("/".join(map(str, e.path)) or "<root>", e.message) for e in found]


class Plan(NamedTuple):
    kind: str
    label: str
    read_path: str
    create_path: str
    body: bytes
    digest: str
    document: dict[str, Any]
    values: dict[str, str]


def _committed(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RenderError("the committed definition {} cannot be read".format(path.name)) from None
    if not isinstance(document, dict):
        raise RenderError("the committed definition {} is not an object".format(path.name))
    return document


def committed_system(exec_root: str) -> dict[str, Any]:
    """The root and working directory the committed system template makes, with the operator's scratch root."""
    template = _committed(COMMITTED_SYSTEM)
    fill = lambda value: string.Template(str(value)).safe_substitute(EXEC_ROOT_DIR=exec_root)  # noqa: E731
    return {
        "rootDir": fill(template.get("rootDir", "")),
        "jobWorkingDir": fill(template.get("jobWorkingDir", "")),
    }


def committed_applications() -> list[dict[str, Any]]:
    """The directory templates of the committed applications, with Tapis's macros unescaped."""
    found = []
    for path in sorted(COMMITTED_APPS.glob("*.json")):
        attributes = _committed(path).get("jobAttributes") or {}
        found.append(
            {
                "jobAttributes": {
                    name: str(attributes.get(name, "")).replace("$$", "$")
                    for name in tapis_policy.DIRECTORY_FIELDS
                }
            }
        )
    return found


def plan(
    kind: str,
    path: Path,
    file_values: Mapping[str, str],
    env: Mapping[str, str],
    released: Mapping[str, Any] | None = None,
) -> Plan:
    """Render one file and check it; raises ``RenderError`` for every way it is not fit to send."""
    spec = KINDS[kind]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise RenderError("{} cannot be read".format(path)) from None
    try:
        document = json.loads(text)
    except ValueError:
        raise RenderError("{} is not JSON".format(path)) from None
    if not isinstance(document, dict):
        raise RenderError("{} is not a JSON object".format(path))
    values = resolve(placeholders(document), env, file_values)
    rendered = render_document(document, values)
    problems = schema_problems(spec.schema, rendered)
    if problems:
        raise RenderError("{} does not match Tapis's schema: {}".format(path, "; ".join(problems)))
    # NEMAR's own rules, the same function the tests run on the committed files: a stale or locally
    # edited definition cannot register something weaker than CI checked.
    # The operator's scratch root, even when an edited template no longer names it. Without it the
    # job directories cannot be checked, and a definition that cannot be checked is refused.
    root = file_values.get("EXEC_ROOT_DIR") or env.get("EXEC_ROOT_DIR")
    if not root:
        raise RenderError(
            "EXEC_ROOT_DIR is required to check the job directories (set it, even when this "
            "definition does not name it)"
        )
    if kind == "app":
        weaker = tapis_policy.app_violations(rendered)
        weaker += tapis_policy.ledger_violations([rendered], released or {})
        weaker += tapis_policy.path_violations(committed_system(root), rendered, root)
    else:
        weaker = tapis_policy.system_violations(rendered, exec_root=root)
        for app in committed_applications():
            weaker += tapis_policy.path_violations(rendered, app, root)
    if weaker:
        raise RenderError("{} is weaker than NEMAR's rules: {}".format(path, "; ".join(weaker)))
    body = canonical(rendered)
    return Plan(
        kind=kind,
        label=spec.label(rendered),
        read_path=spec.read_path(rendered),
        create_path=spec.create_path,
        body=body,
        digest=hashlib.sha256(body).hexdigest(),
        document=rendered,
        values=values,
    )


# ---- talking to the tenant ---------------------------------------------------------------------------


def parse_base(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port) of the tenant's base address, or ``RenderError``."""
    problem = "TAPIS_BASE_URL must be an https address with no credentials, path, query or fragment"
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise RenderError(problem) from None
    plain_ok = parts.scheme == "http" and host in LOOPBACK
    if (
        parts.scheme not in ("https", "http")
        or not host
        or (parts.scheme == "http" and not plain_ok)
    ):
        raise RenderError(problem)
    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path not in ("", "/")
    ):
        raise RenderError(problem)
    return parts.scheme, host, port


class Tenant:
    def __init__(self, base_url: str, token: str) -> None:
        self.scheme, self.host, self.port = parse_base(base_url)
        self.token = token

    def send(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        make = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        connection = make(self.host, self.port, timeout=TIMEOUT_SECONDS)
        headers = {"X-Tapis-Token": self.token, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read(MAX_BODY_BYTES)
        except (OSError, http.client.HTTPException) as problem:
            # The class of the problem says enough; its text could carry an address.
            raise RequestFailed(
                "{} {} failed: {}".format(method, path, type(problem).__name__)
            ) from None
        finally:
            connection.close()
        if 300 <= response.status < 400:
            raise RequestFailed(
                "{} {} answered a redirect, which is not followed".format(method, path)
            )
        return response.status, data

    def describe(self, method: str, path: str, status: int, data: bytes) -> str:
        """``METHOD path answered N: message``, with the token redacted and the message bounded."""
        message = ""
        try:
            parsed = json.loads(data.decode("utf-8", "replace"))
            if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
                message = parsed["message"]
        except ValueError:
            pass
        # Redact the WHOLE message first: truncating first would let a long token slip through.
        message = "".join(c for c in redact(message, self.token) if c.isprintable())[:200]
        return "{} {} answered {}{}".format(method, path, status, ": " + message if message else "")


class Drift(Exception):
    """A definition already registered does not match the one rendered now."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__("drift")
        self.paths = paths


def _stored(data: bytes) -> Any:
    try:
        parsed = json.loads(data.decode("utf-8", "replace"))
    except ValueError:
        return None
    return parsed.get("result") if isinstance(parsed, dict) else None


def apply(tenant: Tenant, item: Plan, out: TextIO) -> None:
    status, data = tenant.send("GET", item.read_path)
    if status == 200:
        stored = _stored(data)
        if isinstance(stored, dict):
            # What the rendered definition sets must be as stored, and what it leaves out must not
            # have been filled with something the rules forbid (Tapis's own additions are ignored).
            paths = sorted(
                set(differences(item.document, stored))
                | set(tapis_policy.stored_violations(stored))
            )
        else:
            paths = ["/"]
        if paths:
            raise Drift(paths)
        print(
            "{}: exists and matches this definition; left unchanged (a registered version is "
            "never modified: publish a new version to change it)".format(item.label),
            file=out,
        )
        return
    if status != 404:
        raise RequestFailed(tenant.describe("GET", item.read_path, status, data))
    status, data = tenant.send("POST", item.create_path, item.body)
    if status == 201:
        print("{}: created".format(item.label), file=out)
        return
    if status == 409:
        raise RequestFailed(
            "{}: it appeared after the read, so this run did not create it and has not looked at "
            "it; nothing was changed, check it by hand ({})".format(
                item.label, tenant.describe("POST", item.create_path, status, data)
            )
        )
    raise RequestFailed(tenant.describe("POST", item.create_path, status, data))


def fetch_digest(url: str, max_bytes: int = MAX_ZIP_BYTES) -> str:
    """The SHA-256 of the file at ``url`` (an https address, or loopback http). No credential is sent.

    Redirects are followed (a release asset is served from object storage), at most five, never
    from https to http, and every hop must itself be https or loopback http. At most 64 MiB are read.
    """
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        try:
            parts = urlsplit(current)
            host, port = parts.hostname, parts.port
        except ValueError:
            raise RequestFailed("containerImage is not an address") from None
        plain_ok = parts.scheme == "http" and host in LOOPBACK
        if (
            (parts.scheme != "https" and not plain_ok)
            or not host
            or parts.username
            or parts.password
        ):
            raise RequestFailed("containerImage is not an https address (or loopback http)")
        make = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        connection = make(host, port, timeout=TIMEOUT_SECONDS)
        target = (parts.path or "/") + ("?" + parts.query if parts.query else "")
        try:
            connection.request(
                "GET", target, headers={"User-Agent": "nemar-register", "Accept": "*/*"}
            )
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                response.read(65536)  # the redirect's own body, so the connection closes cleanly
                following = urljoin(current, response.getheader("Location") or "")
                if parts.scheme == "https" and urlsplit(following).scheme != "https":
                    raise RequestFailed("containerImage redirects from https to http")
                current = following
                continue
            if response.status != 200:
                raise RequestFailed("containerImage answered HTTP {}".format(response.status))
            digest, total = hashlib.sha256(), 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    return digest.hexdigest()
                total += len(chunk)
                if total > max_bytes:
                    raise RequestFailed(
                        "containerImage is larger than {} bytes".format(MAX_ZIP_BYTES)
                    )
                digest.update(chunk)
        except (OSError, http.client.HTTPException) as problem:
            raise RequestFailed(
                "fetching containerImage failed: {}".format(type(problem).__name__)
            ) from None
        finally:
            connection.close()
    raise RequestFailed("containerImage redirects too many times")


def verify_zip(item: Plan) -> None:
    """The zip Tapis will fetch and unzip (with no digest check of its own) is the one the definition records."""
    name, expected = tapis_policy.digest_note(item.document)
    actual = fetch_digest(str(item.document.get("containerImage")))
    if actual != expected:
        raise RequestFailed(
            "{}: the file at containerImage has sha256 {}, but notes.{} records {}; nothing was "
            "registered".format(item.label, actual, name, expected)
        )


# ---- the command line --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="register.py", description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--vars", metavar="FILE", help="NAME=value lines for the placeholders")
    parser.add_argument(
        "--apply", action="store_true", help="register what is absent (default: a dry run)"
    )
    parser.add_argument(
        "--ledger", metavar="FILE", default=str(LEDGER), help="the released-versions ledger"
    )
    kinds = parser.add_subparsers(dest="kind", required=True)
    for kind in KINDS:
        sub = kinds.add_parser(kind, help="register {} definitions".format(kind))
        sub.add_argument("files", nargs="+", metavar="FILE")
    return parser


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    out = out or sys.stdout
    err = err or sys.stderr

    try:
        file_values = load_vars(Path(args.vars).read_text(encoding="utf-8")) if args.vars else {}
        released = load_ledger(Path(args.ledger))
        plans = [plan(args.kind, Path(name), file_values, env, released) for name in args.files]
        tenant = None
        if args.apply:
            tenant = _tenant_from(env)
            stand_ins = reserved_values({k: v for item in plans for k, v in item.values.items()})
            if stand_ins:
                raise RenderError(
                    "--apply refuses the repository's example values; fill in real ones for: {}".format(
                        ", ".join(stand_ins)
                    )
                )
    except (RenderError, OSError) as problem:
        print("register: {}".format(problem), file=err)
        return 2

    if tenant is None:
        print("dry run: nothing is sent; add --apply to register what is absent", file=out)
        for item in plans:
            print(item.label, file=out)
            print("  GET {}".format(item.read_path), file=out)
            print(
                "  POST {} sha256 {} ({} bytes), only if the read finds nothing".format(
                    item.create_path, item.digest, len(item.body)
                ),
                file=out,
            )
            if item.kind == "app":
                print(
                    "  --apply first fetches containerImage and checks its sha256 against the notes",
                    file=out,
                )
        return 0

    try:
        # Before anything reaches the tenant: Tapis fetches and unzips containerImage with no digest
        # check of its own, so this is the only place the recorded digest is compared with the file.
        for item in plans:
            if item.kind == "app":
                verify_zip(item)
    except RequestFailed as problem:
        print("register: {}".format(problem), file=err)
        return 1

    drifted = False
    for item in plans:
        try:
            apply(tenant, item, out)
        except Drift as drift:
            shown = drift.paths[:MAX_DRIFT_PATHS]
            more = len(drift.paths) - len(shown)
            print(
                "register: {}: exists but differs from this definition at {}{} ({} difference(s)); "
                "nothing was changed".format(
                    item.label,
                    ", ".join(shown),
                    " and {} more".format(more) if more else "",
                    len(drift.paths),
                ),
                file=err,
            )
            drifted = True
        except RequestFailed as problem:
            print("register: {}".format(problem), file=err)
            return 1
    return 1 if drifted else 0


def _tenant_from(env: Mapping[str, str]) -> Tenant:
    base = env.get("TAPIS_BASE_URL")
    token = env.get("TAPIS_TOKEN")
    missing = [n for n, v in (("TAPIS_BASE_URL", base), ("TAPIS_TOKEN", token)) if not v]
    if missing:
        raise RenderError("--apply needs {} in the environment".format(" and ".join(missing)))
    assert base is not None
    assert token is not None
    if not re.fullmatch(r"[\x21-\x7e]+", token):
        raise RenderError("TAPIS_TOKEN is not one printable token")
    return Tenant(base, token)


if __name__ == "__main__":
    sys.exit(main())
