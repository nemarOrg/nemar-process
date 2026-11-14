"""NEMAR's runner routes: exchange, claim, heartbeat, complete.

Standard library only. Every request carries `User-Agent: nemar-runner/<version>` and a JSON body
(`protocol` on the exchange and the claim, where the contract says so); the secret is the credential
of the exchange alone, and the access token (kept in memory, renewed 60 s before it expires) is the
credential of the rest. Redirects are never followed, so a credential cannot be sent elsewhere.

Errors carry a status and the Worker's one fixed word (`{"error": <word>}`), never a body or a URL.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import __version__, contract
from .config import Config, Tuning
from .logs import Log

# Words after which a runner can only stop: its secret is not (or no longer) accepted.
FATAL_WORDS = ("runner_unknown", "runner_credential_not_accepted")
# Words (a 426) after which a runner must stop claiming but lets the jobs it holds finish.
DRAIN_WORDS = ("runner_outdated", "protocol_unsupported")
# A job whose lease the Worker no longer honors: stop it, and do not complete it.
LEASE_WORDS = ("lease_lost", "job_not_claimed")

_CLAIM_MAX_BYTES = 8 * 1024 * 1024  # 32 jobs of at most 87,382 characters each, and some room
_SMALL_MAX_BYTES = 64 * 1024


class ApiError(Exception):
    """A request failed. ``status`` is the HTTP status (None for a transport fault), ``word`` the
    Worker's refusal word when it sent one, ``kind`` a fixed label for the log."""

    def __init__(self, kind: str, status: int | None = None, word: str | None = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.status = status
        self.word = word

    @property
    def transient(self) -> bool:
        return self.status is None or self.status >= 500 or self.status == 429

    @property
    def fatal(self) -> bool:
        return self.word in FATAL_WORDS

    @property
    def drain(self) -> bool:
        return self.word in DRAIN_WORDS

    @property
    def lease_lost(self) -> bool:
        return self.word in LEASE_WORDS or self.word == "runner_unknown"

    def fields(self) -> dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "word": self.word}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def parse_timestamp(text: str) -> float:
    """Seconds since the epoch of a UTC timestamp the contract's reader accepted."""
    return contract.timestamp_seconds(text)


class Api:
    def __init__(
        self, config: Config, secret: str, log: Log, tuning: Tuning, version: str = __version__
    ) -> None:
        self._base = config.api_base
        self._secret = secret
        self._log = log
        self._tuning = tuning
        self.version = version
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._lock = threading.Lock()
        self._exchange_lock = threading.Lock()  # one exchange at a time, however many jobs ask
        self._token: str | None = None
        self._expires = 0.0

    # ---- transport ------------------------------------------------------------------------------

    def _send(
        self,
        path: str,
        body: dict[str, Any] | None,
        bearer: str,
        limit: int,
        method: str = "POST",
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self._base + path, data=data, method=method)
        request.add_header("Authorization", "Bearer " + bearer)
        request.add_header("User-Agent", contract.USER_AGENT_PREFIX + self.version)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(request, timeout=self._tuning.http_timeout_seconds) as response:
                raw = response.read(limit + 1)
        except urllib.error.HTTPError as error:
            word = self._word_of(error)
            error.close()
            raise ApiError("http_{}".format(error.code), error.code, word) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            raise ApiError("transport_failed") from None
        except (OSError, ValueError, http.client.HTTPException) as error:
            raise ApiError("transport_" + type(error).__name__.lower()) from None
        if len(raw) > limit:
            raise ApiError("bad_response")
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ApiError("bad_response") from None

    @staticmethod
    def _word_of(error: urllib.error.HTTPError) -> str | None:
        try:
            parsed = json.loads(error.read(_SMALL_MAX_BYTES).decode("utf-8"))
        except (OSError, ValueError):
            return None
        word = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(word, str) and contract.WORD_RE.fullmatch(word):
            return word
        return None

    # ---- the token --------------------------------------------------------------------------------

    def exchange(self) -> str:
        """Buy an access token with the secret. The token is returned and kept in memory."""
        body = {"protocol": contract.PROTOCOL, "runner_version": self.version}
        answer = self._send("/compute/runners/token", body, self._secret, _SMALL_MAX_BYTES)
        try:
            token, expires_at = contract.read_token_response(answer)
        except contract.BadAnswer as error:
            raise ApiError("bad_response") from error
        now = time.time()
        expires = parse_timestamp(expires_at)
        if expires - now <= 2 * self._tuning.token_margin_seconds:
            # The Worker's clock and ours disagree by minutes: renew on a schedule of our own, and
            # let a 401 `runner_token_expired` be the correction.
            expires = now + 2 * self._tuning.token_margin_seconds
        with self._lock:
            self._token, self._expires = token, expires
        return token

    def _valid_token(self) -> str | None:
        with self._lock:
            token, expires = self._token, self._expires
        if token is not None and time.time() < expires - self._tuning.token_margin_seconds:
            return token
        return None

    def _access(self) -> str:
        token = self._valid_token()
        if token is not None:
            return token
        with self._exchange_lock:
            return self._valid_token() or self.exchange()  # another thread may have just done it

    def _renew_after(self, refused: str) -> str:
        """A new token after NEMAR refused ``refused``; one exchange, shared by every thread that saw
        the same refusal."""
        with self._exchange_lock:
            with self._lock:
                current = self._token
            if current is not None and current != refused and self._valid_token() is not None:
                return current
            return self.exchange()

    def _authed(self, path: str, body: dict[str, Any], limit: int) -> Any:
        token = self._access()
        try:
            return self._send(path, body, token, limit)
        except ApiError as error:
            if error.status != 401 or error.word != "runner_token_expired":
                raise
        self._log.info("token_renewed_after_expiry")
        return self._send(path, body, self._renew_after(token), limit)

    # ---- the routes ---------------------------------------------------------------------------------

    def reachable(self) -> bool:
        """Whether the host answers HTTP at all (any status), for `doctor`'s egress check."""
        request = urllib.request.Request(self._base + "/", method="GET")
        request.add_header("User-Agent", contract.USER_AGENT_PREFIX + self.version)
        try:
            with self._opener.open(request, timeout=self._tuning.http_timeout_seconds):
                return True
        except urllib.error.HTTPError as error:
            error.close()
            return True
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
            return False

    def me(self) -> str:
        """The runner's registration id (`rnr_` and 24 hex), from `GET /compute/runners/me`."""
        token = self._access()
        try:
            answer = self._send("/compute/runners/me", None, token, _SMALL_MAX_BYTES, method="GET")
        except ApiError as error:
            if error.status != 401 or error.word != "runner_token_expired":
                raise
            answer = self._send(
                "/compute/runners/me",
                None,
                self._renew_after(token),
                _SMALL_MAX_BYTES,
                method="GET",
            )
        try:
            return contract.read_me_response(answer)
        except contract.BadAnswer as error:
            raise ApiError("bad_response") from error

    def claim(
        self, free_slots: int, runtimes: list[str], arch: str
    ) -> tuple[list[contract.ClaimedJob], str]:
        body = {
            "protocol": contract.PROTOCOL,
            "free_slots": free_slots,
            "runtimes": runtimes,
            "arch": arch,
            "runner_version": self.version,
        }
        answer = self._authed("/compute/runners/claim", body, _CLAIM_MAX_BYTES)
        claimed_at = time.time()
        try:
            return contract.read_claim_response(answer, free_slots, claimed_at)
        except contract.BadAnswer as error:
            self._log.error("claim_answer_refused", where=error.where)
            raise ApiError("bad_response") from None

    def heartbeat(self, job_id: str, lease_id: str, phase: str) -> bool:
        """Renew a lease; True when the Worker says to stop the job."""
        body = {"job_id": job_id, "lease_id": lease_id, "phase": phase}
        answer = self._authed("/compute/runners/heartbeat", body, _SMALL_MAX_BYTES)
        try:
            return contract.read_heartbeat_response(answer)
        except contract.BadAnswer as error:
            raise ApiError("bad_response") from error

    def complete(self, body: dict[str, Any]) -> str:
        """Report a job's end; the state it is in. Retried on a transport fault and a 5xx, with a
        growing pause; a refusal is raised at once."""
        attempts = max(1, self._tuning.complete_attempts)
        for attempt in range(attempts):
            try:
                answer = self._authed("/compute/runners/complete", body, _SMALL_MAX_BYTES)
                try:
                    return contract.read_complete_response(answer)
                except contract.BadAnswer as error:
                    raise ApiError("bad_response") from error
            except ApiError as error:
                if not error.transient or attempt == attempts - 1:
                    raise
                self._log.warn(
                    "complete_retry", job=body.get("job_id"), attempt=attempt + 1, **error.fields()
                )
                pause = min(
                    self._tuning.retry_cap_seconds, self._tuning.retry_base_seconds * 2**attempt
                )
                time.sleep(pause)
        raise ApiError("transport_failed")  # unreachable: the loop returns or raises
