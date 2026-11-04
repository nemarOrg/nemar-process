"""The ``include`` glob dialect of a dataset reference: which patterns it refuses, and matching.

The dialect is the shared contract's (the ``include`` description in the launch-specification
schema). A pattern is matched against a manifest path relative to the dataset root, as a whole:

* ``**/`` at the start of a path segment (the start of the pattern, or just after a ``/``) matches
  zero or more directories, so ``a/**/b`` matches ``a/b``;
* ``**`` anywhere else matches any run of characters, ``/`` included. Three or more stars in a row
  are this too, never ``**/`` (``***/x`` is "any run, then ``/x``");
* ``*`` matches any run of characters and ``?`` exactly one, neither ever ``/``;
* ``[abc]``, ``[a-c]`` and ``[!abc]`` are classes, by code point; a class never matches ``/``, so a
  negated class is any one character except those listed and ``/``;
* a class ends at the first ``]`` after its start (``[`` or ``[!``), so ``[]`` and ``[!]`` are empty
  classes, which are refused; a range whose end is before its start (``[z-a]``) is refused; a ``[``
  with no ``]`` after it is a literal ``[``;
* matching is case-sensitive and a leading dot is not special; there are no escapes.

``include_pattern_problem`` is exactly nemar-cli's ``includePatternProblem``: the refused set is
the same one, found left to right, and ``bad_include`` is that set.

Matching never backtracks. A pattern is compiled to a nondeterministic automaton whose states are
the bits of one integer; each character of the path advances all states at once, so a match costs
the path's length times a few big-integer operations (the integer has one bit per pattern token),
whatever the pattern. A regular expression, or a glob matcher that backs up to an earlier ``*``,
can take exponential or cubic time on a hostile pattern.
"""

from __future__ import annotations

from typing import NamedTuple

# Token kinds. A token is the kind and its payload (the character, or a class).
_LIT = 0  # one literal character
_ONE = 1  # ?  : one character, not /
_CLASS = 2  # [..]: one character of a set, not /
_SEG = 3  # *  : any run of characters that has no /, empty included
_ANY = 4  # ** : any run of characters, / included, empty included
_HEAD = 5  # the start of `**/`: enters the body, or skips it (no directory at all)
_BODY = 6  # inside `**/`: any characters, and a / ends it


class BadInclude(ValueError):
    """A pattern the dialect refuses. ``problem`` is ``empty_class`` or ``inverted_range``."""

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem


class _Set(NamedTuple):
    negate: bool
    ranges: tuple[tuple[str, str], ...]


def _tokenize(pattern: str) -> list[tuple[int, object]]:
    tokens: list[tuple[int, object]] = []
    i = 0
    length = len(pattern)
    while i < length:
        char = pattern[i]
        if char == "[":
            negate = pattern[i + 1 : i + 2] == "!"
            start = i + 2 if negate else i + 1
            end = pattern.find("]", start)
            if end == -1:
                tokens.append((_LIT, "["))  # never closed: an ordinary character
                i += 1
                continue
            if end == start:
                raise BadInclude("empty_class")
            body = pattern[start:end]
            ranges: list[tuple[str, str]] = []
            k = 0
            while k < len(body):
                if k + 2 < len(body) and body[k + 1] == "-":
                    if body[k] > body[k + 2]:
                        raise BadInclude("inverted_range")
                    ranges.append((body[k], body[k + 2]))
                    k += 3
                else:
                    ranges.append((body[k], body[k]))
                    k += 1
            tokens.append((_CLASS, _Set(negate, tuple(ranges))))
            i = end + 1
        elif char == "?":
            tokens.append((_ONE, None))
            i += 1
        elif char == "*":
            j = i
            while j < length and pattern[j] == "*":
                j += 1
            stars = j - i
            at_segment_start = i == 0 or pattern[i - 1] == "/"
            if stars == 1:
                tokens.append((_SEG, None))
                i = j
            elif stars == 2 and at_segment_start and pattern[j : j + 1] == "/":
                tokens.append((_HEAD, None))
                tokens.append((_BODY, None))
                i = j + 1  # the slash belongs to `**/`
            else:
                tokens.append((_ANY, None))
                i = j
        else:
            tokens.append((_LIT, char))
            i += 1
    return tokens


def include_pattern_problem(pattern: str) -> str | None:
    """Why the dialect refuses ``pattern`` (``empty_class`` or ``inverted_range``), or None."""
    try:
        _tokenize(pattern)
    except BadInclude as refused:
        return refused.problem
    return None


class Pattern:
    """One ``include`` pattern, compiled once. ``BadInclude`` if the dialect refuses it."""

    __slots__ = ("_body", "_count", "_head", "_masks", "_rounds", "_skip", "_tokens")

    def __init__(self, pattern: str) -> None:
        tokens = _tokenize(pattern)
        self._tokens = tokens
        self._count = len(tokens)
        skip = head = body = 0
        run = longest = 0
        for index, (kind, _payload) in enumerate(tokens):
            bit = 1 << index
            if kind in (_SEG, _ANY, _HEAD):
                skip |= bit  # may match nothing: the next token's state is reached at once
                run += 1
                longest = max(longest, run)
            else:
                run = 0
            if kind == _HEAD:
                head |= bit  # and, with the body skipped, the token after the body too
            elif kind == _BODY:
                body |= bit
        self._skip = skip
        self._head = head
        self._body = body
        self._rounds = longest + 1
        self._masks: dict[str, tuple[int, int]] = {}

    def _close(self, states: int) -> int:
        """Add every state reachable without reading a character."""
        for _ in range(self._rounds):
            states |= ((states & self._skip) << 1) | ((states & self._head) << 2)
        return states

    def _for_char(self, char: str) -> tuple[int, int]:
        """(states that read ``char`` and move on, states that read it and stay)."""
        moves = stays = 0
        slash = char == "/"
        for index, (kind, payload) in enumerate(self._tokens):
            bit = 1 << index
            if kind == _LIT:
                if payload == char:
                    moves |= bit
            elif kind == _ONE:
                if not slash:
                    moves |= bit
            elif kind == _CLASS:
                assert isinstance(payload, _Set)
                found = any(low <= char <= high for low, high in payload.ranges)
                if not slash and found != payload.negate:
                    moves |= bit
            elif kind == _SEG:
                if not slash:
                    stays |= bit
            elif kind in (_ANY, _BODY):  # any character, and stay (_HEAD reads nothing)
                stays |= bit
        return moves, stays

    def match(self, path: str) -> bool:
        """Whether the whole of ``path`` matches. Never backtracks; see the module docstring."""
        states = self._close(1)
        masks = self._masks
        body = self._body
        for char in path:
            found = masks.get(char)
            if found is None:
                found = masks[char] = self._for_char(char)
            moves, stays = found
            following = ((states & moves) << 1) | (states & stays)
            if char == "/":
                following |= (states & body) << 1  # a `/` ends the body of `**/`
            if not following:
                return False
            states = self._close(following)
        return bool(states >> self._count & 1)
