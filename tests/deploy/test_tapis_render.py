"""Rendering a definition: operator values go into placeholders, and nothing else can happen.

A definition is JSON whose string values may hold ``${NAME}`` placeholders (``string.Template``
syntax). Tapis's own macros (``${JobUUID}``) share that syntax, so a definition writes them
``$${JobUUID}`` and they come out as written. The render works on the parsed document, never on its
text, so a value can fill a placeholder but cannot change the document's structure.
"""

from __future__ import annotations

import json

import pytest
import register
import tapis_support as ts

# ---- the happy path and the escape --------------------------------------------------------------


def test_a_placeholder_is_filled_inside_a_string():
    document = register.render_text('{"a": "${ROOT}/jobs", "n": 3}', {"ROOT": "/p"})
    assert document == {"a": "/p/jobs", "n": 3}


def test_a_doubled_dollar_is_a_tapis_macro_left_as_written():
    document = register.render_text('{"a": "$${JobWorkingDir}/$${JobUUID}"}', {})
    assert document == {"a": "${JobWorkingDir}/${JobUUID}"}


def test_a_macro_that_is_not_escaped_is_an_unresolved_placeholder():
    with pytest.raises(register.RenderError, match="JobUUID"):
        register.render_text('{"a": "${JobUUID}"}', {})


def test_nested_documents_are_rendered_in_every_value():
    text = '{"a": [{"b": "${X}"}, "${X}"], "c": {"d": ["${X}"]}}'
    assert register.render_text(text, {"X": "v"}) == {
        "a": [{"b": "v"}, "v"],
        "c": {"d": ["v"]},
    }


# ---- an unresolved placeholder is an error ------------------------------------------------------


def test_an_unresolved_placeholder_is_an_error_that_names_it():
    with pytest.raises(register.RenderError) as caught:
        register.render_text('{"a": "${EXEC_HOST}", "b": "${SIF_ROOT}"}', {"SIF_ROOT": "/p"})
    assert "EXEC_HOST" in str(caught.value)
    assert "SIF_ROOT" not in str(caught.value)


def test_a_stray_dollar_is_an_error_not_literal_text():
    with pytest.raises(register.RenderError):
        register.render_text('{"a": "cost $5"}', {})


def test_a_placeholder_in_a_key_is_refused():
    with pytest.raises(register.RenderError, match="key"):
        register.render_text('{"${X}": 1}', {"X": "v"})


def test_text_that_is_not_a_json_object_is_refused():
    for text in ("", "not json", "[1]", '"a"', "3"):
        with pytest.raises(register.RenderError):
            register.render_text(text, {})


# ---- a value with a newline (or any control character) is an error -------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "a\nb",
        "a\rb",
        "a\x00b",
        "a\x1bb",
        "a\x7fb",
        "a" + chr(0x2028) + "b",
        "a" + chr(0x2029) + "b",
        "a\x85b",
        "a\tb",
    ],
    ids=["newline", "return", "nul", "escape", "delete", "line-sep", "para-sep", "nel", "tab"],
)
def test_a_value_with_a_control_character_is_refused_and_not_echoed(bad):
    with pytest.raises(register.RenderError) as caught:
        register.render_text('{"a": "${NOTE}"}', {"NOTE": bad})
    assert "NOTE" in str(caught.value)
    assert bad not in str(caught.value)


def test_a_value_cannot_change_the_structure_of_the_document():
    evil = '", "b": "injected'
    document = register.render_text('{"a": "${X}", "b": "kept"}', {"X": evil})
    assert document == {"a": evil, "b": "kept"}


def test_a_value_is_not_substituted_again():
    assert register.render_text('{"a": "${X}"}', {"X": "${Y}"}) == {"a": "${Y}"}


# ---- a definition never takes a credential -------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "TAPIS_TOKEN",
        "API_TOKEN",
        "DB_PASSWORD",
        "SSH_PRIVATE_KEY",
        "ACCESS_SECRET",
        "SITE_CREDENTIAL",
    ],
)
def test_a_placeholder_named_like_a_credential_is_refused_even_with_a_value(name):
    with pytest.raises(register.RenderError, match="credential"):
        register.render_text('{{"a": "${{{}}}"}}'.format(name), {name: "x"})


# ---- the shape of each operator value ------------------------------------------------------------

GOOD = {
    "EXEC_HOST": "login.example.org",
    "COMMUNITY_ACCOUNT": "community",
    "EXEC_ROOT_DIR": "/path/to/root",
    "SIF_ROOT": "/path/to/sif",
    "LOCAL_DATASETS_ROOT": "/path/to/datasets",
    "NODE_LOCAL_TMPDIR": "/path/to/node-local",
    "SLURM_ACCOUNT_CPU": "example-cpu",
    "SLURM_ACCOUNT_GPU": "example-gpu",
    "AUTHN_METHOD": "PKI_KEYS",
}
BAD = {
    "EXEC_HOST": ["login.example.org/x", "a b", "host;ls", "https://login.example.org", ""],
    "COMMUNITY_ACCOUNT": ["a b", "x" * 61, "a/b", ""],
    "EXEC_ROOT_DIR": ["relative/dir", "/a/../b", "/a b", "/a,b", "/a:b", "/", ""],
    "SIF_ROOT": ["sif", "/a/../b", "/a b"],
    "LOCAL_DATASETS_ROOT": ["data", "/a;b"],
    "NODE_LOCAL_TMPDIR": ["tmp", "/a/./b"],
    "SLURM_ACCOUNT_CPU": ["a b", "a;b", "", "x" * 40],
    "SLURM_ACCOUNT_GPU": ["a b", "a,b"],
    "AUTHN_METHOD": ["PASSWORDISH", "pki_keys", ""],
}


def test_the_table_of_shapes_covers_every_known_placeholder():
    assert sorted(register.SHAPES) == sorted(GOOD)


@pytest.mark.parametrize("name", sorted(GOOD))
def test_a_good_value_is_accepted(name):
    assert register.render_text('{{"a": "${{{}}}"}}'.format(name), {name: GOOD[name]}) == {
        "a": GOOD[name]
    }


@pytest.mark.parametrize(
    ("name", "bad"), [(n, b) for n in sorted(BAD) for b in BAD[n]], ids=lambda v: str(v)[:24]
)
def test_a_value_of_the_wrong_shape_is_refused(name, bad):
    with pytest.raises(register.RenderError) as caught:
        register.render_text('{{"a": "${{{}}}"}}'.format(name), {name: bad})
    assert name in str(caught.value)


def test_a_placeholder_with_no_known_shape_still_refuses_control_characters_only():
    assert register.render_text('{"a": "${ODD_ONE}"}', {"ODD_ONE": "anything goes (1)"}) == {
        "a": "anything goes (1)"
    }


# ---- where the values come from ------------------------------------------------------------------


def test_the_vars_file_has_names_and_raw_values():
    text = '# a comment\n\nEXEC_HOST=login.example.org\nNOTE=a=b\nQUOTED="x"\n  SPACED = y  \n'
    assert register.load_vars(text) == {
        "EXEC_HOST": "login.example.org",
        "NOTE": "a=b",
        "QUOTED": '"x"',
        "SPACED": "y",
    }


@pytest.mark.parametrize(
    "text",
    ["EXEC_HOST", "lower=1", "A=1\nA=2", "1A=x", "=x", "A B=1", "export A=1"],
    ids=[
        "no-equals",
        "lowercase-name",
        "duplicate",
        "digit-first",
        "empty-name",
        "space",
        "export",
    ],
)
def test_a_malformed_vars_file_is_refused(text):
    with pytest.raises(register.RenderError):
        register.load_vars(text)


def test_the_vars_file_wins_over_the_environment_and_only_named_variables_are_read():
    env = {"A": "from-env", "B": "from-env", "UNRELATED": "x", "TAPIS_TOKEN": "tok"}
    values = register.resolve({"A", "B"}, env, {"A": "from-file"})
    assert values == {"A": "from-file", "B": "from-env"}


def test_resolve_names_every_missing_placeholder():
    with pytest.raises(register.RenderError) as caught:
        register.resolve({"A", "B", "C"}, {"B": "x"}, {})
    assert "A" in str(caught.value)
    assert "C" in str(caught.value)


def test_placeholders_lists_the_names_a_document_uses_and_skips_escapes():
    document = json.loads('{"a": "${ONE}/$${JobUUID}", "b": ["x$TWO", "$$plain"]}')
    assert register.placeholders(document) == {"ONE", "TWO"}


# ---- the committed definitions -------------------------------------------------------------------


def test_the_example_values_fill_every_placeholder_of_every_definition():
    values = ts.example_values()
    used: set[str] = set()
    for path in (*ts.APP_PATHS, ts.EXEC_SYSTEM):
        used |= register.placeholders(json.loads(path.read_text(encoding="utf-8")))
    assert used, "no definition has a placeholder"
    assert used == set(values), "example.env and the definitions disagree"
    for path in (*ts.APP_PATHS, ts.EXEC_SYSTEM):
        assert "${" not in json.dumps(ts.rendered(path)).replace("${Job", ""), path.name
