"""The configuration and the secret: what the runner starts with, and every refusal."""

from __future__ import annotations

import json
import os
import stat

import pytest
from nemar_runner.config import (
    ConfigError,
    default_config_path,
    load_config,
    private_file_problem,
    read_secret,
)
from runner_support import SECRET, write_private


class Files:
    """A configuration file the tests rewrite: ``files(changes, mode, remove)`` writes it and returns its path."""

    def __init__(self, path, secret, root, document):
        self.path, self.secret, self.root, self.document = path, secret, root, document

    def __call__(self, changes=None, mode=0o600, remove=()):
        body = {**self.document, **(changes or {})}
        for key in remove:
            body.pop(key, None)
        write_private(self.path, json.dumps(body), mode)
        return self.path


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    secret = tmp_path / "secret"
    write_private(secret, SECRET + "\n")
    document = {
        "api_base": "https://api.example.test/",
        "secret_file": str(secret),
        "root": str(root),
        "sif_root": str(tmp_path / "sifs"),
        "runtime": "apptainer",
        "slots": 2,
        "arch": "amd64",
        "max_scratch_gb": 20,
    }
    return Files(tmp_path / "config.json", secret, root, document)


def test_a_good_configuration_loads(files):
    config = load_config(files())
    assert config.api_base == "https://api.example.test"  # no trailing slash
    assert (config.runtime, config.slots, config.arch, config.max_scratch_gb) == (
        "apptainer",
        2,
        "amd64",
        20,
    )
    assert config.root == str(files.root)
    assert config.launcher_dir.endswith("launcher")  # the checkout's launcher by default
    assert read_secret(config) == SECRET


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o660, 0o666, 0o620, 0o602])
def test_a_configuration_group_or_others_can_reach_is_refused(files, mode):
    path = files(mode=mode)
    with pytest.raises(ConfigError, match="configuration file can be reached by group or others"):
        load_config(path)


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o660, 0o666])
def test_a_secret_file_group_or_others_can_reach_is_refused(files, mode):
    path = files()
    files.secret.chmod(mode)
    config = load_config(path)
    with pytest.raises(ConfigError, match="secret file can be reached by group or others"):
        read_secret(config)


def test_a_symlinked_configuration_or_secret_is_refused(files, tmp_path):
    real = files()
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ConfigError, match="not a regular file"):
        load_config(link)
    config = load_config(real)
    real_secret = tmp_path / "real-secret"
    write_private(real_secret, SECRET)
    files.secret.unlink()
    files.secret.symlink_to(real_secret)
    with pytest.raises(ConfigError, match="not a regular file"):
        read_secret(config)


def test_a_root_that_is_not_the_users_own_is_refused(files):
    if os.getuid() == 0:
        pytest.skip("root owns everything: no directory can be someone else's")
    for foreign in ("/usr", "/"):
        path = files({"root": foreign})
        with pytest.raises(ConfigError, match="root is not owned by the user"):
            load_config(path)


def test_a_root_that_group_or_others_can_write_is_refused(files):
    files.root.chmod(0o770)
    with pytest.raises(ConfigError, match="root can be written by group or others"):
        load_config(files())


def test_a_root_that_is_missing_or_a_file_is_refused(files, tmp_path):
    with pytest.raises(ConfigError, match="root does not exist"):
        load_config(files({"root": str(tmp_path / "nowhere")}))
    plain = tmp_path / "plain"
    plain.write_text("x")
    with pytest.raises(ConfigError, match="root is not a directory"):
        load_config(files({"root": str(plain)}))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="cannot be read"):
        load_config(tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("changes", "remove", "message"),
    [
        ({}, ("slots",), "lacks slots"),
        ({}, ("api_base",), "lacks api_base"),
        ({"surprise": 1}, (), "unknown configuration key surprise"),
        ({"runtime": "lxc"}, (), "runtime must be one of"),
        ({"arch": "riscv"}, (), "arch must be one of"),
        ({"slots": 0}, (), "slots must be a whole number from 1 to 32"),
        ({"slots": 33}, (), "slots must be a whole number from 1 to 32"),
        ({"slots": True}, (), "slots must be a whole number"),
        ({"slots": "2"}, (), "slots must be a whole number"),
        ({"max_scratch_gb": 0}, (), "max_scratch_gb must be a whole number"),
        ({"api_base": "http://api.example.test"}, (), "must be https"),
        ({"api_base": "ftp://api.example.test"}, (), "must be https"),
        ({"api_base": "https://user:pw@api.example.test"}, (), "plain URL"),
        ({"api_base": "https://api.example.test/?a=b"}, (), "plain URL"),
        ({"api_base": ""}, (), "must be a URL"),
        ({"root": "relative/path"}, (), "root must be an absolute path"),
        ({"sif_root": "sifs"}, (), "sif_root must be an absolute path"),
        ({"secret_file": 5}, (), "secret_file must be a path"),
    ],
)
def test_a_bad_value_is_refused_with_a_message_that_names_the_key_only(
    files, changes, remove, message
):
    with pytest.raises(ConfigError, match=message) as caught:
        load_config(files(changes, remove=remove))
    assert SECRET not in str(caught.value)


def test_http_is_allowed_only_on_loopback(files):
    for base in ("http://127.0.0.1:8787", "http://localhost:8787", "http://[::1]:8787"):
        assert load_config(files({"api_base": base})).api_base == base
    with pytest.raises(ConfigError):
        load_config(files({"api_base": "http://example.test:8787"}))


def test_a_configuration_that_is_not_json_or_not_an_object_is_refused(files):
    write_private(files.path, "{not json")
    with pytest.raises(ConfigError, match="not JSON"):
        load_config(files.path)
    write_private(files.path, "[1, 2]")
    with pytest.raises(ConfigError, match="one object"):
        load_config(files.path)


@pytest.mark.parametrize(
    "text",
    ["", "nmr_short", "nmra_" + "ab" * 32, "NMR_" + "ab" * 32, "nmr_" + "AB" * 32, SECRET + "00"],
)
def test_a_secret_file_that_does_not_hold_a_secret_is_refused_and_never_echoed(files, text):
    config = load_config(files())
    write_private(files.secret, text)
    with pytest.raises(ConfigError, match="does not hold a runner secret") as caught:
        read_secret(config)
    assert text not in str(caught.value) or text == ""


def test_the_secret_file_may_end_in_a_newline(files):
    config = load_config(files())
    write_private(files.secret, SECRET + "\n\n")
    assert read_secret(config) == SECRET


def test_the_default_path_follows_xdg(tmp_path):
    assert default_config_path({"XDG_CONFIG_HOME": str(tmp_path)}) == str(
        tmp_path / "nemar-runner" / "config.json"
    )
    home_default = default_config_path({})
    assert home_default.endswith(os.path.join(".config", "nemar-runner", "config.json"))
    # the XDG rule: an empty or relative value is ignored
    assert default_config_path({"XDG_CONFIG_HOME": ""}) == home_default
    assert default_config_path({"XDG_CONFIG_HOME": "relative"}) == home_default


def test_private_file_problem_names_each_case(tmp_path):
    path = tmp_path / "f"
    write_private(path, "x")
    assert private_file_problem(str(path)) is None
    path.chmod(0o640)
    assert "group or others" in (private_file_problem(str(path)) or "")
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert private_file_problem(str(tmp_path)) == "is not a regular file"
    assert private_file_problem(str(tmp_path / "gone")) == "cannot be read"
