"""`nemar_runner install` and `self-update`: what is accepted, what is refused, and that nothing is
changed by a refusal.

The installer runs for real against a loopback server that stands in for GitHub's release API and file
downloads, with a stand-in executable for cosign (`tests/install_support.py` says what each is and is
not). Where a real cosign binary is installed it is exercised as well, with a throwaway key pair through
the installer's injected trust parameter, a parameter production code never sets; set
`NEMAR_REQUIRE_COSIGN=1` (CI) to make its absence a failure and not a skip.
"""

from __future__ import annotations

import argparse
import fcntl
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import build_runner_bundle
import install_support as support
import pytest
from install_support import (
    Env,
    keyless_bundle,
    plain,
    runner_identity,
    with_member,
)
from nemar_runner import __version__, images, install
from nemar_runner.__main__ import build_parser, main
from nemar_runner.config import Tuning, load_config
from runner_support import write_private

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_COSIGN = os.environ.get("NEMAR_REQUIRE_COSIGN") == "1"


@pytest.fixture
def env(tmp_path):
    made = Env(tmp_path)
    yield made
    made.close()


def refused(word: str, call, *args, **kwargs) -> install.InstallError:
    with pytest.raises(install.InstallError) as caught:
        call(*args, **kwargs)
    assert caught.value.word == word, str(caught.value)
    return caught.value


# ---- the bundle ----------------------------------------------------------------------------------------


def test_the_bundle_builds_identically_twice(tmp_path):
    first = build_runner_bundle.build(ROOT)
    assert build_runner_bundle.build(ROOT) == first
    # a copy of the tree whose files have other times and another place: the same bytes
    copy = support.tree_with_version(tmp_path / "copy", __version__)
    for path in copy.rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    assert build_runner_bundle.build(copy) == first


def test_the_bundle_check_passes_on_this_tree():
    assert build_runner_bundle.check(ROOT) == 0


def test_the_bundle_holds_only_plain_files_in_its_three_directories():
    members = support.tar_members(build_runner_bundle.build(ROOT))
    names = [info.name for info, _ in members]
    assert names == sorted(names)
    for info, _content in members:
        assert info.type in (tarfile.REGTYPE, tarfile.DIRTYPE), info.name
        assert info.name.split("/")[0] in ("runner", "launcher", "deploy"), info.name
        assert not info.name.startswith("deploy/") or info.name.startswith("deploy/runner"), (
            info.name
        )
        assert "__pycache__" not in info.name
        assert not info.name.endswith(".pyc")
        assert (info.uid, info.gid, info.mtime) == (0, 0, build_runner_bundle.FIXED_MTIME)
        assert info.mode in (0o644, 0o755), info.name
    assert "launcher/tapisjob_app.sh" in names
    assert "deploy/runner/boot.py" in names


def test_the_unpacked_bundle_is_a_tree_the_runner_trusts_and_finds_its_launcher(tmp_path):
    destination = tmp_path / "unpacked"
    install.unpack(build_runner_bundle.build(ROOT), str(destination))
    install.check_layout(str(destination), __version__)
    assert install.tree_problem(str(destination)) is None
    assert images.tree_problem(str(destination / "launcher")) is None  # the runner's own rule
    for path in destination.rglob("*"):
        assert support.mode_of(path) in (0o700, 0o600), path
    assert support.mode_of(destination / "launcher" / "tapisjob_app.sh") == 0o700
    assert support.mode_of(destination / "runner" / "nemar_runner" / "agent.py") == 0o600
    # with no launcher_dir in the configuration, the launcher is the one beside the runner
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from nemar_runner import config; print(config.default_launcher_dir())",
            str(destination / "runner"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == str(destination / "launcher")


# ---- the identity -----------------------------------------------------------------------------------


def test_the_identity_accepts_the_release_workflow_for_a_release_tag_and_nothing_else():
    pattern = re.compile(install.IDENTITY_REGEXP)
    good = (
        "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml@refs/tags/"
    )
    for ok in ("nemar-runner-v0.1.0", "nemar-runner-v10.20.30"):
        assert pattern.search(good + ok), ok
    wrong = [
        good + "nemar-runner-v0.1",
        good + "nemar-runner-v0.1.0-rc1",
        good + "nemar-runner-v0.1.0.1",
        good + "v0.1.0",
        good + "image-nemar-python-v0.1.0",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml@refs/tags/"
        "nemar-runner-v0.1.0",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/auto-tag.yml@refs/tags/"
        "nemar-runner-v0.1.0",
        "https://github.com/nemarOrg/nemar-process-evil/.github/workflows/runner-release.yml@refs/tags/"
        "nemar-runner-v0.1.0",
        "https://github.com/someone/nemar-process/.github/workflows/runner-release.yml@refs/tags/"
        "nemar-runner-v0.1.0",
        "x" + good + "nemar-runner-v0.1.0",
        good + "nemar-runner-v0.1.0/extra",
    ]
    for bad in wrong:
        assert not pattern.search(bad), bad
    assert install.OIDC_ISSUER == "https://token.actions.githubusercontent.com"
    assert install.KEYLESS.flags == (
        "--certificate-identity-regexp",
        install.IDENTITY_REGEXP,
        "--certificate-oidc-issuer",
        install.OIDC_ISSUER,
        "--certificate-github-workflow-repository",
        "nemarOrg/nemar-process",
        "--certificate-github-workflow-trigger",
        "push",
    )


def test_production_trusts_the_keyless_identity_and_github_and_no_option_changes_that():
    for function in (install.install, install.command):
        parameters = inspect.signature(function).parameters
        assert parameters["trust"].default is install.KEYLESS, function
        assert parameters["source"].default is install.GITHUB, function
    assert install.GITHUB.allow_http is False
    assert install.GITHUB.api_url.startswith("https://api.github.com/repos/nemarOrg/nemar-process/")
    assert install.GITHUB.download_base.startswith("https://github.com/nemarOrg/nemar-process/")
    # the command line has no option that reaches the trust, the source or an http address
    parser = build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name in ("install", "self-update"):
        options = {o for a in commands.choices[name]._actions for o in a.option_strings}
        assert options <= {
            "-h",
            "--help",
            "--config",
            "--version",
            "--prefix",
            "--cosign-path",
            "--write-units",
            "--units-dir",
        }, name


def test_the_installer_asks_cosign_to_verify_the_bundle_against_the_identity(env):
    env.publish("0.1.0")
    assert env.install("0.1.0") == "installed"
    [call] = env.cosign.calls()
    assert call[0] == "verify-blob"
    assert call[1].endswith(install.tarball_name("0.1.0"))
    assert call[2] == "--bundle"
    assert call[3].endswith(install.bundle_name("0.1.0"))
    assert call[4:] == [
        "--certificate-identity-regexp",
        install.IDENTITY_REGEXP,
        "--certificate-oidc-issuer",
        install.OIDC_ISSUER,
        "--certificate-github-workflow-repository",
        "nemarOrg/nemar-process",
        "--certificate-github-workflow-trigger",
        "push",
    ]


# ---- accepting ---------------------------------------------------------------------------------------


def test_a_bundle_signed_by_the_release_workflow_is_installed_and_made_current(env):
    data = env.publish("0.1.0")
    assert env.install("0.1.0") == "installed"
    version = env.prefix / "versions" / "0.1.0"
    assert (env.prefix / "current").read_text(encoding="ascii") == "0.1.0\n"
    assert install.read_current(str(env.prefix)) == "0.1.0"
    members = {info.name: content for info, content in support.tar_members(data)}
    boot = members["deploy/runner/boot.py"]
    assert (env.prefix / "bin" / "boot.py").read_bytes() == boot  # from the verified bundle
    assert (version / "runner" / "nemar_runner" / "__init__.py").read_text().count('"0.1.0"') == 1
    assert support.mode_of(env.prefix / "bin" / "boot.py") == 0o700
    assert support.mode_of(env.prefix / "current") == 0o600
    assert support.mode_of(env.prefix) == 0o700
    for path in version.rglob("*"):
        assert support.mode_of(path) in (0o700, 0o600), path
    assert install.tree_problem(str(version)) is None
    assert list((env.prefix / ".staging").iterdir()) == []  # the downloads are not kept
    leftovers = [p.name for p in env.prefix.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_cosign_runs_with_a_minimal_environment_in_the_prefix(env, tmp_path):
    recorder = tmp_path / "environment.json"
    script = env.cosign.path.read_text(encoding="utf-8")
    env.cosign.path.write_text(
        script.replace(
            "args = sys.argv[1:]",
            "import os\njson.dump(dict(os.environ), open({!r}, 'w'))\nargs = sys.argv[1:]".format(
                str(recorder)
            ),
            1,
        ),
        encoding="utf-8",
    )
    env.publish("0.1.0")
    env.install("0.1.0")
    seen = json.loads(recorder.read_text(encoding="utf-8"))
    assert set(seen) - {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"} == {"PATH", "HOME", "LANG"}
    assert seen["HOME"] == str(env.prefix / ".cosign-home")


def test_installing_a_version_that_is_there_downloads_nothing_and_only_switches(env):
    env.publish("0.1.0")
    env.publish("0.1.1")
    env.install("0.1.0")
    env.install("0.1.1")
    before = env.requests()
    assert env.install("0.1.0") == "already_installed"  # a rollback is the same command
    assert env.requests() == before
    assert install.read_current(str(env.prefix)) == "0.1.0"


def test_an_installed_version_that_group_or_others_can_write_is_not_switched_to(env):
    env.publish("0.1.0")
    env.install("0.1.0")
    (env.prefix / "versions" / "0.1.0" / "runner" / "nemar_runner" / "agent.py").chmod(0o620)
    refused("version_untrusted", env.install, "0.1.0")


def test_an_installed_version_that_is_not_whole_is_not_switched_to(env):
    env.publish("0.1.0")
    env.publish("0.1.1")
    env.install("0.1.0")
    env.install("0.1.1")
    # what a delete that stopped half way could leave under a version's name
    (env.prefix / "versions" / "0.1.0" / "runner" / "nemar_runner" / "__main__.py").unlink()
    refused("bundle_refused", env.install, "0.1.0")
    assert install.read_current(str(env.prefix)) == "0.1.1"


@pytest.mark.parametrize(
    "version", ["0.1", "1.2.3.4", "1.2.3-rc1", "v1.2.3", "../1.2.3", "", "1.2.x"]
)
def test_a_version_that_is_not_x_y_z_is_refused_before_anything_is_touched(env, version):
    refused("version_refused", env.install, version)
    assert not env.prefix.exists()
    assert env.requests() == 0


# ---- refusing: the signature ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bundle",
    [b"{}", b"", b"not json", json.dumps({"sha256": "0" * 64}).encode()],
    ids=["empty object", "empty file", "not json", "no certificate"],
)
def test_an_unsigned_bundle_is_refused_and_the_archive_is_never_opened(env, bundle):
    # the archive is garbage: if anything read it before the signature was checked, the word would be
    # bundle_refused and not verify_failed
    env.publish_raw("0.1.0", b"this is not an archive", bundle)
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()
    assert len(env.cosign.calls()) == 1


def test_an_unsigned_real_bundle_is_refused(env):
    data = env.bundle("0.1.0")
    env.publish_raw("0.1.0", data, b"{}")
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


def test_a_signature_that_does_not_match_the_archive_is_refused(env):
    signed = env.bundle("0.1.0")
    other = env.bundle("0.1.1")
    env.publish_raw("0.1.0", other, keyless_bundle(signed, runner_identity("0.1.0")))
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


def test_one_changed_byte_of_a_signed_archive_is_refused(env):
    signed = env.bundle("0.1.0")
    changed = signed[:-8] + bytes([signed[-8] ^ 1]) + signed[-7:]
    env.publish_raw("0.1.0", changed, keyless_bundle(signed, runner_identity("0.1.0")))
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


@pytest.mark.parametrize(
    "identity",
    [
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml"
        "@refs/tags/image-nemar-python-v0.1.0",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/image-nemar-python.yml"
        "@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml"
        "@refs/heads/main",
        "https://github.com/nemarOrg/nemar-process/.github/workflows/runner-release.yml"
        "@refs/tags/v0.1.0",
        "https://github.com/someone-else/nemar-process/.github/workflows/runner-release.yml"
        "@refs/tags/nemar-runner-v0.1.0",
    ],
    ids=["image tag", "image main", "runner main", "release tag", "another repository"],
)
def test_a_bundle_signed_by_another_identity_is_refused(env, identity):
    data = env.bundle("0.1.0")
    env.publish_raw("0.1.0", data, keyless_bundle(data, identity))
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


def test_a_bundle_signed_by_another_issuer_is_refused(env):
    data = env.bundle("0.1.0")
    env.publish_raw(
        "0.1.0", data, keyless_bundle(data, runner_identity("0.1.0"), "https://issuer.example.test")
    )
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


@pytest.mark.parametrize(
    ("repository", "trigger"),
    [
        ("someone-else/nemar-process", "push"),  # the workflow file run by another repository
        ("nemarOrg/nemar-process", "workflow_call"),  # the workflow called by another workflow
        ("nemarOrg/nemar-process", "workflow_dispatch"),
    ],
)
def test_a_certificate_for_another_repository_or_trigger_is_refused_though_its_identity_matches(
    env, repository, trigger
):
    data = env.bundle("0.1.0")
    bundle = keyless_bundle(data, runner_identity("0.1.0"), repository=repository, trigger=trigger)
    env.publish_raw("0.1.0", data, bundle)
    refused("verify_failed", env.install, "0.1.0")
    env.assert_untouched()


def test_a_signed_bundle_of_another_version_is_refused(env):
    # a signature by the workflow for tag 0.1.5, offered as release 0.1.6: replayed, not downgraded
    old = env.bundle("0.1.5")
    env.publish_raw("0.1.6", old, keyless_bundle(old, runner_identity("0.1.5")))
    refused("bundle_refused", env.install, "0.1.6")
    env.assert_untouched()


def test_cosign_that_cannot_be_found_or_trusted_refuses_before_any_download(env, tmp_path):
    env.publish("0.1.0")
    empty = tmp_path / "empty-bin"
    empty.mkdir(mode=0o700)
    tuning = Tuning(launch_path=str(empty))
    refused("cosign_missing", env.install, "0.1.0", cosign_path=None, tuning=tuning)
    assert env.requests() == 0
    writable = env.cosign.path
    writable.chmod(0o770)
    refused("cosign_untrusted", env.install, "0.1.0")
    writable.chmod(0o700)
    link = tmp_path / "cosign-link"
    link.symlink_to(env.cosign.path)
    refused("cosign_untrusted", env.install, "0.1.0", cosign_path=str(link))
    refused("cosign_untrusted", env.install, "0.1.0", cosign_path="cosign")
    assert env.requests() == 0
    env.cosign.directory.chmod(0o770)
    refused("cosign_untrusted", env.install, "0.1.0")
    env.cosign.directory.chmod(0o700)


def test_cosign_on_the_trusted_path_is_found_when_cosign_path_is_not_given(env):
    env.publish("0.1.0")
    tuning = Tuning(launch_path=str(env.cosign.directory))
    assert env.install("0.1.0", cosign_path=None, tuning=tuning) == "installed"


def test_a_cosign_that_hangs_is_cut_off(env, monkeypatch):
    script = env.cosign.path.read_text(encoding="utf-8")
    env.cosign.path.write_text(
        script.replace(
            "args = sys.argv[1:]", "import time\ntime.sleep(60)\nargs = sys.argv[1:]", 1
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(install, "VERIFY_TIMEOUT_SECONDS", 0.5)
    env.publish("0.1.0")
    refused("verify_timeout", env.install, "0.1.0")
    env.assert_untouched()


# ---- refusing: the archive ----------------------------------------------------------------------------

LINK = tarfile.TarInfo("runner/link")
LINK.type = tarfile.SYMTYPE
LINK.linkname = "/etc"
HARDLINK = tarfile.TarInfo("runner/hard")
HARDLINK.type = tarfile.LNKTYPE
HARDLINK.linkname = "runner/nemar_runner/agent.py"
DEVICE = tarfile.TarInfo("runner/device")
DEVICE.type = tarfile.CHRTYPE
FIFO = tarfile.TarInfo("runner/fifo")
FIFO.type = tarfile.FIFOTYPE
DIRLINK = tarfile.TarInfo("launcher/nemar_launch/up")
DIRLINK.type = tarfile.SYMTYPE
DIRLINK.linkname = ".."

EVIL = {
    "a symbolic link": (LINK, None),
    "a hard link": (HARDLINK, None),
    "a link to a parent": (DIRLINK, None),
    "a device": (DEVICE, None),
    "a fifo": (FIFO, None),
    "a .. member": (plain("runner/../../../../escape.txt", 4), b"evil"),
    "a leading .. member": (plain("../escape.txt", 4), b"evil"),
    "a .. directory": (plain("runner/..", 0), None),
    "an absolute name": (plain("/tmp/escape.txt", 4), b"evil"),
    "a name outside the bundle's directories": (plain("etc/escape.txt", 4), b"evil"),
    "a backslash": (plain("runner\\escape.txt", 4), b"evil"),
    "an odd character": (plain("runner/a b.txt", 4), b"evil"),
    "a name with an empty part": (plain("runner//escape.txt", 4), b"evil"),
    "a duplicate": (plain("runner/nemar_runner/agent.py", 4), b"evil"),
    "a file where a directory is": (plain("runner/nemar_runner/agent.py/x", 4), b"evil"),
    "an oversized file": (plain("runner/big.bin", 17 * 1024 * 1024), bytes(17 * 1024 * 1024)),
}


@pytest.mark.parametrize("name", sorted(EVIL))
def test_a_signed_archive_with_one_hostile_member_is_refused_and_nothing_is_installed(env, name):
    info, content = EVIL[name]
    data = with_member(env.bundle("0.1.0"), info, content)
    env.publish_raw("0.1.0", data, keyless_bundle(data, runner_identity("0.1.0")))
    refused("bundle_refused", env.install, "0.1.0")
    env.assert_untouched()
    assert len(env.cosign.calls()) == 1  # it was signed: the refusal is the unpacker's
    assert not (env.prefix / "escape.txt").exists()
    assert not (env.tmp / "escape.txt").exists()


def test_a_signed_archive_that_is_not_a_gzip_tar_is_refused(env):
    data = b"signed, but not an archive"
    env.publish_raw("0.1.0", data, keyless_bundle(data, runner_identity("0.1.0")))
    refused("bundle_refused", env.install, "0.1.0")
    env.assert_untouched()


def test_a_signed_archive_with_too_many_members_is_refused(env, monkeypatch):
    monkeypatch.setattr(install, "MAX_MEMBERS", 10)
    env.publish("0.1.0")
    refused("bundle_refused", env.install, "0.1.0")
    env.assert_untouched()


def test_a_signed_archive_without_a_runner_or_its_service_files_is_refused(env):
    members = [
        (info, content)
        for info, content in support.tar_members(env.bundle("0.1.0"))
        if info.name != "deploy/runner/boot.py"
    ]
    data = support.tar_gz(members)
    env.publish_raw("0.1.0", data, keyless_bundle(data, runner_identity("0.1.0")))
    refused("bundle_refused", env.install, "0.1.0")
    env.assert_untouched()


# ---- refusing: the transport -----------------------------------------------------------------------------


def test_a_missing_release_file_is_a_refusal_and_changes_nothing(env):
    refused("download_failed", env.install, "9.9.9")
    env.assert_untouched()


def test_a_redirect_to_an_address_that_is_not_https_or_loopback_is_not_followed(env):
    env.publish("0.1.0")
    env.releases.redirect_to = "http://downloads.example.test/nemar-runner.tar.gz"
    refused("download_refused", env.install, "0.1.0")
    env.assert_untouched()


def test_production_accepts_https_only(env):
    for url in (
        "http://127.0.0.1:1/x",
        "ftp://example.test/x",
        "file:///etc/passwd",
        "https://u:p@h/x",
    ):
        refused("download_refused", install.fetch, url, install.GITHUB, 10)
    # and the loopback http that tests use is refused unless the source says so
    refused("download_refused", install.fetch, env.server.base_url + "/x", install.GITHUB, 10)
    assert env.requests() == 0


def test_a_download_larger_than_the_limit_is_refused(env):
    env.publish("0.1.0")
    refused(
        "download_failed",
        install.fetch,
        env.source.download_base + "/nemar-runner-v0.1.0/" + install.tarball_name("0.1.0"),
        env.source,
        100,
    )


# ---- the prefix ---------------------------------------------------------------------------------------------


def test_a_prefix_that_group_or_others_can_write_or_that_is_a_link_is_refused(env, tmp_path):
    env.publish("0.1.0")
    env.prefix.mkdir(mode=0o700)
    env.prefix.chmod(0o770)
    refused("prefix_refused", env.install, "0.1.0")
    env.prefix.chmod(0o700)
    (env.prefix / "versions").mkdir(mode=0o700)
    (env.prefix / "versions").chmod(0o777)
    refused("prefix_refused", env.install, "0.1.0")
    (env.prefix / "versions").chmod(0o700)
    link = tmp_path / "link-to-prefix"
    link.symlink_to(env.prefix)
    refused(
        "prefix_refused",
        install.install,
        "0.1.0",
        str(link),
        env.source,
        install.KEYLESS,
        str(env.cosign.path),
    )
    assert env.requests() == 0


def test_a_relative_prefix_and_a_missing_parent_are_refused(env):
    refused("prefix_refused", install.install, "0.1.0", "relative/dir", env.source)
    refused(
        "prefix_refused",
        install.install,
        "0.1.0",
        str(env.tmp / "no" / "such" / "prefix"),
        env.source,
    )


def test_two_installs_at_once_are_not_allowed(env):
    env.publish("0.1.0")
    install.prepare_prefix(str(env.prefix))
    fd = os.open(env.prefix / ".install.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        refused("busy", env.install, "0.1.0")
    finally:
        os.close(fd)
    assert env.install("0.1.0") == "installed"


# ---- older versions ---------------------------------------------------------------------------------------


def test_older_versions_beyond_the_newest_two_are_removed(env):
    for version in ("0.1.0", "0.1.1", "0.1.2", "0.1.3"):
        env.publish(version)
        env.install(version)
    assert env.versions() == ["0.1.2", "0.1.3"]
    assert install.read_current(str(env.prefix)) == "0.1.3"
    assert list((env.prefix / ".staging").iterdir()) == []  # what was removed is gone, not parked


def test_a_version_a_runner_is_using_is_never_removed(env):
    for version in ("0.1.0", "0.1.1"):
        env.publish(version)
        env.install(version)
    # a running boot.py holds its version's lock (shared) for as long as it runs
    fd = os.open(env.prefix / "locks" / "0.1.0.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        for version in ("0.1.2", "0.1.3"):
            env.publish(version)
            env.install(version)
        assert env.versions() == ["0.1.0", "0.1.2", "0.1.3"]  # 0.1.1 went, 0.1.0 stayed
    finally:
        os.close(fd)
    env.publish("0.1.4")
    env.install("0.1.4")
    assert env.versions() == ["0.1.3", "0.1.4"]


def test_the_current_version_is_never_removed_even_when_it_is_old(env):
    for version in ("0.1.0", "0.1.1", "0.1.2"):
        env.publish(version)
        env.install(version)
    env.install("0.1.0")  # a rollback: current is the oldest
    assert env.versions() == ["0.1.0", "0.1.1", "0.1.2"]
    assert install.read_current(str(env.prefix)) == "0.1.0"


# ---- self-update ---------------------------------------------------------------------------------------------


def config_for(env: Env, **extra):
    root = env.tmp / "root"
    root.mkdir(mode=0o700, exist_ok=True)
    document = {
        "api_base": "https://api.example.test",
        "secret_file": str(env.tmp / "secret"),
        "root": str(root),
        "sif_root": str(env.tmp / "sifs"),
        "runtime": "apptainer",
        "slots": 1,
        "arch": "amd64",
        "max_scratch_gb": 1,
        "install_prefix": str(env.prefix),
        "cosign_path": str(env.cosign.path),
    }
    document.update(extra)
    path = env.tmp / "config.json"
    write_private(path, json.dumps(document))
    return path


def update(env: Env, **extra) -> str:
    return install.self_update(load_config(str(config_for(env, **extra))), None, env.source)


def test_self_update_is_off_when_the_configuration_says_so_and_asks_nothing(env, capsys):
    env.publish("0.1.0")
    env.publish("0.1.1")
    env.install("0.1.0")
    before = env.requests()
    path = config_for(env, self_update=False)
    assert main(["self-update", "--config", str(path)]) == 0
    assert capsys.readouterr().out == "self-update: self_update_off\n"
    assert update(env, self_update=False) == "self_update_off"
    assert env.requests() == before  # no listing, no download
    assert install.read_current(str(env.prefix)) == "0.1.0"
    assert len(env.cosign.calls()) == 1  # the one of the first install


def test_self_update_installs_the_highest_release_that_is_newer_not_a_prerelease(env):
    env.publish("0.1.0")
    env.install("0.1.0")
    env.publish("0.1.1")
    env.publish("0.1.3")
    env.publish("0.2.0", prerelease=True)
    env.publish("0.3.0", draft=True)
    env.publish("0.4.0", with_assets=False)  # a release whose files are not all there yet
    env.publish("9.9.9", tag="v9.9.9")  # another artifact's tag
    env.publish("8.8.8", tag="image-nemar-python-v8.8.8")
    env.publish("0.1.2")
    assert update(env) == "updated 0.1.0->0.1.3"
    assert install.read_current(str(env.prefix)) == "0.1.3"
    assert (env.prefix / "versions" / "0.1.3").is_dir()
    assert not (env.prefix / "versions" / "0.2.0").exists()
    assert update(env) == "up_to_date"


def test_self_update_never_goes_backwards_or_sideways(env):
    env.publish("0.2.0")
    env.publish("0.1.0")
    env.install("0.2.0")
    assert update(env) == "up_to_date"
    assert install.read_current(str(env.prefix)) == "0.2.0"


def test_self_update_verifies_like_install_and_a_bad_signature_changes_nothing(env):
    env.publish("0.1.0")
    env.install("0.1.0")
    data = env.bundle("0.1.1")
    env.publish_raw("0.1.1", data, b"{}")
    refused("verify_failed", update, env)
    assert install.read_current(str(env.prefix)) == "0.1.0"
    assert env.versions() == ["0.1.0"]
    assert list((env.prefix / ".staging").iterdir()) == []


def test_self_update_without_an_install_or_a_prefix_says_so(env):
    refused("not_installed", update, env)
    refused(
        "prefix_missing",
        install.self_update,
        load_config(str(config_for(env, install_prefix=None))),
        None,
        env.source,
    )


def test_self_update_reads_every_page_of_releases(env):
    env.publish("0.1.0")
    env.install("0.1.0")
    env.publish("0.1.1")
    for number in range(120):  # a hundred and twenty newer releases of other artifacts
        env.releases.releases.insert(
            0,
            {
                "tag_name": "v1.0.{}".format(number),
                "draft": False,
                "prerelease": False,
                "assets": [],
            },
        )
    assert update(env) == "updated 0.1.0->0.1.1"
    assert env.releases.listings == 2


def test_self_update_runs_cosign_verify_blob_and_no_other_program(env):
    env.publish("0.1.0")
    env.install("0.1.0")
    env.publish("0.1.1")
    update(env)
    assert {call[0] for call in env.cosign.calls()} == {"verify-blob"}  # nothing is restarted


# ---- the command line -----------------------------------------------------------------------------------------


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def test_the_command_installs_and_writes_the_units_from_the_verified_bundle(
    env, monkeypatch, tmp_path, capsys
):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    prefix = home / "nemar-runner"
    env.prefix = prefix
    env.publish("0.1.0")
    units = tmp_path / "units"
    config = home / "config.json"
    args = parse(
        "install",
        "--version",
        "0.1.0",
        "--prefix",
        str(prefix),
        "--write-units",
        "--units-dir",
        str(units),
        "--cosign-path",
        str(env.cosign.path),
        "--config",
        str(config),
    )
    out, err = io.StringIO(), io.StringIO()
    assert install.command(args, str(config), out, err, env.source, install.KEYLESS) == 0, (
        err.getvalue()
    )
    assert out.getvalue() == "install: installed version=0.1.0\n"
    run = (units / "nemar-runner.service").read_text(encoding="utf-8")
    assert (
        "ExecStart=/usr/bin/python3 -I -S %h/nemar-runner/bin/boot.py run --config %h/config.json\n"
        in run
    )
    assert "@PREFIX@" not in run
    assert "@CONFIG@" not in run
    update_unit = (units / "nemar-runner-update.service").read_text(encoding="utf-8")
    assert "boot.py self-update --config %h/config.json\n" in update_unit
    assert sorted(p.name for p in units.iterdir()) == sorted(install.UNIT_FILES)
    for path in units.iterdir():
        assert support.mode_of(path) == 0o600


def test_a_relative_configuration_path_is_written_into_a_unit_as_an_absolute_one(
    env, monkeypatch, tmp_path
):
    work = tmp_path / "work"
    (work / "cfg").mkdir(parents=True)
    monkeypatch.chdir(work)
    env.publish("0.1.0")
    env.install("0.1.0", write_units=("cfg/../config.json", str(tmp_path / "units")))
    run = (tmp_path / "units" / "nemar-runner.service").read_text(encoding="utf-8")
    assert "--config {}/config.json\n".format(work.resolve()) in run or (
        "--config {}/config.json\n".format(work) in run
    )
    assert "cfg/../" not in run
    assert "--config config.json" not in run


def test_a_prefix_or_a_configuration_that_cannot_be_written_into_a_unit_is_refused(env, tmp_path):
    env.publish("0.1.0")
    env.install("0.1.0")
    version = str(env.prefix / "versions" / "0.1.0")
    for bad in (
        "/path with space/x",
        "/path/with%h/x",
        "/path/$HOME/x",
        '/path/"quoted"/x',
        "/path/new\nline",
    ):
        refused("units_refused", install.render_units, version, bad, "/c.json", str(tmp_path / "u"))
        refused(
            "units_refused",
            install.render_units,
            version,
            str(env.prefix),
            bad,
            str(tmp_path / "u"),
        )


def test_a_prefix_outside_the_home_is_written_as_it_is(env, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert install.systemd_path(str(tmp_path / "home" / "r")) == "%h/r"
    assert install.systemd_path("/opt/nemar-runner") == "/opt/nemar-runner"
    assert install.systemd_path(str(tmp_path / "home")) == "%h"
    assert install.systemd_path(str(tmp_path / "homework")) == str(tmp_path / "homework")


def test_the_command_reports_a_refusal_with_a_fixed_word_and_exit_1(env):
    env.publish_raw("0.1.0", b"x", b"{}")
    out, err = io.StringIO(), io.StringIO()
    args = parse(
        "install",
        "--version",
        "0.1.0",
        "--prefix",
        str(env.prefix),
        "--cosign-path",
        str(env.cosign.path),
    )
    assert install.command(args, "/nonexistent", out, err, env.source, install.KEYLESS) == 1
    assert err.getvalue() == "nemar-runner install: refused: verify_failed (cosign exit 1)\n"
    assert out.getvalue() == ""


def test_the_command_exits_2_for_a_configuration_it_will_not_use(env, tmp_path, capsys):
    path = tmp_path / "config.json"
    write_private(path, "{}")
    assert main(["self-update", "--config", str(path)]) == 2
    assert "nemar-runner: the configuration lacks" in capsys.readouterr().err


# ---- the real cosign ---------------------------------------------------------------------------------------------


def _cosign(binary: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    environment = {"PATH": os.environ["PATH"], "HOME": str(cwd), "COSIGN_PASSWORD": ""}
    return subprocess.run(
        [binary, *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def offline_signature(binary: str, work: Path, blob: Path):
    """A throwaway key pair and a bundle for ``blob`` made with it, with no Fulcio and no Rekor; the
    flags `verify-blob` needs to accept it. The spelling of the offline flags differs between cosign
    releases, so the ones this binary accepts are found by trying. None when this binary cannot."""
    if _cosign(binary, ["generate-key-pair"], work).returncode != 0:
        return None
    key, public, bundle = work / "cosign.key", work / "cosign.pub", work / "blob.sigstore.json"
    signed = False
    for flags in (("--tlog-upload=false", "--use-signing-config=false"), ("--tlog-upload=false",)):
        done = _cosign(
            binary,
            ["sign-blob", "--key", str(key), "--bundle", str(bundle), *flags, "--yes", str(blob)],
            work,
        )
        if done.returncode == 0 and bundle.exists():
            signed = True
            break
    if not signed:
        return None
    for flags in (
        ("--insecure-ignore-tlog=true",),
        ("--private-infrastructure",),
        ("--insecure-ignore-tlog=true", "--private-infrastructure"),
    ):
        trust = ("--key", str(public), *flags)
        if (
            _cosign(
                binary, ["verify-blob", str(blob), "--bundle", str(bundle), *trust], work
            ).returncode
            == 0
        ):
            return bundle, trust
    return None


@pytest.fixture
def real(env, tmp_path):
    binary = support.real_cosign()
    if binary is None:
        if REQUIRE_COSIGN:
            pytest.fail("cosign is required here (NEMAR_REQUIRE_COSIGN=1) and is not installed")
        pytest.skip("no cosign binary on PATH: CI installs a pinned one and requires it")
    work = tmp_path / "real"
    work.mkdir(mode=0o700)
    blob = work / "blob.tar.gz"
    blob.write_bytes(env.bundle("0.1.0"))
    made = offline_signature(binary, work, blob)
    if made is None:
        if REQUIRE_COSIGN:
            pytest.fail("this cosign cannot sign and verify with a key pair offline")
        pytest.skip("this cosign cannot sign and verify with a key pair offline")
    bundle, trust = made
    return binary, blob, bundle, install.Trust(trust)


def test_a_bundle_signed_with_a_real_throwaway_key_is_installed_through_the_injected_trust(
    env, real
):
    binary, blob, bundle, trust = real
    env.publish_raw("0.1.0", blob.read_bytes(), bundle.read_bytes())
    assert env.install("0.1.0", trust=trust, cosign_path=binary) == "installed"
    assert install.read_current(str(env.prefix)) == "0.1.0"


def test_real_cosign_refuses_a_changed_archive_an_unsigned_bundle_and_the_wrong_trust(env, real):
    binary, blob, bundle, trust = real
    data = blob.read_bytes()
    env.publish_raw("0.1.0", data[:-8] + bytes([data[-8] ^ 1]) + data[-7:], bundle.read_bytes())
    refused("verify_failed", env.install, "0.1.0", trust=trust, cosign_path=binary)
    env.releases.releases.clear()
    env.releases.files.clear()
    env.publish_raw("0.1.0", data, b"{}")
    refused("verify_failed", env.install, "0.1.0", trust=trust, cosign_path=binary)
    env.releases.releases.clear()
    env.releases.files.clear()
    env.publish_raw("0.1.0", data, bundle.read_bytes())
    # the same good signature, but the production identity is what is asked: a key-signed bundle has none
    refused("verify_failed", env.install, "0.1.0", trust=install.KEYLESS, cosign_path=binary)
    env.assert_untouched()
