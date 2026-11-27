"""The check of a Podman container's OCI configuration before it starts (``runtime.check_container``).

The document is the clean case's ``config.json`` that rootless Podman 4.9.3 wrote in CI under the
launcher's flags (``tests/launcher/fixtures/podman-4.9.3-clean-config.json``), with every host path
replaced by a directory under ``/podman-fixture``: ``static`` is the container's ``StaticDir``, ``run`` the
run directory that holds ``HostsPath``, ``binds/<name>`` the declared bind sources. No Podman is
needed: the placeholder directories are made real under the test's temporary directory, with the
files Podman writes there, and what the engine's user namespace says of the container's own paths
(``runtime.resolve_in_namespace``, which a real Podman answers in ``tests/launcher/test_podman_e2e.py``)
is read from those real files with ``os.path.realpath`` and ``os.stat``. Each refusal case changes one
thing in a copy of the clean document and names the reason the check must give.
"""

from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest
from nemar_launch import run, runtime

FIXTURE = Path(__file__).parent / "fixtures" / "podman-4.9.3-clean-config.json"
PLACEHOLDER = "/podman-fixture"
OWN_FILES = {
    "static": ("passwd", "group"),
    "run": ("hosts", "hostname", "resolv.conf", ".containerenv"),
}


class Case:
    """The fixture made real under ``root``: the documents, the declared binds, the namespace's view."""

    def __init__(self, root: Path) -> None:
        self.root = root / "podman-fixture"
        text = FIXTURE.read_text(encoding="utf-8").replace(PLACEHOLDER, str(self.root))
        self.document = json.loads(text)
        for where, names in OWN_FILES.items():
            (self.root / where).mkdir(parents=True)
            for name in names:
                (self.root / where / name).write_text("" if name == ".containerenv" else name)
        (self.root / "static" / "shm").mkdir()
        for name in ("work", "tmp", "code", "elsewhere"):
            (self.root / "binds" / name).mkdir(parents=True)
        self.binds = [
            run.Bind(str(self.root / "binds" / "work"), "/work", False),
            run.Bind(str(self.root / "binds" / "tmp"), "/tmp", False),
            run.Bind(str(self.root / "binds" / "code"), "/code", True),
        ]
        static = str(self.root / "static")
        self.inspected = {
            "OCIConfigPath": static + "/config.json",
            "StaticDir": static,
            "HostsPath": str(self.root / "run" / "hosts"),
            "State": {"Status": "initialized"},
        }

    def resolved(self, document) -> dict:
        found = {}
        for path in runtime.namespace_paths(self.inspected, document):
            if not os.path.exists(path):
                continue
            mode = os.stat(path).st_mode
            kind = "file" if stat.S_ISREG(mode) else "directory" if stat.S_ISDIR(mode) else "other"
            found[path] = runtime.Resolved(os.path.realpath(path), kind)
        return found

    def check(self, document=None, inspected=None):
        document = self.document if document is None else document
        inspected = self.inspected if inspected is None else inspected
        return runtime.check_container(inspected, document, self.binds, self.resolved(document))

    def mount(self, document, destination: str) -> dict:
        (found,) = [m for m in document["mounts"] if m["destination"] == destination]
        return found


@pytest.fixture
def case(tmp_path) -> Case:
    return Case(tmp_path)


def changed(case: Case, change) -> runtime.Refusal | None:
    document = copy.deepcopy(case.document)
    change(case, document)
    return case.check(document)


# ---- the clean document ----------------------------------------------------------------------------


def test_the_fixture_is_the_clean_case_with_placeholders_only():
    text = FIXTURE.read_text(encoding="utf-8")
    document = json.loads(text)
    sources = {m["source"] for m in document["mounts"] if m["source"].startswith("/")}
    assert all(s.startswith(PLACEHOLDER + "/") for s in sources)
    assert "/home/" not in text
    assert "/run/user/" not in text
    assert document["linux"].get("devices") is None
    assert document.get("hooks") is None


def test_the_clean_configuration_passes(case):
    assert case.check() is None


def test_every_row_of_the_allowlist_is_in_the_clean_configuration(case):
    destinations = {m["destination"] for m in case.document["mounts"]}
    assert {row.destination for row in runtime.ENGINE_DEFAULTS} <= destinations
    assert destinations - {row.destination for row in runtime.ENGINE_DEFAULTS} == {
        "/work",
        "/tmp",
        "/code",
    }


def test_options_are_compared_as_sets(case):
    def reorder(_case, document):
        for mount in document["mounts"]:
            mount["options"] = list(reversed(mount["options"])) + mount["options"][:1]

    assert changed(case, reorder) is None


def test_devpts_takes_rprivate_in_place_of_gid_5_when_few_group_ids_are_mapped(case):
    def few_gids(case, document):
        options = case.mount(document, "/dev/pts")["options"]
        options[options.index("gid=5")] = "rprivate"

    assert changed(case, few_gids) is None


@pytest.mark.parametrize(
    "hooks",
    [{"createRuntime": [], "poststart": []}, {}, None, []],
    ids=["empty-stages", "empty-object", "null", "empty-list"],
)
def test_a_hooks_key_is_refused_even_when_it_holds_nothing(case, hooks):
    """Any `hooks` key at all: Podman's own precreate and poststop hooks never reach config.json, so
    nothing there can be told apart, and the clean case has no such key."""

    def present(_case, document):
        document["hooks"] = hooks

    assert changed(case, present) == runtime.Refusal("hook", "-")


# ---- one change each, and the reason ----------------------------------------------------------------


def add_mount(destination, kind, source, options):
    def change(_case, document):
        document["mounts"].append(
            {"destination": destination, "type": kind, "source": source, "options": options}
        )

    return change


def remove_default(_case, document):
    document["mounts"] = [m for m in document["mounts"] if m["destination"] != "/proc"]


def proc_without_nosuid(case, document):
    case.mount(document, "/proc")["options"].remove("nosuid")


def swap_source(case, document):
    case.mount(document, "/code")["source"] = str(case.root / "binds" / "elsewhere")


def writable(case, document):
    options = case.mount(document, "/code")["options"]
    options[options.index("ro")] = "rw"


def shared(case, document):
    case.mount(document, "/work")["options"].append("rshared")


def duplicate(case, document):
    document["mounts"].append(copy.deepcopy(case.mount(document, "/proc")))


def device(_case, document):
    document["linux"]["devices"] = [
        {"path": "/dev/nemar-extra-device", "type": "c", "major": 1, "minor": 5}
    ]


def hook(_case, document):
    document["hooks"] = {"createRuntime": [{"path": "/usr/local/bin/a-hook"}]}


EXTRA = str(Path("/podman-fixture-elsewhere"))


@pytest.mark.parametrize(
    ("change", "reason", "destination"),
    [
        (add_mount("/nemar-extra", "bind", EXTRA, ["rbind", "ro"]), "extra_mount", "/nemar-extra"),
        (
            add_mount("/run", "tmpfs", "tmpfs", ["nosuid", "nodev", "rprivate", "rw"]),
            "extra_mount",
            "/run",
        ),
        (
            add_mount("/run/secrets", "bind", EXTRA, ["bind", "rprivate", "ro", "nosuid"]),
            "extra_mount",
            "/run/secrets",
        ),
        (remove_default, "missing_default", "/proc"),
        (proc_without_nosuid, "missing_default", "/proc"),
        (swap_source, "declared_bind", "/code"),
        (writable, "declared_bind", "/code"),
        (shared, "declared_bind", "/work"),
        (duplicate, "duplicate_mount", "/proc"),
        (device, "extra_device", "/dev/nemar-extra-device"),
        (hook, "hook", "-"),
    ],
    ids=[
        "mount-at-nemar-extra",
        "tmpfs-at-run",
        "bind-at-run-secrets",
        "required-default-removed",
        "proc-without-nosuid",
        "declared-source-swapped",
        "read-only-made-writable",
        "declared-rshared",
        "duplicate-destination",
        "linux-devices-entry",
        "create-runtime-hook",
    ],
)
def test_one_change_is_refused_with_its_reason(case, change, reason, destination):
    assert changed(case, change) == runtime.Refusal(reason, destination)


def test_an_oci_config_path_outside_the_static_dir_is_unreadable(case):
    inspected = dict(case.inspected, OCIConfigPath=str(case.root / "run" / "config.json"))
    assert case.check(inspected=inspected) == runtime.Refusal("config_unreadable", "-")


# ---- what a configuration Podman reads would add, as Podman 4.9.3 writes it --------------------------


def test_a_configured_device_as_rootless_podman_binds_it_is_an_extra_mount_not_a_duplicate(case):
    entry = ["/dev/nemar-extra-device", "bind", "/dev/zero", ["noexec", "nosuid", "rbind", "rw"]]

    def twice(case, document):
        add_mount(*entry)(case, document)
        add_mount(*entry)(case, document)

    assert changed(case, twice) == runtime.Refusal("extra_mount", "/dev/nemar-extra-device")


def test_a_mounts_conf_entry_copied_into_the_run_directory_is_an_extra_mount(case):
    source = str(case.root / "run" / "nemar-extra-subscription")
    options = ["bind", "nodev", "noexec", "nosuid", "ro", "rprivate"]
    change = add_mount("/nemar-extra-subscription", "bind", source, options)
    assert changed(case, change) == runtime.Refusal("extra_mount", "/nemar-extra-subscription")


def test_a_mounts_conf_entry_at_an_allowed_destination_is_refused(case):
    """A subscription is copied to the run directory plus its destination, never directly into it."""
    (case.root / "run" / "etc").mkdir()
    (case.root / "run" / "etc" / "hosts").write_text("copied")

    def replace(case, document):
        case.mount(document, "/etc/hosts")["source"] = str(case.root / "run" / "etc" / "hosts")

    assert changed(case, replace) == runtime.Refusal("extra_mount", "/etc/hosts")


def test_a_file_of_the_host_at_an_allowed_destination_is_refused(case):
    host = case.root / "host-hosts"
    host.write_text("host")

    def replace(case, document):
        case.mount(document, "/etc/hosts")["source"] = str(host)

    assert changed(case, replace) == runtime.Refusal("extra_mount", "/etc/hosts")


def test_an_own_file_that_is_a_link_out_of_the_container_directories_is_refused(case):
    outside = case.root / "outside"
    outside.write_text("host data")
    own = case.root / "run" / "hosts"
    own.unlink()
    own.symlink_to(outside)
    assert case.check() == runtime.Refusal("extra_mount", "/etc/hosts")


def test_a_link_between_the_container_directories_is_resolved_and_allowed(case):
    own = case.root / "run" / "hostname"
    own.unlink()
    own.symlink_to(case.root / "static" / "passwd")
    assert case.check() is None


def test_a_shm_that_is_not_the_containers_own_is_a_missing_default(case):
    def elsewhere(case, document):
        case.mount(document, "/dev/shm")["source"] = str(case.root / "binds" / "elsewhere")

    assert changed(case, elsewhere) == runtime.Refusal("missing_default", "/dev/shm")


def test_a_writable_tmpfs_at_dev_shm_is_a_missing_default(case):
    def tmpfs(case, document):
        shm = case.mount(document, "/dev/shm")
        shm.update(type="tmpfs", source="shm", options=["nosuid", "noexec", "nodev", "mode=1777"])

    assert changed(case, tmpfs) == runtime.Refusal("missing_default", "/dev/shm")


def test_an_own_file_with_other_options_is_an_extra_mount(case):
    def writable_hosts(case, document):
        options = case.mount(document, "/etc/hosts")["options"]
        options[options.index("ro")] = "rw"

    assert changed(case, writable_hosts) == runtime.Refusal("extra_mount", "/etc/hosts")


def test_a_declared_bind_missing_is_refused(case):
    def drop_code(_case, document):
        document["mounts"] = [m for m in document["mounts"] if m["destination"] != "/code"]

    assert changed(case, drop_code) == runtime.Refusal("declared_bind", "/code")


def test_a_declared_bind_that_is_not_a_bind_is_refused(case):
    def tmpfs_work(case, document):
        case.mount(document, "/work").update(type="tmpfs", source="tmpfs")

    assert changed(case, tmpfs_work) == runtime.Refusal("declared_bind", "/work")


def test_a_read_only_bind_that_also_says_rw_is_refused(case):
    def both(case, document):
        case.mount(document, "/code")["options"].append("rw")

    assert changed(case, both) == runtime.Refusal("declared_bind", "/code")


def test_nothing_resolved_in_the_namespace_allows_none_of_the_containers_own_paths(case):
    refusal = runtime.check_container(case.inspected, case.document, case.binds, {})
    assert refusal is not None
    assert refusal.reason in ("missing_default", "extra_mount")


@pytest.mark.parametrize(
    "document",
    [None, [], {"mounts": "x"}, {"mounts": [{"destination": 5}]}, {"mounts": [], "linux": []}],
)
def test_a_document_that_is_not_an_oci_configuration_is_unreadable(case, document):
    refusal = runtime.check_container(case.inspected, document, case.binds, {})
    assert refusal == runtime.Refusal("config_unreadable", "-")


@pytest.mark.parametrize(
    "inspected",
    [
        None,
        [],
        {"StaticDir": "/a", "OCIConfigPath": "/a/config.json"},
        {"StaticDir": "/a/../b", "OCIConfigPath": "/b/config.json", "HostsPath": "/r/hosts"},
        {"StaticDir": "a", "OCIConfigPath": "a/config.json", "HostsPath": "/r/hosts"},
    ],
)
def test_an_inspect_document_without_the_containers_directories_is_unreadable(case, inspected):
    refusal = runtime.check_container(inspected, case.document, case.binds, {})
    assert refusal == runtime.Refusal("config_unreadable", "-")


@pytest.mark.parametrize(
    "state",
    [
        {"Status": "configured"},
        {"Status": "running"},
        {"Status": "exited"},
        {"Status": "stopped"},
        {},
        None,
        "initialized",
    ],
    ids=["configured", "running", "exited", "stopped", "no-status", "no-state", "not-an-object"],
)
def test_a_container_that_is_not_initialized_is_unreadable(case, state):
    """Only an initialized container starts from the config.json that was read: `start` writes it
    again for a configured, stopped or exited one."""
    inspected = dict(case.inspected)
    if state is None:
        del inspected["State"]
    else:
        inspected["State"] = state
    assert case.check(inspected=inspected) == runtime.Refusal("config_unreadable", "-")


def test_an_inspect_answer_that_is_a_list_of_one_is_read(case):
    assert case.check(inspected=[case.inspected]) is None


def test_the_namespace_is_asked_about_the_container_directories_and_own_sources_only(case):
    paths = runtime.namespace_paths(case.inspected, case.document)
    static, run_dir = case.root / "static", case.root / "run"
    assert paths[:2] == [str(static), str(run_dir)]
    assert sorted(paths[2:]) == sorted(
        [str(static / name) for name in ("passwd", "group", "shm")]
        + [str(run_dir / name) for name in ("hosts", "hostname", "resolv.conf", ".containerenv")]
    )
