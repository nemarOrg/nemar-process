"""The standalone entry's command line: a line that cannot be used exits 2 and creates nothing.
Python 3.9 compatible."""

from __future__ import annotations

import pytest
from local_support import LocalRig


@pytest.fixture
def lrig(tmp_path):
    made = LocalRig(tmp_path)
    made.write_spec(made.script_spec("true"))
    yield made
    made.close()


def bad_lines(lrig):
    base = ["--spec", str(lrig.spec_path), "--out", str(lrig.out)]
    sif = ["--sif-root", str(lrig.sif_root)]
    return {
        "no_spec": ["--out", str(lrig.out), *sif],
        "no_out": ["--spec", str(lrig.spec_path), *sif],
        "no_sif_root": base,
        "docker_without_limits": [*base, "--engine", "docker"],
        "docker_without_memory": [*base, "--engine", "docker", "--cpus", "1"],
        "podman": [*base, "--engine", "podman"],
        "code_not_tar_gz": [*base, *sif, "--code", "code.zip"],
        "data_without_dir": [*base, *sif, "--data", "ds1"],
        "data_bad_name": [*base, *sif, "--data", "../x=/tmp"],
        "data_twice": [*base, *sif, "--data", "a=/x", "--data", "a=/y"],
        "zero_cpus": [*base, *sif, "--cpus", "0"],
        "memory_without_cpus": [*base, *sif, "--memory-mb", "64"],
        "fraction_limit": [*base, *sif, "--time-limit-s", "1.5"],
        "registry_without_docker": [*base, *sif, "--allow-registry", "ghcr.io"],
        "registry_not_a_host": [
            *base,
            "--engine",
            "docker",
            "--cpus",
            "1",
            "--memory-mb",
            "64",
            "--allow-registry",
            "a/b",
        ],
        "unknown_flag": [*base, *sif, "--job-token", "x"],
    }


@pytest.mark.parametrize(
    "name",
    [
        "no_spec",
        "no_out",
        "no_sif_root",
        "docker_without_limits",
        "docker_without_memory",
        "podman",
        "code_not_tar_gz",
        "data_without_dir",
        "data_bad_name",
        "data_twice",
        "zero_cpus",
        "memory_without_cpus",
        "fraction_limit",
        "registry_without_docker",
        "registry_not_a_host",
        "unknown_flag",
    ],
)
def test_a_line_that_cannot_be_used_exits_2_and_creates_nothing(lrig, name):
    launch = lrig.run(bad_lines(lrig)[name])
    assert launch.exit == 2
    assert not lrig.out.exists()
    assert launch.stderr
    assert lrig.calls() == []


def test_help_exits_0(lrig, capsys):
    from nemar_launch import local

    assert local.main(["--help"], lrig.environ()) == 0
