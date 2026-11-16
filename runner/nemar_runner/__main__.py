"""The command line: `nemar-runner doctor | availability | gpu-check | once | run | install | self-update`.

Run it as `python3 -m nemar_runner <command>` with `runner/` on the path. Exit status: 0 done, 1 a
failure, 2 a configuration it will not start with, 3 NEMAR will not serve this runner (revoked,
outdated, wrong protocol). `install` and `self-update` (install.py) put a signed release under a prefix.
`availability` prints whether this host has room for a claim now and exits 0 (idle) or 1 (busy).
`gpu-check` takes the GPU idle check once, whatever `gpu.enabled` says, and prints a line per GPU: exit 0
when a GPU is idle and 1 when none is or the check could not be made.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from . import __version__, availability, doctor, gpu, install
from .agent import EXIT_CONFIG, Agent
from .config import ConfigError, Tuning, default_config_path, load_config, read_secret
from .logs import Log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemar-runner", description="NEMAR's runner agent.")
    parser.add_argument("--version", action="version", version="nemar-runner/" + __version__)
    parser.add_argument(
        "--config",
        help="the configuration file (default: $XDG_CONFIG_HOME/nemar-runner/config.json)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    config_option(commands.add_parser("doctor", help="check this host, one yes or no a line"))
    config_option(
        commands.add_parser("availability", help="say whether this host has room for a claim now")
    )
    config_option(
        commands.add_parser("gpu-check", help="take the GPU idle check once and say what it found")
    )
    config_option(commands.add_parser("once", help="claim once, run what was claimed, and exit"))
    config_option(commands.add_parser("run", help="claim jobs until stopped"))
    install.add_commands(commands, config_option)
    return parser


def config_option(parser: argparse.ArgumentParser) -> None:
    """`--config` after the command as well as before it: the service unit writes it after (`boot.py
    run --config <file>`). Absent here it is left as the top-level option set it."""
    parser.add_argument("--config", default=argparse.SUPPRESS, help="the configuration file")


def availability_command(path: str, tuning: Tuning | None) -> int:
    """One line, `availability: idle ...` or `availability: busy reason=<word> ...`, and the exit status
    0 or 1. The numbers are the readings it was judged on: no secret and no path."""
    try:
        config = load_config(path)
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG
    verdict = availability.Gate(config, tuning or Tuning()).claim_check(config.slots)
    sys.stdout.write(verdict.line() + "\n")
    return 0 if verdict.idle else 1


def _sleep(seconds: float) -> bool:
    """Sleep, and say the check is not being stopped (this command is stopped by its person's Ctrl-C)."""
    time.sleep(seconds)
    return False


def gpu_check_command(path: str, tuning: Tuning | None) -> int:
    """The GPU idle check once, with every sample taken (a decision stops early for a busy GPU; this does
    not), a line per GPU (`gpu<i>: idle ...` or `gpu<i>: busy reason=<word> ...`, then the sizes and the
    counts per sample: no process id, no user name, no path) and the exit status 0 when a GPU is idle and
    1 when none is. When the check cannot be made the line is `gpu: unavailable reason=gpu_check_failed
    why=<word>`. It runs whether or not `gpu.enabled` is true, so that a host can look before it turns the
    check on; a note on standard error says so."""
    try:
        config = load_config(path)
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG
    tuning = tuning or Tuning()
    if not config.gpu.enabled:
        sys.stderr.write("nemar-runner: gpu is not enabled: no job is given a GPU\n")
    pool = gpu.Pool(config, tuning)
    try:
        reports = pool.window(_sleep, float("inf"), settle=False)
    except gpu.CheckFailed as failed:
        sys.stdout.write("gpu: unavailable reason={} why={}\n".format(gpu.FAILED, failed.why))
        return 1
    for line in gpu.lines(reports):
        sys.stdout.write(line + "\n")
    return 0 if any(report.reason is None for report in reports) else 1


def main(argv: list[str] | None = None, tuning: Tuning | None = None) -> int:
    """Run a command. ``tuning`` is for tests: the numbers (and the launch PATH) of a run are fixed by the
    card, and nothing in the configuration file or the environment can change them."""
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    path = args.config or default_config_path()
    if args.command == "doctor":
        return doctor.run(path, sys.stdout, tuning)
    if args.command == "availability":
        return availability_command(path, tuning)
    if args.command == "gpu-check":
        return gpu_check_command(path, tuning)
    if args.command in ("install", "self-update"):
        return install.command(args, path, sys.stdout, sys.stderr, tuning=tuning)
    try:
        config = load_config(path)
        secret = read_secret(config)
        log = Log(os.path.join(config.root, "runner.log"))
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG
    agent = Agent(config, secret, log, tuning)
    try:
        agent.prepare()
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG

    def stopping(_signum: int, _frame: object) -> None:
        agent.stop.set()

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    signal.signal(signal.SIGHUP, stopping)  # a closed terminal or session stops it like SIGTERM
    try:
        return agent.run_once() if args.command == "once" else agent.run_forever()
    finally:
        agent.release()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
