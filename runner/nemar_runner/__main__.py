"""The command line: `nemar-runner doctor | once | run | install | self-update`.

Run it as `python3 -m nemar_runner <command>` with `runner/` on the path. Exit status: 0 done, 1 a
failure, 2 a configuration it will not start with, 3 NEMAR will not serve this runner (revoked,
outdated, wrong protocol). `install` and `self-update` (install.py) put a signed release under a prefix.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from . import __version__, doctor, install
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
    config_option(commands.add_parser("once", help="claim once, run what was claimed, and exit"))
    config_option(commands.add_parser("run", help="claim jobs until stopped"))
    install.add_commands(commands, config_option)
    return parser


def config_option(parser: argparse.ArgumentParser) -> None:
    """`--config` after the command as well as before it: the service unit writes it after (`boot.py
    run --config <file>`). Absent here it is left as the top-level option set it."""
    parser.add_argument("--config", default=argparse.SUPPRESS, help="the configuration file")


def main(argv: list[str] | None = None, tuning: Tuning | None = None) -> int:
    """Run a command. ``tuning`` is for tests: the numbers (and the launch PATH) of a run are fixed by the
    card, and nothing in the configuration file or the environment can change them."""
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    path = args.config or default_config_path()
    if args.command == "doctor":
        return doctor.run(path, sys.stdout, tuning)
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
