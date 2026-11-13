"""The command line: `nemar-runner doctor | once | run`.

Run it as `python3 -m nemar_runner <command>` with `runner/` on the path. Exit status: 0 done, 1 a
failure, 2 a configuration it will not start with, 3 NEMAR will not serve this runner (revoked,
outdated, wrong protocol).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from . import __version__, doctor
from .agent import EXIT_CONFIG, Agent
from .config import ConfigError, default_config_path, load_config, read_secret
from .logs import Log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemar-runner", description="NEMAR's runner agent.")
    parser.add_argument("--version", action="version", version="nemar-runner/" + __version__)
    parser.add_argument(
        "--config",
        help="the configuration file (default: $XDG_CONFIG_HOME/nemar-runner/config.json)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check this host, one yes or no a line")
    commands.add_parser("once", help="claim once, run what was claimed, and exit")
    commands.add_parser("run", help="claim jobs until stopped")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    path = args.config or default_config_path()
    if args.command == "doctor":
        return doctor.run(path, sys.stdout)
    try:
        config = load_config(path)
        secret = read_secret(config)
        log = Log(os.path.join(config.root, "runner.log"))
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG
    agent = Agent(config, secret, log)
    try:
        agent.prepare()
    except ConfigError as error:
        sys.stderr.write("nemar-runner: {}\n".format(error))
        return EXIT_CONFIG

    def stopping(_signum: int, _frame: object) -> None:
        agent.stop.set()

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        return agent.run_once() if args.command == "once" else agent.run_forever()
    finally:
        agent.release()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
