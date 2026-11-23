# The tiny image the runner-stack end to end test runs under real Apptainer (tests/e2e/test_runner_stack.py).
#
# It is a WORKLOAD, not a stand-in for anything of NEMAR: the real launcher starts it, and it does what a
# recipe does at the container boundary (takes the recipe's arguments, writes a product file under --out,
# exits 0) with the standard library alone, so the test spends no time on a recipe's own dependencies.
#
# Build from the repository root:
#   docker build -t tiny:ci tests/images/tiny-recipe
#
# python:3.12-slim (Python 3.12.16, Debian trixie), image index as of 2026-11-04: the same digest as
# images/nemar-python/Dockerfile.
FROM docker.io/library/python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

COPY --chmod=0755 nemar-recipe /usr/local/bin/nemar-recipe
