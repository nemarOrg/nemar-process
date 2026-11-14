"""nemar-launch: the launcher NEMAR runs on a compute node as a Tapis ZIP-runtime application.

Standard library only, Python 3.9 or newer, so it runs under whatever Python the host has. It reads
one launch specification from the environment, fetches a code bundle, stages datasets, runs the
entry in a container isolated from the host, and leaves provenance, a result manifest and one fixed
failure code in a directory the container never had mounted, then uploads outputs and those files
through NEMAR's outputs routes. Nothing the container writes is trusted.
"""

__version__ = "0.1.0"
