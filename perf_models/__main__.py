"""Module entry point: `python -m perf_models <subcommand>`."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
