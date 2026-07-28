"""``atlas-quant`` console-script entry point."""

from __future__ import annotations

import sys
from typing import Sequence

from atlas_quant.cli.filing_momentum import main as _filing_momentum_main


def main(argv: Sequence[str] | None = None) -> int:
    return _filing_momentum_main(argv)


if __name__ == "__main__":
    sys.exit(main())
