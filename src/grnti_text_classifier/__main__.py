"""CLI entrypoint: python -m grnti_text_classifier"""

from __future__ import annotations

import sys


def main() -> int:
    print("grnti-text-classifier — use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
