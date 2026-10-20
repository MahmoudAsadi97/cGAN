"""Score an immutable sealed TEST study once, with no fitting or generation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.frozen_test import evaluate_frozen_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_frozen_test(args.bundle, args.expected_seal_sha256, args.output)
    print(str(args.output.resolve() / "report.json"))


if __name__ == "__main__":
    main()
