"""Evaluate one sealed unified TEST study with no fitting or generation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.frozen_reference_study import evaluate_reference_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_reference_study(args.bundle, args.expected_seal_sha256, args.output)
    print(str(args.output.resolve() / "report.json"))


if __name__ == "__main__":
    main()
