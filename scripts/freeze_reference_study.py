"""Seal the complete candidate/reference study without reading or hashing TEST."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.frozen_reference_study import freeze_reference_study
from fluvius_cgan.frozen_test import _forbid_test_inputs, _json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared96", required=True, type=Path)
    parser.add_argument("--prepared24", required=True, type=Path)
    parser.add_argument("--specification", required=True, type=Path, help="JSON roster, evidence paths, protocols and decision; see FROZEN_REFERENCE_STUDY.md")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.specification.name == "test.npz" or args.specification.resolve().name == "test.npz":
        raise ValueError("specification names a TEST archive")
    for prepared in (args.prepared96, args.prepared24):
        _forbid_test_inputs(prepared.resolve(), [args.specification])
    spec = _json(args.specification)
    result = freeze_reference_study(args.prepared96, args.prepared24, spec["arms"], args.output,
        candidate_kind=spec["candidate_kind"], decision_rationale=spec["decision_rationale"],
        protocol_files=spec["protocol_files"], development_evidence=spec["development_evidence"],
        attempt_history=[*spec["attempt_history"], args.specification])
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
