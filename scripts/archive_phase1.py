"""Bundle completed Phase 1 checkpoints and reports without source datasets.

Only named run/evaluation artifacts and PNG/PDF/CSV figures are included.
The caller supplies a new .zip output path; no training or evaluation is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
ARMS = ("conditional", "unconditional")
RUN_NAMES = tuple(f"{arm}_seed_{seed}" for arm in ARMS for seed in SEEDS)
REPORT_NAMES = (*RUN_NAMES, "mean", "resample", "copula")


def _safe_path(path, *, directory=False):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Source symlinks are not permitted: {component}")
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError(f"Expected a regular {'directory' if directory else 'file'}: {path}")
    return path


def _sha256(path):
    with _safe_path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    value = json.loads(_safe_path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _inventory(runs, evaluations, figures):
    members = {}
    def add(root, prefix, relative):
        path = _safe_path(root / relative)
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact is outside its declared root: {path}")
        member = f"{prefix}/{relative}"
        members[member] = {"path": path, "sha256": _sha256(path), "bytes": path.stat().st_size}
        return path

    def verify(path, expected, description):
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {description}")

    launch = _json(add(runs, "runs", "manifest.json"))
    source_commit = launch.get("git", {}).get("head")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)):
        raise ValueError("Training launch must record its full source Git commit")
    jobs = launch.get("jobs")
    if (launch.get("status") != "completed" or not isinstance(jobs, list) or len(jobs) != 10
            or {job.get("name") for job in jobs} != set(RUN_NAMES)
            or any(job.get("status") != "completed" or job.get("exit_code") != 0 for job in jobs)):
        raise ValueError("Training launch must contain exactly ten completed successful runs")
    launch_jobs = {job["name"]: job for job in jobs}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_seed_{seed}"
            relative = f"{arm}/seed_{seed}"
            manifest = _json(add(runs, "runs", f"{relative}/manifest.json"))
            config = manifest.get("config", {})
            if (manifest.get("status") != "completed" or config.get("seed") != seed
                    or config.get("unconditional") is not (arm == "unconditional")
                    or manifest.get("completed_epochs") != config.get("epochs")):
                raise ValueError(f"Training run is incomplete or has wrong identity: {name}")
            for filename in ("best.pt", "last.pt", "history.json"):
                path = add(runs, "runs", f"{relative}/{filename}")
                verify(path, manifest.get("artifact_hashes", {}).get(filename), f"{name}/{filename}")
            saved_config = _json(add(runs, "runs", f"configs/{name}.json"))
            if saved_config != config or launch_jobs[name].get("config") != config:
                raise ValueError(f"Launch/run configuration mismatch: {name}")
            add(runs, "runs", f"logs/{name}.log")

    completed = _json(add(evaluations, "evaluations", "completed.json"))
    for name in ("summary", "inference"):
        path = add(evaluations, "evaluations", f"{name}.json")
        verify(path, completed.get(f"{name}_sha256"), name)
        execution = _json(add(evaluations, "evaluations", f"{name}_execution.json"))
        if execution.get("returncode") != 0:
            raise ValueError(f"Evaluation {name} did not complete successfully")
    snapshot_path = add(evaluations, "evaluations", "protocol.snapshot.json")
    frozen = _json(snapshot_path)
    if frozen.get("frozen") is not True or frozen.get("phase") != 1 or frozen.get("training_seeds") != list(SEEDS):
        raise ValueError("Evaluation snapshot must be the frozen five-seed Phase 1 protocol")
    protocol_hash = completed.get("protocol_sha256")
    verify(snapshot_path, protocol_hash, "protocol snapshot")
    if launch.get("protocol_sha256") != protocol_hash:
        raise ValueError("Training and evaluation protocols differ")
    plan = _json(add(evaluations, "evaluations", "execution_plan.json"))
    provenance = plan.get("provenance", {})
    if provenance.get("protocol", {}).get("sha256") != protocol_hash:
        raise ValueError("Evaluation execution plan protocol differs")
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_seed_{seed}"
            relative = f"{arm}/seed_{seed}"
            verify(runs / relative / "best.pt", provenance.get("checkpoints", {}).get(name, {}).get("sha256"), name)
            verify(runs / relative / "manifest.json", provenance.get("run_manifests", {}).get(name, {}).get("sha256"), name)
    execution = _json(add(evaluations, "evaluations", "evaluation_execution.json"))
    records = execution.get("jobs")
    if (not isinstance(records, list) or len(records) != 13 or {row.get("name") for row in records} != set(REPORT_NAMES)
            or any(row.get("returncode") != 0 for row in records)):
        raise ValueError("All thirteen final evaluation reports must have completed")
    for record in records:
        name = record["name"]
        path = add(evaluations, "evaluations", f"reports/{name}.json")
        verify(path, record.get("report_sha256"), name)
        report = _json(path)
        if report.get("protocol_sha256") != protocol_hash or report.get("split") != "test":
            raise ValueError(f"Report protocol/split differs: {name}")
        if name in RUN_NAMES and report.get("checkpoint_sha256") != provenance["checkpoints"][name]["sha256"]:
            raise ValueError(f"Report checkpoint differs: {name}")
    for name in (*REPORT_NAMES, "summarize", "inference"):
        for stream in ("stdout", "stderr"):
            add(evaluations, "evaluations", f"logs/{name}.{stream}.log")
    for relative in ("test_access.jsonl", "reports/test_access.jsonl"):
        path = evaluations / relative
        if path.exists() or path.is_symlink():
            add(evaluations, "evaluations", relative)

    figure_paths = []
    for path in sorted(figures.iterdir()):
        if path.suffix.lower() in {".png", ".pdf", ".csv"}:
            figure_paths.append(add(figures, "figures", path.name))
    render_path = figures / "render_manifest.json"
    if render_path.exists() or render_path.is_symlink():
        render = _json(add(figures, "figures", "render_manifest.json"))
        outputs = render.get("output_sha256", {})
        if not isinstance(outputs, dict) or set(outputs) != {path.name for path in figure_paths}:
            raise ValueError("Renderer manifest output inventory differs from allowed figures")
        for path in figure_paths:
            verify(path, outputs[path.name], path.name)
        sources = render.get("source_json_sha256")
        if not isinstance(sources, dict):
            raise ValueError("Renderer manifest must declare source JSON hashes")
        known = {str(value["path"]): value["sha256"] for value in members.values()}
        for path, digest in sources.items():
            # Compare inventory strings only: never dereference manifest paths.
            if known.get(str(Path(os.path.abspath(path)))) != digest:
                raise ValueError(f"Renderer source is outside the verified inventory or changed: {path}")
    return members, launch


def archive(runs, evaluations, figures, output):
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Archive output must be new: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Archive output must have a .zip suffix")
    roots = [_safe_path(path, directory=True) for path in (runs, evaluations, figures)]
    if len(set(roots)) != 3:
        raise ValueError("Runs, evaluations, and figures must be separate directories")
    members, launch = _inventory(*roots)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    bundle_manifest = {
        "schema_version": 1, "phase": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head, "training_source_commit": launch.get("git", {}).get("head"),
        "training_source_py_sha256": launch.get("source_py_sha256", {}),
        "protocol_sha256": launch["protocol_sha256"],
        "members": {name: {key: value[key] for key in ("sha256", "bytes")} for name, value in sorted(members.items())},
        "scope": "Completed checkpoints, configs, logs, saved reports, protocol snapshot and figures; source datasets excluded",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}-", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, value in sorted(members.items()):
                digest, size = hashlib.sha256(), 0
                source_path = _safe_path(value["path"])
                with source_path.open("rb") as source, bundle.open(name, "w", force_zip64=True) as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
                        digest.update(block)
                        size += len(block)
                if digest.hexdigest() != value["sha256"] or size != value["bytes"]:
                    raise ValueError(f"Artifact changed while archiving: {name}")
            bundle.writestr("bundle_manifest.json", json.dumps(bundle_manifest, indent=2, allow_nan=False) + "\n")
        with zipfile.ZipFile(temporary) as bundle:
            bad = bundle.testzip()
            if bad is not None:
                raise ValueError(f"Archive CRC verification failed: {bad}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another output appeared.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {output} ({len(members)} artifacts)")
    return bundle_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "evaluations", "figures", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        archive(args.runs, args.evaluations, args.figures, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"archive_phase1: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
