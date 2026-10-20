"""Inspect a ZIP release without extracting it or loading entire CSVs."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile


def audit(path):
    report = {"archive": Path(path).name, "members": []}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            item = {"name": member.filename, "bytes": member.file_size}
            if member.filename.lower().endswith(".csv"):
                with archive.open(member) as source:
                    stream = io.TextIOWrapper(source, encoding="utf-8-sig", errors="strict", newline="")
                    sample = stream.read(32768)
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    rows = list(csv.reader(io.StringIO(sample), dialect))
                    item.update(delimiter=dialect.delimiter, columns=rows[0], example_rows=rows[1:4])
            elif member.filename.lower().endswith((".json", ".txt", ".md")) and member.file_size < 100000:
                item["text"] = archive.read(member).decode("utf-8-sig")
            report["members"].append(item)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", default="dataset_audit.json")
    args = parser.parse_args()
    result = audit(args.archive)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
