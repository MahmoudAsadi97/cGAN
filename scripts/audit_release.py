"""Inspect a ZIP release without extracting it or loading entire CSVs."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


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
            elif member.filename.lower().endswith(".xlsx"):
                with zipfile.ZipFile(io.BytesIO(archive.read(member))) as book:
                    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    strings = []
                    if "xl/sharedStrings.xml" in book.namelist():
                        root = ET.fromstring(book.read("xl/sharedStrings.xml"))
                        strings = ["".join(t.itertext()) for t in root.findall("x:si", ns)]
                    sheets = {}
                    for name in book.namelist():
                        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                            rows = []
                            for row in ET.fromstring(book.read(name)).findall(".//x:row", ns):
                                values = []
                                for cell in row.findall("x:c", ns):
                                    v = cell.find("x:v", ns)
                                    value = v.text if v is not None else "".join(cell.itertext())
                                    values.append(strings[int(value)] if cell.get("t") == "s" else value)
                                rows.append(values)
                            sheets[name] = rows
                    item["sheets"] = sheets
            elif member.filename.lower().endswith((".json", ".txt", ".md", ".log")) and member.file_size < 100000:
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
