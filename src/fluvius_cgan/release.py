"""Pinned source retrieval and archive inspection; archives are never extracted."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

RELEASE_URL = 'https://github.com/MahmoudAsadi97/cGAN/releases/download/data/raw.zip'
RELEASE_SHA256 = '934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8'
RELEASE_BYTES = 395085174


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: str | Path, expected_sha256: str = RELEASE_SHA256) -> str:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f'SHA-256 mismatch for {Path(path).name}: expected {expected_sha256}, got {digest}')
    return digest


def download_release(destination: str | Path) -> Path:
    """Download the exact published asset atomically, or verify an existing copy."""
    destination = Path(destination)
    if destination.exists():
        verify_archive(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    if partial.exists():
        raise FileExistsError(f'Partial download exists: {partial}; remove it before retrying')
    request = urllib.request.Request(RELEASE_URL, headers={'User-Agent': 'fluvius-cgan/0.2'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as out:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                out.write(block)
        if partial.stat().st_size != RELEASE_BYTES:
            raise ValueError('Downloaded size does not match the pinned release')
        verify_archive(partial)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def audit_archive(path: str | Path) -> dict:
    """Report member sizes and CSV headers, without printing household measurements."""
    path = Path(path)
    report = {'archive': path.name, 'bytes': path.stat().st_size,
              'source_sha256': sha256_file(path), 'members': []}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            item = {'name': info.filename, 'bytes': info.file_size,
                    'compressed_bytes': info.compress_size}
            if info.filename.lower().endswith('.csv'):
                with archive.open(info) as binary:
                    text = io.TextIOWrapper(binary, encoding='utf-8-sig', newline='')
                    header = text.readline(65536)
                    if not header.endswith(('\n', '\r')):
                        raise ValueError(f'Missing or oversized CSV header: {info.filename}')
                    item['columns'] = next(csv.reader([header]))
            report['members'].append(item)
    return report
