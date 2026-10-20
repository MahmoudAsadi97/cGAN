import hashlib
import zipfile
import pytest
from fluvius_cgan.release import audit_archive, verify_archive, download_release


def test_digest_mismatch_is_rejected(tmp_path):
    p = tmp_path / 'raw.zip'
    p.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='SHA-256'):
        verify_archive(p, '0' * 64)


def test_audit_reports_schema_without_extracting(tmp_path):
    p = tmp_path / 'raw.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('dir/profile.csv', 'EAN_ID,Datum\n1,2024-01-01\n')
        z.writestr('../escape.txt', 'never extract me')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert verify_archive(p, digest) == digest
    report = audit_archive(p)
    assert report['members'][0]['columns'] == ['EAN_ID', 'Datum']
    assert report['source_sha256'] == digest
    assert not (tmp_path.parent / 'escape.txt').exists()


def test_existing_download_is_verified_before_reuse(tmp_path):
    p = tmp_path / 'raw.zip'
    p.write_bytes(b'not the published archive')
    with pytest.raises(ValueError, match='SHA-256'):
        download_release(p)
