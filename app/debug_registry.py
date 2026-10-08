"""Canonical scan registry. Preview/alias files never become extra scans."""
import json
from pathlib import Path


def read_registry(root: Path) -> list[dict]:
    path = root / 'sweeps/scan_registry.json'
    if path.stat().st_size > 256_000:
        raise ValueError('Registry exceeds size bound')
    scans = json.loads(path.read_text())['scans']
    labels = [s.get('label') for s in scans]
    shas = [s.get('sha256') for s in scans]
    if not scans or not all(labels) or not all(shas) or len(set(labels)) != len(labels) or len(set(shas)) != len(shas):
        raise ValueError('Registry needs at least one scan with unique non-empty labels and sha256')
    names = {n for s in scans for n in [s.get('file'), s.get('browser_preview'), *(s.get('aliases') or [])] if n}
    if names & set(labels):   # aliases/files/previews are fields of a scan, never extra scans
        raise ValueError('A scan label equals another scan\'s file, preview or alias')
    for scan in scans:
        for key in ('file', 'browser_preview'):
            name = scan.get(key)
            if name and (Path(name).name != name or not (root/'sweeps'/name).resolve().is_relative_to((root/'sweeps').resolve())):
                raise ValueError('Invalid registry file')
    return sorted(scans, key=lambda s: s['label'])
