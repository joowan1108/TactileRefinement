"""Download the pinned Isaac 4.5 task release; reuse only SHA256-verified local files."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = (ROOT / 'dataset_revision.txt').read_text().strip()
DEST = ROOT / 'data/UniVTAC'
EXISTING = Path('/tactile/contrast/data/UniVTAC')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    url = f'https://huggingface.co/api/datasets/byml/UniVTAC/tree/{REVISION}?recursive=true&limit=1000'
    entries = []
    while url:
        with urllib.request.urlopen(url) as r:
            entries.extend(json.load(r))
            url = next((part.split(';')[0].strip().strip('<>') for part in r.headers.get('Link', '').split(',') if 'rel="next"' in part), None)
    files = [x for x in entries if x['type'] == 'file' and x['path'].startswith('isaac45/')]
    (ROOT / 'configs/dataset_inventory.json').write_text(json.dumps(files, indent=2))

    def fetch(item):
        relative = Path(item['path'])
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('oid')
        candidate = EXISTING / relative.parts[1] / 'clean' / relative.name
        source = 'download'
        if target.exists() and target.stat().st_size == item['size'] and (not expected or digest(target) == expected):
            source = 'verified_existing'
        elif candidate.exists() and candidate.stat().st_size == item['size'] and expected and digest(candidate) == expected:
            if target.exists():
                raise RuntimeError(f'Refusing to replace an existing mismatched file: {target}')
            os.link(candidate, target)
            source = 'verified_local_hardlink'
        else:
            if shutil.disk_usage(DEST).free < item['size'] + 25 * 2**30:
                raise RuntimeError('Insufficient disk space; reserving 25 GiB for environments and checkpoints')
            temp = target.with_suffix(target.suffix + '.download')
            remote = f'https://huggingface.co/datasets/byml/UniVTAC/resolve/{REVISION}/{item["path"]}'
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(remote, timeout=120) as r, temp.open('wb') as f:
                        shutil.copyfileobj(r, f, 8 << 20)
                    assert temp.stat().st_size == item['size']
                    assert not expected or digest(temp) == expected
                    temp.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
        return {'path': item['path'], 'bytes': item['size'], 'sha256': expected or digest(target), 'source': source}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch, files):
            records.append(record)
            if len(records) % 25 == 0:
                print(f'Verified {len(records)}/{len(files)}', flush=True)
    assert sum(r['path'].endswith('.hdf5') for r in records) == 800
    report = {'repo': 'byml/UniVTAC', 'revision': REVISION, 'version': 'isaac45', 'files': records, 'total_bytes': sum(r['bytes'] for r in records)}
    (ROOT / 'data/download_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete': True, 'episodes': 800, 'bytes': report['total_bytes'], 'downloaded_files': sum(r['source'] == 'download' for r in records)}), flush=True)


if __name__ == '__main__':
    main()
