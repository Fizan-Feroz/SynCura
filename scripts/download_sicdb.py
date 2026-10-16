"""Download SICdb (Salzburg Intensive Care database) v1.0.8 from PhysioNet.

BLOCKED until a human completes access approval (agents cannot do this):
  1. Create a PhysioNet account and become a credentialed user
     (https://physionet.org/settings/credentialing/ -- identity + training).
  2. Complete CITI "Data or Specimens Only Research" and submit it at
     https://physionet.org/settings/training/.
  3. Request access from the SICdb authors with a research question at
     https://physionet.org/content/sicdb/1.0.8/ and sign the DUA.

Once approved, download into data/ (gitignored) with Basic-Auth credentials
from the environment (never hardcoded, never logged):

    set PHYSIONET_USER=you@example.com
    set PHYSIONET_PASSWORD=********
    python scripts/download_sicdb.py --list-only     # show files + sizes first
    python scripts/download_sicdb.py                 # download + verify

Stdlib only (urllib) so it runs in any venv. Skips files already present
with matching size, so interrupted downloads resume by re-running.
"""
import argparse
import os
import re
import sys
import urllib.request

BASE = 'https://physionet.org/files/sicdb/1.0.8/'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, 'data', 'sicdb-1.0.8')


def opener():
    user = os.environ.get('PHYSIONET_USER')
    password = os.environ.get('PHYSIONET_PASSWORD')
    if not user or not password:
        print('ERROR: set PHYSIONET_USER and PHYSIONET_PASSWORD first.', file=sys.stderr)
        sys.exit(1)
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, 'https://physionet.org', user, password)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(mgr))


def list_dir(op, url, depth=0):
    """Return [(name, url, is_dir, size_or_None)] for one listing level."""
    with op.open(url, timeout=60) as resp:
        html = resp.read().decode('utf-8', 'replace')
    out = []
    for href in re.findall(r'href="([^"?#]+)"', html):
        if href in ('/', '../', './') or href.startswith(('?', '#', 'http', '//')):
            continue
        full = url + href if not href.startswith('/') else 'https://physionet.org' + href
        if href.endswith('/'):
            if depth < 2:
                out.extend(list_dir(op, full, depth + 1))
        else:
            out.append((href.rstrip('/').split('/')[-1], full, False, None))
    return out


def remote_size(op, url):
    req = urllib.request.Request(url, method='HEAD')
    try:
        with op.open(req, timeout=60) as resp:
            size = resp.headers.get('Content-Length')
            return int(size) if size else None
    except Exception:
        return None


def download(op, url, dest):
    tmp = dest + '.part'
    req = urllib.request.Request(url)
    with op.open(req, timeout=300) as resp, open(tmp, 'wb') as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description='Download SICdb v1.0.8 from PhysioNet.')
    ap.add_argument('--list-only', action='store_true', help='list files + sizes, download nothing')
    ap.add_argument('--dest', default=DEST, help='download directory')
    args = ap.parse_args()

    op = opener()
    print(f'listing {BASE}')
    try:
        files = list_dir(op, BASE)
    except Exception as exc:
        print(f'ERROR: cannot list files ({exc}). Access likely not approved yet -- '
              'see the docstring steps at the top of this script.', file=sys.stderr)
        return 1
    if not files:
        print('ERROR: empty listing. Access likely not approved yet.', file=sys.stderr)
        return 1

    total, known = 0, 0
    rows = []
    for name, url, _, _ in files:
        size = remote_size(op, url)
        rows.append((name, size))
        if size:
            total += size
            known += 1
    for name, size in rows:
        print(f'  {name}  ({size / 1e9:.2f} GB)' if size else f'  {name}  (size unknown)')
    print(f'{len(rows)} files, {total / 1e9:.1f} GB across {known} sized files. Dest: {args.dest}')
    if args.list_only:
        return 0

    os.makedirs(args.dest, exist_ok=True)
    bad = []
    for name, url, _, _ in files:
        dest = os.path.join(args.dest, name)
        size = remote_size(op, url)
        if os.path.isfile(dest) and size and os.path.getsize(dest) == size:
            print(f'skip (present): {name}')
            continue
        print(f'downloading: {name}')
        try:
            download(op, url, dest)
        except Exception as exc:
            print(f'FAILED {name}: {exc}', file=sys.stderr)
            bad.append(name)
            continue
        if os.path.getsize(dest) == 0:
            print(f'FAILED {name}: empty file', file=sys.stderr)
            bad.append(name)
    if bad:
        print(f'WARNING: {len(bad)} files failed: {bad}', file=sys.stderr)
        return 1
    print('OK: SICdb download complete. Next: ETL into the 12-feature window format '
          '(see docs/DATASET_SCHEMAS.md); minute data needs the sicdb unpacking script.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
