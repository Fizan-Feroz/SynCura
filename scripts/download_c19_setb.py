"""Download PhysioNet Challenge 2019 (Sepsis) training_setB into PROJ/data/challenge-2019-1.0.0/.

Downloads directly from PhysioNet AWS open repository (unsigned S3):
  s3://physionet-open/challenge-2019/1.0.0/training/training_setB/
Target directory:
  PROJ/data/challenge-2019-1.0.0/training/training_setB/

Usage:
  python scripts/download_c19_setb.py --smoke          # Download 100 files to verify pipeline
  python scripts/download_c19_setb.py                  # Download full 20,000 files (~100 MB)
"""
import os
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import boto3
import botocore
from ml.paths import data_root

BUCKET = 'physionet-open'
PREFIX = 'challenge-2019/1.0.0/training/training_setB/'
DEST_DIR = os.path.join(data_root(), 'challenge-2019-1.0.0', 'training', 'training_setB')


def get_s3_client():
    return boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))


def list_s3_files(s3, max_files=None):
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)
    keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.psv'):
                keys.append(key)
                if max_files and len(keys) >= max_files:
                    return keys
    return keys


def download_one(s3, key, dest_dir):
    fname = os.path.basename(key)
    target = os.path.join(dest_dir, fname)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return True
    try:
        s3.download_file(BUCKET, key, target)
        return True
    except Exception as e:
        print(f"Error downloading {fname}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Download PhysioNet Challenge 2019 Set B into project folder.')
    parser.add_argument('--smoke', action='store_true', help='Download first 100 files as smoke test')
    parser.add_argument('--max-files', type=int, default=None, help='Limit number of files to download')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel download threads')
    args = parser.parse_args()

    max_files = 100 if args.smoke else args.max_files

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f'Target dataset directory: {DEST_DIR}')
    print('Connecting to PhysioNet AWS open repository (s3://physionet-open)...')
    s3 = get_s3_client()

    print('Listing files in training_setB...')
    keys = list_s3_files(s3, max_files=max_files)
    print(f'Found {len(keys)} files to download.')

    # Filter out already existing files
    needed = [k for k in keys if not (os.path.exists(os.path.join(DEST_DIR, os.path.basename(k))) and os.path.getsize(os.path.join(DEST_DIR, os.path.basename(k))) > 0)]
    print(f'{len(keys) - len(needed)} files already downloaded. {len(needed)} files remaining.')

    if needed:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            thread_s3 = [get_s3_client() for _ in range(args.workers)]
            futures = {
                executor.submit(download_one, thread_s3[i % args.workers], k, DEST_DIR): k
                for i, k in enumerate(needed)
            }
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(needed), desc='Downloading Set B'):
                pass

    downloaded = [f for f in os.listdir(DEST_DIR) if f.endswith('.psv')]
    print(f'\nTotal files in {DEST_DIR}: {len(downloaded)}')
    print('Download completed successfully.')


if __name__ == '__main__':
    main()
