"""Fetch VitalDB datasets and clinical outcomes into PROJ/data/vitaldb/.

Downloads:
- cases.csv (6,388 cases, 74 clinical parameters + death_inhosp, icu_days, demographics)
- labs.csv (long-format laboratory results: BUN, Creatinine, Glucose, WBC, Platelets, etc.)
- trks.csv (registry of all available high-frequency and numeric vital tracks)

Usage:
    python scripts/download_vitaldb.py
"""
import os
import sys

# Ensure PROJ root is in python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gzip
import urllib.request
import pandas as pd

from ml.paths import data_root

VITALDB_DIR = os.path.join(data_root(), 'vitaldb')
BASE_API = 'https://api.vitaldb.net'

FILES = [
    ('cases.csv', f'{BASE_API}/cases'),
    ('labs.csv', f'{BASE_API}/labs'),
    ('trks.csv', f'{BASE_API}/trks'),
]


def download_file(filename, url):
    dest = os.path.join(VITALDB_DIR, filename)
    print(f'Downloading {filename} from {url}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
        # If gzipped, decompress
        if content[:2] == b'\x1f\x8b':
            content = gzip.decompress(content)
        with open(dest, 'wb') as f:
            f.write(content)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f'Saved decompressed {filename} to {dest} ({size_mb:.2f} MB)')
    return dest


def main():
    os.makedirs(VITALDB_DIR, exist_ok=True)
    print(f'Storing VitalDB datasets in project directory: {VITALDB_DIR}')
    for fname, url in FILES:
        dest = os.path.join(VITALDB_DIR, fname)
        # Always redownload or verify clean read
        try:
            pd.read_csv(dest, nrows=5)
            print(f'File {fname} is valid ({os.path.getsize(dest) / (1024*1024):.2f} MB), skipping.')
        except Exception:
            download_file(fname, url)

    # Verification
    cases_path = os.path.join(VITALDB_DIR, 'cases.csv')
    df_cases = pd.read_csv(cases_path)
    print(f'\n[VitalDB Verification]')
    print(f'Total cases: {len(df_cases)}')
    if 'death_inhosp' in df_cases.columns:
        deaths = df_cases['death_inhosp'].sum()
        print(f'In-hospital mortality: {deaths}/{len(df_cases)} ({deaths/len(df_cases)*100:.2f}%)')
    if 'icu_days' in df_cases.columns:
        icu_cases = (df_cases['icu_days'] > 0).sum()
        print(f'ICU-admitted cases: {icu_cases}/{len(df_cases)}')
    print('\nVitalDB tables downloaded successfully.')


if __name__ == '__main__':
    main()
