# -*- coding: utf-8 -*-
"""Round-2 autonomous campaign: target LOCKED holdout AUC >= 0.86.

Eligibility rule (v5 hard rule): only seeds trained AFTER both locked
slices existed (i.e. with the multi-lock exclusion active) may enter the
final ensemble. That excludes seeds 11/12/13 — their role is reference
only. Fresh seeds: baseline 14,15 (E1b) + gap 21-23 (E2a) + gap128 24,25
+ ablations + fusion.

Phases:
  B0  carve locked-B (seed 777, disjoint from locked-A)
  B1  fresh baseline 14,15 (tag E1b)
  B2  gap 21,22,23 (tag E2a)
  B3  gap128 24,25, gated on gap val >= baseline val - 0.002 (tag E2b)
  B4  greedy ensemble over eligible dirs -> candidate
  B5  ablations if best working < 0.869 (proxy: locked reads ~0.01 below
      working, so 0.869 working is the on-track bar for 0.86 locked)
  B6  fusion 41,42 (gapfuse) if still short, then re-ensemble
  B7  SINGLE eval_locked on locked-B -> victory iff >= 0.86

Always writes ml/training_runs/ROUND2_RESULTS.md.
"""
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_LOCKED = 0.86
WORKING_BAR = 0.869
LOCK = os.path.join(REPO, 'ml', 'training_runs', '.campaign.lock')
LOCK_B = os.path.join(REPO, 'ml', 'locked_holdout_B.json')
PHASE_TIMEOUT = 6 * 3600


def log(msg):
    print(f'[R2 {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def wait_for_lock():
    while os.path.exists(LOCK):
        log('waiting for in-flight campaign...')
        time.sleep(60)


def run_phase(args, timeout=PHASE_TIMEOUT):
    log('RUN: python -u -m ml.improve_results ' + ' '.join(args))
    try:
        subprocess.run([sys.executable, '-u', '-m', 'ml.improve_results'] + args,
                       cwd=REPO, timeout=timeout)
    except subprocess.TimeoutExpired:
        log('phase timed out; continuing with whatever finished')


def edirs(*tags):
    out = []
    for t in tags:
        out += sorted(glob.glob(os.path.join(REPO, 'ml', 'training_runs', f'improve_*_{t}')))
    return [d for d in out if os.path.isdir(d)]


def scan_seeds(dirs):
    rows = []
    for d in dirs:
        for s in sorted(os.listdir(d)):
            mp = os.path.join(d, s, 'metrics.json')
            if os.path.basename(s).startswith('seed') and os.path.exists(mp):
                try:
                    m = json.load(open(mp))
                    rows.append({'key': f'{os.path.basename(d)}/{os.path.basename(s)}',
                                 'val': m['val_auc'], 'ho': m['fresh_holdout_auc'],
                                 'cfg': m.get('config', '?')})
                except Exception:
                    continue
    return rows


def best(rows, key):
    return max(rows, key=lambda r: r[key]) if rows else None


def report(rows, verdict):
    lines = ['# Round-2 campaign results (target: locked-B >= 0.86)', '',
             '| run/seed | config | val | working-holdout |',
             '|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['ho']):
        lines.append(f"| {r['key']} | {r['cfg']} | {r['val']:.4f} | {r['ho']:.4f} |")
    lines += ['', f'**Verdict: {verdict}**', '']
    with open(os.path.join(REPO, 'ml', 'training_runs', 'ROUND2_RESULTS.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('wrote ROUND2_RESULTS.md')


def main():
    wait_for_lock()
    log('TARGET: locked-B holdout >= 0.86 (single final eval only)')

    # B0: carve locked-B disjoint from locked-A
    if not os.path.exists(LOCK_B):
        log('carving locked-B...')
        subprocess.run([sys.executable, '-u', '-m', 'ml.carve_locked_holdout',
                        '--seed', '777', '--out', LOCK_B, '--frac', '0.10',
                        '--exclude', os.path.join(REPO, 'ml', 'locked_holdout.json')],
                       cwd=REPO)
    lb = json.load(open(LOCK_B))
    log(f"locked-B: {lb['n_patients']} patients, seed {lb['seed']}")

    # B1: fresh baseline
    run_phase(['--config', 'baseline', '--seeds', '14', '15', '--tag', 'E1b'])
    rows = scan_seeds(edirs('E1b'))

    # B2: gap channels
    run_phase(['--config', 'gap', '--seeds', '21', '22', '23', '--tag', 'E2a'])
    rows = scan_seeds(edirs('E1b', 'E2a'))

    # B3: width, gated
    bv = best([r for r in rows if r['cfg'].startswith('baseline')], 'val')
    gv = best([r for r in rows if r['cfg'].startswith('gap') and 'gap128' not in r['cfg'] and 'gapfuse' not in r['cfg']], 'val')
    if gv and bv and gv['val'] >= bv['val'] - 0.002:
        run_phase(['--config', 'gap128', '--seeds', '24', '25', '--tag', 'E2b'])
        rows = scan_seeds(edirs('E1b', 'E2a', 'E2b'))
    else:
        log('skip B3 (gap did not earn width)')

    # B4: ensemble over eligible dirs
    run_phase(['--ensemble'] + edirs('E1b', 'E2a', 'E2b'))
    cand = None
    for d in edirs('E1b', 'E2a', 'E2b'):
        cp = os.path.join(d, 'candidate_ensemble.json')
        if os.path.exists(cp):
            c = json.load(open(cp))
            if cand is None or c['val_auc'] > cand['val_auc']:
                cand = c
    rows = scan_seeds(edirs('E1b', 'E2a', 'E2b'))
    if cand:
        log(f"B4 candidate: {cand['members']} val={cand['val_auc']:.4f} ho={cand['fresh_holdout_auc']:.4f}")

    # B5: ablations if working bar not met
    bh = best(rows, 'ho')
    if (not bh) or bh['ho'] < WORKING_BAR:
        for tag, extra in [('horizon6', ['--horizon', '6', '--seeds', '31']),
                           ('horizon24', ['--horizon', '24', '--seeds', '32']),
                           ('window120', ['--window', '120', '--seeds', '33']),
                           ('dropout02', ['--dropout', '0.2', '--seeds', '34']),
                           ('dropout04', ['--dropout', '0.4', '--seeds', '35']),
                           ('lr3e-4', ['--lr', '3e-4', '--seeds', '36'])]:
            run_phase(['--config', 'gap', '--tag', 'E4' + tag] + extra)
        rows = scan_seeds(edirs('E1b', 'E2a', 'E2b', 'E4horizon6', 'E4horizon24',
                                'E4window120', 'E4dropout02', 'E4dropout04', 'E4lr3e-4'))
    else:
        log(f"skip B5: working best {bh['ho']:.4f} >= bar {WORKING_BAR}")

    # B6: fusion if still short
    bh = best(rows, 'ho')
    if (not bh) or bh['ho'] < WORKING_BAR:
        run_phase(['--config', 'gapfuse', '--seeds', '41', '42', '--tag', 'FUSE'])
        rows = scan_seeds(edirs('E1b', 'E2a', 'E2b', 'FUSE'))
    else:
        log('skip B6 fusion (bar already met)')

    # Re-ensemble over everything eligible, then the ONE locked eval
    all_dirs = edirs('E1b', 'E2a', 'E2b', 'E4horizon6', 'E4horizon24', 'E4window120',
                     'E4dropout02', 'E4dropout04', 'E4lr3e-4', 'FUSE')
    run_phase(['--ensemble'] + all_dirs)
    cand, cand_path = None, None
    for d in all_dirs:
        cp = os.path.join(d, 'candidate_ensemble.json')
        if os.path.exists(cp):
            c = json.load(open(cp))
            if cand is None or c['val_auc'] > cand['val_auc']:
                cand, cand_path = c, cp
    if not cand:
        report(rows, 'NO CANDIDATE PRODUCED — campaign failed before final eval.')
        return
    log(f"FINAL candidate: {cand['members']} val={cand['val_auc']:.4f} ho={cand['fresh_holdout_auc']:.4f}")
    log('ONE-TIME locked-B evaluation...')
    subprocess.run([sys.executable, '-u', '-m', 'ml.eval_locked', cand_path,
                    '--locked', LOCK_B,
                    '--out', os.path.join(REPO, 'ml', 'LOCKED_EVAL_B.json')],
                   cwd=REPO)
    fin = json.load(open(os.path.join(REPO, 'ml', 'LOCKED_EVAL_B.json')))
    if fin['auc'] >= TARGET_LOCKED:
        report(rows, f"VICTORY: locked-B AUC {fin['auc']:.4f} >= {TARGET_LOCKED} "
                     f"(members {cand['members']}, acc {fin['accuracy']:.3f})")
    else:
        bb = best(rows, 'ho')
        report(rows, f"TARGET MISSED: locked-B AUC {fin['auc']:.4f} < {TARGET_LOCKED}. "
                     f"Best working: {bb['key']} {bb['ho']:.4f}. See log for next options (GRU-D, pooling).")


if __name__ == '__main__':
    main()
