# -*- coding: utf-8 -*-
"""Carve the LOCKED final-validation slice (v5 order item #1).

- 10% of set-b patients, GroupShuffleSplit random_state=999
  (distinct from the val seed 42 and the working-holdout seed 123).
- Nothing trains on these patients and nothing gates on them.
- The campaign winner is evaluated here EXACTLY ONCE, at the very end,
  via ml/eval_locked.py. That number is the paper/report headline.
- Saves ml/locked_holdout.json durably: seed + full patient-ID list.

Run once, before P2. Refuses to run twice (would invalidate the lock).
"""
import datetime
import json
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from ml.paths import physionet2012_root as _pn_root
BASE = _pn_root()
OUT = 'ml/locked_holdout.json'
LOCK_SEED = 999
TEST_SIZE = 0.10


def main():
    if os.path.exists(OUT):
        print(f'{OUT} already exists — refusing to re-carve (that would invalidate the lock).')
        raise SystemExit(2)
    from ml.dataset import load_physionet_batch
    data = load_physionet_batch(
        os.path.join(BASE, 'set-b_full', 'set-b'),
        os.path.join(BASE, 'Outcomes-b.txt'))
    pids = np.array([pid for _, _, pid in data])
    upb = np.unique(pids)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=LOCK_SEED)
    _, lock_idx = next(gss.split(np.zeros(len(upb)), np.zeros(len(upb)), groups=upb))
    locked = sorted(upb[lock_idx].tolist())
    payload = {
        'seed': LOCK_SEED,
        'test_size': TEST_SIZE,
        'n_patients': len(locked),
        'n_setb_patients': len(upb),
        'patients': locked,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': ('FINAL validation only. No training, no hyperparameter/ensemble/early-'
                    'stopping gating, evaluated exactly once via ml/eval_locked.py.'),
        'rule': 'distinct from val seed 42 and working-holdout seed 123',
    }
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'locked {len(locked)}/{len(upb)} set-b patients -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
