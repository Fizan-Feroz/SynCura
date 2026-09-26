# Round-2 campaign results (target: locked-B >= 0.86)

| run/seed | config | val | working-holdout |
|---|---|---|---|
| E1b/seed14 | baseline | 0.8165 | 0.8315 |
| E1b/seed15 | baseline | 0.8263 | 0.8356 |
| E2a/seed21 | gap | 0.8159 | 0.8395 |
| E2a/seed22 | gap | 0.8215 | 0.8328 |
| E2a/seed23 | gap | 0.8330 | 0.8326 |
| E2b/seed24 | gap128 | 0.8284 | 0.8343 |
| E2b/seed25 | gap128 | 0.8270 | 0.8324 |
| E4horizon6/seed31 | gap+h6 | 0.8249 | 0.8309 |
| E4horizon24/seed32 | gap+h24 | 0.8083 | 0.8306 |
| E4window120/seed33 | gap+w120 | 0.8200 | 0.8375 |
| E4dropout02/seed34 | gap+do02 | 0.8221 | 0.8375 |
| E4dropout04/seed35 | gap+do04 | 0.8222 | 0.8310 |
| E4lr3e-4/seed36 | gap+lr | 0.8082 | 0.8316 |
| FUSE/seed41 | gapfuse | 0.8298 | 0.8281 |
| FUSE/seed42 | gapfuse | 0.8255 | 0.8296 |

Greedy selection: start seed23 (0.8330); best add seed15 → 0.8326, stop.
Winner = single seed23. Ensemble added nothing (members too correlated).

**Locked-B (seed 777, 360 pts, one-time): AUC 0.8178**, acc 0.733,
sens 0.777, spec 0.724, prec 0.365. See `ml/LOCKED_EVAL_B.json`.

**Verdict: TARGET MISSED (0.8178 < 0.86).** 15 seeds across 6 config
families converge to val ~0.82–0.83, working holdout ~0.83–0.84, locked
~0.82. Gap channels, width, fusion, and all ablations move numbers
within noise. PhysioNet-only AttentionLSTM has a ceiling here; 0.86
needs multi-dataset pooling (S13 recipe) and/or pretraining — round 3.
