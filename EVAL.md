# Evaluation: picking the best model + thresholds from a new train run

Given a finished training run `<RUN>` (e.g. `.../runs/segment/train-8`, containing
`weights/epochN.pt` + `results.csv` + `args.yaml`), do the following with `evalFalsePositives.py`.

Run with the project venv: `source ~/Programs/.venv/bin/activate` (`pydl`).
All commands write a self-describing folder to `<RUN>/eval/<tag>/` (see `manifest.json` for provenance).

Defaults: **class-aware NMS** (per-class — for clean class-independent thresholds + standard mAP),
**iou 0.7, conf_min 0.15**. Production deploys with class-agnostic NMS (high-score bias); to get a
deployment-faithful FP number, add `--agnostic-nms` to the `--mode eval` check (step 3). Agnostic NMS
distorts per-class thresholds (it couples classes), so it is NOT used for select/sweep.

## 1. Select the epoch — `--mode select`

Shortlists epochs by mAP50(B) (top-5 overall ∪ top-5 post-closure), then compares them on each
epoch's **per-class-optimal** recall-vs-NoLitter-FP frontier and auto-picks the winner.

```bash
python evalFalsePositives.py --mode select --run <RUN> \
  --neg <NOLITTER>/train/split/images \
  --valid <DATASET>/data.yaml \
  --fp-budget 0.15 --tag v1
```

Read `<RUN>/eval/v1/summary.txt`:
- per-epoch table (mAP50, recall@budget) + the **WINNER** epoch,
- **DID-IT-MATTER** diagnostic (whether the FP frontier changed the pick vs argmax-mAP50, and the
  mAP50↔recall@budget rank correlation),
- `frontiers.png` overlays every epoch's frontier — eyeball the operating point, adjust `--fp-budget` if needed.

Output also includes `class_thresholds_ep<W>.yaml` (the winner's deployable per-class thresholds).

> Epoch numbering: `results.csv`/`--epoch` are 1-based; the file is `weights/epoch{N-1}.pt`. The yaml records both.

**Explicit checkpoints instead of a shortlist** — to get the same per-class-optimal frontier on
specific `.pt` files (e.g. compare `best.pt` vs `last.pt`), use `--mode models`. It recomputes mAP50
from the val pass (no `results.csv` needed) and writes per-model thresholds + stripped copies:

```bash
python evalFalsePositives.py --mode models \
  --models <RUN>/weights/best.pt <RUN>/weights/last.pt \
  --valid <RUN_DATA_YAML> --neg <NOLITTER>/train/split/images \
  --fp-budget 0.15 --tag bestlast
```

## 2. (optional) Full threshold report on the winner — `--mode sweep`

Detailed per-class threshold analysis (global vs budget vs F-beta, per-class table, plots):

```bash
python evalFalsePositives.py --mode sweep --run <RUN> --epoch <W> \
  --neg <NOLITTER>/train/split/images --valid <DATASET>/data.yaml
```

## 3. Confirm real FP at the chosen thresholds — `--mode eval`

```bash
python evalFalsePositives.py --mode eval --no-strip --agnostic-nms \
  --model <RUN>/eval/v1/best_ep<W>.pt \
  --class-thr <RUN>/eval/v1/class_thresholds_ep<W>.yaml \
  --neg <NOLITTER>/train/split/images
```

`--agnostic-nms` here mirrors production; `best_ep<W>.pt` is the optimizer-stripped winner already
copied into the eval folder (so `--no-strip`). Prints the actual NoLitter image-FP rate and saves
annotated FP images — should be **≤** the frontier's (conservative, class-aware) prediction. Deploy
`best_ep<W>.pt` + that yaml.

## Notes
- Re-running `--mode select` with the same `--tag` reuses cached `curves/*.npy` (no re-inference) — cheap to re-analyze at a different `--fp-budget`.
- Knobs: `--top-map`/`--top-post` (shortlist size), `--beta` (<1 favors precision), `--iou` (set if production NMS iou ≠ 0.7).
