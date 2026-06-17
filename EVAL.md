# Evaluation: picking the best model + thresholds from a new train run

Given a finished training run `<RUN>` (e.g. `.../runs/segment/train-8`, containing
`weights/epochN.pt` + `results.csv` + `args.yaml`), do the following with `evalFalsePositives.py`.

Run with the project venv: `source ~/Programs/.venv/bin/activate` (`pydl`).
All commands write a self-describing folder to `<RUN>/eval/<tag>/` (see `manifest.json` for provenance).

Defaults match production: **class-agnostic NMS, iou 0.7, conf_min 0.15**.

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

## 2. (optional) Full threshold report on the winner — `--mode sweep`

Detailed per-class threshold analysis (global vs budget vs F-beta, per-class table, plots):

```bash
python evalFalsePositives.py --mode sweep --run <RUN> --epoch <W> \
  --neg <NOLITTER>/train/split/images --valid <DATASET>/data.yaml
```

## 3. Confirm real FP at the chosen thresholds — `--mode eval`

```bash
python evalFalsePositives.py --mode eval \
  --model <RUN>/weights/epoch<W-1>.pt \
  --class-thr <RUN>/eval/v1/class_thresholds_ep<W>.yaml \
  --neg <NOLITTER>/train/split/images
```

Prints the actual NoLitter image-FP rate and saves annotated FP images — sanity-check it matches
the frontier's prediction. Deploy `epoch<W-1>.pt` + that yaml.

## Notes
- Re-running `--mode select` with the same `--tag` reuses cached `curves/*.npy` (no re-inference) — cheap to re-analyze at a different `--fp-budget`.
- Knobs: `--top-map`/`--top-post` (shortlist size), `--beta` (<1 favors precision), `--iou` (set if production NMS iou ≠ 0.7).
