from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
import csv
import cv2
import base64
import requests
import json
import time
import sys

def load_class_thresholds(spec, model_names=None):
    """Load per-class confidence thresholds as {class_id: thr}.

    spec may be: a path to a YAML/JSON file written by threshold_analysis (keys 'names' +
    'thresholds' list, id-aligned), an id-aligned list, or a {class_name|class_id: thr} dict.
    If the file carries 'names' and model_names is given, a mismatch is warned about (alignment guard).
    """
    import yaml
    data = spec
    if isinstance(spec, str):
        with open(spec) as f:
            data = yaml.safe_load(f)
    if isinstance(data, dict) and "thresholds" in data:
        if model_names is not None and "names" in data:
            mn = [model_names[i] for i in range(len(model_names))]
            if list(data["names"]) != mn:
                print("WARNING: threshold file 'names' != model names — check class alignment!")
        return {i: float(t) for i, t in enumerate(data["thresholds"])}
    if isinstance(data, (list, tuple)):
        return {i: float(t) for i, t in enumerate(data)}
    if isinstance(data, dict):
        name2id = {v: k for k, v in model_names.items()} if model_names else {}
        out = {}
        for k, v in data.items():
            if isinstance(k, int) or (isinstance(k, str) and str(k).isdigit()):
                out[int(k)] = float(v)
            elif k in name2id:
                out[name2id[k]] = float(v)
        return out
    raise ValueError(f"Unrecognized class-threshold spec: {type(spec)}")


def eval_fp(model_path, images_folder, conf_threshold=0.25, save_images=True, verbose=True, class_thr=None):
    # Load model and run inference. class_thr (path/list/dict) applies PER-CLASS thresholds.
    model = YOLO(model_path)
    thr_map = load_class_thresholds(class_thr, model.names) if class_thr is not None else None
    predict_conf = max(min(thr_map.values()) - 1e-3, 1e-3) if thr_map else conf_threshold
    results = model.predict(source=images_folder, conf=predict_conf, verbose=False)

    # Count false positives
    total_fps = 0
    images_with_fps = 0
    fps_per_class = defaultdict(int)
    fps_per_image = []

    if save_images:
        from ultralytics.utils.files import increment_path
        save_dir = increment_path(Path("runs/detect/eval_fp"), exist_ok=False)  # eval_fp, eval_fp2, ...
        save_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        # Apply per-class thresholds: keep a box only if its conf >= that class's threshold
        if thr_map is not None:
            import torch
            keep = [i for i, b in enumerate(result.boxes)
                    if float(b.conf[0]) >= thr_map.get(int(b.cls[0]), predict_conf)]
            result = result[torch.tensor(keep, dtype=torch.long)]
        # Patch names dict so plot() can label all detected class IDs
        for box in result.boxes:
            cid = int(box.cls[0])
            if cid not in result.names:
                result.names[cid] = f"class_{cid}"

        num_detections = len(result.boxes)
        detections = []
        image_fps = {'path': result.path, 'count': num_detections, 'classes': defaultdict(int), 'detections': detections}

        if num_detections > 0:
            images_with_fps += 1
            total_fps += num_detections

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                fps_per_class[class_name] += 1
                image_fps['classes'][class_name] += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                })

            # Save annotated image
            if save_images:
                annotated = result.plot()
                out_path = save_dir / Path(result.path).name
                cv2.imwrite(str(out_path), annotated)

        fps_per_image.append(image_fps)

    # Print summary
    if verbose:
        print(f"\nTotal images: {len(results)}")
        print(f"Images with FPs: {images_with_fps}")
        print(f"Total FPs: {total_fps}")
        print(f"FP rate: {images_with_fps/len(results)*100:.2f}%")
        print(f"\nFPs per class:")
        for class_name, count in sorted(fps_per_class.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")

    # Save to CSV (and annotated images) only outside the sweep
    if save_images:
        csv_path = save_dir / 'fps_per_image.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Total FPs'] + sorted(fps_per_class.keys()))

            for img_data in fps_per_image:
                row = [img_data['path'], img_data['count']]
                for class_name in sorted(fps_per_class.keys()):
                    row.append(img_data['classes'].get(class_name, 0))
                writer.writerow(row)

        print(f"\nAnnotated images saved to {save_dir}/")
        print(f"Saved to {csv_path}")

    return fps_per_image, results


def vlm_verify_detection(image_path, detection, vlm_url="http://127.0.0.1:1234/v1/chat/completions", model="lfm2.5-vl-1.6b", context_pad=0.5):
    """Ask a VLM whether a single YOLO detection is actually the detected class.

    Args:
        image_path: path to the original image
        detection: dict with 'class_name', 'confidence', 'bbox' [x1, y1, x2, y2]
        vlm_url: LM Studio chat completions endpoint
        model: VLM model name
        context_pad: padding as a fraction of bbox size (0.5 = 50% extra on each side)

    Returns:
        dict with keys: 'litter_confirmed' (bool), 'explanation' (str), 'raw_response' (str)
    """
    img = cv2.imread(image_path)
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = detection['bbox']
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * context_pad), int(bh * context_pad)
    crop = img[max(0, y1 - pad_y):min(ih, y2 + pad_y), max(0, x1 - pad_x):min(iw, x2 + pad_x)]

    max_side = 1024
    ch, cw = crop.shape[:2]
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    class_name = detection['class_name']
    prompt = (
        f"A litter-detection model detected \"{class_name}\" in this region of an image "
        f"(confidence: {detection['confidence']:.0%}).\n\n"
        "Look carefully at the center of this cropped image. "
        f"First, describe what you actually see in this region. "
        f"Then, decide: is this actually \"{class_name}\" (a piece of litter/trash)? "
        "Or is it something else being misidentified?\n\n"
        "End your answer with EXACTLY one of these two lines:\n"
        "VERDICT: LITTER CONFIRMED\n"
        "VERDICT: NO LITTER\n"
    )

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                    {'type': 'text', 'text': prompt},
                ]
            }
        ],
        'max_tokens': 300,
        'temperature': 0.1,
    }

    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.post(vlm_url, json=payload, timeout=60)
        data = resp.json()
        if 'choices' in data:
            raw = data['choices'][0]['message']['content']
            litter_confirmed = "VERDICT: LITTER CONFIRMED" in raw.upper()
            return {
                'litter_confirmed': litter_confirmed,
                'explanation': raw.strip(),
                'raw_response': raw,
            }
        if attempt < max_retries - 1:
            time.sleep(1)

    error_msg = data.get('error', {}).get('message', str(data)) if isinstance(data, dict) else str(data)
    raise RuntimeError(f"API error after {max_retries} retries: {error_msg}")


def verify_all_fps(fps_per_image, vlm_url="http://127.0.0.1:1234/v1/chat/completions", model="lfm2.5-vl-1.6b"):
    """Run VLM verification on each individual detection across all images.

    Args:
        fps_per_image: list of dicts from eval_fp (each has 'path', 'count', 'classes', 'detections')

    Returns:
        list of verification results (one per detection)
    """
    all_detections = []
    for img_data in fps_per_image:
        for det in img_data['detections']:
            all_detections.append((img_data['path'], det))

    total = len(all_detections)
    print(f"\nVerifying {total} individual detections via VLM...")

    litter_confirmed_count = 0
    no_litter_count = 0
    errors = 0
    results = []
    t_start = time.time()

    for i, (img_path, detection) in enumerate(all_detections):
        try:
            t0 = time.time()
            result = vlm_verify_detection(img_path, detection, vlm_url, model)
            dt = time.time() - t0
            result['image_path'] = img_path
            result['class_name'] = detection['class_name']
            result['confidence'] = detection['confidence']
            result['bbox'] = detection['bbox']
            results.append(result)

            if result['litter_confirmed']:
                litter_confirmed_count += 1
                status = "LITTER"
            else:
                no_litter_count += 1
                status = "NO LITTER"
        except Exception as e:
            errors += 1
            dt = time.time() - t0
            status = "ERROR"
            print(f"\n  ERROR on {Path(img_path).name} [{detection['class_name']}]: {e}")

        done = i + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        eta = avg * (total - done)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.0f}s"
        bar_len = 30
        filled = int(bar_len * done / total)
        bar = "█" * filled + "░" * (bar_len - filled)

        sys.stdout.write(
            f"\r  {bar} {done}/{total} | {status:9s} | {dt:.1f}s | ETA {eta_str} | litter:{litter_confirmed_count} none:{no_litter_count} err:{errors}"
        )
        sys.stdout.flush()

    print(f"\n\nVLM Verification Summary:")
    print(f"  Litter confirmed: {litter_confirmed_count}")
    print(f"  No litter found: {no_litter_count}")
    print(f"  Errors: {errors}")
    print(f"  Total verified: {len(results)}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    # Save verification results to CSV
    with open('vlm_verification.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Class', 'Confidence', 'BBox', 'VLM Verdict', 'Explanation'])
        for r in results:
            verdict = "LITTER_CONFIRMED" if r['litter_confirmed'] else "NO_LITTER"
            writer.writerow([r['image_path'], r['class_name'], f"{r['confidence']:.2f}",
                             str(r['bbox']), verdict, r['explanation']])

    print("Saved to vlm_verification.csv")
    return results


def sweep_frontier(run_dir, neg_folder, results_csv=None, conf=0.334,
                   epochs=None, close_epoch=80,
                   out_csv="fp_frontier.csv", out_png="fp_frontier.png"):
    """Sweep saved checkpoints to build the FP-vs-recall frontier.

    FP-rate is computed on `neg_folder` via eval_fp (no image plotting); the recall/mAP50
    axis is read for free from the run's results.csv. Lets you pick a checkpoint by a
    deployment criterion instead of Ultralytics' mask+mAP95 fitness.

    Args:
        run_dir: training run dir (contains weights/epochN.pt and results.csv)
        neg_folder: folder of hard-negative images (should produce zero detections)
        conf: operating confidence threshold (matches eval_fp default in deployment)
        epochs: 1-based epochs to evaluate (matches results.csv); default = dense post-closure
        close_epoch: mosaic-close boundary, only used to annotate the plot/summary

    Returns:
        pandas.DataFrame with epoch, fp_rate, tot_fp, recall, map50
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics.utils.torch_utils import strip_optimizer

    run_dir = Path(run_dir)
    df = pd.read_csv(results_csv or run_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    if epochs is None:
        epochs = sorted(set([20, 28, 36, 44, 49, 52, 56, 60, 63, 68, 72, 76] + list(range(78, 101))))

    rows = []
    for ep in epochs:
        ck = run_dir / "weights" / f"epoch{ep - 1}.pt"   # results epoch is 1-based; files are epoch0..N-1
        if not ck.exists():
            print("skip (missing):", ck)
            continue
        # Strip to a temp file so we evaluate the deployed EMA (FP16) weights, not raw training weights.
        # epochN.pt is left intact; production best.pt is built the same way (strip_optimizer at train end).
        strip_optimizer(str(ck), s="/tmp/_eval_ckpt.pt")
        fpi, _ = eval_fp("/tmp/_eval_ckpt.pt", neg_folder, conf, save_images=False, verbose=False)
        n = len(fpi)
        imgs_fp = sum(1 for d in fpi if d['count'] > 0)
        tot_fp = sum(d['count'] for d in fpi)
        row = df[df["epoch"] == ep]
        rec = float(row["metrics/recall(B)"].iloc[0]) if len(row) else float("nan")
        m50 = float(row["metrics/mAP50(B)"].iloc[0]) if len(row) else float("nan")
        rows.append(dict(epoch=ep, fp_rate=100 * imgs_fp / n, tot_fp=tot_fp, recall=rec, map50=m50))
        print(f"ep{ep:3d}  FP_rate={rows[-1]['fp_rate']:5.2f}%  tot_fp={tot_fp:4d}  "
              f"recall={rec:.4f}  mAP50={m50:.4f}")

    R = pd.DataFrame(rows)
    R.to_csv(out_csv, index=False)

    print("\n=== SUMMARY ===")
    best = R.loc[R.fp_rate.idxmin()]
    print(f"min FP_rate : {best.fp_rate:.2f}% @ep{int(best.epoch)} (recall {best.recall:.3f})")
    pre, post = R[R.epoch <= close_epoch], R[R.epoch > close_epoch]
    if len(pre):
        print(f"PRE-closure : FP {pre.fp_rate.min():.2f}-{pre.fp_rate.max():.2f}%  max recall {pre.recall.max():.3f}")
    if len(post):
        print(f"POST-closure: FP {post.fp_rate.min():.2f}-{post.fp_rate.max():.2f}%  max recall {post.recall.max():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sc = ax[0].scatter(R.fp_rate, R.recall, c=R.epoch, cmap="viridis", s=60)
    for _, r in R.iterrows():
        ax[0].annotate(int(r.epoch), (r.fp_rate, r.recall), fontsize=7)
    ax[0].set_xlabel(f"FP rate % (conf={conf})")
    ax[0].set_ylabel("val recall(B)")
    ax[0].set_title("FP-recall frontier (color=epoch)")
    ax[0].grid(alpha=.3)
    plt.colorbar(sc, ax=ax[0])
    ax[1].plot(R.epoch, R.fp_rate, "o-", label="FP rate %")
    ax[1].axvline(close_epoch, color="g", ls="--", label="mosaic close")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("FP rate %")
    ax[1].set_title("FP rate vs epoch")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    print(f"saved {out_csv} and {out_png}")
    return R


def threshold_analysis(model_path, valid_data, neg_folder,
                       fp_budget=0.15, w_neg=1.0, beta=0.5, conf_min=0.05, recall_mode="micro",
                       strip=False, out_csv="thresholds.csv", out_png="thresholds.png",
                       out_thr="class_thresholds.yaml"):
    """Pick operating thresholds from a deployment trade-off, not Ultralytics' F1-max-on-val.

    Combines two precision signals on a shared confidence grid:
      - val precision  P_val  = TP / (TP + FP_val)            -> precision inside litter scenes (valid set)
      - combined prec  P_comb = TP / (TP + FP_val + w*FP_neg) -> adds empty-scene FPs (NoLitter set)
    Recall comes from valid; the operational FP metric (images-with-FP rate) comes from NoLitter.

    Reports three GLOBAL thresholds for comparison, and a PER-CLASS threshold table:
      - ultra : Ultralytics F1-max conf on valid (the inherited ~0.334)
      - budget: lowest conf with NoLitter image-FP-rate <= fp_budget (=> max recall under budget)
      - fbeta : argmax F-beta(P_comb, recall)  (beta<1 favors precision)

    Args:
        model_path : checkpoint to analyze (pass a deployable/stripped .pt, or set strip=True)
        valid_data : data.yaml (recall + val-precision axis)
        neg_folder : NoLitter images (empty-scene FP axis)
        fp_budget  : target fraction of NoLitter images allowed to fire (e.g. 0.15)
        w_neg      : weight of empty-scene FPs in P_comb (deployment empty-frame prior)
        beta       : F-beta beta (<1 precision-favoring)
        conf_min   : low conf for the single NoLitter inference pass (thresholds swept offline)
        recall_mode: "micro" (instance-weighted, deployment-faithful) or "macro" (mean of
                     per-class curves; matches Ultralytics recall(B) for comparing to old results)
        strip      : strip_optimizer to EMA/FP16 temp before analysis (use for raw epochN.pt)

    Returns:
        dict with global thresholds, per-class DataFrame, and the raw curves.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics.utils.metrics import smooth

    if strip:
        from ultralytics.utils.torch_utils import strip_optimizer
        strip_optimizer(str(model_path), s="/tmp/_thr_ckpt.pt")
        model_path = "/tmp/_thr_ckpt.pt"

    # ---- 1) VALID: per-class recall/precision curves over the 1000-pt conf grid ----
    m = YOLO(model_path)
    names = m.names                                   # {id: name}
    name2id = {v: k for k, v in names.items()}
    res = m.val(data=valid_data, plots=False, verbose=False)
    px = np.asarray(res.box.px)                       # conf grid, shape (1000,) == linspace(0,1,1000)
    rc = np.asarray(res.box.r_curve)                  # (n_present, 1000) recall @ IoU0.5
    pc = np.asarray(res.box.p_curve)                  # (n_present, 1000) precision @ IoU0.5
    f1c = np.asarray(res.box.f1_curve)
    present = list(res.box.ap_class_index)            # class ids aligned to curve rows
    nt = np.asarray(res.nt_per_class)                 # GT count per class id (len nc)

    eps = 1e-9
    # reconstruct per-class TP/FP counts from the curves: tpc = recall*n_l ; fpc = tpc*(1-p)/p
    n_l = {c: float(nt[c]) for c in present}
    tpc = {c: rc[i] * n_l[c] for i, c in enumerate(present)}
    fpc = {c: np.where(pc[i] > eps, tpc[c] * (1 - pc[i]) / np.clip(pc[i], eps, 1), 0.0)
           for i, c in enumerate(present)}

    # ---- 2) NOLITTER: one low-conf pass, then sweep thresholds offline ----
    fpi, _ = eval_fp(str(model_path), neg_folder, conf_min, save_images=False, verbose=False)
    N_neg = len(fpi)
    all_conf = []                                     # every detection conf (for FP_neg count)
    img_max = []                                      # per-image max conf (for global image-FP rate)
    cls_conf = defaultdict(list)                      # class_id -> all det confs
    cls_img_max = defaultdict(list)                   # class_id -> per-image max conf for that class
    for img in fpi:
        if not img['detections']:
            continue
        img_max.append(max(d['confidence'] for d in img['detections']))
        per_cls = defaultdict(float)
        for d in img['detections']:
            cid = name2id.get(d['class_name'])
            if cid is None:
                continue
            all_conf.append(d['confidence']); cls_conf[cid].append(d['confidence'])
            per_cls[cid] = max(per_cls[cid], d['confidence'])
        for cid, mx in per_cls.items():
            cls_img_max[cid].append(mx)

    def ge_count(arr, grid):                          # count(values >= t) for each t in grid
        a = np.sort(np.asarray(arr, dtype=float)) if len(arr) else np.zeros(0)
        return len(a) - np.searchsorted(a, grid, side="left")

    fp_neg_tot = ge_count(all_conf, px)               # total FP detections vs conf
    fp_img_tot = ge_count(img_max, px)                # images firing vs conf
    fp_img_rate = fp_img_tot / max(N_neg, 1)

    # ---- 3) GLOBAL recall / precision / objective (micro or macro) ----
    fpn_c = {c: ge_count(cls_conf.get(c, []), px) for c in present}   # per-class FP_neg counts vs conf
    if recall_mode == "macro":   # mean of per-class curves -> matches Ultralytics recall(B)
        R = rc.mean(0)
        P_val = pc.mean(0)
        P_comb = np.mean([tpc[c] / (tpc[c] + fpc[c] + w_neg * fpn_c[c] + eps) for c in present], axis=0)
    else:                        # micro: instance-weighted (deployment-faithful)
        TP = np.sum([tpc[c] for c in present], axis=0)
        FPv = np.sum([fpc[c] for c in present], axis=0)
        R = TP / (np.sum([n_l[c] for c in present]) + eps)
        P_val = TP / (TP + FPv + eps)
        P_comb = TP / (TP + FPv + w_neg * fp_neg_tot + eps)
    fb = (1 + beta**2) * P_comb * R / (beta**2 * P_comb + R + eps)

    i_ultra = int(smooth(f1c.mean(0), 0.1).argmax())
    bud = np.where(fp_img_rate <= fp_budget)[0]
    i_bud = int(bud[0]) if len(bud) else len(px) - 1   # lowest conf under budget => max recall
    i_fb = int(np.nanargmax(fb))

    def at(i):
        return dict(conf=float(px[i]), recall=float(R[i]), P_val=float(P_val[i]),
                    P_comb=float(P_comb[i]), fp_img_rate=float(fp_img_rate[i]))
    glob = {"ultra_F1": at(i_ultra), "fp_budget": at(i_bud), "fbeta": at(i_fb)}
    print(f"=== GLOBAL thresholds (recall_mode={recall_mode}) ===")
    for k, v in glob.items():
        print(f"  {k:10s} conf={v['conf']:.3f}  recall={v['recall']:.3f}  "
              f"P_val={v['P_val']:.3f}  P_comb={v['P_comb']:.3f}  FP_img={v['fp_img_rate']*100:.1f}%")

    # ---- 4) PER-CLASS thresholds (F-beta on per-class P_comb; budget fallback if absent in val) ----
    rows, agg = [], []
    for cid in sorted(set(present) | set(cls_img_max)):
        cname = names[cid]
        fpn_cur = ge_count(cls_conf.get(cid, []), px)
        fpimg_c = ge_count(cls_img_max.get(cid, []), px) / max(N_neg, 1)
        if cid in present:
            tp_c, fpv_c = tpc[cid], fpc[cid]
            r_c = tp_c / (n_l[cid] + eps)
            pcomb_c = tp_c / (tp_c + fpv_c + w_neg * fpn_cur + eps)
            fb_c = (1 + beta**2) * pcomb_c * r_c / (beta**2 * pcomb_c + r_c + eps)
            j = int(np.nanargmax(fb_c))
            rows.append(dict(cls=cname, n_gt=int(n_l[cid]), thr=float(px[j]),
                             recall=float(r_c[j]), P_val=float(pc[present.index(cid)][j]),
                             P_comb=float(pcomb_c[j]), fp_img_rate=float(fpimg_c[j])))
            agg.append(dict(cls=cname, n_l=n_l[cid], tp=float(tp_c[j]),
                            fpv=float(fpv_c[j]), fpn=float(fpn_cur[j])))   # counts at the chosen thr
        else:  # only fires on negatives, no val recall -> threshold by FP budget alone
            b = np.where(fpimg_c <= fp_budget)[0]
            j = int(b[0]) if len(b) else len(px) - 1
            rows.append(dict(cls=cname, n_gt=0, thr=float(px[j]), recall=float("nan"),
                             P_val=float("nan"), P_comb=float("nan"), fp_img_rate=float(fpimg_c[j])))
    per_class = pd.DataFrame(rows).sort_values("fp_img_rate", ascending=False)
    per_class.to_csv(out_csv, index=False)
    print("\n=== PER-CLASS thresholds (sorted by FP exposure) ===")
    print(per_class.to_string(index=False))

    # ---- 4b) GLOBAL metrics AT the per-class thresholds, aggregated two ways ----
    # micro = instance-weighted (sum counts across classes); macro = plain mean of per-class metrics (w=1/class)
    # precision = TP/(TP+FP_val) [litter scenes] ; precision_fp = TP/(TP+FP_neg) [empty scenes]
    A = pd.DataFrame(agg)                                  # present classes only (have GT recall)
    f1 = lambda p, r: 2 * p * r / (p + r + eps)
    TPs, FPvs, FPns, NL = A.tp.sum(), A.fpv.sum(), A.fpn.sum(), A.n_l.sum()
    micro = dict(recall=TPs/(NL+eps), precision=TPs/(TPs+FPvs+eps),
                 precision_fp=TPs/(TPs+FPns+eps), P_comb=TPs/(TPs+FPvs+w_neg*FPns+eps))
    micro["F1"] = f1(micro["precision"], micro["recall"])
    r_c, p_c, pfp_c = A.tp/(A.n_l+eps), A.tp/(A.tp+A.fpv+eps), A.tp/(A.tp+A.fpn+eps)
    macro = dict(recall=r_c.mean(), precision=p_c.mean(), precision_fp=pfp_c.mean(),
                 P_comb=(A.tp/(A.tp+A.fpv+w_neg*A.fpn+eps)).mean(), F1=f1(p_c, r_c).mean())
    summary = pd.DataFrame({"micro(inst-wt)": micro, "macro(w=1/cls)": macro}).T[
        ["recall", "precision", "precision_fp", "P_comb", "F1"]]
    print("\n=== GLOBAL metrics at per-class thresholds ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- 4c) write per-class threshold config, id-aligned to data.yaml (consumed by eval_fp) ----
    import yaml
    nc = len(names)
    default_thr = float(px[i_fb])                         # for classes absent from valid AND negatives
    thr_by_id = {name2id[r.cls]: float(r.thr) for r in per_class.itertuples()}
    thr_list = [round(thr_by_id.get(c, default_thr), 4) for c in range(nc)]
    with open(out_thr, "w") as f:
        yaml.safe_dump({"model": str(model_path), "recall_mode": recall_mode, "beta": beta,
                        "names": [names[c] for c in range(nc)], "thresholds": thr_list},
                       f, sort_keys=False)
    print(f"saved {out_thr}  (per-class thresholds, id-aligned — edit freely, feed to eval_fp)")

    # ---- 5) plots: DET trade-off + precision/recall vs conf ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(fp_img_rate * 100, R, "-", lw=1.5)
    for k, c in [("ultra_F1", "r"), ("fp_budget", "g"), ("fbeta", "b")]:
        v = glob[k]; ax[0].scatter(v["fp_img_rate"]*100, v["recall"], c=c, s=80, label=f"{k} (conf {v['conf']:.2f})")
    ax[0].axvline(fp_budget*100, color="g", ls=":", alpha=.5)
    ax[0].set_xlabel("NoLitter image-FP rate %"); ax[0].set_ylabel("val recall")
    ax[0].set_title("Deployment trade-off (parametric in conf)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(px, R, label="recall (valid)")
    ax[1].plot(px, P_val, label="P_val (litter scenes)")
    ax[1].plot(px, P_comb, label=f"P_comb (w={w_neg})")
    ax[1].plot(px, fp_img_rate, "--", color="gray", label="FP image rate (neg)")
    for k, c in [("ultra_F1", "r"), ("fp_budget", "g"), ("fbeta", "b")]:
        ax[1].axvline(glob[k]["conf"], color=c, ls=":", label=f"{k}")
    ax[1].set_xlabel("confidence"); ax[1].set_xlim(0, 0.8); ax[1].set_title("Curves vs conf")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=110)
    print(f"\nsaved {out_csv} and {out_png}")
    return dict(globals=glob, per_class=per_class, summary=summary, thresholds=thr_list,
                curves=dict(px=px, recall=R, P_val=P_val, P_comb=P_comb, fp_img_rate=fp_img_rate))


if __name__ == "__main__":
    # sweep (per-class thresholds + YAML):  python evalFalsePositives.py --mode sweep --epoch 85
    # eval  (FP at those thresholds):        python evalFalsePositives.py --mode eval --class-thr class_thresholds.yaml
    import argparse
    p = argparse.ArgumentParser(description="Sweep per-class thresholds, or eval_fp at a threshold config.")
    p.add_argument("--mode", choices=["sweep", "eval"], default="sweep",
                   help="sweep = threshold_analysis (writes YAML); eval = eval_fp on the negatives")
    p.add_argument("--run", default="/home/charles/Programs/ultralytics-fork/runs/segment/train-4",
                   help="training run dir (contains weights/epochN.pt)")
    p.add_argument("--epoch", default=85, help="checkpoint file number, i.e. weights/epoch<N>.pt")
    p.add_argument("--model", default=None, help="explicit checkpoint path (overrides --run/--epoch, e.g. best.pt)")
    p.add_argument("--recall-mode", default="micro", choices=["micro", "macro"],
                   help="[sweep] micro = instance-weighted (deployment); macro = mean per-class (matches recall(B))")
    p.add_argument("--neg", default="/home/charles/Programs/datasetManipulation/NoLitter-3/train/split/images",
                   help="NoLitter hard-negative images folder")
    p.add_argument("--valid", default="/home/charles/Programs/datasetManipulation/datasets/Dataset-ViPARE-33-split/data.yaml",
                   help="[sweep] data.yaml for the recall/precision axis")
    p.add_argument("--fp-budget", type=float, default=0.15, help="[sweep] max NoLitter image-FP rate for the budget threshold")
    p.add_argument("--w-neg", type=float, default=1.0, help="[sweep] weight of empty-scene FPs (deployment empty-frame prior)")
    p.add_argument("--beta", type=float, default=0.5, help="[sweep] F-beta beta (<1 favors precision)")
    p.add_argument("--class-thr", default=None, help="[eval] per-class threshold YAML/list (else use --conf)")
    p.add_argument("--conf", type=float, default=0.334, help="[eval] global conf when no --class-thr")
    p.add_argument("--no-strip", action="store_true", help="skip strip_optimizer (pass an already-deployable .pt)")
    a = p.parse_args()

    ckpt = a.model or f"{a.run}/weights/epoch{a.epoch}.pt"

    if a.mode == "sweep":
        threshold_analysis(ckpt, a.valid, a.neg, fp_budget=a.fp_budget, w_neg=a.w_neg,
                           beta=a.beta, recall_mode=a.recall_mode, strip=not a.no_strip)
    else:  # eval: run eval_fp at a per-class config (or a global conf)
        model = ckpt
        if not a.no_strip:
            from ultralytics.utils.torch_utils import strip_optimizer
            strip_optimizer(ckpt, s="/tmp/_eval_ckpt.pt")   # EMA/FP16 deployable weights
            model = "/tmp/_eval_ckpt.pt"
        if a.class_thr:
            eval_fp(model, a.neg, class_thr=a.class_thr)
        else:
            eval_fp(model, a.neg, conf_threshold=a.conf)
