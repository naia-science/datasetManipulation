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


def eval_fp(model_path, images_folder, conf_threshold=0.25, save_images=True, verbose=True, class_thr=None,
            agnostic_nms=False, iou=0.7):
    # Load model and run inference. class_thr (path/list/dict) applies PER-CLASS thresholds.
    # agnostic_nms/iou mirror production NMS: agnostic keeps one box per region (argmax class).
    model = YOLO(model_path)
    thr_map = load_class_thresholds(class_thr, model.names) if class_thr is not None else None
    predict_conf = max(min(thr_map.values()) - 1e-3, 1e-3) if thr_map else conf_threshold
    results = model.predict(source=images_folder, conf=predict_conf, iou=iou,
                            agnostic_nms=agnostic_nms, verbose=False)

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


def _compute_curves(model_path, valid_data, neg_folder, conf_min=0.15, strip=False, cache=None,
                    agnostic_nms=False, iou=0.7):
    """Shared per-checkpoint data extraction for threshold_analysis / frontier_select.

    Runs the only expensive work once: one val pass (per-class recall/precision curves over
    the 1000-pt conf grid) and one low-conf NoLitter pass (every empty-scene detection conf).
    Every threshold/frontier number is derived offline from these arrays, so results are a
    deterministic function of this cache. When `cache` (a .npy path) is given it is loaded if
    present, else computed and saved (allow_pickle).

    Returns dict: model_path (resolved/stripped), names, px, rc, pc, f1c, present, nt,
    N_neg, all_conf, img_max, cls_conf, cls_img_max, neg_img_cls_max (per-image {cid: maxconf}).
    """
    import numpy as np
    if cache is not None and Path(cache).exists():
        return np.load(cache, allow_pickle=True).item()

    if strip:
        from ultralytics.utils.torch_utils import strip_optimizer
        strip_optimizer(str(model_path), s="/tmp/_thr_ckpt.pt")
        model_path = "/tmp/_thr_ckpt.pt"

    # ---- 1) VALID: per-class recall/precision curves over the 1000-pt conf grid ----
    m = YOLO(model_path)
    names = dict(m.names)                             # {id: name}
    name2id = {v: k for k, v in names.items()}
    res = m.val(data=valid_data, plots=False, verbose=False, agnostic_nms=agnostic_nms, iou=iou)
    out = dict(
        model_path=str(model_path), names=names,
        px=np.asarray(res.box.px),                    # conf grid (1000,) == linspace(0,1,1000)
        rc=np.asarray(res.box.r_curve),               # (n_present, 1000) recall @ IoU0.5
        pc=np.asarray(res.box.p_curve),               # (n_present, 1000) precision @ IoU0.5
        f1c=np.asarray(res.box.f1_curve),
        present=list(res.box.ap_class_index),         # class ids aligned to curve rows
        nt=np.asarray(res.nt_per_class),              # GT count per class id (len nc)
        map50=float(res.box.map50),                   # mAP50(B) recomputed on THIS valid_data + NMS
    )

    # ---- 2) NOLITTER: one low-conf pass, then sweep thresholds offline ----
    fpi, _ = eval_fp(str(model_path), neg_folder, conf_min, save_images=False, verbose=False,
                     agnostic_nms=agnostic_nms, iou=iou)
    all_conf, img_max = [], []                        # every det conf; per-image max conf
    cls_conf, cls_img_max = defaultdict(list), defaultdict(list)
    neg_img_cls_max = []                              # per neg image: {cid: max conf} ({} if no det)
    for img in fpi:
        per_cls = defaultdict(float)
        if img['detections']:
            img_max.append(max(d['confidence'] for d in img['detections']))
            for d in img['detections']:
                cid = name2id.get(d['class_name'])
                if cid is None:
                    continue
                all_conf.append(d['confidence']); cls_conf[cid].append(d['confidence'])
                per_cls[cid] = max(per_cls[cid], d['confidence'])
            for cid, mx in per_cls.items():
                cls_img_max[cid].append(mx)
        neg_img_cls_max.append(dict(per_cls))
    out.update(N_neg=len(fpi), all_conf=all_conf, img_max=img_max,
               cls_conf=dict(cls_conf), cls_img_max=dict(cls_img_max),
               neg_img_cls_max=neg_img_cls_max)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out, allow_pickle=True)
    return out


def threshold_analysis(model_path, valid_data, neg_folder,
                       fp_budget=0.15, w_neg=1.0, beta=0.5, conf_min=0.15, recall_mode="micro",
                       strip=False, agnostic_nms=False, iou=0.7,
                       out_csv="thresholds.csv", out_png="thresholds.png",
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

    # Shared expensive step (one val pass + one low-conf NoLitter pass), reusable + cacheable.
    cv = _compute_curves(model_path, valid_data, neg_folder, conf_min=conf_min, strip=strip,
                         agnostic_nms=agnostic_nms, iou=iou)
    model_path = cv["model_path"]
    names = cv["names"]                               # {id: name}
    name2id = {v: k for k, v in names.items()}
    px, rc, pc, f1c = cv["px"], cv["rc"], cv["pc"], cv["f1c"]
    present, nt = cv["present"], cv["nt"]             # curve-aligned class ids; GT count per id
    N_neg = cv["N_neg"]
    all_conf, img_max = cv["all_conf"], cv["img_max"]
    cls_conf, cls_img_max = cv["cls_conf"], cv["cls_img_max"]

    eps = 1e-9
    # reconstruct per-class TP/FP counts from the curves: tpc = recall*n_l ; fpc = tpc*(1-p)/p
    n_l = {c: float(nt[c]) for c in present}
    tpc = {c: rc[i] * n_l[c] for i, c in enumerate(present)}
    fpc = {c: np.where(pc[i] > eps, tpc[c] * (1 - pc[i]) / np.clip(pc[i], eps, 1), 0.0)
           for i, c in enumerate(present)}

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


def frontier_thresholds(cv, r_target, absent_thr=0.5):
    """Id-aligned per-class thresholds at a target per-class recall (feed straight to eval_fp).

    Each val-present class is thresholded at the highest conf still achieving recall>=r_target;
    classes absent from val (fire only on negatives, no recall) get a fixed `absent_thr`.
    """
    import numpy as np
    px, rc, present, names = cv["px"], cv["rc"], cv["present"], cv["names"]
    thr = {}
    for i, c in enumerate(present):
        idx = np.where(rc[i] >= r_target)[0]
        j = int(idx[-1]) if len(idx) else len(px) - 1   # rc is non-increasing in conf
        thr[c] = float(px[j])
    return [round(thr.get(c, absent_thr), 4) for c in range(len(names))]


def epoch_frontier(cv, w_grid=None, beta=0.5, absent_thr=1.01):
    """Per-class-OPTIMAL deployable frontier for one checkpoint (threshold/calibration-free).

    Sweeps the empty-scene FP penalty w_neg (a Lagrange multiplier). At each w_neg every
    val-present class is thresholded INDEPENDENTLY at the argmax of its F-beta on combined
    precision P_comb = TP / (TP + FP_val + w*FP_neg) vs its own recall — the same per-class
    objective threshold_analysis uses, here swept into a curve. Classes absent from val
    (recall 0, FP only) are suppressed (absent_thr>1). Each w_neg yields one (micro-recall,
    NoLitter image-FP) point under those per-class thresholds; the sweep is the optimal envelope.

    Returns list of dicts: w_neg, recall, fp_img_rate (%), fp_fires, thr_list (id-aligned).
    """
    import numpy as np
    px, rc, pc = cv["px"], cv["rc"], cv["pc"]
    present, nt, names = cv["present"], cv["nt"], cv["names"]
    cls_conf, neg, N_neg = cv["cls_conf"], cv["neg_img_cls_max"], cv["N_neg"]
    nc = len(names); eps = 1e-9
    if w_grid is None:
        w_grid = np.concatenate([[0.0], np.geomspace(0.1, 64.0, 19)])

    def ge_count(arr, grid):                          # count(values >= t) for each t in grid
        a = np.sort(np.asarray(arr, dtype=float)) if len(arr) else np.zeros(0)
        return len(a) - np.searchsorted(a, grid, side="left")

    n_l = {c: float(nt[c]) for c in present}
    NL = sum(n_l.values()) + eps
    tpc = {c: rc[i] * n_l[c] for i, c in enumerate(present)}          # per-class TP vs conf
    fpc = {c: np.where(pc[i] > eps, tpc[c] * (1 - pc[i]) / np.clip(pc[i], eps, 1), 0.0)
           for i, c in enumerate(present)}                            # per-class val FP vs conf
    fpn = {c: ge_count(cls_conf.get(c, []), px) for c in present}     # per-class neg FP-dets vs conf
    r_c = {c: tpc[c] / (n_l[c] + eps) for c in present}               # per-class recall vs conf

    rows = []
    for w in w_grid:
        thr = [absent_thr] * nc                                       # absent classes suppressed
        for i, c in enumerate(present):
            pcomb = tpc[c] / (tpc[c] + fpc[c] + w * fpn[c] + eps)
            fb = (1 + beta**2) * pcomb * r_c[c] / (beta**2 * pcomb + r_c[c] + eps)
            thr[c] = float(px[int(np.argmax(np.nan_to_num(fb, nan=0.0)))])
        tp_tot = 0.0                                                  # micro recall at thr vector
        for i, c in enumerate(present):
            j = min(int(np.searchsorted(px, thr[c])), len(px) - 1)
            tp_tot += float(rc[i][j]) * n_l[c]
        fires = tot = 0                                              # image fires if ANY class fires
        for d in neg:
            hit = False
            for cid, mc in d.items():
                if cid < nc and mc >= thr[cid]:
                    hit = True; tot += 1
            fires += int(hit)
        rows.append(dict(w_neg=float(w), recall=tp_tot / NL,
                         fp_img_rate=100.0 * fires / max(N_neg, 1), fp_fires=int(tot),
                         thr_list=[round(t, 4) for t in thr]))
    return rows


def frontier_select(run_dir, valid_data, neg_folder, conf_min=0.15, top_map=5, top_post=5,
                    close_epoch=None, fp_budget=10.0, beta=0.5, agnostic_nms=False, iou=0.7,
                    tag=None, out_dir=None):
    """Shortlist epochs by mAP50(B), then compare them on their own per-class FP frontiers.

    Selection: top `top_map` epochs by box mAP50 over the whole run UNION the top `top_post`
    post-closure epochs by mAP50 (dedup). mAP50 is a free, threshold-free quality gate; it is
    blind to empty-scene FPs, so the actual pick comes from each model's recall-vs-NoLitter-FP
    frontier (epoch_frontier), every model judged at its own per-class thresholds under
    production-matched NMS. Auto-pick = max micro-recall with image-FP rate <= fp_budget (%).

    Writes a self-describing run under <run>/eval/<tag>/ : manifest.json (argv, git commit,
    versions, inputs, params), frontiers.csv, frontiers.png, summary.txt, class_thresholds_ep*.yaml,
    and curves/epochNN.npy caches (the only expensive artifact; re-analysis is then free).
    """
    import json, subprocess, datetime, sys
    import numpy as np, pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ultralytics.utils.torch_utils import strip_optimizer
    import ultralytics, torch

    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "results.csv"); df.columns = [c.strip() for c in df.columns]
    if close_epoch is None:
        close_epoch = _derive_close_epoch(run_dir) or int(df["epoch"].max())
    mcol = "metrics/mAP50(B)"
    top_a = df.nlargest(top_map, mcol)["epoch"].astype(int).tolist()
    post = df[df["epoch"] > close_epoch]
    top_b = post.nlargest(top_post, mcol)["epoch"].astype(int).tolist()
    shortlist = sorted(set(top_a) | set(top_b))

    tag = tag or "select_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir) if out_dir else run_dir / "eval" / tag
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)

    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(Path(__file__).resolve().parent),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    manifest = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv, cwd=str(Path.cwd()), code_file=str(Path(__file__).resolve()),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--", Path(__file__).name)),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      ultralytics=ultralytics.__version__),
        run_dir=str(run_dir), valid_data=str(valid_data), neg_folder=str(neg_folder),
        neg_count=len(list(Path(neg_folder).glob("*"))),
        params=dict(conf_min=conf_min, top_map=top_map, top_post=top_post, fp_budget=fp_budget,
                    beta=beta, close_epoch=int(close_epoch), agnostic_nms=agnostic_nms, iou=iou),
        shortlist=shortlist, shortlist_topmap=top_a, shortlist_postclosure=top_b,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[select] tag={tag}  out_dir={out_dir}")
    print(f"[select] shortlist={shortlist}  (top_map={top_a}, post>{close_epoch}={top_b})")

    rows, frontiers, op_thr = [], {}, {}
    for ep in shortlist:
        ck = run_dir / "weights" / f"epoch{ep - 1}.pt"   # results epoch 1-based; files epoch0..N-1
        if not ck.exists():
            print("skip (missing):", ck); continue
        strip_optimizer(str(ck), s="/tmp/_sel_ckpt.pt")  # deployable EMA/FP16, like best.pt
        cv = _compute_curves("/tmp/_sel_ckpt.pt", valid_data, neg_folder, conf_min=conf_min,
                             strip=False, cache=out_dir / "curves" / f"epoch{ep}.npy",
                             agnostic_nms=agnostic_nms, iou=iou)
        fr = pd.DataFrame(epoch_frontier(cv, beta=beta)); fr.insert(0, "epoch", ep)
        frontiers[ep] = fr
        feas = fr[fr.fp_img_rate <= fp_budget]
        op = feas.loc[feas.recall.idxmax()] if len(feas) else fr.loc[fr.fp_img_rate.idxmin()]
        op_thr[ep] = list(op.thr_list)                   # winner's deployable per-class thresholds
        # mAP50 recomputed on THIS valid_data + prod NMS (consistent w/ the frontier); results.csv
        # only gates the shortlist. Old caches without map50 fall back to results.csv.
        m50 = float(cv.get("map50", df.loc[df.epoch == ep, mcol].iloc[0]))
        rows.append(dict(epoch=ep, map50=m50, recall_at_budget=float(op.recall),
                         fp_at_op=float(op.fp_img_rate), feasible=bool(len(feas))))
        print(f"  ep{ep:3d}  mAP50={m50:.4f}  recall@FP<={fp_budget:g}%={op.recall:.3f}  "
              f"(FP={op.fp_img_rate:.1f}% @ w_neg={op.w_neg:g})")

    pd.concat(frontiers.values(), ignore_index=True).to_csv(out_dir / "frontier_points.csv", index=False)
    R = pd.DataFrame(rows)
    feasR = R[R.feasible]
    R = R.sort_values(["recall_at_budget", "map50"], ascending=False)
    R.to_csv(out_dir / "frontiers.csv", index=False)
    win = (feasR if len(feasR) else R).sort_values(["recall_at_budget", "map50"],
                                                   ascending=False).iloc[0]
    win_ep = int(win.epoch)

    # "did it matter": does the FP frontier pick differ from argmax-mAP50, and how rank-correlated?
    argmax_map_ep = int(R.loc[R.map50.idxmax(), "epoch"])
    rho = float(R[["map50", "recall_at_budget"]].corr(method="spearman").iloc[0, 1]) if len(R) > 1 else float("nan")
    changed = win_ep != argmax_map_ep

    # winner's per-class-optimal thresholds at its operating point -> id-aligned YAML for eval_fp
    names = np.load(out_dir / "curves" / f"epoch{win_ep}.npy", allow_pickle=True).item()["names"]
    thr_list = op_thr[win_ep]
    # copy the deployable (optimizer-stripped EMA/FP16) winner checkpoint into the eval folder
    best_pt = out_dir / f"best_ep{win_ep}.pt"
    strip_optimizer(str(run_dir / "weights" / f"epoch{win_ep - 1}.pt"), s=str(best_pt))
    import yaml
    thr_path = out_dir / f"class_thresholds_ep{win_ep}.yaml"
    with open(thr_path, "w") as f:
        # NOTE: 'epoch' is the 1-based results.csv row; 'weights_file' is the 0-based checkpoint
        # name for that epoch (Ultralytics saves the Nth epoch as epoch{N-1}.pt). They differ by 1.
        f.write(f"# epoch {win_ep} = results.csv row (1-based); weights/epoch{win_ep - 1}.pt is its "
                f"checkpoint (0-based file naming, off by one). 'model' below is the stripped copy.\n")
        yaml.safe_dump({"model": str(best_pt),
                        "source_weights": str(run_dir / "weights" / f"epoch{win_ep - 1}.pt"),
                        "epoch": win_ep, "weights_file": f"epoch{win_ep - 1}.pt",
                        "fp_budget": fp_budget, "beta": beta,
                        "agnostic_nms": agnostic_nms, "iou": iou,
                        "names": [names[c] for c in range(len(names))],
                        "thresholds": thr_list}, f, sort_keys=False)

    # overlay plot: every model's frontier + chosen operating points + budget line
    fig, ax = plt.subplots(figsize=(9, 6))
    for ep, fr in frontiers.items():
        ax.plot(fr.fp_img_rate, fr.recall, "-", lw=1, alpha=.7, label=f"ep{ep}")
        o = fr[fr.fp_img_rate <= fp_budget]
        if len(o):
            o = o.loc[o.recall.idxmax()]; ax.scatter(o.fp_img_rate, o.recall, s=30, zorder=3)
    ax.axvline(fp_budget, color="g", ls="--", alpha=.6, label=f"FP budget {fp_budget:g}%")
    ax.set_xlabel("NoLitter image-FP rate %  (per-class-optimal thresholds, prod NMS)")
    ax.set_ylabel("micro recall (valid)")
    ax.set_title(f"Per-epoch deployable frontier — winner ep{win_ep}")
    ax.set_xlim(0, max(2 * fp_budget, 5)); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(out_dir / "frontiers.png", dpi=120); plt.close()

    # human-readable summary
    lines = [f"frontier_select  tag={tag}", f"run_dir={run_dir}", f"timestamp={manifest['timestamp']}",
             f"git_commit={manifest['git_commit']}  dirty={manifest['git_dirty']}",
             f"params: conf_min={conf_min} agnostic_nms={agnostic_nms} iou={iou} beta={beta} "
             f"fp_budget={fp_budget}% close_epoch={close_epoch}",
             f"shortlist (top_map {top_a} U post-closure {top_b}) = {shortlist}",
             "  [shortlist gated by results.csv mAP50; the mAP50 column below is RECOMPUTED on "
             "valid_data + prod NMS]", "",
             "epoch  mAP50   recall@budget  FP%@op  feasible", "-" * 48]
    for _, r in R.iterrows():
        lines.append(f"{int(r.epoch):5d}  {r.map50:.4f}  {r.recall_at_budget:11.3f}  "
                     f"{r.fp_at_op:6.1f}  {bool(r.feasible)}")
    lines += ["", f"WINNER: epoch {win_ep}  (max recall under {fp_budget:g}% FP, tie-break mAP50)",
              f"  recall={float(win.recall_at_budget):.3f}  mAP50={float(win.map50):.4f}",
              f"  deployable checkpoint -> {best_pt.name}  (optimizer-stripped)",
              f"  per-class thresholds  -> {thr_path.name}", "",
              "DID-IT-MATTER diagnostic:",
              f"  argmax-mAP50 epoch = {argmax_map_ep};  FP-frontier winner = {win_ep};  "
              f"frontier changed the pick = {changed}",
              f"  Spearman(mAP50, recall@budget) across shortlist = {rho:.3f}  "
              f"(near 1.0 => FP axis redundant for epoch choice; thresholds still required)",
              f"  eval at deploy: python evalFalsePositives.py --mode eval --no-strip "
              f"--model {best_pt} --class-thr {thr_path}"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-9:]))
    print(f"\n[select] wrote {out_dir}/ (manifest.json, frontiers.csv/png, summary.txt, "
          f"{thr_path.name}, {best_pt.name})")
    return R


def frontier_models(checkpoints, valid_data, neg_folder, conf_min=0.15, fp_budget=10.0, beta=0.5,
                    agnostic_nms=False, iou=0.7, tag=None, out_dir=None):
    """Same per-class-optimal frontier stats as frontier_select, but on explicit checkpoints
    (e.g. best.pt, last.pt) instead of an mAP50 epoch shortlist.

    Each checkpoint is stripped (deployable EMA/FP16), validated (mAP50 + recall curves) and run
    through epoch_frontier; per-model deployable thresholds (at the fp_budget operating point) are
    written for every model, and the model with the highest recall under budget is flagged winner.
    Outputs go to <run>/eval/<tag>/ (run dir inferred from the first checkpoint's grandparent).
    """
    import json, subprocess, datetime, sys, yaml
    import numpy as np, pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ultralytics.utils.torch_utils import strip_optimizer
    import ultralytics, torch

    cks = [Path(c) for c in checkpoints]
    run_dir = cks[0].resolve().parent.parent              # .../run/weights/x.pt -> .../run
    tag = tag or "models_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir) if out_dir else run_dir / "eval" / tag
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)

    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(Path(__file__).resolve().parent),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    manifest = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv, cwd=str(Path.cwd()), code_file=str(Path(__file__).resolve()),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--", Path(__file__).name)),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      ultralytics=ultralytics.__version__),
        run_dir=str(run_dir), valid_data=str(valid_data), neg_folder=str(neg_folder),
        neg_count=len(list(Path(neg_folder).glob("*"))),
        params=dict(conf_min=conf_min, fp_budget=fp_budget, beta=beta,
                    agnostic_nms=agnostic_nms, iou=iou),
        checkpoints=[str(c) for c in cks],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[models] tag={tag}  out_dir={out_dir}")
    print(f"[models] checkpoints={[c.name for c in cks]}")

    rows, frontiers, thr_paths = [], {}, {}
    for ck in cks:
        if not ck.exists():
            print("skip (missing):", ck); continue
        label = ck.stem                                   # best, last, ...
        deploy_pt = out_dir / f"{label}_stripped.pt"      # deployable copy kept in the eval folder
        strip_optimizer(str(ck), s=str(deploy_pt))
        cv = _compute_curves(str(deploy_pt), valid_data, neg_folder, conf_min=conf_min,
                             strip=False, cache=out_dir / "curves" / f"{label}.npy",
                             agnostic_nms=agnostic_nms, iou=iou)
        fr = pd.DataFrame(epoch_frontier(cv, beta=beta)); fr.insert(0, "model", label)
        frontiers[label] = fr
        feas = fr[fr.fp_img_rate <= fp_budget]
        op = feas.loc[feas.recall.idxmax()] if len(feas) else fr.loc[fr.fp_img_rate.idxmin()]
        m50 = float(cv["map50"])
        rows.append(dict(model=label, source=str(ck), map50=m50, recall_at_budget=float(op.recall),
                         fp_at_op=float(op.fp_img_rate), feasible=bool(len(feas))))
        # per-model deployable thresholds at its operating point
        names = cv["names"]
        thr_path = out_dir / f"class_thresholds_{label}.yaml"
        with open(thr_path, "w") as f:
            yaml.safe_dump({"model": str(deploy_pt), "source_weights": str(ck),
                            "label": label, "map50": m50, "fp_budget": fp_budget, "beta": beta,
                            "agnostic_nms": agnostic_nms, "iou": iou,
                            "names": [names[c] for c in range(len(names))],
                            "thresholds": list(op.thr_list)}, f, sort_keys=False)
        thr_paths[label] = thr_path
        print(f"  {label:6s}  mAP50={m50:.4f}  recall@FP<={fp_budget:g}%={op.recall:.3f}  "
              f"(FP={op.fp_img_rate:.1f}% @ w_neg={op.w_neg:g})")

    pd.concat(frontiers.values(), ignore_index=True).to_csv(out_dir / "frontier_points.csv", index=False)
    R = pd.DataFrame(rows).sort_values(["recall_at_budget", "map50"], ascending=False)
    R.to_csv(out_dir / "frontiers.csv", index=False)
    feasR = R[R.feasible]
    win = (feasR if len(feasR) else R).iloc[0]
    win_label = str(win.model)

    # overlay plot
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, fr in frontiers.items():
        ax.plot(fr.fp_img_rate, fr.recall, "-", lw=1.3, alpha=.8, label=label)
        o = fr[fr.fp_img_rate <= fp_budget]
        if len(o):
            o = o.loc[o.recall.idxmax()]; ax.scatter(o.fp_img_rate, o.recall, s=40, zorder=3)
    ax.axvline(fp_budget, color="g", ls="--", alpha=.6, label=f"FP budget {fp_budget:g}%")
    ax.set_xlabel("NoLitter image-FP rate %  (per-class-optimal thresholds)")
    ax.set_ylabel("micro recall (valid)")
    ax.set_title(f"Per-model deployable frontier — winner {win_label}")
    ax.set_xlim(0, max(2 * fp_budget, 5)); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / "frontiers.png", dpi=120); plt.close()

    lines = [f"frontier_models  tag={tag}", f"run_dir={run_dir}", f"timestamp={manifest['timestamp']}",
             f"git_commit={manifest['git_commit']}  dirty={manifest['git_dirty']}",
             f"params: conf_min={conf_min} agnostic_nms={agnostic_nms} iou={iou} beta={beta} "
             f"fp_budget={fp_budget}%", "",
             "model   mAP50   recall@budget  FP%@op  feasible", "-" * 48]
    for _, r in R.iterrows():
        lines.append(f"{str(r.model):6s}  {r.map50:.4f}  {r.recall_at_budget:11.3f}  "
                     f"{r.fp_at_op:6.1f}  {bool(r.feasible)}")
    lines += ["", f"WINNER: {win_label}  (max recall under {fp_budget:g}% FP, tie-break mAP50)",
              f"  recall={float(win.recall_at_budget):.3f}  mAP50={float(win.map50):.4f}",
              f"  deployable checkpoint -> {win_label}_stripped.pt",
              f"  per-class thresholds  -> {thr_paths[win_label].name}", "",
              f"  eval at deploy: python evalFalsePositives.py --mode eval --no-strip --agnostic-nms "
              f"--model {out_dir / (win_label + '_stripped.pt')} --class-thr {thr_paths[win_label]}"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))
    print(f"\n[models] wrote {out_dir}/ (manifest.json, frontiers.csv/png, summary.txt, "
          f"per-model class_thresholds_*.yaml, *_stripped.pt)")
    return R


def _resolve_epochs(spec, run_dir):
    """Parse --epochs into a list of 1-based epochs. 'all' = every row in results.csv."""
    import pandas as pd
    if spec == "all":
        df = pd.read_csv(Path(run_dir) / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        return [int(e) for e in df["epoch"].tolist()]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _derive_close_epoch(run_dir):
    """mosaic-close boundary = epochs - close_mosaic, read from the run's args.yaml (None if unavailable)."""
    import yaml
    ap = Path(run_dir) / "args.yaml"
    if not ap.exists():
        return None
    with open(ap) as f:
        args = yaml.safe_load(f)
    if args.get("epochs") is not None and args.get("close_mosaic") is not None:
        return int(args["epochs"]) - int(args["close_mosaic"])
    return None


if __name__ == "__main__":
    # frontier (best-epoch FP-vs-recall sweep):  python evalFalsePositives.py --mode frontier --run <run>
    # sweep    (per-class thresholds + YAML):     python evalFalsePositives.py --mode sweep --epoch 85
    # eval     (FP at those thresholds):          python evalFalsePositives.py --mode eval --class-thr class_thresholds.yaml
    import argparse
    p = argparse.ArgumentParser(description="Select epoch (mAP50 shortlist + per-class FP frontier), sweep thresholds, or eval_fp.")
    p.add_argument("--mode", choices=["select", "models", "frontier", "sweep", "eval"], default="select",
                   help="select = frontier_select (mAP50 shortlist -> per-class FP frontier); "
                        "models = frontier_models (same frontier on explicit checkpoints via --models, e.g. best.pt last.pt); "
                        "frontier = sweep_frontier (coarse fixed-conf FP-vs-recall); sweep = threshold_analysis; eval = eval_fp")
    p.add_argument("--run", default="/home/charles/Programs/ultralytics-fork/runs/segment/train-8",
                   help="training run dir (contains weights/epochN.pt)")
    p.add_argument("--epoch", default=85, help="checkpoint file number, i.e. weights/epoch<N>.pt")
    p.add_argument("--model", default=None, help="explicit checkpoint path (overrides --run/--epoch, e.g. best.pt)")
    p.add_argument("--recall-mode", default="micro", choices=["micro", "macro"],
                   help="[sweep] micro = instance-weighted (deployment); macro = mean per-class (matches recall(B))")
    p.add_argument("--neg", default="/home/charles/Programs/datasetManipulation/NoLitter-3/train/split/images",
                   help="NoLitter hard-negative images folder")
    p.add_argument("--valid", default="/home/charles/Programs/datasetManipulation/datasets/Dataset-ViPARE-34-split/data.yaml",
                   help="[sweep] data.yaml for the recall/precision axis")
    p.add_argument("--fp-budget", type=float, default=0.15, help="[sweep] max NoLitter image-FP rate for the budget threshold")
    p.add_argument("--w-neg", type=float, default=1.0, help="[sweep] weight of empty-scene FPs (deployment empty-frame prior)")
    p.add_argument("--beta", type=float, default=0.5, help="[sweep] F-beta beta (<1 favors precision)")
    p.add_argument("--epochs", default="all",
                   help="[frontier] 1-based epochs to evaluate: 'all', comma list, or range a-b (e.g. 70-80)")
    p.add_argument("--close-epoch", type=int, default=None,
                   help="[frontier] mosaic-close boundary for annotation; default derived from args.yaml (epochs - close_mosaic)")
    p.add_argument("--out-csv", default="fp_frontier.csv", help="[frontier] output CSV path")
    p.add_argument("--out-png", default="fp_frontier.png", help="[frontier] output plot path")
    p.add_argument("--top-map", type=int, default=5, help="[select] top-N epochs by mAP50(B) over the whole run")
    p.add_argument("--top-post", type=int, default=5, help="[select] top-N post-closure epochs by mAP50(B)")
    p.add_argument("--conf-min", type=float, default=0.15, help="[select/sweep] low conf for the single NoLitter+val pass")
    p.add_argument("--models", nargs="+", default=None,
                   help="[models] explicit checkpoint paths to compare on the frontier (e.g. best.pt last.pt)")
    p.add_argument("--tag", default=None, help="[select] eval subfolder name under <run>/eval/ (default: timestamp)")
    p.add_argument("--out-dir", default=None, help="[select] explicit output dir (default <run>/eval/<tag>)")
    p.add_argument("--class-aware-nms", dest="agnostic_nms", action="store_false", default=False,
                   help="DEFAULT: class-aware NMS (per-class), for clean class-independent thresholds + standard mAP")
    p.add_argument("--agnostic-nms", dest="agnostic_nms", action="store_true",
                   help="class-agnostic NMS (production-faithful, one box per region) — use for a deployment eval check")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (production value)")
    p.add_argument("--class-thr", default=None, help="[eval] per-class threshold YAML/list (else use --conf)")
    p.add_argument("--conf", type=float, default=0.334, help="operating conf ([eval] global conf / [frontier] FP operating point)")
    p.add_argument("--no-strip", action="store_true", help="skip strip_optimizer (pass an already-deployable .pt)")
    a = p.parse_args()

    ckpt = a.model or f"{a.run}/weights/epoch{a.epoch}.pt"

    if a.mode == "select":
        frontier_select(a.run, a.valid, a.neg, conf_min=a.conf_min, top_map=a.top_map,
                        top_post=a.top_post, close_epoch=a.close_epoch, fp_budget=a.fp_budget * 100,
                        agnostic_nms=a.agnostic_nms, iou=a.iou, tag=a.tag, out_dir=a.out_dir)
    elif a.mode == "models":
        if not a.models:
            p.error("--mode models requires --models <ckpt> [<ckpt> ...]")
        frontier_models(a.models, a.valid, a.neg, conf_min=a.conf_min, fp_budget=a.fp_budget * 100,
                        beta=a.beta, agnostic_nms=a.agnostic_nms, iou=a.iou, tag=a.tag, out_dir=a.out_dir)
    elif a.mode == "frontier":
        epochs = _resolve_epochs(a.epochs, a.run)
        close_epoch = a.close_epoch if a.close_epoch is not None else (_derive_close_epoch(a.run) or 80)
        print(f"frontier: {len(epochs)} epochs ({epochs[0]}..{epochs[-1]})  conf={a.conf}  close_epoch={close_epoch}")
        sweep_frontier(a.run, a.neg, conf=a.conf, epochs=epochs, close_epoch=close_epoch,
                       out_csv=a.out_csv, out_png=a.out_png)
    elif a.mode == "sweep":
        threshold_analysis(ckpt, a.valid, a.neg, fp_budget=a.fp_budget, w_neg=a.w_neg,
                           beta=a.beta, conf_min=a.conf_min, recall_mode=a.recall_mode,
                           agnostic_nms=a.agnostic_nms, iou=a.iou, strip=not a.no_strip)
    else:  # eval: run eval_fp at a per-class config (or a global conf)
        model = ckpt
        if not a.no_strip:
            from ultralytics.utils.torch_utils import strip_optimizer
            strip_optimizer(ckpt, s="/tmp/_eval_ckpt.pt")   # EMA/FP16 deployable weights
            model = "/tmp/_eval_ckpt.pt"
        if a.class_thr:
            eval_fp(model, a.neg, class_thr=a.class_thr, agnostic_nms=a.agnostic_nms, iou=a.iou)
        else:
            eval_fp(model, a.neg, conf_threshold=a.conf, agnostic_nms=a.agnostic_nms, iou=a.iou)
