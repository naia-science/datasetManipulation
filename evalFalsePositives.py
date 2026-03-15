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

def eval_fp(model_path, images_folder, conf_threshold=0.25):
    # Load model and run inference
    model = YOLO(model_path)
    results = model.predict(source=images_folder, conf=conf_threshold, verbose=False)

    # Count false positives
    total_fps = 0
    images_with_fps = 0
    fps_per_class = defaultdict(int)
    fps_per_image = []

    save_dir = Path("runs/detect/eval_fp")
    save_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
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
            annotated = result.plot()
            out_path = save_dir / Path(result.path).name
            cv2.imwrite(str(out_path), annotated)

        fps_per_image.append(image_fps)

    # Print summary
    print(f"\nTotal images: {len(results)}")
    print(f"Images with FPs: {images_with_fps}")
    print(f"Total FPs: {total_fps}")
    print(f"FP rate: {images_with_fps/len(results)*100:.2f}%")
    print(f"\nFPs per class:")
    for class_name, count in sorted(fps_per_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")

    # Save to CSV
    with open('fps_per_image.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Total FPs'] + sorted(fps_per_class.keys()))
        
        for img_data in fps_per_image:
            row = [img_data['path'], img_data['count']]
            for class_name in sorted(fps_per_class.keys()):
                row.append(img_data['classes'].get(class_name, 0))
            writer.writerow(row)

    print(f"\nAnnotated images saved to {save_dir}/")
    print("Saved to fps_per_image.csv")

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