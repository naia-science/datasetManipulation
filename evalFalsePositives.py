from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
import csv
import cv2

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
        image_fps = {'path': result.path, 'count': num_detections, 'classes': defaultdict(int)}

        if num_detections > 0:
            images_with_fps += 1
            total_fps += num_detections

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                fps_per_class[class_name] += 1
                image_fps['classes'][class_name] += 1

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