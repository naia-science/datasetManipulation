from ultralytics import YOLO
from collections import defaultdict
import csv

def eval_fp(model_path, images_folder, conf_threshold=0.25):
    # Load model and run inference
    model = YOLO(model_path)
    results = model.predict(source=images_folder, conf=conf_threshold, verbose=False)

    # Count false positives
    total_fps = 0
    images_with_fps = 0
    fps_per_class = defaultdict(int)
    fps_per_image = []

    for result in results:
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
        
        fps_per_image.append(image_fps)

    # Print summary
    print(f"\nTotal images: {len(results)}")
    print(f"Images with FPs: {images_with_fps}")
    print(f"Total FPs: {total_fps}")
    print(f"FP rate: {images_with_fps/len(results)*100:.2f}%")
    print(f"\nPer class: {dict(fps_per_class)}")

    # Save to CSV
    with open('fps_per_image.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Total FPs'] + sorted(fps_per_class.keys()))
        
        for img_data in fps_per_image:
            row = [img_data['path'], img_data['count']]
            for class_name in sorted(fps_per_class.keys()):
                row.append(img_data['classes'].get(class_name, 0))
            writer.writerow(row)

    print("\nSaved to fps_per_image.csv")