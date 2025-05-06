import math
import numpy as np
from functools import reduce
import cv2

def resize_image_cv2(image, max_size=1280):
    original_h, original_w = image.shape[:2]

    # Determine the scaling factor based on the larger dimension
    if original_w > original_h:
        scale_factor = max_size / original_w
    else:
        scale_factor = max_size / original_h
    
    # Only downsize
    if scale_factor > 1.0:
        return image

    # Calculate new dimensions
    new_w = int(original_w * scale_factor)
    new_h = int(original_h * scale_factor)

    # Resize the image using cv2
    try:
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except:
        print(f"cannot resize image {original_w}x{original_h} to {new_w}x{new_h}")
        resized_image = None
    return resized_image


def clip_coords(coords):
    coords[..., 0] = coords[..., 0].clip(0, 1.0)  # x
    coords[..., 1] = coords[..., 1].clip(0, 1.0)  # y
    # Check that the annotation is useful
    if np.any(coords[..., 0] > 0) and np.any(coords[..., 1] > 0) and np.any(coords[..., 0]<1.0) and np.any(coords[..., 1]<1.0):
        return coords
    else:
        return None
    
def remove_zero_area_points(polygon):
    """
    Removes points that create zero-area zones in the given polygon.
    A polygon is represented as a NumPy array of shape (n_points, 2).

    Parameters:
        polygon (np.ndarray): Input polygon, shape (n_points, 2).

    Returns:
        np.ndarray: The cleaned polygon with no zero-area zones.
    """
    if polygon is None:
        return None
    # deduplicate points
    unique_polygon = []
    for i, point in enumerate(polygon):
        if i == 0 or not np.allclose(point, unique_polygon[-1]):
            unique_polygon.append(point)
    
    # case where first point = last point
    if np.allclose(unique_polygon[0], unique_polygon[-1]):
        unique_polygon = unique_polygon[:-1]
    unique_polygon = np.array(unique_polygon)
    
    cleaned_polygon = []
    n = len(unique_polygon)
    # remove points in zero area triangles
    prev_idx = -1
    for i in range(n):
        p_prev = unique_polygon[prev_idx]  # Previous point
        p_curr = unique_polygon[i]      # Current point
        p_next = unique_polygon[(i + 1) % n]  # Next point (wrap-around)

        # Calculate the signed area of the triangle formed by these three points
        area = 0.5 * np.abs(
            p_prev[0] * (p_curr[1] - p_next[1]) +
            p_curr[0] * (p_next[1] - p_prev[1]) +
            p_next[0] * (p_prev[1] - p_curr[1])
        )

        # If the area is non-zero, keep the current point
        if area > 1e-15:
            cleaned_polygon.append(p_curr)
            prev_idx = i
            
    if len(cleaned_polygon) > 2:
        return np.array(cleaned_polygon)
    else:
        return None


def get_annotation_properties(annotations):
    """
    Calculate the ratio of the smallest annotation area to image size, and the normalized englobing rectangle.
    
    Args:
        annotations (list): List of annotations, where each annotation is a list of [x, y] coordinates (normalized).
    
    Returns:
        smallest_area_ratio (float): Ratio of the smallest annotation (bounding box area) over image size.
        global_rect (tuple): Normalized bounding box that covers all annotations (x_min, y_min, x_max, y_max).
    """
    if not annotations:
        return 0.0, (0.0, 0.0, 1.0, 1.0)
    smallest_area = 1.0
    
    # Initialize values for global bounding box in normalized coordinates
    x_min_global, y_min_global = 1.0, 1.0
    x_max_global, y_max_global = 0.0, 0.0

    for annotation in annotations:
        coords = np.array(annotation)
        
        # Get bounding box for this annotation in normalized coordinates
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        # Calculate area of the bounding box in normalized coordinates
        area = (x_max - x_min) * (y_max - y_min)
        smallest_area = min(smallest_area, area)
        
        # Update global bounding box (englobing rect in normalized coordinates)
        x_min_global = min(x_min_global, x_min)
        y_min_global = min(y_min_global, y_min)
        x_max_global = max(x_max_global, x_max)
        y_max_global = max(y_max_global, y_max)
    
    smallest_area_ratio = math.sqrt(smallest_area)  

    # Global bounding box in normalized coordinates
    global_rect = (x_min_global, y_min_global, x_max_global, y_max_global)
    
    return smallest_area_ratio, global_rect


def crop_image_with_margin(image, annotations, global_rect, margin=0.1):
    """
    Crops the image based on the englobing rectangle of annotations, with a margin, and adjusts annotations accordingly.
    
    Args:
        image (numpy.ndarray): The input image to crop.
        annotations (list): List of annotations, where each annotation is a list of [x, y] coordinates (normalized).
        global_rect (tuple): The englobing rectangle of all annotations in normalized coordinates (x_min, y_min, x_max, y_max).
        margin (float): The margin to apply around the englobing rectangle (in normalized coordinates).
    
    Returns:
        cropped_image (numpy.ndarray): The cropped image.
        new_annotations (list): The adjusted annotations for the cropped image (normalized).
    """
    h, w, _ = image.shape  # Get image dimensions

    # Extract the global bounding box from the normalized coordinates
    x_min, y_min, x_max, y_max = global_rect

    # Apply margin to the bounding box in normalized coordinates
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(1, x_max + margin)
    y_max = min(1, y_max + margin)


    # Convert the normalized coordinates back to absolute pixel coordinates for cropping
    x_min_abs = int(x_min * w)
    y_min_abs = int(y_min * h)
    x_max_abs = int(x_max * w)
    y_max_abs = int(y_max * h)

    # Calculate the aspect ratio of the cropped region
    aspect_ratio = (x_max_abs - x_min_abs) / (y_max_abs - y_min_abs)

    if aspect_ratio > 2:  # Width is more than twice the height
        desired_height = (x_max_abs - x_min_abs) / 2
        y_center = int((y_min_abs + y_max_abs) / 2)
        half_height = int(desired_height / 2)
        if y_center - half_height < 0:
            y_min_abs = 0
            y_max = min(h, 2 * half_height)
        elif y_center + half_height > h:
            y_max_abs = h
            y_min_abs = max(0, h - 2 * half_height)
        else:
            y_min_abs = y_center - half_height
            y_max_abs = y_center + half_height
    elif aspect_ratio < 0.5:  # Height is more than twice the width        
        desired_width = (y_max_abs - y_min_abs) / 2
        x_center = int((x_min_abs + x_max_abs) / 2)
        half_width = int(desired_width / 2)
        if x_center - half_width < 0:
            x_min_abs = 0
            x_max_abs = min(w, 2 * half_width)
        elif x_center + half_width > w:
            x_max_abs = w
            x_min_abs = max(0, w - 2 * half_width)
        else:
            x_min_abs = x_center - half_width
            x_max_abs = x_center + half_width


    # Crop the image
    cropped_image = image[y_min_abs:y_max_abs, x_min_abs:x_max_abs]

    # Adjust annotations to fit within the cropped image
    new_annotations = []
    for annotation in annotations:
        new_coords = np.array(annotation)
        
        # Shift coordinates by the top-left corner of the crop
        new_coords[:, 0] -= x_min  # Adjust x by the crop margin in normalized coords
        new_coords[:, 1] -= y_min  # Adjust y by the crop margin in normalized coords
        
        # Normalize coordinates with respect to the new cropped size
        new_coords[:, 0] /= (x_max - x_min)  # New width in normalized coords
        new_coords[:, 1] /= (y_max - y_min)  # New height in normalized coords
        
        new_annotations.append(new_coords)

    return cropped_image, new_annotations


def split_img(img, ann, max_size=1280, max_split=2):
    w,h,_ = img.shape
    nb_splits_w, nb_splits_h = min(w // max_size, max_split-1), min(h // max_size, max_split-1)
    print(nb_splits_h)
    print(nb_splits_w)
    if nb_splits_w == 0 and nb_splits_h == 0:
        # no split, return just the basic image
        # we need to convert the ann to dictionnary to match the expected output
        new_ann = {i:a for i,a in enumerate(ann)}
        return [img], [new_ann]
    else:
        nw, nh = int(w / (nb_splits_w + 1)), int(h / (nb_splits_h + 1))
        wcoords = [(i * nw, (i+1) * nw) for i in range(nb_splits_w +1 )]
        hcoords = [(i * nh, (i+1) * nh) for i in range(nb_splits_h +1 )]
        imgs = []
        new_anns = []
        for wc in wcoords:
            for hc in hcoords:
                # split images
                imgs.append(img[wc[0]:wc[1], hc[0]:hc[1]])
                new_ann = {}
                xymin = np.array([wc[0]/w, hc[0]/h])[::-1]
                xyscale = np.array([w/(wc[1]-wc[0]), h/(hc[1]-hc[0])])[::-1]
                # get ann coords
                for i, s in enumerate(ann):
                    coords = np.array(s)
                    coords -= xymin
                    coords *= xyscale
                    coords = clip_coords(coords)
                    coords = remove_zero_area_points(coords)
                    if coords is not None:
                        new_ann[i] = coords
                new_anns.append(new_ann)
        return imgs, new_anns

from pathlib import Path

def split_large_images(im_dir, max_size=1280, max_splits=2):
    """
    Convert segmentation dataset splitting images that have a size larger than 1280 into several

    Args:
        im_dir (str | Path): Path to image directory to convert.
        will generate a new folder "split" with the same image and labels directories
    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from tqdm import tqdm

    from ultralytics.data import YOLODataset
    from ultralytics.utils import LOGGER

    # NOTE: add placeholder to pass class index check
    data = dict(names=list(range(1000)))
    data["channels"] = 3
    dataset = YOLODataset(im_dir, data=data)
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected")
    else:
        LOGGER.error("Detection labels detected")
        return 

    save_dir = Path(im_dir).parent / "split"
    save_dir.mkdir(parents=True, exist_ok=True)
    new_im_dir = save_dir / "images"
    new_im_dir.mkdir(parents=True, exist_ok=True)
    new_label_dir = save_dir / "labels"
    new_label_dir.mkdir(parents=True, exist_ok=True)
    
    total_num_processed = 0
    total_num_generated = 0
    total_cropped = []
    total_split = []
    
    for l in tqdm(dataset.labels, total=len(dataset.labels), desc="splitting images"):
        h, w = l["shape"]
        boxes = l["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        total_num_processed += 1
        # why unnormalize inplace?
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        smallest, englobing = get_annotation_properties(l["segments"])
        im = cv2.imread(l["im_file"])
        if smallest < 0.01: # threshold for cropping then splitting
            im, new_ann = crop_image_with_margin(im, l["segments"], englobing)
            total_cropped += [l["im_file"]]

            smallest, englobing = get_annotation_properties(new_ann)
        else:
            new_ann = l["segments"]
        
        if smallest < 0.01: # still small
            imgs, new_anns = split_img(im, new_ann, max_splits)
            total_cropped.remove(l["im_file"])
            total_split += [l["im_file"]]
        else:
            im = resize_image_cv2(im, max_size=max_size)
            if im is None:
                print("problem with "+l["im_file"])
                continue
            new_anns = [{i:n for i, n in enumerate(new_ann)}]
            imgs = [im]
        for i, (img, new_ann) in enumerate(zip(imgs, new_anns)):
            total_num_generated += 1
            
            texts = []
            name = Path(l["im_file"]).stem + "_" + str(i)
            img_file = new_im_dir / (name + Path(l["im_file"]).suffix)
            txt_file = new_label_dir / (name + ".txt")
            
            cls = l["cls"]
            for k, v in new_ann.items():
                v = v.flatten()
                line = (int(cls[k]), *v)
                texts.append(("%g " * len(line)).rstrip() % line)
            if texts:
                with open(txt_file, "a") as f:
                    f.writelines(text + "\n" for text in texts)
            cv2.imwrite(str(img_file.resolve()), img)

            
    LOGGER.info(f"Generated {total_num_generated} images and labels from {total_num_processed} original images, saved in {save_dir}")
    LOGGER.info(f"Num of images  split: {len(total_split)}, num just cropped: {len(total_cropped)}")
    """print("split:")
    for i in total_split:
        print(i)
    print("cropped:")
    for i in total_cropped: 
        print(i)"""
    # returns the last ones for display
    return imgs, new_anns


if __name__ == "__main__":
    # get arguments from command line, directory to split, and max size
    import argparse
    parser = argparse.ArgumentParser(description="Split large images into smaller ones")
    parser.add_argument("im_dir", type=str, help="Path to image directory to split")
    parser.add_argument("--max_size", type=int, default=1280, help="Maximum size of images to split")
    args = parser.parse_args()

    split_large_images(args.im_dir, args.max_size)
