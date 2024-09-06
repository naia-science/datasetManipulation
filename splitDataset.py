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
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized_image


def clip_coords(coords):
    coords[..., 0] = coords[..., 0].clip(0, 1.0)  # x
    coords[..., 1] = coords[..., 1].clip(0, 1.0)  # y
    # Check that the annotation is useful
    if np.any(coords[..., 0] > 0) and np.any(coords[..., 1] > 0) and np.any(coords[..., 0]<1.0) and np.any(coords[..., 1]<1.0):
        return coords
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
    # print(f"img:{h}x{w} new_img:{x_min_abs} {y_min_abs} {x_max_abs} {y_max_abs}")

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


def split_img(img, ann, max_size=1280):
    w,h,_ = img.shape
    nb_splits_w, nb_splits_h = w // max_size, h // max_size
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
                    if coords is not None:
                        new_ann[i] = coords
                new_anns.append(new_ann)
        
        return imgs, new_anns

from pathlib import Path

def split_large_images(im_dir, max_size=1280):
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
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected")
    else:
        LOGGER.info("Detection labels detected")

    save_dir = Path(im_dir).parent / "split"
    save_dir.mkdir(parents=True, exist_ok=True)
    new_im_dir = save_dir / "images"
    new_im_dir.mkdir(parents=True, exist_ok=True)
    new_label_dir = save_dir / "labels"
    new_label_dir.mkdir(parents=True, exist_ok=True)
    
    total_num_processed = 0
    total_num_generated = 0
    
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
            smallest, englobing = get_annotation_properties(new_ann)
        else:
            new_ann = l["segments"]
        
        if smallest < 0.01: # still small
            imgs, new_anns = split_img(im, new_ann)
        else:
            im = resize_image_cv2(im, max_size=max_size)
            new_anns = [{i:n for i, n in enumerate(new_ann)}]
            imgs = [im]
    
        for k, (img, new_ann) in enumerate(zip(imgs, new_anns)):
            total_num_generated += 1
            texts = []
            name = Path(l["im_file"]).stem + "_" + str(k)
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
