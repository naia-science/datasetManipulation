import numpy as np
from functools import reduce
import cv2

def clip_coords(coords):
    coords[..., 0] = coords[..., 0].clip(0, 1.0)  # x
    coords[..., 1] = coords[..., 1].clip(0, 1.0)  # y
    # Check that the annotation is useful
    if np.any(coords[..., 0] > 0) and np.any(coords[..., 1] > 0) and np.any(coords[..., 0]<1.0) and np.any(coords[..., 1]<1.0):
        return coords
    else:
        return None

def split_img(img, ann, max_size=1280):
    w,h,_ = img.shape
    nb_splits_w, nb_splits_h = w // max_size, h // max_size
    if nb_splits_w == 0 and nb_splits_h == 0:
        # no split, return just the basic image
        return [img], [ann]
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
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(l["im_file"])
        imgs, new_anns = split_img(im, l["segments"])
    
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
