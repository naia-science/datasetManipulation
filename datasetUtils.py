import os

os.system('pip install roboflow==0.2.29')
os.system('pip install ultralytics==8.0.196')

from roboflow import Roboflow
import ultralytics
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from pycocotools.coco import COCO
from PIL import Image, ExifTags
import numpy as np
import colorsys
from matplotlib import pylab
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import splitfolders
import shutil

# ------------------------------------------------- Get Roboflow dataset functions -------------------------------------------------

def dl_roboflow_dataset(ver):
    """Downloads the Roboflow dataset."""
    rf = Roboflow(api_key="REDACTED")
    project = rf.workspace("naia-science").project("dataset-vipare")
    dataset = project.version(ver).download("yolov8")

    # replace the following lines with python calls for creating directories and moving files
    os.makedirs(f"./datasets/", exist_ok=True)
    # move the downloaded dataset to the new directory
    os.rename(f"./Dataset-ViPARE-{ver}/", f"./datasets/Dataset-ViPARE-{ver}/")
    # move the data.yaml file to the new directory
    os.rename(f"./datasets/Dataset-ViPARE-{ver}/data.yaml", f"./datasets/data.yaml")


def display_test_image(roboVer, path):
    """Displays a random image from the Roboflow dataset."""
    im_path = os.path.join(path, 'Dataset-ViPARE-' + str(roboVer) + '/valid/images/')
    lab_path = os.path.join(path, 'Dataset-ViPARE-' + str(roboVer) + '/valid/labels/')
    
    #get random image from image directory
    image = random.choice(os.listdir(im_path))

    #get associated label by removing .jpg, adding .txt
    label = os.path.splitext(image)[0] + '.txt'

    #display image at im_path + image using plt
    
    img = mpimg.imread(im_path + image)
    imgplot = plt.imshow(img)

    #draw segmentation from label at lab_path + label using plt - it is a polygon, not a rectangle !

    with open(lab_path + label, 'r') as f:
        for line in f:
            polygon = line.split()[1:]
            polygon = [float(i) for i in polygon]

            #reshape polygon to be a list of tuples, each tuple being a point and knowing polygon values are between 0 and 1, need to multiply by image size
            polygon = [(int(polygon[i]*img.shape[1]), int(polygon[i+1]*img.shape[0])) for i in range(0, len(polygon), 2)]

            #draw polygon - TODO : use different color depending on class, add transparency for a better display?
            plt.fill(*zip(*polygon), 'r')
    plt.show()


def delete_roboflow_dataset(ver):
    """Deletes the Roboflow dataset."""
    os.system(f'rm -rf ./datasets/Dataset-ViPARE-{ver}')

# ------------------------------------------------- Get TACO dataset functions -------------------------------------------------
    
def dl_taco_dataset():
    """Downloads the TACO dataset."""
    # clone repo, install requirements
    os.system('git clone https://github.com/pedropro/TACO.git')
    os.system('pip install -r ./TACO/requirements.txt')
    
    # replace download.py for a paralellized + functional version from pr
    os.system('rm -rf ./TACO/download.py')
    os.chdir('./TACO/')
    os.system('git checkout 94031c21be6c6a9db247bf8284a55c96c21cfcf9 download.py')
    # launch download
    os.system('python download.py')
    os.chdir('..')

def delete_TACO_dataset():
    """Deletes the TACO dataset."""
    os.system('rm -rf ./TACO/')

def display_test_image_seg_TACO():
    """Displays a random image from the TACO dataset with its segmentation."""
    dataset_path = './TACO/data'
    anns_file_path = dataset_path + '/' + 'annotations.json'
    
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    
    #select image
    image_filepath = 'batch_1/000014.jpg'
    pylab.rcParams['figure.figsize'] = (28,28)


    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    # Loads dataset as a coco object
    coco = COCO(anns_file_path)

    # Find image id
    img_id = -1
    for img in imgs:
        if img['file_name'] == image_filepath:
            img_id = img['id']
            break

    # Show image and corresponding annotations
    if img_id == -1:
        print('Incorrect file name')
    else:

        # Load image
        print(image_filepath)
        I = Image.open(dataset_path + '/' + image_filepath)

        # Load and process image metadata
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)

        # Show image
        fig,ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(I)

        # Load mask ids
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns_sel = coco.loadAnns(annIds)

        # Show annotations
        for ann in anns_sel:
            color = colorsys.hsv_to_rgb(np.random.random(),1,1)
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
                p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
                ax.add_collection(p)
            [x, y, w, h] = ann['bbox']
            rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
                             facecolor='none', alpha=0.7, linestyle = '--')
            ax.add_patch(rect)

        plt.show()

# -------------------------------------------------  COCO to YOLO util functions -------------------------------------------------

def cocoToYolo(dir_path):
    """Converts the COCO dataset at dir_path to Yolov8 format."""
    import os
    import json
    import cv2
    import numpy as np
    import shutil
    from tqdm import tqdm
    from collections import defaultdict

    # Load the COCO annotations
    with open(os.path.join(dir_path, 'annotations.json')) as f:
        data = json.load(f)

    # Create a dictionary to map class names to class ids
    class_map = {}
    for i, category in enumerate(data['categories']):
        class_map[category['id']] = i

    # Create a dictionary to map image ids to image file names
    image_map = {}
    for image in data['images']:
        image_map[image['id']] = image['file_name']

    # Create a dictionary to map image file names to image ids
    image_id_map = {}
    for image in data['images']:
        image_id_map[image['file_name']] = image['id']

    # Create a dictionary to map image ids to image sizes
    image_size_map = {}
    for image in data['images']:
        image_size_map[image['id']] = (image['width'], image['height'])

    # Create a dictionary to map image ids to bounding boxes
    segs = defaultdict(list)
    for annotation in data['annotations']:
        
        image_id = annotation['image_id']
        class_id = class_map[annotation['category_id']]
    
        # Convert COCO segmentation to Yolov8 segmentation (no bounding box, just polygon)
        polygon = annotation['segmentation'][0]
        segs[image_id].append((class_id, polygon))

    # Create a directory to store the Yolov8 annotations
    yolo_dir = os.path.join(dir_path, 'yolo')
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
    os.makedirs(yolo_dir)
    
    # Create a directory to store the Yolov8 images
    yolo_img_dir = os.path.join(yolo_dir, 'images')
    os.makedirs(yolo_img_dir)

    # Create a directory to store the Yolov8 labels
    yolo_label_dir = os.path.join(yolo_dir, 'labels')
    os.makedirs(yolo_label_dir)

    # Convert the COCO annotations to Yolov8 annotations
    for image_id, seg in tqdm(segs.items()):
        # Load the image
        img = cv2.imread(os.path.join(dir_path, image_map[image_id]))
        if (img is None):
            print(f"Image {image_map[image_id]} not found")
            continue
        img_h, img_w, _ = img.shape

        # Create a file to store the Yolov8 annotations

        yolo_label_file = os.path.join(yolo_label_dir, image_map[image_id].replace('jpg', 'txt').replace('/', '_').replace('JPG', 'txt'))
        with open(yolo_label_file, 'w') as f:
            for class_id, polygon in seg:
                # Convert COCO polygon to Yolov8 polygon

                polygon = np.array(polygon).reshape(-1, 2).astype(float)
                polygon[:, 0] = polygon[:, 0] / (1.0 * img_w)
                polygon[:, 1] = polygon[:, 1] / (1.0 * img_h)
                polygon = polygon.reshape(-1)
                polygon = ' '.join([str(p) for p in polygon])

                # Write the Yolov8 annotation to the file
                f.write(f'{class_id} {polygon}\n')

        # Save the Yolov8 image)
        cv2.imwrite(os.path.join(yolo_img_dir, image_map[image_id].replace('/', '_')), img)

    print('Done!')

def split_dataset(dir_path, train_ratio, test_ratio, val_ratio):
    """Splits the dataset into train, test and validation sets."""
    splitfolders.ratio(dir_path, output=dir_path, seed=1337, ratio=(train_ratio, test_ratio, val_ratio))

# -------------------------------------------------  classes Adjustment util functions -------------------------------------------------
    
def tacoClassMatch(tacoClassID):
    """Matches the TACO class ID to the Naia class ID."""

    # May have to be updated if Naia classes or TACO classes are changed

    # bon au final je tire pas vraiment profit du dictionnaire des classes naia, mais remplir à la mano les liens Taco -> Naia aurait été beaucoup plus long
    # que de faire un matching comme j'ai fait, et certainement moins facilement maintenable
    
    # On crée le dictionari des classes Naia, pour pouvoir le modifier simplement si besoin
    NaiaClassesValue = { 0: "autre",
                    1: "autre-papier-carton",
                    2: "autre-plastique-fragments",
                    3: "autre-polystyrene",
                    4: "bouteille-en-plastique",
                    5: "bouteille-en-verre",
                    6: "cannette",
                    7:"emballage-alimentaire-papier",
                    8:"emballage-alimentaire-plastique",
                    9: "indefini",
                    10 : "megot",
                    11 : "sac-ordures-menageres"}


    # On l'inverse, pour pouvoir accéder à l'ID de la classe à partir de son nom, et rendre le matching plus lisible et maintenable
    
    NaiaClassesKey = {v: k for k, v in NaiaClassesValue.items()}
    match tacoClassID:
        case 59:
            return NaiaClassesKey["megot"]
        case 10 | 11 | 12:
            return NaiaClassesKey["cannette"]
        case 41 | 40 | 38:
            return NaiaClassesKey["sac-ordures-menageres"]
        case 6 | 26:
            return NaiaClassesKey["bouteille-en-verre"]
        case 4 | 5:
            return NaiaClassesKey["bouteille-en-plastique"]
        case 15 | 56 | 20 | 19 | 18 | 16:
            return NaiaClassesKey["emballage-alimentaire-papier"]
        case 55 | 49 | 47 | 22 | 46 | 45 | 44 | 43 | 42 | 39 | 37 | 24 | 21:
            return NaiaClassesKey["emballage-alimentaire-plastique"]
        case 7 | 48 | 36 | 29 | 27:
            return NaiaClassesKey["autre-plastique-fragments"]
        case 14 | 34 | 33 | 32 | 31 | 30 | 17:
            return NaiaClassesKey["autre-papier-carton"]
        case 57:
            return NaiaClassesKey["autre-polystyrene"]
        case 0 | 1 | 8 | 9 | 13 | 58 | 53 | 52 | 51 | 28 | 25:
            return NaiaClassesKey["autre"]
        case _:
            return NaiaClassesKey["indefini"]



def tacoClassesToNaia(path_to_Yolo):
    """Takes the path to the Yolo dataset, and changes the class number in the labels files to match the Naia classes."""
    # for train, test and val directories in yolo, open their subdirectory "labels".
    # for each file in the labels directories, open it, and for each line in the file, convert the taco class to a Naia class
    # (change the class number using the tacoClassMatch function)
    # write the new class number in the file, and save it in the same directory

#make sure to only replace the first instance of the number in the line, and not the potential second one (for the segmentation points)
    
    for directory in ["train", "test", "val"]:
        for file in os.listdir(path_to_Yolo + directory + "/labels"):
            with open(path_to_Yolo + directory + "/labels/" + file, "r") as f:
                lines = f.readlines()
            with open(path_to_Yolo + directory + "/labels/" + file, "w") as f:
                for line in lines:
                    tacoClassID = int(line.split()[0])
                    newClassID = tacoClassMatch(tacoClassID)
                    line = line.replace(str(tacoClassID), str(newClassID), 1)
                    f.write(line)

# -------------------------------------------------  Merge datasets util functions -------------------------------------------------
                    
def mergeDatasets(dataset1, dataset2, output):
    """Merges two datasets to the output directory."""

    if not os.path.exists(output):
        os.makedirs(output)

    # Copy the datasets val, train and test directories to the output directory
    for directory in ["train", "test", "val"]:
        if os.path.exists(dataset1 + "/" + directory):
            shutil.copytree(dataset1 + "/" + directory, output + "/" + directory)
        if os.path.exists(dataset2 + "/" + directory):
            shutil.copytree(dataset2 + "/" + directory, output + "/" + directory)
    
    if os.path.exists(dataset1 + "/data.yaml"):
        shutil.copy(dataset1 + "/data.yaml", output + "/data.yaml")
    elif os.path.exists(dataset2 + "/data.yaml"):
        shutil.copy(dataset2 + "/data.yaml", output + "/data.yaml")
    # replace the train, val and test directories in the data.yaml file with the new ones
    with open(output + "/data.yaml", "r") as f:
        lines = f.readlines()
    with open(output + "/data.yaml", "w") as f:
        for line in lines:
            if "train:" in line:
                f.write("train: " + output + "/train\n")
            elif "val:" in line:
                f.write("val: " + output + "/val\n")
            elif "test:" in line:
                f.write("test: " + output + "/test\n")
            else:
                f.write(line)