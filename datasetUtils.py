import os
from roboflow import Roboflow
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def dl_roboflow_dataset(ver):
    """
    Description:
      Downloads the Roboflow dataset.
    Usage:
      dl_roboflow_dataset(ver)
    Arguments:
      ver: Roboflow version to download
    """
    # check if file roboflowAPIKey.txt exists and read the key from it, first line only
    api_key = ""
    if os.path.exists("./roboflowAPIkey.txt"):
        with open("roboflowAPIkey.txt", "r") as f:
            api_key = f.readline().strip()
    else:
        print("No Roboflow API key file found, please create a roboflowAPIkey.txt file with your key in it.")
        return
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("naia-science").project("dataset-vipare")
    dataset = project.version(ver).download("yolov8")

    # replace the following lines with python calls for creating directories and moving files
    os.makedirs(f"./datasets/", exist_ok=True)
    # move the downloaded dataset to the new directory
    os.rename(f"./Dataset-ViPARE-{ver}/", f"./datasets/Dataset-ViPARE-{ver}/")
    # move the data.yaml file to the new directory
    os.rename(f"./datasets/Dataset-ViPARE-{ver}/data.yaml", f"./datasets/data.yaml")


def display_image(path):
    """
    Description:
        Displays a random image from a dataset with annotations.
    Usage:
        display_image(path) 
    Arguments:
        path: Path to the dataset folder split
    """
    im_path = os.path.join(path, '/images/')
    lab_path = os.path.join(path, '/labels/')
    
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

    
def dl_taco_dataset(ver = 3):
    """
    Description:
        Downloads the TACO dataset from roboflow Universe.
    Usage:
        dl_taco_dataset(dataset_version)
    Arguments:
        None
    """
    api_key = ""
    if os.path.exists("./roboflowAPIkey.txt"):
        with open("roboflowAPIkey.txt", "r") as f:
            api_key = f.readline().strip()
    else:
        print("No Roboflow API key file found, please create a roboflowAPIkey.txt file with your key in it.")
        return
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("naia-science").project("vipare-taco-class-match")
    dataset = project.version(ver).download("yolov8")

    # replace the following lines with python calls for creating directories and moving files
    os.makedirs(f"./datasets/", exist_ok=True)
    # move the downloaded dataset to the new directory
    os.rename(f"./vipare-taco-class-match-{ver}/", f"./datasets/Dataset-TACO-{ver}/")


def dl_robouni_dataset(ver = 4):
    """
    Description:
        Downloads the ROBOUNI dataset from roboflow Universe.
    Usage:
        dl_robouni_dataset(dataset_version)
    Arguments:
        None
    """
    api_key = ""
    if os.path.exists("./roboflowAPIkey.txt"):
        with open("roboflowAPIkey.txt", "r") as f:
            api_key = f.readline().strip()
    else:
        print("No Roboflow API key file found, please create a roboflowAPIkey.txt file with your key in it.")
        return
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("naia-science").project("vipare-robouni-objectdetect")
    dataset = project.version(ver).download("yolov8")

    os.makedirs(f"./datasets/", exist_ok=True)
    os.rename(f"./Vipare-RoboUni-ObjectDetect-{ver}/", f"./datasets/Dataset-ROBOUNI-{ver}/")



# -------------------------------------------------  Display util functions -------------------------------------------------
def colorFromClass(classID):
    """
    Description:
        Returns a color for a class ID. Colors are selected among the list available here : https://matplotlib.org/stable/gallery/color/named_colors.html
    Usage:
        colorFromClass(classID)
    Arguments:
        classID: Class ID
    """
    match classID:
        case '0':
            return 'lightgray', 'darkgray'
        case '1':
            return 'goldenrod', 'darkgoldenrod'
        case '2':
            return 'mediumblue', 'darkblue'
        case '3':
            return 'beige', 'darkkhaki'
        case '4':
            return 'lightskyblue', 'deepskyblue'
        case '5':
            return 'seagreen', 'darkgreen'
        case '6':
            return 'magenta', 'darkmagenta'
        case '7':
            return 'yellow', 'goldenrod'
        case '8':
            return 'blueviolet', 'rebeccapurple'
        case '9':
            return 'darkslategrey', 'black' 
        case '10':
            return 'darkorange', 'peru'
        case '11':
            return 'brown', 'darkred'
        case _:
            #here just in case, but should not happen
            return 'red', 'darkred'