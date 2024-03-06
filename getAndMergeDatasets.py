#import datasetUtils module
import datasetUtils as du

import os
import shutil


def launch_full_download(roboflow_version, delete_datasets_after_merge, get_all_fresh_datasets, outputPath):
    # Select roboflow version here

    if get_all_fresh_datasets:
        print('Deleting datasets')
        du.delete_roboflow_dataset()
        du.delete_TACO_dataset()
        du.delete_merged_datasets(outputPath)

    if not (os.path.exists('./datasets')):
        os.mkdir('./datasets')
    else:
        print('Datasets folder already exists')

    if not (os.path.exists('./datasets/Dataset-ViPARE-' + str(roboflow_version) + '/images')):
        print('Downloading Roboflow dataset version ' + str(roboflow_version) + '...')
        du.dl_roboflow_dataset(roboflow_version)
    else:
        print('Roboflow dataset version ' + str(roboflow_version) + ' already exists')

    if not (os.path.exists('./datasets/TACO')):
        print('Downloading TACO dataset')
        du.dl_taco_dataset()
    else:
        print('TACO dataset already exists')

    if not (os.path.exists('./datasetsTACO/data/yolo')):
        print('Converting TACO dataset to YOLO format')
        du.cocoToYolo('./datasets/TACO/data')
        du.split_dataset('./datasets/TACO/data/yolo', 0.7, 0.2, 0.1)
        du.tacoClassesToNaia('./datasets/TACO/data/yolo/')
    else:
        print('TACO dataset already in YOLO format')

    if not (os.path.exists('./datasets/mergeDataset')):
        print('Merging datasets')

        path = './datasets/Dataset-ViPARE-' + str(roboflow_version)
        # safe rename for VIPARE dataset and complete merge
        if os.path.exists(path + "/valid"):
            os.rename(path + "/valid", path + "/val")
            with open('./datasets/data.yaml', 'r') as f:
                lines = f.readlines()
            with open('./datasets/data.yaml', 'w') as f:
                for line in lines:
                    if "valid" in line :
                        f.write('val: ./Dataset-ViPARE-' + str(roboflow_version) + '/val/images')
                    else :
                        f.write(line)
            

        shutil.copy("./datasets/data.yaml", path + "/data.yaml")
        du.mergeDatasets('./datasets/Dataset-ViPARE-' + str(roboflow_version), './datasets/TACO/data/yolo', './datasets/mergeDataset')
    else:
        print('Merged dataset already exists, not modifying it')

    if delete_datasets_after_merge:
        print('Deleting datasets')
        du.delete_roboflow_dataset()
        du.delete_TACO_dataset()

if __name__ == '__main__':
    roboflow_version = 4
    delete_datasets_after_merge = False
    get_all_fresh_datasets = False
    outputPath = './datasets/mergeDataset'
    #TODO : add arge parser to get roboflow_version and delete_datasets_after_merge and get_all_fresh_datasets from command line
    launch_full_download(roboflow_version, delete_datasets_after_merge, get_all_fresh_datasets, outputPath)