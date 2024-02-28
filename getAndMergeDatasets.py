#import datasetUtils module
import datasetUtils as du

import os

if __name__ == '__main__':
    # Select roboflow version here
    roboflow_version = 2
    delete_datasets_after_merge = False

    if not (os.path.exists('./datasets')):
        print('Downloading Roboflow dataset')
        du.dl_roboflow_dataset(roboflow_version)

    if not (os.path.exists('./TACO')):
        print('Downloading TACO dataset')
        du.dl_taco_dataset()

    if not (os.path.exists('./TACO/data/yolo')):
        print('Converting TACO dataset to YOLO format')
        du.cocoToYolo('./TACO/data')
        du.split_dataset('./TACO/data/yolo', 0.7, 0.2, 0.1)
        du.tacoClassesToNaia('./TACO/data/yolo/')

    if not (os.path.exists('./mergeDataset')):
        print('Merging datasets')
        du.mergeDatasets('./datasets', './TACO/data/yolo', './mergeDataset')

    if delete_datasets_after_merge:
        print('Deleting datasets')
        du.delete_roboflow_dataset()
        du.delete_TACO_dataset()

