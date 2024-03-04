#import datasetUtils module
import datasetUtils as du

import os
import shutil

if __name__ == '__main__':
    # Select roboflow version here
    roboflow_version = 2
    delete_datasets_after_merge = False

    if not (os.path.exists('./datasets')):
        os.mkdir('./datasets')

    if not (os.path.exists('./datasets/Dataset-ViPARE-' + str(roboflow_version) + '/images')):
        print('Downloading Roboflow dataset version ' + str(roboflow_version) + '...')
        du.dl_roboflow_dataset(roboflow_version)

    if not (os.path.exists('./datasets/TACO')):
        print('Downloading TACO dataset')
        du.dl_taco_dataset()

    if not (os.path.exists('./datasetsTACO/data/yolo')):
        print('Converting TACO dataset to YOLO format')
        du.cocoToYolo('./datasets/TACO/data')
        du.split_dataset('./datasets/TACO/data/yolo', 0.7, 0.2, 0.1)
        du.tacoClassesToNaia('./datasets/TACO/data/yolo/')

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

    if delete_datasets_after_merge:
        print('Deleting datasets')
        du.delete_roboflow_dataset()
        du.delete_TACO_dataset()

